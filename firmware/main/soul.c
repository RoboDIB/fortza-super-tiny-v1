/*
 * FORTZA Soul — implementation
 */
#include "soul.h"
#include <string.h>
#include <math.h>
#include "nvs_flash.h"
#include "nvs.h"
#include "esp_log.h"

#define TAG        "fortza.soul"
#define NVS_NS     "fortza"
#define NVS_KEY    "soul"

/* ------------------------------------------------------------------ */

void soul_init(SoulState *s, float fou)
{
    memset(s->fortza, 0, sizeof(s->fortza));
    memset(s->ds,     0, sizeof(s->ds));
    s->fou        = fou;
    s->turn_steps = 0;
}

/* ------------------------------------------------------------------ */

void soul_load(SoulState *s)
{
    nvs_handle_t h;
    esp_err_t err = nvs_open(NVS_NS, NVS_READONLY, &h);
    if (err != ESP_OK) return;   /* first boot, no soul saved yet */

    size_t len = SOUL_DIM * sizeof(float);
    err = nvs_get_blob(h, NVS_KEY, s->fortza, &len);
    nvs_close(h);

    if (err == ESP_OK)
        ESP_LOGI(TAG, "soul loaded from NVS");
    else
        ESP_LOGI(TAG, "no soul in NVS, starting fresh");
}

/* ------------------------------------------------------------------ */

void soul_save(const SoulState *s)
{
    nvs_handle_t h;
    if (nvs_open(NVS_NS, NVS_READWRITE, &h) != ESP_OK) return;
    nvs_set_blob(h, NVS_KEY, s->fortza, SOUL_DIM * sizeof(float));
    nvs_commit(h);
    nvs_close(h);
}

/* ------------------------------------------------------------------ */

void soul_zero_grad(SoulState *s)
{
    memset(s->ds, 0, sizeof(s->ds));
    s->turn_steps = 0;
}

/* ------------------------------------------------------------------ */

void soul_accum_grad(SoulState *s,
                     const float *out_W_soul_cols,
                     const float *probs,
                     uint16_t chosen,
                     uint32_t vocab_size)
{
    /*
     * Cross-entropy gradient w.r.t. soul vector:
     *   err[i] = probs[i] - (i == chosen ? 1.0 : 0.0)
     *   ds    += out_W_soul_cols^T · err
     *
     * out_W_soul_cols layout: [vocab_size][SOUL_DIM]
     * i.e. row i = output weight row for token i, soul columns only
     */
    for (int j = 0; j < SOUL_DIM; j++) {
        float grad = 0.0f;
        for (uint32_t i = 0; i < vocab_size; i++) {
            float err = probs[i] - (i == (uint32_t)chosen ? 1.0f : 0.0f);
            grad += out_W_soul_cols[i * SOUL_DIM + j] * err;
        }
        s->ds[j] += grad;
    }
    s->turn_steps++;
}

/* ------------------------------------------------------------------ */

void rul_update(SoulState *s, int feedback)
{
    if (s->turn_steps == 0) return;

    float scale = s->fou / (float)s->turn_steps;
    if (feedback == -1) scale = -scale;   /* inhibit: push away */

    /* Apply gradient */
    for (int i = 0; i < SOUL_DIM; i++)
        s->fortza[i] -= scale * s->ds[i];

    /* L2 decay */
    for (int i = 0; i < SOUL_DIM; i++)
        s->fortza[i] *= (1.0f - SOUL_DECAY);

    /* Clip norm */
    float norm_sq = 0.0f;
    for (int i = 0; i < SOUL_DIM; i++)
        norm_sq += s->fortza[i] * s->fortza[i];
    float norm = sqrtf(norm_sq);
    if (norm > SOUL_CLIP) {
        float inv = SOUL_CLIP / norm;
        for (int i = 0; i < SOUL_DIM; i++)
            s->fortza[i] *= inv;
    }

    soul_zero_grad(s);
}
