/*
 * FORTZA Learner — implementation
 */
#include "learner.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "esp_log.h"

#define TAG "fortza.learner"

/* ------------------------------------------------------------------ */
/* Feedback keyword tables                                              */
/* ------------------------------------------------------------------ */

static const char *POSITIVE[] = {
    "good", "yes", "nice", "correct", "right", "great",
    "perfect", "exactly", "yep", "sure", NULL
};
static const char *NEGATIVE[] = {
    "no", "wrong", "bad", "nope", "not right", "stop",
    "incorrect", "nah", "that's wrong", NULL
};

int learner_detect_feedback(const char *input)
{
    for (int i = 0; POSITIVE[i]; i++)
        if (strstr(input, POSITIVE[i])) return +1;
    for (int i = 0; NEGATIVE[i]; i++)
        if (strstr(input, NEGATIVE[i])) return -1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Init                                                                 */
/* ------------------------------------------------------------------ */

void learner_init(LearnerState *L, FORTZAModel *model)
{
    memset(L, 0, sizeof(*L));
    L->model    = model;
    L->wo_dirty = false;

    soul_init(&L->soul, FOU_DEFAULT);
    soul_load(&L->soul);

    /* Dequantize soul columns of out_W into float32 for rul gradient.
     * out_W shape: (vocab, dec_hidden + soul_dim)
     * Soul columns: last soul_dim columns of each row.
     */
    uint32_t V = model->vocab_size;
    uint32_t D = model->dec_hidden;
    uint32_t S = model->soul_dim;

    L->out_W_soul_cols = malloc(V * S * sizeof(float));
    for (uint32_t row = 0; row < V; row++) {
        for (uint32_t col = 0; col < S; col++) {
            int8_t q = model->out_W[row * (D + S) + D + col];
            L->out_W_soul_cols[row * S + col] =
                (float)(q + 128) * model->out_W_scale + model->out_W_zp;
        }
    }

    /* Try to load previously learned Wo */
    learner_load_wo(L);
}

void learner_free(LearnerState *L)
{
    free(L->out_W_soul_cols);
    free(L->dWo);
    free(L->dbo);
}

/* ------------------------------------------------------------------ */
/* Per-turn lifecycle                                                   */
/* ------------------------------------------------------------------ */

void learner_begin_turn(LearnerState *L)
{
    soul_zero_grad(&L->soul);
    /* Zero gradient accumulators if allocated */
    if (L->dWo) {
        uint32_t V = L->model->vocab_size;
        uint32_t cols = L->model->dec_hidden + L->model->soul_dim;
        memset(L->dWo, 0, V * cols * sizeof(float));
        memset(L->dbo, 0, V * sizeof(float));
    }
}

void learner_step(LearnerState *L,
                  const float *h_dec,
                  const float *probs,
                  uint16_t chosen)
{
    uint32_t V = L->model->vocab_size;
    uint32_t D = L->model->dec_hidden;
    uint32_t S = L->model->soul_dim;

    /* Save h_combined = cat(h_dec, soul) for fine-tune gradient */
    memcpy(L->h_combined,     h_dec,          D * sizeof(float));
    memcpy(L->h_combined + D, L->soul.fortza, S * sizeof(float));

    /* RUL: accumulate soul gradient */
    soul_accum_grad(&L->soul, L->out_W_soul_cols, probs, chosen, V);

    /* Fine-tune gradient accumulation (only if buffers allocated) */
    if (L->dWo) {
        for (uint32_t i = 0; i < V; i++) {
            float err = probs[i] - (i == (uint32_t)chosen ? 1.0f : 0.0f);
            /* dWo[i,:] += err * h_combined */
            for (uint32_t j = 0; j < D + S; j++)
                L->dWo[i * (D + S) + j] += err * L->h_combined[j];
            L->dbo[i] += err;
        }
    }
}

/* ------------------------------------------------------------------ */
/* End of turn — apply learning                                         */
/* ------------------------------------------------------------------ */

static void apply_output_finetune(LearnerState *L)
{
    FORTZAModel *m = L->model;
    uint32_t V     = m->vocab_size;
    uint32_t cols  = m->dec_hidden + m->soul_dim;
    int      T     = L->soul.turn_steps > 0 ? L->soul.turn_steps : 1;

    /* Lazy-allocate fine-tune buffers */
    if (!L->dWo) {
        L->dWo = calloc(V * cols, sizeof(float));
        L->dbo = calloc(V,        sizeof(float));
        if (!L->dWo || !L->dbo) {
            ESP_LOGE(TAG, "OOM: cannot allocate fine-tune buffers");
            return;
        }
    }

    /* Dequantize current Wo to float32 */
    float *Wo_f = malloc(V * cols * sizeof(float));
    float *bo_f = malloc(V * sizeof(float));
    if (!Wo_f || !bo_f) { free(Wo_f); free(bo_f); return; }

    for (uint32_t i = 0; i < V * cols; i++)
        Wo_f[i] = (float)(m->out_W[i] + 128) * m->out_W_scale + m->out_W_zp;
    for (uint32_t i = 0; i < V; i++)
        bo_f[i] = (float)(m->out_b[i] + 128) * m->out_b_scale + m->out_b_zp;

    /* SGD step */
    float lr = LR_OUTPUT / (float)T;
    for (uint32_t i = 0; i < V * cols; i++)
        Wo_f[i] -= lr * L->dWo[i];
    for (uint32_t i = 0; i < V; i++)
        bo_f[i] -= lr * L->dbo[i];

    /* Re-quantize Wo back to INT8 */
    float mn = Wo_f[0], mx = Wo_f[0];
    for (uint32_t i = 1; i < V * cols; i++) {
        if (Wo_f[i] < mn) mn = Wo_f[i];
        if (Wo_f[i] > mx) mx = Wo_f[i];
    }
    float scale = (mx - mn) / 255.0f;
    if (scale == 0.0f) scale = 1.0f;
    m->out_W_scale = scale;
    m->out_W_zp    = mn;
    for (uint32_t i = 0; i < V * cols; i++) {
        int q = (int)roundf((Wo_f[i] - mn) / scale) - 128;
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        m->out_W[i] = (int8_t)q;
    }

    /* Update soul columns float32 cache */
    uint32_t D = m->dec_hidden, S = m->soul_dim;
    for (uint32_t row = 0; row < V; row++)
        for (uint32_t col = 0; col < S; col++) {
            int8_t qv = m->out_W[row * (D + S) + D + col];
            L->out_W_soul_cols[row * S + col] =
                (float)(qv + 128) * m->out_W_scale + m->out_W_zp;
        }

    free(Wo_f);
    free(bo_f);
    L->wo_dirty = true;
    ESP_LOGI(TAG, "output layer fine-tuned and re-quantized");
}

void learner_end_turn(LearnerState *L, int feedback)
{
    rul_update(&L->soul, feedback);
    soul_save(&L->soul);

    if (feedback == +1)
        apply_output_finetune(L);

    if (L->wo_dirty)
        learner_save_wo(L);
}

/* ------------------------------------------------------------------ */
/* SPIFFS Wo persistence                                                */
/* ------------------------------------------------------------------ */

void learner_save_wo(LearnerState *L)
{
    FORTZAModel *m = L->model;
    uint32_t V     = m->vocab_size;
    uint32_t cols  = m->dec_hidden + m->soul_dim;

    FILE *f = fopen(LEARNED_WO_PATH, "wb");
    if (!f) { ESP_LOGW(TAG, "cannot write %s", LEARNED_WO_PATH); return; }

    fwrite(&V,              4, 1, f);
    fwrite(&cols,           4, 1, f);
    fwrite(&m->out_W_scale, 4, 1, f);
    fwrite(&m->out_W_zp,    4, 1, f);
    fwrite(m->out_W,        1, V * cols, f);
    fwrite(m->out_b,        1, V,        f);
    fclose(f);
    ESP_LOGI(TAG, "saved learned Wo to %s", LEARNED_WO_PATH);
}

int learner_load_wo(LearnerState *L)
{
    FORTZAModel *m = L->model;
    FILE *f = fopen(LEARNED_WO_PATH, "rb");
    if (!f) return -1;   /* no learned weights yet — fine */

    uint32_t V, cols;
    fread(&V,    4, 1, f);
    fread(&cols, 4, 1, f);
    if (V != m->vocab_size || cols != m->dec_hidden + m->soul_dim) {
        ESP_LOGW(TAG, "learned Wo shape mismatch, ignoring");
        fclose(f); return -2;
    }
    fread(&m->out_W_scale, 4, 1, f);
    fread(&m->out_W_zp,    4, 1, f);
    fread(m->out_W,        1, V * cols, f);
    fread(m->out_b,        1, V,        f);
    fclose(f);
    ESP_LOGI(TAG, "loaded learned Wo from %s", LEARNED_WO_PATH);
    return 0;
}
