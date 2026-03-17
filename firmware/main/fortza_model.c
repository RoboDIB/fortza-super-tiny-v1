/*
 * FORTZA Model — inference engine
 */
#include "fortza_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "esp_spiffs.h"

#define TAG "fortza"

/* ------------------------------------------------------------------ */
/* File helpers                                                         */
/* ------------------------------------------------------------------ */

static void rd_u32(FILE *f, uint32_t *v) { fread(v, 4, 1, f); }
static void rd_f32(FILE *f, float    *v) { fread(v, 4, 1, f); }

static int8_t *rd_layer(FILE *f,
                         uint32_t *rows_out, uint32_t *cols_out,
                         float *scale, float *zp)
{
    uint32_t r, c;
    rd_u32(f, &r); rd_u32(f, &c);
    rd_f32(f, scale); rd_f32(f, zp);
    int8_t *d = malloc(r * c);
    fread(d, 1, r * c, f);
    if (rows_out) *rows_out = r;
    if (cols_out) *cols_out = c;
    return d;
}

/* ------------------------------------------------------------------ */
/* Load                                                                 */
/* ------------------------------------------------------------------ */

int fortza_model_load(FORTZAModel *m, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { ESP_LOGE(TAG, "cannot open %s", path); return -1; }

    uint32_t magic, vocab, enc_h, dec_h, soul_d, ngram_b, n_layers;
    rd_u32(f, &magic);
    if (magic != FZSU_MAGIC) {
        ESP_LOGE(TAG, "bad magic 0x%08X (expected 0x%08X)", magic, FZSU_MAGIC);
        fclose(f); return -2;
    }
    rd_u32(f, &vocab);
    rd_u32(f, &enc_h);
    rd_u32(f, &dec_h);
    rd_u32(f, &soul_d);
    rd_u32(f, &ngram_b);
    rd_u32(f, &n_layers);

    m->vocab_size    = vocab;
    m->enc_hidden    = enc_h;
    m->dec_hidden    = dec_h;
    m->soul_dim      = soul_d;
    m->ngram_buckets = ngram_b;

    float sc, zp;
    uint32_t r, c;

    /* 0: enc_W, 1: enc_b */
    int8_t *enc_W = rd_layer(f, &r, &c, &sc, &zp);
    lstm_init(&m->encoder, vocab + ngram_b, enc_h,
              enc_W, sc, zp, NULL, 0, 0);
    int8_t *enc_b = rd_layer(f, NULL, NULL, &sc, &zp);
    m->encoder.b = enc_b; m->encoder.b_scale = sc; m->encoder.b_zp = zp;

    /* 2: sg_W, 3: sg_b */
    m->sg_W = rd_layer(f, NULL, NULL, &m->sg_W_scale, &m->sg_W_zp);
    m->sg_b = rd_layer(f, NULL, NULL, &m->sg_b_scale, &m->sg_b_zp);

    /* 4: Wc_W, 5: Wc_b */
    m->Wc_W = rd_layer(f, NULL, NULL, &m->Wc_W_scale, &m->Wc_W_zp);
    m->Wc_b = rd_layer(f, NULL, NULL, &m->Wc_b_scale, &m->Wc_b_zp);

    /* 6: dec_W, 7: dec_b — long_hair */
    int8_t *dec_W = rd_layer(f, &r, &c, &sc, &zp);
    lstm_init(&m->long_hair, vocab, dec_h,
              dec_W, sc, zp, NULL, 0, 0);
    int8_t *dec_b = rd_layer(f, NULL, NULL, &sc, &zp);
    m->long_hair.b = dec_b; m->long_hair.b_scale = sc; m->long_hair.b_zp = zp;

    /* 8: out_W, 9: out_b */
    m->out_W = rd_layer(f, NULL, NULL, &m->out_W_scale, &m->out_W_zp);
    m->out_b = rd_layer(f, NULL, NULL, &m->out_b_scale, &m->out_b_zp);

    fclose(f);
    ESP_LOGI(TAG, "model loaded: vocab=%u enc=%u dec=%u soul=%u",
             vocab, enc_h, dec_h, soul_d);
    return 0;
}

void fortza_model_free(FORTZAModel *m)
{
    free(m->encoder.W);   free(m->encoder.b);
    free(m->encoder.h);   free(m->encoder.c);
    free(m->sg_W);        free(m->sg_b);
    free(m->Wc_W);        free(m->Wc_b);
    free(m->long_hair.W); free(m->long_hair.b);
    free(m->long_hair.h); free(m->long_hair.c);
    free(m->out_W);       free(m->out_b);
}

/* ------------------------------------------------------------------ */
/* Dequant helper                                                        */
/* ------------------------------------------------------------------ */

static inline float dq(int8_t q, float scale, float zp)
{
    return (float)(q + 128) * scale + zp;
}

/* INT8 matrix-vector multiply: y[rows] = W[rows,cols] * x[cols] + b[rows] */
static void matvec(const int8_t *W, float W_sc, float W_zp,
                   const int8_t *b, float b_sc, float b_zp,
                   const float  *x, int rows, int cols,
                   float *y)
{
    for (int r = 0; r < rows; r++) {
        float acc = dq(b[r], b_sc, b_zp);
        for (int c = 0; c < cols; c++)
            acc += dq(W[r * cols + c], W_sc, W_zp) * x[c];
        y[r] = acc;
    }
}

/* ------------------------------------------------------------------ */
/* Encode                                                               */
/* ------------------------------------------------------------------ */

void fortza_encode(FORTZAModel *m,
                   const uint16_t *indices, int len,
                   float *zia_out)
{
    uint32_t V = m->vocab_size;
    uint32_t N = m->ngram_buckets;

    zia(indices, len, zia_out);
    lstm_reset(&m->encoder);

    float *x = calloc(V + N, sizeof(float));
    for (int t = 0; t < len; t++) {
        memset(x, 0, (V + N) * sizeof(float));
        if (indices[t] < V) x[indices[t]] = 1.0f;
        memcpy(x + V, zia_out, N * sizeof(float));
        lstm_step(&m->encoder, x);
    }
    free(x);
}

/* ------------------------------------------------------------------ */
/* Soul gate — update fortza                                            */
/* ------------------------------------------------------------------ */

void fortza_soul_gate(FORTZAModel *m,
                      const float *h_enc,
                      float *soul)
{
    uint32_t E = m->enc_hidden;
    uint32_t S = m->soul_dim;
    uint32_t in_dim = E + S;

    float *combined = malloc(in_dim * sizeof(float));
    memcpy(combined,     h_enc, E * sizeof(float));
    memcpy(combined + E, soul,  S * sizeof(float));

    float *out = malloc(S * 2 * sizeof(float));
    matvec(m->sg_W, m->sg_W_scale, m->sg_W_zp,
           m->sg_b, m->sg_b_scale, m->sg_b_zp,
           combined, S * 2, in_dim, out);

    /* gate = sigmoid(out[0..S]), update = tanh(out[S..2S]) */
    for (int i = 0; i < (int)S; i++) {
        float gate   = 1.0f / (1.0f + expf(-out[i]));
        float update = tanhf(out[S + i]);
        soul[i] = gate * soul[i] + (1.0f - gate) * update;
    }

    free(combined);
    free(out);
}

/* ------------------------------------------------------------------ */
/* Wc bridge — init long_hair from encoder + zia                        */
/* ------------------------------------------------------------------ */

void fortza_bridge(FORTZAModel *m,
                   const float *h_enc,
                   const float *zia_feats)
{
    uint32_t E = m->enc_hidden;
    uint32_t N = m->ngram_buckets;
    uint32_t D = m->dec_hidden;
    uint32_t in_dim = E + N;

    float *ctx = malloc(in_dim * sizeof(float));
    memcpy(ctx,     h_enc,    E * sizeof(float));
    memcpy(ctx + E, zia_feats, N * sizeof(float));

    float *dh = m->long_hair.h;
    matvec(m->Wc_W, m->Wc_W_scale, m->Wc_W_zp,
           m->Wc_b, m->Wc_b_scale, m->Wc_b_zp,
           ctx, D, in_dim, dh);
    for (int i = 0; i < (int)D; i++) dh[i] = tanhf(dh[i]);
    memset(m->long_hair.c, 0, D * sizeof(float));

    free(ctx);
}

/* ------------------------------------------------------------------ */
/* Decode step                                                          */
/* ------------------------------------------------------------------ */

uint16_t fortza_decode_step(FORTZAModel *m,
                             uint16_t prev_idx,
                             const float *soul,
                             float *probs_out)
{
    uint32_t V = m->vocab_size;
    uint32_t D = m->dec_hidden;
    uint32_t S = m->soul_dim;

    /* One-hot input for long_hair decoder */
    float *x = calloc(V, sizeof(float));
    if (prev_idx < V) x[prev_idx] = 1.0f;
    lstm_step(&m->long_hair, x);
    free(x);

    /* Output: cat(h_dec, soul) → logits */
    float *h_soul = malloc((D + S) * sizeof(float));
    memcpy(h_soul,     m->long_hair.h, D * sizeof(float));
    memcpy(h_soul + D, soul,           S * sizeof(float));

    float *logits = malloc(V * sizeof(float));
    matvec(m->out_W, m->out_W_scale, m->out_W_zp,
           m->out_b, m->out_b_scale, m->out_b_zp,
           h_soul, V, D + S, logits);

    /* Softmax + argmax */
    float max_v = logits[0];
    for (uint32_t i = 1; i < V; i++) if (logits[i] > max_v) max_v = logits[i];

    float sum = 0.0f;
    for (uint32_t i = 0; i < V; i++) {
        logits[i] = expf(logits[i] - max_v);
        sum       += logits[i];
    }

    uint16_t best = 0;
    float best_p  = 0.0f;
    for (uint32_t i = 0; i < V; i++) {
        logits[i] /= sum;
        if (probs_out) probs_out[i] = logits[i];
        if (logits[i] > best_p) { best_p = logits[i]; best = (uint16_t)i; }
    }

    free(logits);
    free(h_soul);
    return best;
}
