/*
 * FORTZA Model — inference engine header
 *
 * Components:
 *   fortza     soul vector (32 floats, persistent, learned via rul)
 *   zia        n-gram context hash (parameter-free)
 *   long_hair  asymmetric decoder (hidden=128)
 *   rul        on-device learning (soul.h)
 *   fou        soul drift factor (learner.h)
 */
#pragma once
#include <stdint.h>
#include "lstm.h"
#include "ngram.h"

#define FZSU_MAGIC       0x465A5355u
#define SOUL_DIM         32
#define ENC_HIDDEN       64
#define DEC_HIDDEN       128   /* long_hair */
#define MAX_VOCAB        128

typedef struct {
    uint32_t vocab_size;
    uint32_t enc_hidden;     /* 64  */
    uint32_t dec_hidden;     /* 128 — long_hair */
    uint32_t soul_dim;       /* 32  — fortza */
    uint32_t ngram_buckets;  /* 32  — zia */

    /* Encoder LSTM */
    LSTMState encoder;

    /* Soul gate: Linear(enc+soul → soul*2) */
    int8_t  *sg_W;
    float    sg_W_scale, sg_W_zp;
    int8_t  *sg_b;
    float    sg_b_scale, sg_b_zp;

    /* Wc bridge: Linear(enc+ngram → dec) */
    int8_t  *Wc_W;
    float    Wc_W_scale, Wc_W_zp;
    int8_t  *Wc_b;
    float    Wc_b_scale, Wc_b_zp;

    /* Decoder LSTM (long_hair) */
    LSTMState long_hair;

    /* Output: Linear(dec+soul → vocab) */
    int8_t  *out_W;
    float    out_W_scale, out_W_zp;
    int8_t  *out_b;
    float    out_b_scale, out_b_zp;
} FORTZAModel;

/* Load weights.bin from SPIFFS. Returns 0 on success. */
int  fortza_model_load(FORTZAModel *m, const char *path);
void fortza_model_free(FORTZAModel *m);

/*
 * Encode user input. Fills encoder.h/c and computes zia_out.
 * zia_out: caller-provided float[NGRAM_BUCKETS].
 */
void fortza_encode(FORTZAModel *m,
                   const uint16_t *indices, int len,
                   float *zia_out);

/*
 * Update fortza (soul vector) using soul gate.
 * soul: float[SOUL_DIM], updated in-place.
 */
void fortza_soul_gate(FORTZAModel *m,
                      const float *h_enc,
                      float *soul);

/*
 * Initialize long_hair (decoder) state from encoder + zia via Wc bridge.
 * Call after fortza_encode + fortza_soul_gate.
 */
void fortza_bridge(FORTZAModel *m,
                   const float *h_enc,
                   const float *zia_feats);

/*
 * One decoder step. Returns predicted token index.
 * probs_out: optional float[vocab_size] — probabilities for rul learning.
 *            Pass NULL to skip (greedy only, no learning).
 * soul:      float[SOUL_DIM] — current fortza, injected at output.
 */
uint16_t fortza_decode_step(FORTZAModel *m,
                             uint16_t prev_idx,
                             const float *soul,
                             float *probs_out);
