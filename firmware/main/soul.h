/*
 * FORTZA Soul — persistent conversation identity
 *
 * fortza[] lives in SRAM, survives power cycles via NVS.
 * rul_update() applies the gradient after each conversation turn.
 * fou controls how fast the soul drifts (learning rate).
 */
#pragma once
#include <stdint.h>
#include <stdbool.h>

#define SOUL_DIM    32
#define FOU_DEFAULT 0.01f   /* soul learning rate (fou) */
#define SOUL_DECAY  0.001f
#define SOUL_CLIP   2.0f

typedef struct {
    float  fortza[SOUL_DIM];   /* persistent soul vector */
    float  ds[SOUL_DIM];       /* gradient accumulator for current turn */
    float  fou;                /* drift factor — soul learning rate */
    int    turn_steps;         /* decode steps accumulated this turn */
} SoulState;

/* Initialize with default fou. Call once at boot. */
void soul_init(SoulState *s, float fou);

/* Load soul from NVS. No-op if not found (soul stays zero). */
void soul_load(SoulState *s);

/* Save soul to NVS. Call after each turn. */
void soul_save(const SoulState *s);

/* Reset gradient accumulator at start of each turn. */
void soul_zero_grad(SoulState *s);

/*
 * Accumulate soul gradient for one decode step.
 * out_W_soul_cols: float[vocab * SOUL_DIM] — soul columns of output weight
 * probs:           float[vocab_size] — softmax output this step
 * chosen:          token chosen this step
 * vocab_size:      number of tokens
 */
void soul_accum_grad(SoulState *s,
                     const float *out_W_soul_cols,
                     const float *probs,
                     uint16_t chosen,
                     uint32_t vocab_size);

/*
 * RUL — Residual Update Loop
 * Apply accumulated gradient to fortza. Call once per turn after decode.
 * feedback: +1 reinforce, -1 inhibit, 0 implicit (self-supervised)
 */
void rul_update(SoulState *s, int feedback);
