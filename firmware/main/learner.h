/*
 * FORTZA Learner — output layer fine-tuning + feedback detection
 *
 * On explicit positive feedback ("good", "yes", "nice" etc.):
 *   - Lazily allocates float32 Wo gradient buffer (~31KB)
 *   - Runs one SGD step on the output projection
 *   - Re-quantizes Wo to INT8 in-place
 *   - Saves updated Wo to /spiffs/learned_weights.bin
 *   - Frees gradient buffers
 *
 * On negative feedback ("no", "wrong", "bad" etc.):
 *   - Calls rul_update with feedback=-1 to inhibit soul
 */
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "soul.h"
#include "fortza_model.h"

#define LR_OUTPUT       1e-4f
#define LEARNED_WO_PATH "/spiffs/learned_wo.bin"

typedef struct {
    SoulState       soul;
    FORTZAModel    *model;       /* pointer to main model */

    /* Soul columns of Wo in float32 (vocab * SOUL_DIM) — for rul grad */
    float          *out_W_soul_cols;   /* allocated once at init */

    /* Fine-tune state (lazy, allocated only on positive feedback) */
    float          *dWo;         /* gradient accumulator (vocab * (dec+soul)) */
    float          *dbo;         /* bias gradient (vocab) */
    bool            wo_dirty;    /* true if Wo has been fine-tuned this session */

    /* h_combined buffer for gradient: (dec_hidden + soul_dim) floats */
    float           h_combined[DEC_HIDDEN + SOUL_DIM];
} LearnerState;

/* Init. Dequantizes soul columns of Wo into float32 for rul grad. */
void learner_init(LearnerState *L, FORTZAModel *model);
void learner_free(LearnerState *L);

/* Call at start of each turn. */
void learner_begin_turn(LearnerState *L);

/*
 * Call during decode for each step.
 * h_dec:  decoder hidden state this step (DEC_HIDDEN floats)
 * probs:  softmax output (vocab_size floats)
 * chosen: token selected
 */
void learner_step(LearnerState *L,
                  const float *h_dec,
                  const float *probs,
                  uint16_t chosen);

/*
 * Call after decode. Applies rul to soul.
 * feedback: +1 positive, -1 negative, 0 implicit
 * If positive: also runs output layer SGD.
 */
void learner_end_turn(LearnerState *L, int feedback);

/* Save fine-tuned Wo to SPIFFS if dirty. */
void learner_save_wo(LearnerState *L);

/* Load previously fine-tuned Wo from SPIFFS into model. */
int  learner_load_wo(LearnerState *L);

/* Feedback detection from raw user string. Returns +1, -1, or 0. */
int  learner_detect_feedback(const char *input);
