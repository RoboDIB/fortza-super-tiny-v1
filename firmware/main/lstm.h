/*
 * fortza-super-tiny
 * INT8 LSTM cell — header
 */
#pragma once
#include <stdint.h>

typedef struct {
    int     input_size;
    int     hidden_size;
    /* weights: (4*H, I+H) stored row-major, INT8 */
    int8_t *W;
    float   W_scale;
    float   W_zp;
    /* bias: (4*H,), INT8 */
    int8_t *b;
    float   b_scale;
    float   b_zp;
    /* working buffers allocated once */
    float  *h;
    float  *c;
} LSTMState;

void lstm_init(LSTMState *s, int input_size, int hidden_size,
               int8_t *W, float W_scale, float W_zp,
               int8_t *b, float b_scale, float b_zp);

void lstm_reset(LSTMState *s);

/* one step: x is one-hot float vec of size input_size */
void lstm_step(LSTMState *s, const float *x);
