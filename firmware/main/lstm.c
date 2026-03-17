/*
 * fortza-super-tiny
 * INT8 LSTM cell — implementation
 * No floats in the inner loop except activation functions.
 */
#include "lstm.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* 256-entry sigmoid lookup, built once at startup */
static float _sig_lut[256];
static float _tanh_lut[256];
static int   _lut_ready = 0;

static void build_luts(void) {
    if (_lut_ready) return;
    for (int i = 0; i < 256; i++) {
        float x = (i - 128) / 16.0f;   /* maps 0-255 → ~-8..+8 */
        _sig_lut[i]  = 1.0f / (1.0f + expf(-x));
        _tanh_lut[i] = tanhf(x);
    }
    _lut_ready = 1;
}

static inline float lut_sigmoid(float x) {
    int idx = (int)(x * 16.0f + 128.5f);
    if (idx < 0)   idx = 0;
    if (idx > 255) idx = 255;
    return _sig_lut[idx];
}

static inline float lut_tanh(float x) {
    int idx = (int)(x * 16.0f + 128.5f);
    if (idx < 0)   idx = 0;
    if (idx > 255) idx = 255;
    return _tanh_lut[idx];
}

static inline float dequant(int8_t q, float scale, float zp) {
    return (q + 128) * scale + zp;
}

void lstm_init(LSTMState *s, int input_size, int hidden_size,
               int8_t *W, float W_scale, float W_zp,
               int8_t *b, float b_scale, float b_zp) {
    build_luts();
    s->input_size  = input_size;
    s->hidden_size = hidden_size;
    s->W = W;  s->W_scale = W_scale;  s->W_zp = W_zp;
    s->b = b;  s->b_scale = b_scale;  s->b_zp = b_zp;
    s->h = calloc(hidden_size, sizeof(float));
    s->c = calloc(hidden_size, sizeof(float));
}

void lstm_reset(LSTMState *s) {
    memset(s->h, 0, s->hidden_size * sizeof(float));
    memset(s->c, 0, s->hidden_size * sizeof(float));
}

void lstm_step(LSTMState *s, const float *x) {
    int I = s->input_size;
    int H = s->hidden_size;

    /* gates[4*H]: f, i, g, o */
    float *gates = malloc(4 * H * sizeof(float));

    for (int row = 0; row < 4 * H; row++) {
        float acc = dequant(s->b[row], s->b_scale, s->b_zp);
        /* input part */
        for (int col = 0; col < I; col++) {
            if (x[col] != 0.0f) {   /* one-hot: only one nonzero */
                acc += dequant(s->W[row * (I + H) + col], s->W_scale, s->W_zp) * x[col];
            }
        }
        /* hidden part */
        for (int col = 0; col < H; col++) {
            acc += dequant(s->W[row * (I + H) + I + col], s->W_scale, s->W_zp) * s->h[col];
        }
        gates[row] = acc;
    }

    for (int j = 0; j < H; j++) {
        float f = lut_sigmoid(gates[j]);
        float i = lut_sigmoid(gates[H + j]);
        float g = lut_tanh   (gates[2*H + j]);
        float o = lut_sigmoid(gates[3*H + j]);
        s->c[j] = f * s->c[j] + i * g;
        s->h[j] = o * lut_tanh(s->c[j]);
    }

    free(gates);
}
