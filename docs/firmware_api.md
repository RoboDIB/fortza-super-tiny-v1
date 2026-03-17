# Firmware API Reference

## fortza_model.h — Inference Engine

### `fortza_model_load`
```c
int fortza_model_load(FORTZAModel *m, const char *path);
```
Load `weights.bin` from SPIFFS. Returns 0 on success, negative on error.

---

### `fortza_encode`
```c
void fortza_encode(FORTZAModel *m,
                   const uint16_t *indices, int len,
                   float *zia_out);
```
Encode user input. Runs ZIA hash then encoder LSTM.
- `indices`: token index array (from `char_to_idx`)
- `len`: number of tokens
- `zia_out`: caller-allocated `float[NGRAM_BUCKETS]`, filled with ZIA features

After this call, `m->encoder.h` holds the final encoder hidden state.

---

### `fortza_soul_gate`
```c
void fortza_soul_gate(FORTZAModel *m,
                      const float *h_enc,
                      float *soul);
```
Update the soul vector in-place.
- `h_enc`: encoder final hidden state (`float[ENC_HIDDEN]`)
- `soul`: current soul, updated in-place (`float[SOUL_DIM]`)

---

### `fortza_bridge`
```c
void fortza_bridge(FORTZAModel *m,
                   const float *h_enc,
                   const float *zia_feats);
```
Initialize decoder (long_hair) state from encoder output + ZIA. Call after `fortza_soul_gate`.

---

### `fortza_decode_step`
```c
uint16_t fortza_decode_step(FORTZAModel *m,
                             uint16_t prev_idx,
                             const float *soul,
                             float *probs_out);
```
One autoregressive decode step.
- `prev_idx`: previous token (use `idx_start` for first step)
- `soul`: current fortza vector (injected at output)
- `probs_out`: optional `float[vocab_size]` — softmax probabilities (for RUL). Pass `NULL` to skip.
- Returns: predicted next token index

---

## soul.h — Soul Vector + RUL

### `soul_init`
```c
void soul_init(SoulState *s, float fou);
```
Initialize soul state with given drift factor. Call once at boot.

---

### `soul_load` / `soul_save`
```c
void soul_load(SoulState *s);
void soul_save(const SoulState *s);
```
Load/save soul from/to NVS. `soul_load` is a no-op if no soul has been saved yet (fresh device).

---

### `soul_zero_grad`
```c
void soul_zero_grad(SoulState *s);
```
Reset gradient accumulator `ds` to zero. Call at start of each turn.

---

### `soul_accum_grad`
```c
void soul_accum_grad(SoulState *s,
                     const float *out_W_soul_cols,
                     const float *probs,
                     uint16_t chosen,
                     uint32_t vocab_size);
```
Accumulate soul gradient for one decode step.
- `out_W_soul_cols`: soul columns of output weight matrix (`float[vocab_size * SOUL_DIM]`)
- `probs`: softmax output this step (`float[vocab_size]`)
- `chosen`: token index selected this step

---

### `rul_update`
```c
void rul_update(SoulState *s, int feedback);
```
Apply accumulated soul gradient. Call once per turn after decoding.
- `feedback`: `+1` reinforce, `-1` inhibit, `0` implicit (self-supervised)

---

## learner.h — On-Device Learning

### `learner_init`
```c
void learner_init(LearnerState *L, FORTZAModel *model);
```
Initialize learner. Dequantizes soul columns of `W_o` into float32 for gradient computation.
Calls `soul_load()` internally. Calls `learner_load_wo()` to restore any previous fine-tuning.

---

### `learner_begin_turn`
```c
void learner_begin_turn(LearnerState *L);
```
Reset gradient accumulators. Call before encoding each user input.

---

### `learner_step`
```c
void learner_step(LearnerState *L,
                  const float *h_dec,
                  const float *probs,
                  uint16_t chosen);
```
Accumulate gradients for one decode step.
- `h_dec`: decoder hidden state this step (`float[DEC_HIDDEN]`)
- `probs`: softmax output (`float[vocab_size]`)
- `chosen`: selected token

---

### `learner_end_turn`
```c
void learner_end_turn(LearnerState *L, int feedback);
```
Apply soul update via RUL. If `feedback == +1`, also fine-tunes output layer and saves to SPIFFS.
Saves soul to NVS regardless of feedback.

---

### `learner_detect_feedback`
```c
int learner_detect_feedback(const char *input);
```
Detect feedback in user input string. Returns `+1`, `-1`, or `0`.

Positive keywords: `good`, `yes`, `nice`, `correct`, `right`, `great`, `perfect`, `exactly`, `yep`, `sure`

Negative keywords: `no`, `wrong`, `bad`, `nope`, `not right`, `stop`, `incorrect`, `nah`

---

## ngram.h — ZIA

### `zia`
```c
void zia(const uint16_t *indices, int len, float *out);
```
Compute normalized bigram hash features. `out` must be `float[NGRAM_BUCKETS]`.

---

## lstm.h — INT8 LSTM Cell

### `lstm_init`
```c
void lstm_init(LSTMState *s, int input_size, int hidden_size,
               int8_t *W, float W_scale, float W_zp,
               int8_t *b, float b_scale, float b_zp);
```

### `lstm_reset`
```c
void lstm_reset(LSTMState *s);
```
Zero h and c states. Call before encoding each new input.

### `lstm_step`
```c
void lstm_step(LSTMState *s, const float *x);
```
One LSTM step. Updates `s->h` and `s->c` in-place. `x` must be `float[input_size]`.

---

## Constants

```c
#define SOUL_DIM      32    /* fortza dimension */
#define ENC_HIDDEN    64    /* encoder hidden size */
#define DEC_HIDDEN   128    /* long_hair hidden size */
#define NGRAM_BUCKETS 32    /* zia buckets */
#define FOU_DEFAULT  0.01f  /* soul learning rate */
#define SOUL_DECAY   0.001f /* L2 decay per turn */
#define SOUL_CLIP    2.0f   /* max soul norm */
#define FZSU_MAGIC   0x465A5355u  /* weights.bin magic */
```
