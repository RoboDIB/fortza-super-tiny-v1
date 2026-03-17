# FORTZA Architecture

## Overview

FORTZA is a character-level seq2seq LSTM with five named components, each solving a specific problem for conversational AI on microcontrollers.

```
user input (chars)
      │
      ├──→ [ZIA]  bigram hash → 32-dim features (no parameters)
      │
      └──→ [one-hot chars]
              │
         cat(char, zia)
              │
        [Encoder LSTM]  hidden=64
              │
           h_enc (64)
              │
     ┌────────┴────────────┐
     │                     │
[Soul Gate]          cat(h_enc, zia)
fortza_old ──→ fortza_new       │
     │                  [Wc Bridge]
     │                     │
     │              h_dec_init (128)
     │                     │
     └──────→ [Decoder LSTM (long_hair)]  hidden=128
                 each step:
                 cat(h_dec, fortza) → [Output Linear] → vocab logits
                                              │
                                        response chars
              after turn:
              [RUL] accumulate soul gradient → update fortza
              fortza saved to NVS
```

---

## Components

### `fortza` — Soul Vector

**What:** A 32-float vector in SRAM. The model's persistent identity across conversation turns.

**Where:** `firmware/main/soul.h`, `firmware/main/soul.c`, `training/fortza_model.py`

**How it works:**

The soul gate is a learned gating function (analogous to a GRU cell at turn level):
```
[gate; update] = W_sg · cat(h_enc, soul_old) + b_sg
soul_new = sigmoid(gate) * soul_old + (1 - sigmoid(gate)) * tanh(update)
```

After training and quantization, `W_sg` is frozen in flash. The soul vector itself is the only thing that changes after deployment.

**Persistence:** Saved to NVS after every turn. Loaded on boot. Survives power cycles.

**Effect:** After several turns, the soul encodes the current conversation's topic, tone, and the user's vocabulary patterns. Responses become more contextually appropriate without the model ever "seeing" earlier turns explicitly.

---

### `zia` — N-gram Context Hash

**What:** Parameter-free character bigram feature extractor. Returns a 32-dim normalized count vector.

**Where:** `training/ngram.py`, `firmware/main/ngram.c`

**Hash function (identical in Python and C):**
```
bucket = (char_a * 31337 + char_b * 17) % 32
```

**Example:** Input "hello" (indices [h=7, e=5, l=11, l=11, o=14]):
- Pair (h,e): bucket = (7*31337 + 5*17) % 32 = ...
- Pair (e,l): bucket = ...
- etc. → normalized 32-dim vector

**Why it helps:** The encoder now knows that the input contains mostly common English bigrams ("he", "ll") vs. rare ones, giving it word-shape awareness without any learned vocabulary.

**Zero cost:** No weights, no gradients. Pure math. Works identically in Python (training) and C (firmware).

---

### `long_hair` — Asymmetric Decoder

**What:** The decoder LSTM with hidden size 128, larger than the encoder's 64.

**Where:** `training/fortza_model.py` (`self.long_hair`), `firmware/main/fortza_model.c` (`model->long_hair`)

**Why asymmetric:**
- Encoding = compression. One context vector from many inputs. Hidden=64 is sufficient.
- Decoding = generation. Must produce coherent character sequences. Needs more capacity.
- This saves ~61 KB vs a symmetric hidden=128 encoder.

**Bridge:** `W_c ∈ R^{128×96}` connects the encoder output to the decoder:
```
h_dec_0 = tanh(W_c · cat(h_enc[64], zia[32]))
```

---

### `rul` — Residual Update Loop

**What:** On-device learning algorithm. Updates `fortza` (and optionally the output layer) after every conversation turn.

**Where:** `firmware/main/soul.c` (`rul_update()`), `firmware/main/learner.c`

**Track 1 — Implicit soul adaptation (every turn):**

Accumulates cross-entropy gradient w.r.t. soul during decode:
```
∂L_t/∂s = W_o_soul^T · (probs_t − one_hot(chosen_t))
```

After decoding:
```
soul -= fou * (mean gradient over T steps)
soul *= (1 - 0.001)     # L2 decay
soul  = clip_norm(soul, max=2.0)
```

**Track 2 — Output layer fine-tuning (explicit positive feedback):**

Triggered when user says "good", "yes", "nice", etc.:
```
∇W_o = mean over steps of: (probs_t − one_hot(chosen_t)) ⊗ cat(h_dec_t, soul)
W_o_float -= lr_out * ∇W_o     (lr_out = 1e-4)
```
Then re-quantize W_o to INT8 and save to SPIFFS.

**Memory requirements:**
- Soul gradient `ds`: 128 bytes
- Soul columns `W_o_soul` (float32): V × 32 × 4 bytes ≈ 6 KB
- Fine-tune `∇W_o` (lazy, float32): V × 160 × 4 bytes ≈ 31 KB
- All freed after update

---

### `fou` — Drift Factor

**What:** The soul learning rate. Controls how fast the soul adapts.

**Default:** `0.01` (`FOU_DEFAULT` in `soul.h`)

**Effect:**
- `fou = 0.0`: Soul never changes. Model is static (defeats the purpose).
- `fou = 0.01`: Gentle drift. Soul stabilizes after ~20 turns.
- `fou = 0.1`: Fast adaptation. Soul may drift too far from initial training distribution.

**Tuning:** If the model's responses feel unstable or random over time, lower `fou`. If it never seems to adapt, raise it.

---

## Parameter Count

| Layer | Shape | Params |
|---|---|---|
| Encoder LSTM | (256, 81) | 20,736 W + 256 b |
| Soul gate | (64, 96) | 6,144 W + 64 b |
| Wc bridge | (128, 96) | 12,288 W + 128 b |
| Decoder LSTM | (512, 177) | 90,624 W + 512 b |
| Output | (49, 160) | 7,840 W + 49 b |
| **Total** | | **~138K weights + ~1K biases ≈ 139K params** |

INT8 size: ~139 KB weights + 28 byte header + per-layer metadata ≈ **~144 KB**

---

## Data Flow (Detailed)

```python
# Per-turn inference (pseudocode matching firmware)

def fortza_turn(input_str, soul):
    # 1. Tokenize
    indices = tokenize(input_str)          # list of int

    # 2. ZIA
    zia_feats = zia(indices)               # (32,) float, no params

    # 3. Encode
    h, c = zeros(64), zeros(64)
    for idx in indices:
        x = cat(one_hot(idx, V), zia_feats)   # (V+32,)
        h, c = encoder_lstm(x, h, c)

    # 4. Soul gate
    soul = soul_gate(h, soul)             # (32,) updated

    # 5. Bridge
    ctx = cat(h, zia_feats)               # (96,)
    dh = tanh(Wc @ ctx + Wc_b)           # (128,) decoder init

    # 6. Decode
    response = []
    tok = START
    while tok != END and len(response) < MAX_LEN:
        dh, dc = decoder_lstm(one_hot(tok, V), dh, dc)
        h_soul = cat(dh, soul)            # (160,)
        logits = Wo @ h_soul + bo         # (V,)
        probs = softmax(logits)
        tok = argmax(probs)               # greedy
        response.append(tok)
        # RUL: accumulate soul gradient
        ds += Wo_soul_cols.T @ (probs - one_hot(tok))

    # 7. RUL update
    soul -= fou * (ds / len(response))
    soul *= (1 - decay)
    soul = clip_norm(soul, 2.0)

    # 8. Save soul to NVS
    nvs_write(soul)

    return response, soul
```
