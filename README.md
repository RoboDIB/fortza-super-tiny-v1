# fortza-super-tiny — v1

**A self-learning conversational AI that runs entirely on a $3 ESP32 microcontroller.**

No cloud. No Wi-Fi. No reflashing to improve. The model learns from you, on-chip, in real time.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/paper-PAPER.md-green.svg)](PAPER.md)
[![Hardware](https://img.shields.io/badge/hardware-ESP32-orange.svg)](docs/deployment.md)

---

## What is FORTZA?

FORTZA is a novel conversational AI architecture built from scratch for microcontrollers with less than 1 MB of RAM. It introduces four innovations that have never appeared together on such constrained hardware:

| Innovation | What it does |
|---|---|
| **Soul Vector** (`fortza`) | 32-float persistent conversation identity — survives power cycles |
| **N-gram Context Hash** (`zia`) | Parameter-free bigram features — word-awareness at zero cost |
| **Asymmetric Codec** (`long_hair`) | Decoder (hidden=128) larger than encoder (hidden=64) |
| **Residual Update Loop** (`rul`) | On-device soul gradient learning after every single turn |

The result: a model that remembers what you said last week, adapts to your style, and gets better every conversation — running on a chip that costs less than a coffee.

---

## Architecture at a glance

```
user input
  ├→ zia (bigram hash, zero params) → 32-dim features
  └→ one-hot chars
  → cat(char, zia) → [Encoder LSTM, hidden=64] → h_enc
  → cat(h_enc, zia) → [Wc bridge] → h_dec_init
  → soul_gate(h_enc, fortza_old) → fortza_new     ← soul updated here
  → [Decoder LSTM (long_hair), hidden=128]
      each step: cat(h_dec, fortza) → [Output Linear] → char
  → response
  → rul: soul gradient accumulated, applied after turn
  → fortza saved to NVS (flash)
```

**Model size:** ~167K parameters | ~144 KB INT8 on device | ~237 KB SRAM during inference

---

## Quickstart

### 1. Install dependencies

```bash
pip install torch numpy datasets
```

### 2. Prepare dataset

```bash
cd training
python data/prepare.py      # downloads PersonaChat (~10K conversation pairs)
```

Or write your own `training/data/data.txt`:
```
you: hey
bot: hey what's up
you: how are you
bot: doing great, you?
```

### 3. Train

```bash
python train.py --epochs 1000
```

GPU recommended. On NVIDIA L4: ~38s/epoch. Loss drops from ~3.5 → ~0.89 over 1000 epochs.

### 4. Test on PC

```bash
python chat.py
```

The soul vector persists across turns — type `soul` to see its norm grow. Type `reset` to clear it.

### 5. Quantize for ESP32

```bash
python quantize.py
# outputs: checkpoints/weights.bin (~144 KB)
```

### 6. Flash to ESP32

```bash
cd ../firmware
idf.py set-target esp32
idf.py build
idf.py -p /dev/ttyUSB0 flash

cd ..
python tools/flash_weights.py --port /dev/ttyUSB0
```

### 7. Chat

Open any serial monitor at 115200 baud:

```
[fortza-super-tiny] ready
[fortza soul norm: 0.000]
say something...

you: hey
bot: hey how are you doing

you: i like hiking
bot: that sounds really fun

you: good
[learning: good response remembered]
```

---

## On-device learning

Every turn, silently:
- Soul vector updated via RUL gradient (128 bytes in NVS)

When you say `good` / `yes` / `nice`:
- Output layer fine-tuned and saved to SPIFFS

When you say `bad` / `wrong`:
- Soul gradient inhibited

After power cycle:
- Soul and learned weights restored automatically

---

## Special commands

| Command | Effect |
|---|---|
| `.reset` | Clear soul vector to zero |
| `.soul` | Print current soul norm |
| `.save` | Force save soul + weights |
| `good` / `yes` / `nice` | Positive feedback → fine-tune |
| `bad` / `wrong` / `no` | Negative feedback → inhibit |

---

## Hardware

| | Minimum | Recommended |
|---|---|---|
| Board | ESP32 (520 KB SRAM) | ESP32-S3 or ESP32 + PSRAM |
| Flash | 4 MB | 8 MB |
| Connection | USB-to-serial | Built-in USB (S3) |
| Training | Any CPU | NVIDIA GPU (CUDA) |

Any generic ESP32 dev board from AliExpress (~$3) works.

---

## Project structure

```
fortza-super-tiny/
├── training/
│   ├── fortza_model.py      PyTorch FORTZA model
│   ├── ngram.py             ZIA parameter-free bigram hash
│   ├── tokenizer.py         Character tokenizer
│   ├── train.py             Training loop (GPU)
│   ├── quantize.py          INT8 export → weights.bin
│   ├── chat.py              PC chat with persistent soul
│   ├── test.py              Quality evaluation suite
│   └── data/
│       ├── prepare.py       PersonaChat downloader
│       └── data.txt         Training pairs
├── firmware/
│   └── main/
│       ├── main.c           UART loop, commands
│       ├── fortza_model.c   Inference engine (encode/decode)
│       ├── soul.c           Soul vector + RUL
│       ├── learner.c        Output layer fine-tuning
│       ├── ngram.c          ZIA in C
│       └── lstm.c           INT8 LSTM cell
├── tools/
│   └── flash_weights.py     SPIFFS uploader
├── docs/
│   ├── architecture.md      Deep architecture docs
│   ├── training.md          Training guide
│   ├── deployment.md        ESP32 deployment
│   └── firmware_api.md      C API reference
├── PAPER.md                 Full research paper
└── MODEL_CARD.md            Model card (HuggingFace-style)
```

---

## Performance

| Metric | Value |
|---|---|
| Parameters | ~167K |
| Quantized size | ~144 KB INT8 |
| SRAM usage (inference) | ~237 KB / 520 KB |
| SRAM usage (fine-tuning peak) | ~299 KB / 520 KB |
| Training loss (final, 1000 epochs) | 0.8858 |
| Coherent responses | ~76% (cold start) |
| Soul update time | <1 ms/turn |

---

## Citation

```bibtex
@article{pradhan2025fortza,
  title   = {FORTZA: A Soul-Vector Architecture for Persistent Conversational AI
             on Ultra-Low-Resource Microcontrollers},
  author  = {Pradhan, Dibyaprakash},
  journal = {arXiv preprint},
  year    = {2025},
  url     = {https://github.com/RoboDIB/fortza-super-tiny-v1}
}
```

---

## Author

**Dibyaprakash Pradhan** — Independent Research

---

## License

MIT — see [LICENSE](LICENSE).
Training data (PersonaChat) is CC BY-NC 4.0.

---

## What's next — FORTZA v2

v2 introduces **Recurrent Depth** — one shared LSTM layer run 96 times with per-depth steering vectors, dynamic width expansion, and a reasoning scratchpad. 96 reasoning steps. ~56 KB total weights. Still fits in ESP32 SRAM.

See [`../fortza-v2/`](../fortza-v2/) — coming soon at github.com/RoboDIB.
# fortza-super-tiny-v1
