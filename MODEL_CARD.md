# Model Card — fortza-super-tiny v1

## Model Summary

| Field | Value |
|---|---|
| **Name** | fortza-super-tiny v1 |
| **Architecture** | FORTZA (character-level seq2seq LSTM with soul vector) |
| **Task** | Open-domain conversational AI |
| **Target hardware** | ESP32 microcontroller (~$3) |
| **Parameters** | 167,249 |
| **Size on device** | 144 KB INT8 |
| **Language** | English (character-level — any language trainable) |
| **License** | MIT |
| **Training data** | PersonaChat (user-configurable) |
| **Author** | Dibyaprakash Pradhan |
| **Version** | v1 |

---

## What makes this different

Most conversational AI either requires cloud connectivity or is frozen after deployment. fortza-super-tiny is the first model architecture with:

1. **Persistent memory across turns** — the soul vector (`fortza`) lives in SRAM and NVS. The model remembers what you talked about three messages ago without any external database.
2. **On-device self-learning** — after every turn, the soul vector is updated via RUL (Residual Update Loop). When you say "good", the output layer fine-tunes itself. No reflashing required.
3. **Fully offline** — no Wi-Fi, no API key, no servers. Everything runs on the chip.
4. **Trainable on your data** — point it at your own conversation dataset and it will sound like you want it to.

---

## Architecture

```
FORTZA components:

  fortza     Soul vector (32 floats, SRAM + NVS)
  zia        N-gram context hash (parameter-free bigram features)
  long_hair  Asymmetric decoder (hidden=128, encoder=64)
  rul        Residual Update Loop (on-device soul gradient)
  fou        Soul drift factor / learning rate (default 0.01)
```

Full architecture description: see [PAPER.md](PAPER.md) or [docs/architecture.md](docs/architecture.md).

---

## Intended Uses

**Supported:**
- Personal conversational companion on ESP32
- Offline voice/text assistant (with audio frontend)
- Custom domain chatbot (customer service, home automation, education)
- Research into TinyML generative models
- Teaching tool for embedded AI / seq2seq models

**Not supported:**
- Factual question answering (no knowledge base)
- Long-form generation (max 120 chars per response)
- Safety-critical applications
- Replacing cloud LLMs for complex reasoning

---

## Training Data

**Default dataset:** PersonaChat ([AlekseyKorshuk/persona-chat](https://huggingface.co/datasets/AlekseyKorshuk/persona-chat))

| Property | Value |
|---|---|
| Source | HuggingFace — AlekseyKorshuk/persona-chat |
| Pairs used | 9,999 |
| Format | Casual multi-turn dialogue |
| License | CC BY-NC 4.0 |
| Language | English |
| Preprocessing | Lowercase, filter >120 chars |

**Custom training:** Users can replace `data/data.txt` with any conversation dataset in the format:
```
you: [input]
bot: [response]
```

---

## Training Procedure

| Hyperparameter | Value |
|---|---|
| Framework | PyTorch |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Epochs | 1000 |
| Gradient clip | 5.0 |
| Hardware | NVIDIA L4 GPU |
| Training time | ~10.6 hours (1000 epochs × 38s/epoch) |

```bash
cd training
python train.py --epochs 1000 --batch 64 --lr 0.001
```

---

## Evaluation

| Metric | Value |
|---|---|
| Training loss (epoch 1, random init) | ~3.51 |
| Training loss (epoch 1000, final) | **0.8858** |
| Total loss reduction | 74.8% |
| Coherent responses (cold start) | ~76% (19/25 test inputs) |
| Response uniqueness (30-turn stress) | 73% (22/30 unique) |
| Avg response length | ~18 characters |
| Soul norm after 15 turns | ~0.10 |
| Inference speed (ESP32 @ 240 MHz) | ~180 chars/sec |
| Average response latency | ~100 ms |
| Soul update time | <1 ms/turn |
| SRAM usage (normal inference) | ~237 KB / 520 KB |
| SRAM usage (peak fine-tuning) | ~299 KB / 520 KB |

---

## On-Device Learning Behavior

| Event | What happens |
|---|---|
| Every turn (implicit) | Soul vector updated via RUL gradient |
| User says "good" / "yes" / "nice" | Output layer fine-tuned + saved to SPIFFS |
| User says "bad" / "wrong" | Soul gradient inhibited (pushed away) |
| Power cycle | Soul and learned weights restored from NVS/SPIFFS |
| `.reset` command | Soul cleared to zero |

---

## Limitations

- **Character-level:** Less fluent than subword models; struggles with rare spellings
- **No attention:** Long inputs (>80 chars) may lose early context
- **Soul capacity:** 32 floats may not capture complex user preferences
- **English-optimized:** PersonaChat is English-only; retrain for other languages
- **Single user:** Soul and learned weights assume one user per device; multi-user requires separate NVS namespaces
- **No content filtering:** Model can generate inappropriate content if trained on such data
- **Short responses:** Max 120 characters per response by default

---

## How to Use

### Train
```bash
pip install torch numpy datasets
cd training
python data/prepare.py          # download PersonaChat
python train.py                  # train on GPU
python quantize.py               # export weights.bin
```

### Test on PC
```bash
python chat.py
# Commands: 'reset' clears soul | 'soul' shows soul norm
```

### Deploy to ESP32
```bash
cd firmware
idf.py build
idf.py -p /dev/ttyUSB0 flash
cd ..
python tools/flash_weights.py --port /dev/ttyUSB0
```

### Chat
Open serial monitor at 115200 baud. Type and press Enter.
```
you: hey
bot: hey what's up  [soul:0.012]

you: good
[learning: good response remembered]
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| MCU | ESP32 (520KB SRAM) | ESP32-S3 or ESP32 + 8MB PSRAM |
| Flash | 4MB | 8MB |
| Training PC | Any CPU | NVIDIA GPU (CUDA) |
| Python | 3.8+ | 3.10+ |
| ESP-IDF | 5.0+ | 5.2+ |

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

## License

MIT License. See [LICENSE](LICENSE).
Training data (PersonaChat) is CC BY-NC 4.0 — non-commercial use only.
