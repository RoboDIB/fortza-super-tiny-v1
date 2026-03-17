# Changelog

## v1.0.0 — 2025-03-17 (fortza-super-tiny)

Initial release of the FORTZA architecture.

### Architecture (new — not in any prior work)
- Soul Vector (`fortza`): 32-float persistent conversation identity in SRAM + NVS
- N-gram Context Hash (`zia`): parameter-free bigram features (zero parameters)
- Asymmetric Codec (`long_hair`): encoder hidden=64, decoder hidden=128
- Residual Update Loop (`rul`): two-track on-device learning
  - Track 1: implicit soul gradient (every turn, always-on)
  - Track 2: output layer SGD (triggered by positive user feedback)
- Soul drift factor (`fou`): 0.01 default learning rate for soul adaptation

### Training
- PyTorch GPU training pipeline (NVIDIA L4, ~38s/epoch)
- PersonaChat dataset (9,999 pairs, CC BY-NC 4.0)
- Adam optimizer, 1000 epochs, final loss 0.8858
- INT8 quantization with per-tensor scale/zero_point
- Binary export format: `weights.bin` (magic `0x465A5355`, 10 layers, ~144 KB)

### Firmware (ESP32, ESP-IDF 5.x)
- INT8 inference engine with float32 accumulation
- Soul persistence: NVS (non-volatile storage, survives power cycles)
- Learned weights persistence: SPIFFS (`/spiffs/learned_wo.bin`)
- UART interface: 115200 baud, dot commands (`.reset`, `.soul`, `.save`)
- Feedback detection: keyword-based (`good`/`yes`/`nice` → +1, `bad`/`wrong` → -1)
- Repeat-loop protection: breaks if same char repeats 3+ times

### Performance (v1 baseline)
- 167,249 parameters | 144 KB INT8 on device
- 237 KB SRAM normal inference (46% of ESP32's 520 KB)
- 299 KB SRAM peak with fine-tuning (58%)
- ~180 chars/sec decode on ESP32 @ 240 MHz
- ~100 ms average response latency
- 76% coherent responses on cold start (PersonaChat)

---

## v2.0.0 — planned (fortza-v2)

Recurrent Depth architecture:
- Single shared LSTM layer run 96× per token
- Per-depth steering vectors (96 × 256 bytes = 24 KB)
- Dynamic width expansion across depth phases
- Reasoning scratchpad: 96 key vectors (3 KB)
- Estimated total weights: ~56 KB (fits entirely in SRAM)
- 96 reasoning steps per token
