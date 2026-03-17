# Training Guide

## Prerequisites

```bash
pip install torch numpy datasets
```

A CUDA GPU is strongly recommended. Training on CPU with 9,999 pairs takes days. On an NVIDIA L4 it takes ~1–2 hours.

---

## Step 1 — Prepare your dataset

### Option A: Use PersonaChat (default)

```bash
cd training
python data/prepare.py
```

This downloads ~10K casual conversation pairs from HuggingFace and writes them to `data/data.txt`.

### Option B: Use your own data

Write `training/data/data.txt` in this format:

```
you: hey how are you
bot: doing well! just had coffee. you?
you: same, a bit tired
bot: yeah mornings are rough
you: what are you up to
bot: not much, just chilling honestly
```

**Rules:**
- One `you:` line followed immediately by one `bot:` line
- Lowercase preferred (model lowercases all input anyway)
- Keep lines under 120 characters
- 1,000+ pairs minimum. 5,000–15,000 is ideal for a $3 chip.
- The more consistent the "voice", the stronger the personality

**Domain tips:**

| Goal | Dataset suggestion |
|---|---|
| General friend | PersonaChat, DailyDialog |
| Technical assistant | Your own Q&A pairs |
| Customer service | FAQ pairs from your product |
| Character voice | Dialogue from books/scripts |

---

## Step 2 — Train

```bash
python train.py
```

### What you'll see

```
[fortza-super-tiny] device: cuda
  loading data from data/data.txt
  found 9999 conversation pairs
  vocab size: 49 characters
  model: enc=64 dec=128 soul=32 params≈138,609
  training for 1000 epochs, batch=64

  epoch 1/1000  [10/157  6%]  loss 3.4821
  epoch 1/1000  loss 3.2104  (2.1s)
  you: hey what's up
  bot: [early gibberish]

  epoch 50/1000  loss 1.8441  (1.8s)
  you: what are you doing
  bot: just hanging out, you know
  ...
```

### Loss interpretation

| Loss | State |
|---|---|
| ~3.5 | Random gibberish |
| ~2.5 | Starts forming real words |
| ~1.8 | Recognizable sentences |
| ~1.2 | Coherent replies |
| ~0.8 | Talking like a friend |
| <0.5 | Potentially overfitting — check diversity |

### Key flags

```bash
python train.py \
  --data    data/data.txt \   # dataset path
  --out     checkpoints \     # where to save weights
  --epochs  1000 \            # training iterations
  --batch   64 \              # batch size (lower if OOM)
  --lr      0.001 \           # learning rate
  --log-every 1 \             # print sample every N epochs
  --temperature 0.8           # sampling temp for log samples
```

---

## Step 3 — Evaluate on PC

```bash
python chat.py
```

This loads the best checkpoint and lets you chat. The soul vector persists across turns in the terminal session — same as on ESP32.

```
you: hey
bot: hey what's going on  [soul:0.000]

you: not much, just thinking
bot: yeah same, mind just wanders  [soul:0.031]

you: soul
  [fortza norm: 0.031  dim: 32]

you: reset
  [fortza reset — soul cleared]
```

The `[soul:X.XXX]` after each response shows the soul vector norm. It should grow slowly as the conversation progresses. If it stays at 0.000, something is wrong with the RUL gradient path.

---

## Step 4 — Quantize

```bash
python quantize.py
```

Converts `checkpoints/weights_best.npz` to `checkpoints/weights.bin` (INT8, ~144 KB).

```
[fortza-super-tiny] loading weights...
  vocab_size   : 49
  enc_hidden   : 64
  dec_hidden   : 128  (long_hair)
  soul_dim     : 32   (fortza)
  ngram_buckets: 32   (zia)

  written: checkpoints/weights.bin  (144.2 KB)
  flash with: python tools/flash_weights.py --port /dev/ttyUSB0
```

---

## Retraining on new data

To retrain on different data without starting from scratch:

1. Replace `data/data.txt` with your new dataset
2. Run `python train.py` — checkpoints are overwritten
3. Run `python quantize.py`
4. Reflash

The soul vector on the device is not affected by reflashing — it lives in NVS. To clear it, send `.reset` over serial.

---

## Troubleshooting

**Loss not decreasing after 100 epochs:**
- Try a lower learning rate: `--lr 0.0003`
- Check dataset quality — very short or repetitive pairs hurt training

**Loss decreasing but responses are gibberish:**
- Lower temperature: `--temperature 0.5`
- Train longer

**Out of memory during training:**
- Reduce batch size: `--batch 32` or `--batch 16`

**Training is very slow:**
- Verify GPU is being used: output should say `device: cuda`
- If `device: cpu`, install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
