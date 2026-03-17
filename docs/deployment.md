# Deployment Guide

## Hardware

| Component | Minimum | Recommended |
|---|---|---|
| Board | ESP32 dev board | ESP32-S3-DevKitC or ESP32 + 8MB PSRAM |
| Flash | 4MB | 8MB |
| Connection | USB-to-serial | built-in USB (S3) |
| Power | 5V USB | any 3.3–5V supply |

Any generic ESP32 dev board from AliExpress (~$3) works. The ESP32-S3 gives more SRAM headroom and is recommended if you plan to increase hidden sizes.

---

## Prerequisites

```bash
# Install ESP-IDF 5.x
# https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/

# Verify
idf.py --version   # should print 5.x.x
```

---

## Step 1 — Build firmware

```bash
cd firmware
idf.py set-target esp32    # or esp32s3
idf.py build
```

Build output: `firmware/build/fortza-super-tiny.bin` (~600 KB)

---

## Step 2 — Flash firmware

```bash
idf.py -p /dev/ttyUSB0 flash
# or on macOS: -p /dev/cu.usbserial-*
# or on Windows: -p COM3
```

---

## Step 3 — Flash model weights

Train first if you haven't:
```bash
cd ../training
python train.py && python quantize.py
cd ..
```

Then flash weights and vocab to SPIFFS:
```bash
python tools/flash_weights.py --port /dev/ttyUSB0
```

This uploads `weights.bin` and `vocab.json` to the SPIFFS partition at 0x310000.

---

## Step 4 — Open serial monitor

```bash
idf.py -p /dev/ttyUSB0 monitor
# or use any terminal at 115200 baud: screen, minicom, PuTTY
```

You should see:
```
[fortza-super-tiny] ready
[fortza soul norm: 0.000]
say something...

you:
```

---

## Talking to it

Type a message and press Enter:
```
you: hey
bot: hey what's up

you: not much
bot: nice, same here honestly
```

---

## Special commands

| Command | Effect |
|---|---|
| `.reset` | Clear soul vector to zero |
| `.soul` | Print current soul norm |
| `.save` | Force save soul + learned weights to flash |
| `good` / `yes` / `nice` | Positive feedback — fine-tunes output layer |
| `bad` / `wrong` / `no` | Negative feedback — inhibits soul |

---

## Understanding the on-device learning

**After every turn:**
The soul vector is updated silently via RUL. You won't see anything — it just happens. The model is slowly adapting to your conversation style.

**After "good":**
```
you: good
[learning: good response remembered]
```
The output layer fine-tunes itself to reinforce what it just said. Saved to `/spiffs/learned_wo.bin`.

**After power cycle:**
Soul and learned weights are restored automatically. The model picks up exactly where it left off.

**After many conversations:**
The soul norm will be non-zero on boot. The model will feel slightly different than a fresh model — it's been shaped by your conversations.

---

## SPIFFS partition layout

```
/spiffs/weights.bin       — base model weights (flashed, read-only)
/spiffs/vocab.json        — character vocabulary
/spiffs/learned_wo.bin    — fine-tuned output layer (written by learner)
```

---

## Partition table

The default ESP-IDF partition table may not have enough space for SPIFFS. If `idf.py build` warns about partition size, add a custom `partitions.csv`:

```csv
# Name,   Type, SubType, Offset,  Size,    Flags
nvs,      data, nvs,     0x9000,  0x6000,
phy_init, data, phy,     0xf000,  0x1000,
factory,  app,  factory, 0x10000, 0x300000,
spiffs,   data, spiffs,  0x310000,0xCF000,
```

Then in `firmware/sdkconfig` (or `idf.py menuconfig`):
```
CONFIG_PARTITION_TABLE_CUSTOM=y
CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="partitions.csv"
```

---

## Troubleshooting

**"weights.bin not found"**
Run `python tools/flash_weights.py` to upload the model files.

**"bad magic 0x..."**
Old weights from the previous architecture. Re-run `python quantize.py` and reflash.

**Garbage output / no response**
- Check baud rate is 115200
- Try `.reset` to clear soul
- Verify vocab.json matches the weights (same training run)

**"OOM: cannot allocate fine-tune buffers"**
ESP32 is out of heap during fine-tuning. Reduce `MAX_OUTPUT` in `main.c` or switch to ESP32-S3 with PSRAM.

**Soul not saving (ESP_ERR_NVS_NOT_FOUND on next boot)**
NVS partition may be corrupted. Run: `idf.py -p /dev/ttyUSB0 erase-flash` then reflash everything.
