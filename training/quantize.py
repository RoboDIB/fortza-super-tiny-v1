"""
fortza-super-tiny — Quantize
Converts float32 weights → INT8 and exports weights.bin for ESP32.

Binary format (v2, magic 0x465A5355 "FZSU"):
  [header]
    uint32  magic        = 0x465A5355
    uint32  vocab_size
    uint32  enc_hidden   = 64
    uint32  dec_hidden   = 128
    uint32  soul_dim     = 32
    uint32  ngram_buckets= 32
    uint32  n_layers     = 10
  [per layer, 10 total]
    uint32  rows
    uint32  cols
    float32 scale
    float32 zero_point
    int8[]  data  (rows*cols bytes)

Layer order (matches fortza_model.export_weights()):
  0: enc_W   (4*64, V+32)    encoder LSTM weights
  1: enc_b   (4*64,)         encoder LSTM bias
  2: sg_W    (64,  96)       soul_gate weights
  3: sg_b    (64,)           soul_gate bias
  4: Wc_W    (128, 96)       context bridge weights
  5: Wc_b    (128,)          context bridge bias
  6: dec_W   (4*128, V+128)  long_hair LSTM weights
  7: dec_b   (4*128,)        long_hair LSTM bias
  8: out_W   (V,   160)      output projection weights
  9: out_b   (V,)            output projection bias

Usage:
  python quantize.py
  python quantize.py --weights checkpoints/weights_best.npz --vocab checkpoints/vocab.json
"""

import argparse
import json
import os
import struct
import numpy as np

MAGIC    = 0x465A5355   # "FZSU" — FORTZA Soul Unit
N_LAYERS = 10
SOUL_DIM = 32
NGRAM_BUCKETS = 32
ENC_HIDDEN    = 64
DEC_HIDDEN    = 128


def quantize_array(arr):
    flat = arr.flatten().astype(np.float32)
    mn, mx = flat.min(), float(flat.max())
    scale  = (mx - mn) / 255.0 if mx != mn else 1.0
    zp     = mn
    q      = np.round((flat - zp) / scale).clip(0, 255).astype(np.uint8)
    q_s8   = (q.astype(np.int16) - 128).astype(np.int8)
    return q_s8.reshape(arr.shape), scale, zp


def write_layer(f, arr):
    arr = np.atleast_2d(arr) if arr.ndim == 1 else arr
    rows, cols = arr.shape
    q, scale, zp = quantize_array(arr)
    f.write(struct.pack("<II", rows, cols))
    f.write(struct.pack("<ff", scale, zp))
    f.write(q.tobytes())


def quantize(args):
    print("[fortza-super-tiny] loading weights...")
    data   = np.load(args.weights)
    layers = [data[f"arr_{i}"] for i in range(len(data.files))]

    assert len(layers) == N_LAYERS, \
        f"expected {N_LAYERS} weight arrays, got {len(layers)}\n" \
        f"make sure you trained with fortza_model.py (not old seq2seq.py)"

    with open(args.vocab) as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    enc_W = layers[0]
    print(f"  vocab_size   : {vocab_size}")
    print(f"  enc_hidden   : {ENC_HIDDEN}")
    print(f"  dec_hidden   : {DEC_HIDDEN}  (long_hair)")
    print(f"  soul_dim     : {SOUL_DIM}    (fortza)")
    print(f"  ngram_buckets: {NGRAM_BUCKETS} (zia)")

    with open(args.out, "wb") as f:
        f.write(struct.pack("<IIIIIII",
                            MAGIC, vocab_size,
                            ENC_HIDDEN, DEC_HIDDEN,
                            SOUL_DIM, NGRAM_BUCKETS,
                            N_LAYERS))
        for layer in layers:
            write_layer(f, layer)

    size_kb = os.path.getsize(args.out) / 1024
    print(f"\n  written: {args.out}  ({size_kb:.1f} KB)")
    print(f"  flash with: python tools/flash_weights.py --port /dev/ttyUSB0")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Quantize fortza-super-tiny weights")
    p.add_argument("--weights", default="checkpoints/weights_best.npz")
    p.add_argument("--vocab",   default="checkpoints/vocab.json")
    p.add_argument("--out",     default="checkpoints/weights.bin")
    quantize(p.parse_args())
