"""
fortza-super-tiny
Flash weights.bin and vocab.json to ESP32 SPIFFS partition.

Usage:
  python tools/flash_weights.py
  python tools/flash_weights.py --port /dev/ttyUSB0 --weights training/checkpoints/weights.bin
"""

import argparse
import os
import subprocess
import sys


def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"error: command failed (exit {result.returncode})")
        sys.exit(1)


def flash(args):
    weights = args.weights
    vocab   = args.vocab
    port    = args.port
    idf     = os.environ.get("IDF_PATH", "")

    if not os.path.exists(weights):
        print(f"error: weights not found at {weights}")
        print("  run: python training/quantize.py  first")
        sys.exit(1)

    if not os.path.exists(vocab):
        print(f"error: vocab not found at {vocab}")
        sys.exit(1)

    # Build SPIFFS image using spiffsgen.py from ESP-IDF
    spiffsgen = os.path.join(idf, "components", "spiffs", "spiffsgen.py") if idf else "spiffsgen.py"
    image     = "/tmp/fortza_spiffs.bin"
    data_dir  = "/tmp/fortza_data"

    os.makedirs(data_dir, exist_ok=True)
    import shutil
    shutil.copy(weights, os.path.join(data_dir, "weights.bin"))
    shutil.copy(vocab,   os.path.join(data_dir, "vocab.json"))

    print("[fortza-super-tiny] building SPIFFS image...")
    run([sys.executable, spiffsgen, "0x100000", data_dir, image])

    print(f"[fortza-super-tiny] flashing to {port}...")
    run(["esptool.py", "--port", port, "write_flash", "0x310000", image])

    print("[fortza-super-tiny] done! open serial monitor at 115200 baud")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Flash fortza-super-tiny weights to ESP32")
    p.add_argument("--port",    default="/dev/ttyUSB0")
    p.add_argument("--weights", default="training/checkpoints/weights.bin")
    p.add_argument("--vocab",   default="training/checkpoints/vocab.json")
    flash(p.parse_args())
