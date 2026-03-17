"""
fortza-super-tiny — PC Chat
Test the FORTZA model on your machine before flashing to ESP32.

fortza (soul vector) persists across turns — the model remembers
what you talked about earlier in the session, just like it will on-chip.
Type 'reset' to clear the soul. Type 'soul' to see its current norm.

Usage:
  python chat.py
  python chat.py --weights checkpoints/weights_best.npz --vocab checkpoints/vocab.json
"""

import argparse
import json
import numpy as np
import torch

from tokenizer import Tokenizer
from fortza_model import FORTZAModel, SOUL_DIM, ENC_HIDDEN, DEC_HIDDEN


def load_model(weights_path, vocab_path):
    tok = Tokenizer()
    tok.load(vocab_path)

    data   = np.load(weights_path)
    layers = [data[f"arr_{i}"] for i in range(len(data.files))]

    model = FORTZAModel(vocab_size=tok.vocab_size)
    params = model.export_weights()
    assert len(layers) == len(params), \
        f"weight count mismatch: file has {len(layers)}, model expects {len(params)}"

    # Load weights back into model
    def load_lstm(cell, W_np, b_np):
        V_plus_H = cell.weight_ih.shape[1] + cell.weight_hh.shape[1]
        W = torch.tensor(W_np)
        cell.weight_ih.data = W[:, :cell.weight_ih.shape[1]]
        cell.weight_hh.data = W[:, cell.weight_ih.shape[1]:]
        # bias_ih = b/2, bias_hh = b/2 (they sum to b during export)
        b = torch.tensor(b_np) / 2.0
        cell.bias_ih.data = b
        cell.bias_hh.data = b

    def load_linear(layer, W_np, b_np):
        layer.weight.data = torch.tensor(W_np)
        layer.bias.data   = torch.tensor(b_np)

    enc_W, enc_b, sg_W, sg_b, Wc_W, Wc_b, dec_W, dec_b, out_W, out_b = layers
    load_lstm(model.encoder,    enc_W, enc_b)
    load_linear(model.soul_gate, sg_W,  sg_b)
    load_linear(model.Wc,        Wc_W,  Wc_b)
    load_lstm(model.long_hair,  dec_W, dec_b)
    load_linear(model.output,    out_W, out_b)

    model.eval()
    return model, tok


def chat(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fortza-super-tiny] loading model on {device}...")
    model, tok = load_model(args.weights, args.vocab)
    model = model.to(device)

    # fortza — persistent soul vector, starts at zero (cold start)
    fortza = torch.zeros(1, SOUL_DIM, device=device)

    print("  ready. fortza (soul) starts fresh.")
    print("  commands: 'reset' clears soul | 'soul' shows soul norm\n")

    turn = 0
    while True:
        try:
            user_input = input("you: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nbye!")
            break

        if not user_input:
            continue

        if user_input == "reset":
            fortza = torch.zeros(1, SOUL_DIM, device=device)
            print("  [fortza reset — soul cleared]\n")
            continue

        if user_input == "soul":
            norm = fortza.norm().item()
            print(f"  [fortza norm: {norm:.4f}  dim: {SOUL_DIM}]\n")
            continue

        inp_idx = tok.encode(user_input)
        out_idx, fortza = model.respond(
            inp_idx, fortza, tok.start_idx, tok.end_idx,
            max_len=120, temperature=args.temperature, device=device
        )
        response = tok.decode(out_idx)
        turn += 1

        soul_norm = fortza.norm().item()
        print(f"bot: {response}  [soul:{soul_norm:.3f}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Chat with fortza-super-tiny on PC")
    p.add_argument("--weights",     default="checkpoints/weights_best.npz")
    p.add_argument("--vocab",       default="checkpoints/vocab.json")
    p.add_argument("--temperature", type=float, default=0.8)
    chat(p.parse_args())
