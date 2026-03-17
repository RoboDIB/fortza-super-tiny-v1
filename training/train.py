"""
fortza-super-tiny — Training Script
Uses the FORTZA architecture: fortza (soul), zia (ngram), long_hair (decoder), rul (learning).

Usage:
  python train.py
  python train.py --epochs 500 --batch 64 --lr 0.001
"""

import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizer import Tokenizer
from fortza_model import FORTZAModel, SOUL_DIM, ENC_HIDDEN, DEC_HIDDEN


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

class ConversationDataset(Dataset):
    def __init__(self, pairs, tok, max_len=100):
        self.samples = []
        end   = tok.end_idx
        start = tok.start_idx
        for inp_str, resp_str in pairs:
            enc     = tok.encode(inp_str)[:max_len]
            dec_in  = [start] + tok.encode(resp_str)[:max_len - 1]
            dec_tgt = tok.encode(resp_str)[:max_len - 1] + [end]
            self.samples.append((enc, dec_in, dec_tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch, vocab_size, device):
    enc_seqs, dec_in_seqs, dec_tgt_seqs = zip(*batch)

    def pad_onehot(seqs):
        max_t = max(len(s) for s in seqs)
        out = torch.zeros(len(seqs), max_t, vocab_size, device=device)
        for i, s in enumerate(seqs):
            for t, idx in enumerate(s):
                out[i, t, idx] = 1.0
        return out

    def pad_tgt(seqs):
        max_t = max(len(s) for s in seqs)
        out = torch.zeros(len(seqs), max_t, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            for t, idx in enumerate(s):
                out[i, t] = idx
        return out

    enc_oh  = pad_onehot(enc_seqs)
    dec_oh  = pad_onehot(dec_in_seqs)
    dec_tgt = pad_tgt(dec_tgt_seqs)
    # Return raw indices for ZIA computation
    return enc_oh, list(enc_seqs), dec_oh, dec_tgt


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_pairs(path):
    pairs = []
    lines = open(path, encoding="utf-8").read().strip().splitlines()
    i = 0
    while i < len(lines) - 1:
        a, b = lines[i].strip(), lines[i + 1].strip()
        if a.startswith("you:") and b.startswith("bot:"):
            inp  = a[4:].strip().lower()
            resp = b[4:].strip().lower()
            if inp and resp:
                pairs.append((inp, resp))
            i += 2
        else:
            i += 1
    return pairs


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train(args):
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fortza-super-tiny] device: {device}")
    print(f"  loading data from {args.data}")

    pairs = load_pairs(args.data)
    print(f"  found {len(pairs)} conversation pairs")

    all_text = " ".join(inp + " " + resp for inp, resp in pairs)
    tok = Tokenizer()
    tok.build(all_text)
    print(f"  vocab size: {tok.vocab_size} characters")

    os.makedirs(args.out, exist_ok=True)
    tok.save(os.path.join(args.out, "vocab.json"))

    model = FORTZAModel(vocab_size=tok.vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: enc={ENC_HIDDEN} dec={DEC_HIDDEN} soul={SOUL_DIM} params≈{n_params:,}")
    print(f"  training for {args.epochs} epochs, batch={args.batch}\n")

    dataset = ConversationDataset(pairs, tok)
    V = tok.vocab_size
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loader = DataLoader(
            dataset, batch_size=args.batch, shuffle=True,
            collate_fn=lambda b: collate(b, V, device)
        )
        total_loss = 0.0
        t0 = time.time()

        for step, (enc_oh, enc_idx, dec_oh, dec_tgt) in enumerate(loader):
            B = enc_oh.size(0)
            # fortza starts at zero each batch (cold start training)
            # model learns to work with soul=0 and improve with non-zero soul
            fortza = torch.zeros(B, SOUL_DIM, device=device)

            optimizer.zero_grad()
            logits, _ = model(enc_oh, enc_idx, dec_oh, fortza)

            B, T, _ = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                dec_tgt.reshape(B * T),
                ignore_index=0
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

            if (step + 1) % 10 == 0 or (step + 1) == len(loader):
                pct = (step + 1) / len(loader) * 100
                print(f"\r  epoch {epoch}/{args.epochs}  "
                      f"[{step+1}/{len(loader)}  {pct:.0f}%]  "
                      f"loss {total_loss/(step+1):.4f}", end="", flush=True)

        avg_loss = total_loss / len(loader)
        elapsed  = time.time() - t0
        print(f"\r  epoch {epoch}/{args.epochs}  loss {avg_loss:.4f}  ({elapsed:.1f}s)          ")

        if epoch % args.log_every == 0:
            model.eval()
            sample_inp, _ = random.choice(pairs)
            fortza_single = torch.zeros(1, SOUL_DIM, device=device)
            out_idx, _    = model.respond(
                tok.encode(sample_inp.lower()), fortza_single,
                tok.start_idx, tok.end_idx,
                temperature=args.temperature, device=device
            )
            print(f"  you: {sample_inp}")
            print(f"  bot: {tok.decode(out_idx)}\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_weights(model, os.path.join(args.out, "weights_best.npz"))

    save_weights(model, os.path.join(args.out, "weights_final.npz"))
    print(f"\n[fortza-super-tiny] done. best loss: {best_loss:.4f}")
    print(f"  run: python quantize.py")


def save_weights(model, path):
    np.savez(path, *model.export_weights())


# -----------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train fortza-super-tiny")
    p.add_argument("--data",        default="data/data.txt")
    p.add_argument("--out",         default="checkpoints")
    p.add_argument("--epochs",      type=int,   default=1000)
    p.add_argument("--batch",       type=int,   default=64)
    p.add_argument("--lr",          type=float, default=0.001)
    p.add_argument("--log-every",   type=int,   default=1)
    p.add_argument("--temperature", type=float, default=0.8)
    train(p.parse_args())
