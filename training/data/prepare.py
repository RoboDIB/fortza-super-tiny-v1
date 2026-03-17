"""
fortza-super-tiny
Dataset preparation — downloads and converts top conversational datasets
to the fortza format (you: ...\nbot: ...\n)

Dataset: PersonaChat (default)
  HuggingFace: AlekseyKorshuk/persona-chat
  ~131k dialogue pairs, personality-consistent conversation

Usage:
  pip install datasets
  python data/prepare.py                        # PersonaChat, default cap
  python data/prepare.py --max 10000            # cap at 10k pairs
  python data/prepare.py --source daily         # switch to DailyDialog
  python data/prepare.py --source empathetic    # switch to Empathetic Dialogues
"""

import argparse
import random
import os

random.seed(42)


# -----------------------------------------------------------------------
# Writers
# -----------------------------------------------------------------------

def write_pairs(pairs, path):
    with open(path, "w", encoding="utf-8") as f:
        for inp, resp in pairs:
            inp  = inp.strip().lower()
            resp = resp.strip().lower()
            if not inp or not resp:
                continue
            # skip very long turns — tiny model can't handle them well
            if len(inp) > 120 or len(resp) > 120:
                continue
            f.write(f"you: {inp}\nbot: {resp}\n")
    print(f"  wrote {len(pairs)} pairs → {path}")


# -----------------------------------------------------------------------
# DailyDialog
# HuggingFace: li2017dailydialog/daily_dialog
# Format: each example has a `dialog` list of utterances (alternating speakers)
# -----------------------------------------------------------------------

def load_daily_dialog(max_pairs=None):
    from datasets import load_dataset
    print("[DailyDialog] loading...")
    ds = load_dataset("li2017dailydialog/daily_dialog", split="train",
                      trust_remote_code=True)
    pairs = []
    for ex in ds:
        turns = ex["dialog"]
        for i in range(len(turns) - 1):
            pairs.append((turns[i], turns[i + 1]))
    random.shuffle(pairs)
    if max_pairs:
        pairs = pairs[:max_pairs]
    print(f"  {len(pairs)} pairs extracted")
    return pairs


# -----------------------------------------------------------------------
# PersonaChat
# HuggingFace: AlekseyKorshuk/persona-chat
# Format: utterances[i].history = list of prior turns,
#         utterances[i].candidates[-1] = gold response
# -----------------------------------------------------------------------

def load_persona_chat(max_pairs=None):
    from datasets import load_dataset
    print("[PersonaChat] loading...")
    ds = load_dataset("AlekseyKorshuk/persona-chat", split="train")
    pairs = []
    for ex in ds:
        for utt in ex["utterances"]:
            history    = utt["history"]
            candidates = utt["candidates"]
            if not history or not candidates:
                continue
            inp  = history[-1]
            resp = candidates[-1]          # last candidate is the gold response
            pairs.append((inp, resp))
    random.shuffle(pairs)
    if max_pairs:
        pairs = pairs[:max_pairs]
    print(f"  {len(pairs)} pairs extracted")
    return pairs


# -----------------------------------------------------------------------
# Empathetic Dialogues
# HuggingFace: facebook/empathetic_dialogues
# Format: rows are individual utterances; alternate speaker_idx 0/1
# -----------------------------------------------------------------------

def load_empathetic(max_pairs=None):
    from datasets import load_dataset
    print("[Empathetic Dialogues] loading...")
    ds = load_dataset("facebook/empathetic_dialogues", split="train")

    # Group utterances by conv_id
    convs = {}
    for row in ds:
        cid = row["conv_id"]
        if cid not in convs:
            convs[cid] = []
        convs[cid].append((row["utterance_idx"], row["utterance"]))

    pairs = []
    for cid, turns in convs.items():
        turns.sort(key=lambda t: t[0])
        utts = [t[1] for t in turns]
        for i in range(len(utts) - 1):
            pairs.append((utts[i], utts[i + 1]))

    random.shuffle(pairs)
    if max_pairs:
        pairs = pairs[:max_pairs]
    print(f"  {len(pairs)} pairs extracted")
    return pairs


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

LOADERS = {
    "daily":      (load_daily_dialog, 15000),
    "persona":    (load_persona_chat, 10000),
    "empathetic": (load_empathetic,    5000),
}


def main(args):
    out_dir = os.path.dirname(os.path.abspath(__file__))

    source = args.source
    loader, default_max = LOADERS[source]
    cap = args.max if args.max else default_max
    pairs = loader(max_pairs=cap)

    out_path = os.path.join(out_dir, "data.txt")
    write_pairs(pairs, out_path)
    print(f"\n[fortza-super-tiny] dataset ready → data/data.txt")
    print("next: python train.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare datasets for fortza-super-tiny")
    p.add_argument("--source", choices=["daily", "persona", "empathetic"],
                   default="persona",
                   help="which dataset to use (default: persona)")
    p.add_argument("--max",   type=int, default=0,
                   help="cap total pairs (0 = use dataset default)")
    main(p.parse_args())
