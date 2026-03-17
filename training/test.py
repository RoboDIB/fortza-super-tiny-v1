"""
fortza-super-tiny — Test Script
Evaluates model quality before flashing to ESP32.

Tests:
  1. Cold start responses (soul=0)
  2. Multi-turn soul evolution (does soul actually change responses?)
  3. Repetition rate (is it stuck in loops?)
  4. Vocabulary diversity (unique chars/words per response)
  5. Response length distribution
  6. Soul stability (does soul norm grow and stabilize?)

Usage:
  python test.py
  python test.py --weights checkpoints/weights_best.npz --turns 20
"""

import argparse
import json
import math
import numpy as np
import torch
from collections import Counter

from tokenizer import Tokenizer
from fortza_model import FORTZAModel, SOUL_DIM
from chat import load_model


# -----------------------------------------------------------------------
# Test conversations
# -----------------------------------------------------------------------

COLD_START_INPUTS = [
    "hey",
    "how are you",
    "what do you do for fun",
    "do you have any pets",
    "what kind of music do you like",
    "do you like to cook",
    "what is your favorite movie",
    "tell me something interesting",
    "do you exercise",
    "what did you do today",
]

MULTI_TURN = [
    "hey how are you",
    "i love hiking in the mountains",
    "do you like the outdoors",
    "i have a dog named max",
    "what do you do for work",
    "i work in tech, it is pretty fun",
    "do you like music",
    "i play guitar sometimes",
    "cool, what else do you do",
    "i like cooking too, especially pasta",
    "nice, sounds like you stay busy",
    "yeah but i always make time for friends",
    "that is important",
    "what about you, any hobbies",
    "do you watch movies",
]


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def repetition_rate(text):
    """Fraction of repeated bigrams in response."""
    words = text.split()
    if len(words) < 2:
        return 0.0
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    if not bigrams:
        return 0.0
    return 1.0 - len(set(bigrams)) / len(bigrams)


def word_diversity(text):
    """Unique word ratio."""
    words = text.split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def is_gibberish(text, threshold=0.35):
    """
    Simple gibberish detector.
    Checks what fraction of character bigrams are common English bigrams.
    """
    common_bigrams = {
        'th','he','in','er','an','re','on','en','at','es',
        'ed','nd','to','ou','ea','ha','ng','as','or','it',
        'is','hi','et','ar','ne','st','nt','al','le','se',
        'de','me','ve','ly','ri','ro','li','ll','la','si',
        'ti','ma','ca','io','ra','no','so','go','do','ho',
        'be','we','te','ge','le','ke','pe','fe','ce','re'
    }
    if len(text) < 2:
        return True
    chars = text.replace(' ', '').lower()
    bigrams = [chars[i:i+2] for i in range(len(chars)-1)]
    if not bigrams:
        return True
    matches = sum(1 for b in bigrams if b in common_bigrams)
    return (matches / len(bigrams)) < threshold


def perplexity_approx(text, char_freq):
    """Approximate perplexity using unigram character model."""
    if not text:
        return float('inf')
    log_prob = 0.0
    for ch in text:
        p = char_freq.get(ch, 1e-6)
        log_prob += math.log(p)
    return math.exp(-log_prob / len(text))


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

def test_cold_start(model, tok, device, temperature):
    print("\n" + "="*60)
    print("TEST 1: Cold Start Responses (soul=0)")
    print("="*60)

    results = []
    for inp in COLD_START_INPUTS:
        fortza = torch.zeros(1, SOUL_DIM, device=device)
        out_idx, _ = model.respond(
            tok.encode(inp), fortza,
            tok.start_idx, tok.end_idx,
            temperature=temperature, device=device
        )
        resp = tok.decode(out_idx)
        rep  = repetition_rate(resp)
        div  = word_diversity(resp)
        gib  = is_gibberish(resp)
        results.append((inp, resp, rep, div, gib))
        flag = "⚠ " if gib else "  "
        print(f"  {flag}you: {inp}")
        print(f"     bot: {resp}")

    good   = sum(1 for r in results if not r[4])
    avg_len = sum(len(r[1].split()) for r in results) / len(results)
    avg_rep = sum(r[2] for r in results) / len(results)
    print(f"\n  coherent: {good}/{len(results)}  "
          f"avg_words: {avg_len:.1f}  avg_repetition: {avg_rep:.3f}")
    return results


def test_soul_evolution(model, tok, device, temperature):
    print("\n" + "="*60)
    print("TEST 2: Multi-Turn Soul Evolution")
    print("="*60)
    print("  (watching how soul changes response quality over turns)")

    fortza = torch.zeros(1, SOUL_DIM, device=device)
    soul_norms = []

    for turn, inp in enumerate(MULTI_TURN):
        out_idx, fortza = model.respond(
            tok.encode(inp), fortza,
            tok.start_idx, tok.end_idx,
            temperature=temperature, device=device
        )
        resp = tok.decode(out_idx)
        norm = fortza.norm().item()
        soul_norms.append(norm)
        gib  = is_gibberish(resp)
        flag = "⚠" if gib else "✓"
        print(f"  [{flag}] turn {turn+1:>2}  soul:{norm:.4f}  you: {inp[:40]}")
        print(f"            bot: {resp}")

    print(f"\n  soul norm start: {soul_norms[0]:.4f} → end: {soul_norms[-1]:.4f}")
    growing = soul_norms[-1] > soul_norms[0]
    print(f"  soul growing: {'✓ yes' if growing else '✗ no — RUL may not be working'}")
    return soul_norms


def test_repetition_stress(model, tok, device, temperature, n=30):
    print("\n" + "="*60)
    print(f"TEST 3: Repetition Stress ({n} turns, same input)")
    print("="*60)
    print("  (does soul prevent the model getting stuck?)")

    inp = "hey how are you doing"
    fortza = torch.zeros(1, SOUL_DIM, device=device)
    responses = []

    for _ in range(n):
        out_idx, fortza = model.respond(
            tok.encode(inp), fortza,
            tok.start_idx, tok.end_idx,
            temperature=temperature, device=device
        )
        responses.append(tok.decode(out_idx))

    unique = len(set(responses))
    most_common, count = Counter(responses).most_common(1)[0]
    print(f"  unique responses: {unique}/{n}")
    print(f"  most repeated: '{most_common}' ({count}x)")
    print(f"  diversity: {'✓ good' if unique > n * 0.4 else '⚠ low — model may be stuck'}")
    return responses


def test_quantitative(model, tok, device, temperature):
    print("\n" + "="*60)
    print("TEST 4: Quantitative Summary")
    print("="*60)

    all_inputs = COLD_START_INPUTS + MULTI_TURN
    fortza = torch.zeros(1, SOUL_DIM, device=device)
    responses = []

    for inp in all_inputs:
        out_idx, fortza = model.respond(
            tok.encode(inp), fortza,
            tok.start_idx, tok.end_idx,
            temperature=temperature, device=device
        )
        responses.append(tok.decode(out_idx))

    lengths     = [len(r.split()) for r in responses]
    char_lengths= [len(r) for r in responses]
    rep_rates   = [repetition_rate(r) for r in responses]
    diversities = [word_diversity(r) for r in responses]
    gibberish_n = sum(1 for r in responses if is_gibberish(r))
    empty_n     = sum(1 for r in responses if len(r.strip()) == 0)

    print(f"  total responses     : {len(responses)}")
    print(f"  empty responses     : {empty_n}")
    print(f"  gibberish (est.)    : {gibberish_n} / {len(responses)}  "
          f"({100*gibberish_n/len(responses):.0f}%)")
    print(f"  avg word length     : {sum(lengths)/len(lengths):.1f} words")
    print(f"  avg char length     : {sum(char_lengths)/len(char_lengths):.1f} chars")
    print(f"  avg repetition rate : {sum(rep_rates)/len(rep_rates):.3f}  (0=none, 1=all repeated)")
    print(f"  avg word diversity  : {sum(diversities)/len(diversities):.3f}  (1=all unique)")

    verdict = "PASS" if gibberish_n < len(responses) * 0.5 else "NEEDS MORE TRAINING"
    print(f"\n  VERDICT: {verdict}")
    if verdict != "PASS":
        print("  → Try: python train.py --epochs 300 (continue training)")
        print("  → Or:  use a smaller, cleaner dataset")

    return responses


# -----------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fortza-super-tiny test] device: {device}")
    print(f"  weights : {args.weights}")
    print(f"  temp    : {args.temperature}")

    model, tok = load_model(args.weights, args.vocab)
    model = model.to(device)
    model.eval()

    test_cold_start(model, tok, device, args.temperature)
    soul_norms = test_soul_evolution(model, tok, device, args.temperature)
    test_repetition_stress(model, tok, device, args.temperature, n=args.turns)
    test_quantitative(model, tok, device, args.temperature)

    print("\n" + "="*60)
    print("Done. If results look good → python quantize.py → flash to ESP32")
    print("="*60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test fortza-super-tiny quality")
    p.add_argument("--weights",     default="checkpoints/weights_best.npz")
    p.add_argument("--vocab",       default="checkpoints/vocab.json")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--turns",       type=int,   default=30)
    main(p.parse_args())
