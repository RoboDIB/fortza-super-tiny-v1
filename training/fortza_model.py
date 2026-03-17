"""
FORTZA Model
------------
Novel conversational architecture for microcontrollers.
First self-learning, persistent-memory chatbot designed for <1MB RAM chips.

Components:
  fortza     — Soul vector (32 floats, persists across turns, learned via RUL)
  zia        — N-gram Context Hash (parameter-free bigram features)
  long_hair  — Asymmetric decoder (hidden=128 > encoder hidden=64)
  rul        — On-device learning update (Residual Update Loop)
  fou        — Soul drift factor (learning rate for fortza)

Architecture per turn:
  input chars
    → zia (bigram hash, param-free) → 32-dim features
    → cat(one-hot, zia) → [Encoder LSTM, hidden=64] → h_enc
    → cat(h_enc, zia_global) = context[96]
    → [Wc bridge: Linear(96→128)] → h_dec_init   (long_hair init)
    → soul_gate(h_enc, fortza_old) → fortza_new
    → [Decoder LSTM (long_hair), hidden=128]
        each step: cat(h_dec, fortza) → [Output Linear(160→vocab)]
    → response chars
    → rul: accumulate soul gradient, apply after turn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ngram import zia_batch, NGRAM_BUCKETS

SOUL_DIM    = 32
ENC_HIDDEN  = 64
DEC_HIDDEN  = 128   # long_hair
FOU_DEFAULT = 0.01  # default soul learning rate


class FORTZAModel(nn.Module):
    def __init__(self, vocab_size,
                 enc_hidden=ENC_HIDDEN,
                 dec_hidden=DEC_HIDDEN,
                 soul_dim=SOUL_DIM,
                 ngram_buckets=NGRAM_BUCKETS):
        super().__init__()
        self.vocab_size    = vocab_size
        self.enc_hidden    = enc_hidden
        self.dec_hidden    = dec_hidden   # long_hair hidden size
        self.soul_dim      = soul_dim
        self.ngram_buckets = ngram_buckets

        # Encoder: cat(one-hot[V], zia[32]) → hidden[64]
        self.encoder = nn.LSTMCell(vocab_size + ngram_buckets, enc_hidden)

        # Soul gate: updates fortza (persistent conversation identity)
        # Input: cat(h_enc[64], fortza[32]) → gate[32] + update[32]
        self.soul_gate = nn.Linear(enc_hidden + soul_dim, soul_dim * 2)

        # Wc bridge: cat(h_enc[64], zia_global[32]) → h_dec_init[128]
        self.Wc = nn.Linear(enc_hidden + ngram_buckets, dec_hidden)

        # long_hair decoder: one-hot[V] → hidden[128]
        self.long_hair = nn.LSTMCell(vocab_size, dec_hidden)

        # Output: cat(h_dec[128], fortza[32]) → vocab logits
        self.output = nn.Linear(dec_hidden + soul_dim, vocab_size)

    # ------------------------------------------------------------------
    # Soul gate — gated update of fortza vector
    # ------------------------------------------------------------------
    def soul_update(self, h_enc, fortza):
        """
        h_enc:  (B, enc_hidden)
        fortza: (B, soul_dim)
        returns fortza_new (B, soul_dim)
        """
        x      = torch.cat([h_enc, fortza], dim=-1)
        out    = self.soul_gate(x)
        gate   = torch.sigmoid(out[:, :self.soul_dim])
        update = torch.tanh(out[:, self.soul_dim:])
        return gate * fortza + (1.0 - gate) * update

    # ------------------------------------------------------------------
    # Forward pass (teacher forcing, batched)
    # inp_onehot:  (B, T_in, V)
    # inp_indices: list[list[int]]  for zia computation
    # tgt:         (B, T_out, V)   decoder inputs (teacher forced)
    # fortza:      (B, soul_dim)   current soul state (zeros for cold start)
    # returns: logits (B, T_out, V), fortza_new (B, soul_dim)
    # ------------------------------------------------------------------
    def forward(self, inp_onehot, inp_indices, tgt, fortza):
        B, T_in, V = inp_onehot.shape
        device = inp_onehot.device

        # ZIA: bigram hash features (B, ngram_buckets)
        zia_np  = zia_batch(inp_indices, self.ngram_buckets)
        zia_t   = torch.tensor(zia_np, dtype=torch.float32, device=device)
        # Expand across encoder timesteps
        zia_exp = zia_t.unsqueeze(1).expand(B, T_in, -1)           # (B, T_in, N)
        enc_in  = torch.cat([inp_onehot, zia_exp], dim=-1)         # (B, T_in, V+N)

        # Encode
        h = torch.zeros(B, self.enc_hidden, device=device)
        c = torch.zeros(B, self.enc_hidden, device=device)
        for t in range(T_in):
            h, c = self.encoder(enc_in[:, t, :], (h, c))

        # Update fortza (soul vector)
        fortza_new = self.soul_update(h, fortza)

        # Wc bridge → long_hair initial state
        ctx = torch.cat([h, zia_t], dim=-1)                        # (B, enc+N=96)
        dh  = torch.tanh(self.Wc(ctx))
        dc  = torch.zeros_like(dh)

        # Decode (long_hair) with teacher forcing
        # fortza injected at output each step
        logits = []
        for t in range(tgt.size(1)):
            dh, dc = self.long_hair(tgt[:, t, :], (dh, dc))
            out_in = torch.cat([dh, fortza_new], dim=-1)           # (B, dec+soul=160)
            logits.append(self.output(out_in))

        return torch.stack(logits, dim=1), fortza_new              # (B, T, V), (B, S)

    # ------------------------------------------------------------------
    # Inference: single sample, fortza passed in and returned
    # ------------------------------------------------------------------
    @torch.no_grad()
    def respond(self, inp_indices, fortza, start_idx, end_idx,
                max_len=120, temperature=0.8, device="cpu"):
        """
        inp_indices: list of int
        fortza:      (1, soul_dim) — current soul, returns updated soul
        returns: (output_indices list, fortza_new tensor)
        """
        V = self.vocab_size
        N = self.ngram_buckets

        from ngram import zia as zia_fn
        zia_np = zia_fn(inp_indices, N)
        zia_t  = torch.tensor(zia_np, dtype=torch.float32, device=device).unsqueeze(0)

        # Encode
        h = torch.zeros(1, self.enc_hidden, device=device)
        c = torch.zeros(1, self.enc_hidden, device=device)
        for idx in inp_indices:
            x_char = F.one_hot(torch.tensor([idx], device=device), V).float()
            x      = torch.cat([x_char, zia_t], dim=-1)
            h, c   = self.encoder(x, (h, c))

        # Update fortza
        fortza_new = self.soul_update(h, fortza)

        # Bridge → long_hair init
        ctx = torch.cat([h, zia_t], dim=-1)
        dh  = torch.tanh(self.Wc(ctx))
        dc  = torch.zeros_like(dh)

        # Decode
        out = []
        tok = torch.tensor([start_idx], device=device)
        repeat_count = 0
        last_idx = -1
        for _ in range(max_len):
            x      = F.one_hot(tok, V).float()
            dh, dc = self.long_hair(x, (dh, dc))
            out_in = torch.cat([dh, fortza_new], dim=-1)
            logits = self.output(out_in).squeeze(0) / temperature
            probs  = torch.softmax(logits, dim=-1)
            tok    = torch.multinomial(probs, 1)
            idx    = tok.item()
            if idx == end_idx:
                break
            # Break on character repeat loop (e.g. "! ! ! ! !")
            if idx == last_idx:
                repeat_count += 1
                if repeat_count >= 3:
                    break
            else:
                repeat_count = 0
                last_idx = idx
            out.append(idx)

        return out, fortza_new

    # ------------------------------------------------------------------
    # Export weights — 8 numpy arrays for quantize.py
    # Order: enc_W, enc_b, soul_gate_W, soul_gate_b,
    #        Wc_W, Wc_b, dec_W, dec_b, out_W, out_b
    # Wait — plan says 8 layers but we have 5 components × 2 = 10.
    # Keep enc, Wc, dec, output (8 arrays = 4 layers × W+b).
    # soul_gate exported separately as layers 5+6 → total 10 arrays, n_layers=10.
    # ------------------------------------------------------------------
    def export_weights(self):
        def lstm_wb(cell):
            W = torch.cat([cell.weight_ih, cell.weight_hh], dim=1)
            b = cell.bias_ih + cell.bias_hh
            return W.detach().cpu().float().numpy(), b.detach().cpu().float().numpy()

        def linear_wb(layer):
            return (layer.weight.detach().cpu().float().numpy(),
                    layer.bias.detach().cpu().float().numpy())

        enc_W,  enc_b  = lstm_wb(self.encoder)
        sg_W,   sg_b   = linear_wb(self.soul_gate)
        Wc_W,   Wc_b   = linear_wb(self.Wc)
        dec_W,  dec_b  = lstm_wb(self.long_hair)
        out_W,  out_b  = linear_wb(self.output)

        # Order matches firmware fortza_model_load() exactly
        return [enc_W, enc_b, sg_W, sg_b, Wc_W, Wc_b, dec_W, dec_b, out_W, out_b]
