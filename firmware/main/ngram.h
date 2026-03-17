/*
 * ZIA — N-gram Context Hash (firmware)
 * Parameter-free bigram features. Hash constants MUST match ngram.py exactly:
 *   bucket = (a * 31337 + b * 17) % NGRAM_BUCKETS
 */
#pragma once
#include <stdint.h>

#define NGRAM_BUCKETS 32

/*
 * Compute normalized bigram hash features from a token index array.
 * out: float array of size NGRAM_BUCKETS, caller-allocated.
 */
void zia(const uint16_t *indices, int len, float *out);
