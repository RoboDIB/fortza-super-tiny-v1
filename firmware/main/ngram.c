/*
 * ZIA — N-gram Context Hash (firmware)
 * Identical hash constants to training/ngram.py.
 */
#include "ngram.h"
#include <string.h>

void zia(const uint16_t *indices, int len, float *out)
{
    memset(out, 0, NGRAM_BUCKETS * sizeof(float));
    if (len < 2) return;

    float total = 0.0f;
    for (int i = 0; i < len - 1; i++) {
        uint32_t bucket = ((uint32_t)indices[i] * 31337u
                         + (uint32_t)indices[i + 1] * 17u) % NGRAM_BUCKETS;
        out[bucket] += 1.0f;
        total        += 1.0f;
    }
    if (total > 0.0f) {
        for (int i = 0; i < NGRAM_BUCKETS; i++)
            out[i] /= total;
    }
}
