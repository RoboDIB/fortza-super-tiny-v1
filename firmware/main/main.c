/*
 * fortza-super-tiny — main
 *
 * FORTZA architecture on ESP32:
 *   fortza  — soul vector, persists in NVS across power cycles
 *   zia     — n-gram context hash features
 *   long_hair — asymmetric decoder (hidden=128)
 *   rul     — on-device soul learning after every turn
 *   fou     — soul drift factor (FOU_DEFAULT = 0.01)
 *
 * Commands over UART (115200 baud):
 *   Anything → bot responds and soul updates (rul)
 *   "good" / "yes" / "nice" → positive feedback: fine-tunes output layer
 *   "no" / "wrong" / "bad"  → negative feedback: inhibits soul
 *   ".reset"  → clear soul vector
 *   ".soul"   → print soul norm
 *   ".save"   → force save learned weights
 */
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_spiffs.h"
#include "nvs_flash.h"
#include "fortza_model.h"
#include "learner.h"

#define TAG         "fortza"
#define VOCAB_PATH  "/spiffs/vocab.json"
#define MODEL_PATH  "/spiffs/weights.bin"
#define MAX_INPUT   128
#define MAX_OUTPUT  200
#define UART_NUM    UART_NUM_0

/* ------------------------------------------------------------------ */
/* Minimal vocab loader (char↔index map from vocab.json)               */
/* ------------------------------------------------------------------ */

#define MAX_VOCAB_SIZE 128
static char vocab_char[MAX_VOCAB_SIZE];
static int  vocab_len = 0;
static int  idx_start = 1, idx_end = 2, idx_unk = 3;

static void vocab_load(void)
{
    FILE *f = fopen(VOCAB_PATH, "r");
    if (!f) { ESP_LOGE(TAG, "vocab not found"); return; }
    char line[64];
    while (fgets(line, sizeof(line), f)) {
        char *q1 = strchr(line, '"'); if (!q1) continue;
        char *q2 = strchr(q1 + 1, '"'); if (!q2) continue;
        if (q2 - q1 - 1 != 1) continue;   /* skip multi-char special tokens */
        char ch = q1[1];
        char *colon = strchr(q2, ':'); if (!colon) continue;
        int idx = atoi(colon + 1);
        if (idx >= 0 && idx < MAX_VOCAB_SIZE) {
            vocab_char[idx] = ch;
            if (idx + 1 > vocab_len) vocab_len = idx + 1;
        }
    }
    fclose(f);
    ESP_LOGI(TAG, "vocab: %d entries", vocab_len);
}

static uint16_t char_to_idx(char c)
{
    for (int i = 4; i < vocab_len; i++)
        if (vocab_char[i] == c) return (uint16_t)i;
    return (uint16_t)idx_unk;
}

/* ------------------------------------------------------------------ */
/* UART                                                                 */
/* ------------------------------------------------------------------ */

static void uart_setup(void)
{
    uart_config_t cfg = {
        .baud_rate  = 115200,
        .data_bits  = UART_DATA_8_BITS,
        .parity     = UART_PARITY_DISABLE,
        .stop_bits  = UART_STOP_BITS_1,
        .flow_ctrl  = UART_HW_FLOWCTRL_DISABLE,
    };
    uart_param_config(UART_NUM, &cfg);
    uart_driver_install(UART_NUM, 256, 0, 0, NULL, 0);
}

static int uart_readline(char *buf, int maxlen)
{
    int n = 0; uint8_t ch;
    while (n < maxlen - 1)
        if (uart_read_bytes(UART_NUM, &ch, 1, portMAX_DELAY) > 0) {
            if (ch == '\n' || ch == '\r') break;
            buf[n++] = (char)ch;
        }
    buf[n] = '\0';
    return n;
}

static void uart_print(const char *s)
{
    uart_write_bytes(UART_NUM, s, strlen(s));
}

/* ------------------------------------------------------------------ */
/* SPIFFS + NVS                                                         */
/* ------------------------------------------------------------------ */

static void storage_init(void)
{
    nvs_flash_init();

    esp_vfs_spiffs_conf_t conf = {
        .base_path              = "/spiffs",
        .partition_label        = NULL,
        .max_files              = 6,
        .format_if_mount_failed = false,
    };
    esp_vfs_spiffs_register(&conf);
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */

void app_main(void)
{
    uart_setup();
    storage_init();
    vocab_load();

    FORTZAModel model;
    if (fortza_model_load(&model, MODEL_PATH) != 0) {
        uart_print("error: weights.bin not found. flash it first.\r\n");
        return;
    }

    LearnerState learner;
    learner_init(&learner, &model);

    char soul_msg[64];
    float norm = 0.0f;
    for (int i = 0; i < SOUL_DIM; i++)
        norm += learner.soul.fortza[i] * learner.soul.fortza[i];
    norm = sqrtf(norm);
    snprintf(soul_msg, sizeof(soul_msg),
             "[fortza soul norm: %.3f]\r\n", norm);

    uart_print("\r\n[fortza-super-tiny] ready\r\n");
    uart_print(soul_msg);
    uart_print("say something...\r\n\r\n");

    char    input[MAX_INPUT];
    char    output[MAX_OUTPUT];
    uint16_t enc_buf[MAX_INPUT];
    float   zia_feats[NGRAM_BUCKETS];
    float   probs[MAX_VOCAB_SIZE];

    while (1) {
        uart_print("you: ");
        int len = uart_readline(input, MAX_INPUT);
        if (len == 0) continue;
        for (int i = 0; i < len; i++) input[i] = tolower((unsigned char)input[i]);
        uart_print("\r\n");

        /* --- dot commands --- */
        if (strcmp(input, ".reset") == 0) {
            memset(learner.soul.fortza, 0, sizeof(learner.soul.fortza));
            soul_save(&learner.soul);
            uart_print("[fortza reset]\r\n\r\n");
            continue;
        }
        if (strcmp(input, ".soul") == 0) {
            float n2 = 0.0f;
            for (int i = 0; i < SOUL_DIM; i++)
                n2 += learner.soul.fortza[i] * learner.soul.fortza[i];
            snprintf(soul_msg, sizeof(soul_msg),
                     "[soul norm: %.4f]\r\n\r\n", sqrtf(n2));
            uart_print(soul_msg);
            continue;
        }
        if (strcmp(input, ".save") == 0) {
            learner_save_wo(&learner);
            soul_save(&learner.soul);
            uart_print("[saved]\r\n\r\n");
            continue;
        }

        /* --- feedback detection --- */
        int feedback = learner_detect_feedback(input);
        if (feedback != 0) {
            learner_end_turn(&learner, feedback);
            if (feedback == +1)
                uart_print("[learning: good response remembered]\r\n\r\n");
            else
                uart_print("[learning: noted, won't repeat]\r\n\r\n");
            continue;
        }

        /* --- encode --- */
        for (int i = 0; i < len; i++) enc_buf[i] = char_to_idx(input[i]);
        learner_begin_turn(&learner);
        fortza_encode(&model, enc_buf, len, zia_feats);
        fortza_soul_gate(&model, model.encoder.h, learner.soul.fortza);
        fortza_bridge(&model, model.encoder.h, zia_feats);

        /* --- decode --- */
        uint16_t tok = (uint16_t)idx_start;
        int out_len  = 0;
        int repeat_count = 0;
        uint16_t last_tok = 0;
        while (out_len < MAX_OUTPUT - 1) {
            tok = fortza_decode_step(&model, tok, learner.soul.fortza, probs);
            if (tok == (uint16_t)idx_end || tok == 0) break;

            /* Break on character repeat loop (e.g. "! ! ! ! !") */
            if (tok == last_tok) { if (++repeat_count >= 3) break; }
            else                 { repeat_count = 0; last_tok = tok; }

            /* RUL: accumulate soul gradient this step */
            learner_step(&learner, model.long_hair.h, probs, tok);

            char ch = (tok < (uint16_t)vocab_len) ? vocab_char[tok] : '?';
            output[out_len++] = ch;
        }
        output[out_len] = '\0';

        /* --- apply implicit soul learning (rul, no fine-tune) --- */
        learner_end_turn(&learner, 0);

        uart_print("bot: ");
        uart_print(output);
        uart_print("\r\n\r\n");
    }
}
