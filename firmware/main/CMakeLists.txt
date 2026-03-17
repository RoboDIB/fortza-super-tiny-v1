idf_component_register(
    SRCS
        "main.c"
        "lstm.c"
        "ngram.c"
        "fortza_model.c"
        "soul.c"
        "learner.c"
    INCLUDE_DIRS "."
    REQUIRES spiffs driver nvs_flash
)
