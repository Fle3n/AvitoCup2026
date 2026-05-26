"""
Avito ML Cup 2026 — End-to-End Neural Recommender.

Подпакеты пайплайна (вызываются из train.py / predict.py в указанном
порядке):

    build_vocab     -> item_vocab.parquet + feature_dims.json
    build_user_seq  -> user_seq.parquet
    train_model     -> model.pt
    predict_model   -> submission.csv

Каждый модуль самодостаточен и читает/пишет в пути из переменных
окружения  (AVITO_DATA, AVITO_CACHE, AVITO_OUT).
"""
