"""
train.py — entry-point верхнего уровня обучения.

Требование номинации: «python train.py» должен прочитать сырые данные
и положить артефакт обученной модели.

Запускает три стадии пайплайна по очереди:

    1.  src.build_vocab        scan train_data → item_vocab.parquet
                                                  feature_dims.json
                                            (≈5 минут, peak RAM ~20 ГБ)

    2.  src.build_user_seq     поартиционно собирает per-user истории
                                для всех пользователей с >= 1 контактом
                                в pre-threshold окне.
                                                  user_seq.parquet
                                            (≈3-5 минут, peak RAM ~20 ГБ)

    3.  src.train_model        обучение TwoTower (ItemTower + UserTower)
                                3 эпохи AdamW + warmup + cosine.
                                                  model.pt
                                            (≈10-15 минут на RTX 5090)

Все промежуточные артефакты пишутся в $AVITO_CACHE
(по умолчанию /workspace/cache).  Данные ожидаются в $AVITO_DATA
(по умолчанию /data).

ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ
--------------------
    AVITO_DATA          — где лежат данные   (default: /data)
    AVITO_CACHE         — куда писать кэш    (default: /workspace/cache)
    AVITO_THRESHOLD_MS  — отсечка времени для local-validation:
                          0 (по умолчанию) — брать все события (production);
                          1775606400000    — 2026-04-08 00:00 UTC
                                              (synth threshold организатора).

Пример запуска в Docker-контейнере:
    docker run --gpus all \\
        -v /path/to/data:/data \\
        -v /path/to/cache:/workspace/cache \\
        avito-nn python train.py
"""
from src import build_vocab, build_user_seq, train_model

if __name__ == "__main__":
    print("=" * 64)
    print("STEP 1/3 — build_vocab")
    print("Сканирует все события и составляет таблицу объявлений-кандидатов.")
    print("=" * 64)
    build_vocab.main()

    print()
    print("=" * 64)
    print("STEP 2/3 — build_user_seq")
    print("Поартиционно собирает истории пользователей с >=1 контактом.")
    print("=" * 64)
    build_user_seq.main()

    print()
    print("=" * 64)
    print("STEP 3/3 — train_model")
    print("Обучает TwoTower NN: ItemTower + UserTower → dot-product.")
    print("=" * 64)
    train_model.train(train_model.parse_args())
