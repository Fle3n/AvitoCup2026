"""
predict.py — entry-point верхнего уровня инференса.

Требование номинации: «python predict.py» должен прочитать обученную
модель и положить рядом submission.csv в формате соревнования
(колонки user_id, item_id; не более 160 строк на пользователя).

Что внутри:
    src.predict_model.main() — единственный шаг.
        1) загружает обученную TwoTower из $AVITO_CACHE/model.pt;
        2) ItemTower → эмбеддит весь каталог (≈30 ГБ fp16 на GPU);
        3) сканирует pre-threshold события для всех eval-юзеров и
           считает per-user recency-взвешенный user_emb;
        4) скорит чанками `user_emb · item_emb.T` с BM25-нормализацией
           и vertical/region-бонусами;
        5) убирает уже виденные item'ы и пишет top-160 в submission.csv.

ВХОДНЫЕ ФАЙЛЫ ($AVITO_DATA, по умолчанию /data):
    eval_users.csv             — список user_id для предсказания
    eval_user_events.pq        — их история взаимодействий
    train_data/part_*.parquet  — общая история всех юзеров
    item_features.parquet      — метаданные объявлений

КЭШ ($AVITO_CACHE, по умолчанию /workspace/cache):
    item_vocab.parquet         — словарь объявлений (из build_vocab)
    feature_dims.json          — cardinality фич
    model.pt                   — обученная модель (из train_model)

ВЫХОД:
    $AVITO_OUT  (по умолчанию /workspace/submission.csv)

Параметры окружения, которые меняют поведение:
    AVITO_THRESHOLD_MS=1775606400000   — режим local validation
                                          (берём только pre-threshold события)
"""
from src import predict_model

if __name__ == "__main__":
    print("=" * 64)
    print("predict_model — NN-скоринг всех eval-юзеров против каталога")
    print("=" * 64)
    predict_model.main()
