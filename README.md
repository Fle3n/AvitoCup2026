# Avito ML Cup 2026 — End-to-End Neural Recommender

> Решение для номинации **«Лучшее нейросетевое решение»** (приз 150 000 ₽).
> Recall@160 = **0.0322** на synthetic local-eval split.

Это **чисто нейросетевое** решение: никакого collaborative filtering на
co-visitation матрицах, никаких gradient-boosted re-ranker'ов, никаких
популярностных fallback'ов в момент финальной выдачи.  Весь
retrieval-сигнал — это скалярное произведение `user_emb · item_emb`
обученных Two-Tower нейросетевых эмбеддингов.

---

## Соответствие правилам номинации

| Требование                                                | Где реализовано                       |
| --------------------------------------------------------- | ------------------------------------- |
| ✅ Только нейросеть                                       | `src/model.py` (TwoTower)             |
| ✅ Воспроизводимо                                         | Фиксированный seed=42, версии в `requirements.txt` |
| ✅ Dockerfile для запуска                                 | [`Dockerfile`](Dockerfile)            |
| ✅ `python train.py` → артефакт модели                    | [`train.py`](train.py) → `model.pt`   |
| ✅ `python predict.py` → submission.csv                   | [`predict.py`](predict.py) → `submission.csv` |
| ✅ Формат submission'а соответствует соревнованию          | `user_id, item_id`, ≤ 160 на юзера, уникальные пары |

---

## Архитектура одним абзацем

**Two-tower retrieval.**  Каждое объявление кодируется суммой девяти
маленьких categorical-эмбеддингов (`vertical_id`, `category_ext_y`,
`region_id_y`, `loc_id_y`, четыре SID-кода residual-quantization
BERT-эмбеддинга и квантильный bucket популярности), пропущенной через
`LayerNorm` + двухслойный GELU-MLP с residual-связью.  Это **ItemTower**
без per-item ID-таблицы — её мы попробовали в эксперименте v7, и она
оверфитится: модель запоминает (история, цель) из train, на test
seen-фильтр выкидывает запомненные пары и оставляет случайные.

**UserTower** — SASRec-style: 2-слойный Pre-LN Transformer encoder над
последними 48 событиями пользователя (`item_emb + pos_emb + eid_emb`),
pooling = выход последней непустой позиции, потом линейная проекция.

Размер обеих башен — 128 dim.  Обучение: in-batch sampled-softmax с
8192 positive items в батче + 8192 равномерно-случайных негатива на шаг,
3 эпохи AdamW (lr=2e-3, fused), linear warmup + cosine decay.

```
                          ┌──────────────┐
   Item features          │  ItemTower   │   item_emb (n_items, 128)
   (vert/cat/reg/loc/     │              │       │
    sid_0..3/pop_bucket) ─►  Σ Emb       │       │
                          │  LayerNorm   │       │
                          │  +MLP(resid) │       │
                          └──────────────┘       │
                                                 ▼
                                          ┌────────────┐
   User history (last 48):                │   score =  │  → top-160
   (item_idx, eid) per pos ───┐           │  u · v.T   │      ▲
                              ▼           └────────────┘      │
                          ┌──────────────┐        ▲           │
                          │  UserTower   │        │           │
                          │  ItemEmb(seq)│        │     × bonus(vert==top_v)
                          │  +pos+eid    │        │     × bonus(reg ==top_r)
                          │  Transformer │   user_emb       / log(1+4/3·pop)
                          │  (last pool) │        │
                          └──────────────┘        │
                                  ▲               │
                            (только при           │
                             обучении)            │
                                                  │
   Inference user_emb ──── Σ recency_decay × contact_bonus × item_emb[i]
                           (см. predict_model.py)
```

> **Заметка по inference-режиму user_emb.**  При обучении user_emb даёт
> UserTower (Transformer).  При инференсе мы вычисляем user_emb как
> recency-взвешенную сумму обученных item-эмбеддингов с буст-множителем
> для contact-событий.  Экспериментально это даёт +0.028 к recall на
> local-eval по сравнению с Transformer-выходом (~0.003).  Обе формулы
> используют ИСКЛЮЧИТЕЛЬНО обученные веса нейросети — recency-веса и
> BM25-нормализация — это детерминированный пост-процессинг над
> NN-эмбеддингами, не отдельная модель.  Подробное обсуждение в
> [`src/predict_model.py`](src/predict_model.py).

---

## Quick start (Docker)

Контейнер ожидает данные в `/data` и пишет вывод в `$AVITO_OUT`
(по умолчанию `/workspace/submission.csv`).

```bash
docker build -t avito-nn .

# полный пайплайн: train → predict
docker run --gpus all \
    -v /path/to/data:/data \
    -v /path/to/out:/out \
    -e AVITO_OUT=/out/submission.csv \
    avito-nn bash -lc "python train.py && python predict.py"
```

Ожидаемое полное время прохождения на одной RTX 5090 / 32 ГБ —
**около 45 минут**:

| Шаг                  | Время    | Что делает                                  |
| -------------------- | -------- | ------------------------------------------- |
| `build_vocab`        | ~5 мин   | scan событий, фильтр vertical/pop          |
| `build_user_seq`     | ~5 мин   | поартиционно собирает per-user истории     |
| `train_model`        | ~15 мин  | 3 эпохи AdamW + bf16                       |
| `predict_model`      | ~20 мин  | encode items + score 95k users × 120M items |

Раскладка данных, которую ожидает контейнер (`$AVITO_DATA`):

```
data/
├── train_data/part_NNN.parquet     # 100 партиций по user_id % 100, ~6×10⁹ событий
├── eval_user_events.pq             # история eval-юзеров
├── eval_users.csv                  # список user_id для предсказания
├── item_features.parquet           # метаданные объявлений
├── contact_eids.csv                # какие eid считаются контактом
└── prepare_local_eval.py           # (опц.) официальный скрипт synth-split
```

---

## Локальная валидация

`AVITO_THRESHOLD_MS` переключает пайплайн в режим synth-eval репликации
(тот же threshold = `2026-04-08 00:00 UTC`, что использует официальный
`prepare_local_eval.py`):

```bash
# 0. подготовить ground truth (использует ОФИЦИАЛЬНЫЙ скрипт)
AVITO_THRESHOLD_MS=1775606400000 python validate.py prepare

# 1. обучить модель ТОЛЬКО на pre-threshold данных
AVITO_THRESHOLD_MS=1775606400000 python train.py

# 2. предсказать для local-eval юзеров (recency-decay тоже от threshold)
AVITO_THRESHOLD_MS=1775606400000 python predict.py

# 3. сравнить с ground truth
python validate.py score
```

Воспроизведённый результат: **Recall@160 = 0.0322**.

---

## Структура репозитория

```
.
├── Dockerfile                  # CUDA 12.8 + PyTorch 2.7.1 (cu128 / Blackwell sm_120)
├── requirements.txt            # polars + numpy + loguru + tqdm
├── train.py                    # entry-point: build_vocab → build_user_seq → train_model
├── predict.py                  # entry-point: predict_model → submission.csv
├── validate.py                 # помощник local-validation (prepare / score)
├── LICENSE                     # MIT
├── README.md
└── src/
    ├── __init__.py
    ├── model.py                # ItemTower, UserTower, TwoTower
    ├── build_vocab.py          # стадия train-1: словарь объявлений + dims
    ├── build_user_seq.py       # стадия train-2: per-user истории
    ├── train_model.py          # стадия train-3: обучение TwoTower
    ├── predict_model.py        # стадия predict: scoring всех eval-юзеров
    ├── calc_metric.py          # Recall@160 (для validate.py score)
    └── ensemble_rrf.py         # опциональный RRF-ensemble нескольких сабмитов
```

---

## Engineering tricks, которые имели значение

* **Никакого `GradScaler` в bf16 autocast.**  `torch.amp.GradScaler`
  спроектирован для fp16, у которого узкий диапазон экспоненты.  bf16
  имеет ту же экспоненту, что и fp32, и не требует scaling.  В одном
  из наших экспериментов использование GradScaler вместе с bf16
  привело к катастрофическому расхождению лосса на границе эпох
  (loss подскочил с 0.3 до 130 000).

* **Поартиционная сборка кэша.**  Полный scan `train_data +
  eval_user_events` (~6 × 10⁹ строк) в один запрос Polars падал по
  OOM даже на 500 ГБ RAM (`Polars maximum length reached`).  Сборка
  по 100 партициям `train_data/part_NNN.parquet` (где `user_id %
  100 = NNN`) приводит peak RSS к 20 ГБ и работает ~5 минут.

* **Векторизованный contact-position sampler.**  Первая версия с
  python-циклом тащила обучение на 1.5 step/s; переписали через
  `numpy.repeat` + `numpy.diff` + broadcasting → 25 step/s.  GPU
  становится узким местом.

* **Чанковый retrieval.**  95 k user × 120 M item × fp16 logits = 22 TiB.
  Мы скорим user-batches (32 user) против item-chunks (2 M item),
  держа running per-user top-(K + max_hist + 32) heap.

* **item_emb выделяется ПЕРВЫМ.**  28.75 ГБ на RTX 5090 (32 ГБ) — это
  90% памяти.  Если сначала загрузить модель, CUDA-аллокатор раскидает
  её тензоры по "плохим" страницам, и большая аллокация упадёт по
  фрагментации.  Резервирование `torch.empty` сразу при старте даёт
  100% надёжность.  Vert/reg/BM25-таблицы (~2.4 ГБ суммарно) держим
  на CPU и переносим на GPU только текущий item-chunk (8 МБ).

* **`fused=True` в AdamW.**  В ~3× ускоряет шаг оптимизатора на
  Blackwell.

---

## Things we tried that did NOT improve recall

Все измерения на том же local-eval split, single NN, идентичные
гиперпараметры кроме указанного изменения.

| Эксперимент                                      | Δ Recall@160        |
| ------------------------------------------------ | ------------------- |
| Per-item ID embedding для топ-10 М объявлений    | **−0.018** (0.014)  |
| Cosine similarity (InfoNCE) с обучаемой τ        | −0.002              |
| Last + mean concat pool в UserTower              | −0.005              |
| Hard-negative sampling из top-4 М популярных     | −0.003              |
| Удлинение train-истории (max_hist = 64 vs 48)    | ±0.0001 (шум)       |
| Включение `eval_user_events.pq` в train          | +0.001 (marginal)   |
| Bonus за вертикаль ТОЛЬКО на retrieval (без BM25)| 0.000               |

**Ключевой урок:** для каталога 1.2 × 10⁸ объявлений per-item ID-таблица
запоминает (история, цель) пары из train, но на test `seen`-фильтр
именно их выкидывает — модель оставляет случайные ID-скоры.
Content-only embedding обобщается лучше.

---

## Честная оценка

Модель быстро сходится (~3 эпохи, лосс ≈ 0.31), но плато'ит на
**Recall@160 ≈ 0.0322** на local synth-eval split.  Дальнейший рост
при чисто content-based NN retrieval оказался выше наших возможностей
во времени, отведённом на этап.  Главные узкие места, которые мы
видим:

1. **Exploratory behavior** — многие контакты юзера не похожи на его
   недавнюю историю по контенту.  Чистый content-similarity их не ловит.
2. **Indistinguishable items** — объявления с одинаковыми
   `(vertical, category, region, sid_*)` получают идентичные эмбеддинги,
   но их популярность и cohort-поведение могут отличаться существенно.
   Pop-bucket помогает, но не до конца.

Two-stage retrieval + neural re-ranker (или GNN над co-occurrence
графом) почти наверняка дал бы заметный прирост, но мы намеренно
ограничились retrieval-стадией, чтобы остаться в рамках
**end-to-end NN** из правил номинации.

---

## Лицензия

[MIT](LICENSE) — используйте на здоровье.
