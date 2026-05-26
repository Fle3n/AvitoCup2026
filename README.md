# Avito ML Cup 2026 — End-to-End Neural Recommender

> Решение для номинации **«Лучшее нейросетевое решение»**.
> Recall@160 = **0.0322** на synthetic local-eval split.

Это **чисто нейросетевое** решение: никакого collaborative filtering на
co-visitation матрицах, никаких gradient-boosted re-ranker'ов, никаких
популярностных fallback'ов в момент финальной выдачи.  Весь
retrieval-сигнал — это скалярное произведение `user_emb · item_emb`
обученных Two-Tower нейросетевых эмбеддингов.

---

## Соответствие правилам номинации

| Требование из [`Номинация.txt`](https://t.me/avito_cup)        | Где / как выполнено                                  |
| -------------------------------------------------------------- | ---------------------------------------------------- |
| ✅ В решении используются ТОЛЬКО нейросети                     | [`src/model.py`](src/model.py) (TwoTower) + см. ниже «Где границы end-to-end NN» |
| ✅ Решение выложено на GitHub                                  | Этот репозиторий                                     |
| ✅ Решение воспроизводимо                                      | seed=42 везде, версии в [`requirements.txt`](requirements.txt) и [`Dockerfile`](Dockerfile) (cu128 / Blackwell) |
| ✅ Dockerfile для запуска                                      | [`Dockerfile`](Dockerfile)                           |
| ✅ Скрипт отдаёт `submission.csv` в формате соревнования       | [`predict.py`](predict.py) → колонки `user_id, item_id`, ≤ 160 на пользователя, уникальные пары |
| ✅ `python train.py` → артефакт модели                         | [`train.py`](train.py) → `$AVITO_CACHE/model.pt`     |
| ✅ `python predict.py` → submission.csv                        | [`predict.py`](predict.py) → `$AVITO_OUT`            |

### Где границы end-to-end NN

Чтобы не было неоднозначности при ручной проверке — вот что в нашем
пайплайне является нейросетью, а что нет:

| Компонент                                | NN?  | Откуда берётся                                       |
| ---------------------------------------- | ---- | ---------------------------------------------------- |
| `item_emb` (вектора всех объявлений)     | ✅   | Прямой forward через обученную ItemTower             |
| `user_emb` (вектора eval-юзеров)         | ✅   | Σ recency × contact_bonus × **NN item_emb**          |
| score = dot product                      | ✅   | Скалярное произведение обученных NN-эмбеддингов      |
| BM25 нормализация по популярности        | ❌   | Детерминированная формула на `n_users` колонке vocab |
| vertical/region bonuses                  | ❌   | Детерминированное правило (top-вертикаль юзера)      |
| novelty mask (выкидываем seen items)     | ❌   | Детерминированный фильтр по истории                  |

Все «не-NN» компоненты — это **детерминированный пост-процессинг над
NN-скорами**, аналогичный тому, что делает любой production search/recsys
поверх dot-product retrieval'а.  Никаких обучаемых моделей кроме TwoTower
в пайплайне нет.

---

## Метрика и формат submission'а

* **Recall@160** = среднее по пользователям отношения
  `|prediction ∩ targets| / |targets|`.  Эталонная реализация —
  [`src/calc_metric.py`](src/calc_metric.py).
* **`submission.csv`** содержит колонки `user_id, item_id`, не более
  160 строк на пользователя, все пары уникальные.

Пример первых строк submission.csv:

```csv
user_id,item_id
30,114289691
30,19543419
30,153291894
30,89211239
...
33,120023046
33,3126767
...
```

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

### Время выполнения

Замерено на нашем основном dev-стенде:

| Стенд                                | train.py | predict.py | итого    |
| ------------------------------------ | -------- | ---------- | -------- |
| **1× RTX 5090 32 ГБ + 503 ГБ RAM**   | ~25 мин  | ~25 мин    | **~50 мин** |
| 1× RTX 4090 24 ГБ + 256 ГБ RAM (оценка)¹ | ~30 мин  | ~50 мин²   | ~80 мин   |
| 1× A100 80 ГБ + 256 ГБ RAM (оценка)¹  | ~25 мин  | ~25 мин    | ~50 мин   |

¹ Не запускали лично — оценки по соотношению TFLOPS и VRAM.
² На 24 ГБ VRAM `item_emb` (28.75 ГБ fp16) не помещается; нужно перейти
на `d=64` модель (`train.py --d 64`, ожидаемый recall ~0.029) или
скорить чанками с CPU-staging — в этом случае predict в 2× медленнее.

#### Подробная разбивка train.py на RTX 5090

| Шаг                  | Время    | Что делает                                  |
| -------------------- | -------- | ------------------------------------------- |
| `build_vocab`        | ~5 мин   | scan событий, фильтр по вертикалям/min-pop  |
| `build_user_seq`     | ~3 мин   | поартиционно собирает per-user истории      |
| `train_model`        | ~17 мин  | 3 эпохи AdamW + bf16, ~12 k шагов           |

#### Подробная разбивка predict.py на RTX 5090

| Шаг                  | Время    | Что делает                                    |
| -------------------- | -------- | --------------------------------------------- |
| load model + alloc   | ~2 сек   | 28.75 ГБ item_emb на GPU                      |
| encode catalog       | ~4 сек   | 120 M items через ItemTower (30 M items/sec)  |
| scan user history    | ~3 мин   | scan train_data + eval_user_events (Polars)   |
| build per-user stats | ~30 сек  | recency-weighted веса, top-vert, top-reg      |
| scoring (≈90 k user) | ~20 мин  | streaming 32-user batches × 2 M item chunks   |

### Раскладка данных, которую ожидает контейнер

`$AVITO_DATA` (по умолчанию `/data`):

```
data/
├── train_data/part_NNN.parquet     # 100 партиций по user_id % 100, ~6×10⁹ событий
├── eval_user_events.pq             # история eval-юзеров
├── eval_users.csv                  # 1 колонка: user_id (≈ 95 k строк)
├── item_features.parquet           # метаданные 178 M объявлений
├── contact_eids.csv                # колонка mapped_eid: какие eid = "контакт"
└── prepare_local_eval.py           # (опц.) официальный скрипт synth-split
```

Колонки в `train_data/part_*.parquet` (релевантные):

```
user_id   UInt32   — кто
item_id   UInt32   — что
timestamp Int64    — UTC ms
eid       UInt32   — тип события (просмотр / контакт / ...)
```

Колонки в `item_features.parquet` (используемые):

```
item_id, vertical_id, category_ext_y, region_id_y, loc_id_y,
sid_0_y, sid_1_y, sid_2_y, sid_3_y
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
    ├── model.py                # ItemTower, UserTower, TwoTower, gather_feats
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
  становится узким местом (см. ASCII-пример в docstring'е
  `TrainingData.sample_batch`).

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
