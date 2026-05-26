# ─────────────────────────────────────────────────────────────────────────────
# Avito ML Cup 2026 — end-to-end neural-network recommender
# ─────────────────────────────────────────────────────────────────────────────
# Использование:
#     docker build -t avito-nn .
#     docker run --gpus all \
#         -v /path/to/data:/data \
#         -v /path/to/out:/out \
#         -e AVITO_OUT=/out/submission.csv \
#         avito-nn bash -lc "python train.py && python predict.py"
#
# Что внутри:
#     python train.py     →  /workspace/cache/model.pt + промежуточные кэши
#     python predict.py   →  $AVITO_OUT  (по умолчанию /workspace/submission.csv)
#
# Раскладка данных, которую ожидает контейнер ($AVITO_DATA, default /data):
#     /data/train_data/part_NNN.parquet    (100 партиций по user_id % 100)
#     /data/eval_user_events.pq            (история eval-юзеров)
#     /data/eval_users.csv                 (список user_id для предсказания)
#     /data/item_features.parquet          (метаданные объявлений)
#     /data/contact_eids.csv               (какие eid считаются контактом)

# CUDA 12.8 с cuDNN — поддерживает Blackwell (RTX 5090, sm_120).
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Стандартные env для воспроизводимости и минимума мусора в логах.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

# Python 3.12 — самый свежий на момент сабмита, поддерживается всеми
# зависимостями.  Дополнительно ставим build-essentials для случая,
# когда колесо для какой-то либы недоступно.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        git wget curl ca-certificates \
 && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Сначала ставим PyTorch отдельным шагом — это самый тяжёлый wheel
# (≈4 ГБ), Docker закэширует слой и не будет переустанавливать,
# если изменится только наш код.  cu128 нужен, потому что мы целимся
# в Blackwell (RTX 5090, sm_120) — стандартный cu121 не поддерживает.
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir \
        torch==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu128

# Остальные (лёгкие) зависимости отдельным слоем — обновляем чаще,
# чем PyTorch, но реже, чем сам код src/.
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# Код пайплайна.  src/ реже меняется, чем верхнеуровневые entry-points,
# поэтому копируем в отдельный слой для лучшего docker-cache'а.
COPY src/                        /workspace/src/
COPY train.py predict.py validate.py /workspace/

# Все пути переопределяемы env-переменными.  По умолчанию:
ENV AVITO_DATA=/data \
    AVITO_CACHE=/workspace/cache \
    AVITO_OUT=/workspace/submission.csv

# По умолчанию запускаем оба шага последовательно — это что требуют
# правила номинации.  Можно переопределить bash-команду при `docker run`.
CMD ["bash", "-lc", "python train.py && python predict.py"]
