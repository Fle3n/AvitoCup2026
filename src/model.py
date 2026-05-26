"""
model.py — архитектура нейросетевого рекомендера для Avito Cup 2026.

ОБЩАЯ ИДЕЯ
----------
Это классическая two-tower (двухбашенная) модель, как у YouTube/Yandex.
- **ItemTower**  превращает фичи объявления в вектор размерности d.
- **UserTower**  превращает последовательность последних K событий
                  пользователя в вектор того же размера.
- score(user, item) = u · v  — обычное скалярное произведение.

Почему two-tower, а не end-to-end ранкер?
- У нас 1.2×10⁸ кандидатов в каталоге.  Перерасчёт сложной (user,item)
  модели для каждой пары стоит O(N).  Two-tower позволяет один раз
  посчитать item-эмбеддинги (~30 МБ в fp16), потом для каждого
  пользователя топ-160 за один matmul.
- Это стандартная архитектура для retrieval-стадии; для production она
  всегда дополняется re-ranker'ом второй стадии, но в номинации мы
  обязаны использовать **только нейросеть**, поэтому останавливаемся
  на retrieval.

КЛЮЧЕВОЕ АРХИТЕКТУРНОЕ РЕШЕНИЕ: ContentTower без per-item ID-таблицы.
В Avito 1.2×10⁸ валидных объявлений.  Per-item ID-embedding на 64 dim
fp32 = ~30 ГБ только под параметры, плюс ~60 ГБ под состояние AdamW —
ни на одну GPU не помещается.  Кроме того, у такой таблицы возникает
overfitting (мы проверили в v7-эксперименте: recall падает с 0.031
до 0.014 — модель запоминает пары (история, цель) из train, а на test'е
seen-фильтр выкидывает именно эти пары и оставляет случайные).

Поэтому каждое объявление — это сумма девяти маленьких эмбеддингов:
  vertical_id, category_ext_y, region_id_y, loc_id_y,
  sid_0_y..sid_3_y, popularity_bucket.
Все девять таблиц вместе занимают <1 МБ, а четыре SID-кода (по 1024
значения каждый) дают >10¹² уникальных комбинаций — модель легко
различает похожие объявления.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ItemTower(nn.Module):
    """
    Кодировщик объявлений: словарь категориальных фич → вектор размерности d.

    Что внутри:
      1) Девять `nn.Embedding`-таблиц (одна на каждую фичу).
      2) Их сумма + `LayerNorm`.
      3) Двухслойный MLP с GELU и остаточной связью.
         MLP даёт нелинейное смешивание фич: без него модель эквивалентна
         линейной FM, что недостаточно для богатых сигналов от SID-кодов.

    Все эмбеддинги инициализируются нормальным шумом со стандартным
    отклонением 0.02 (как в BERT/GPT); это стандартное значение для
    Transformer-моделей и стабильно работает для нашего размера.
    """

    def __init__(self, dims: dict, d: int, n_pop_buckets: int = 128):
        super().__init__()
        # "+1" в каждой таблице — небольшая страховка: если в каких-то
        # партициях встретится значение чуть больше, чем dims[col] из
        # build_vocab, мы не упадём.  Память от этого почти не страдает.
        self.e_vert = nn.Embedding(dims["vertical_id"]    + 1, d)
        self.e_cat  = nn.Embedding(dims["category_ext_y"] + 1, d)
        self.e_reg  = nn.Embedding(dims["region_id_y"]    + 1, d)
        self.e_loc  = nn.Embedding(dims["loc_id_y"]       + 1, d)
        # sid_0..sid_3 — это коды residual quantization от BERT-эмбеддинга
        # объявления.  Каждый из четырёх кодов имеет 1024 уникальных
        # значения, кодируя последовательно всё более тонкие отличия в
        # семантике.  Их сумма работает как "tokenization" объявления.
        self.e_s0 = nn.Embedding(dims["sid_0_y"] + 1, d)
        self.e_s1 = nn.Embedding(dims["sid_1_y"] + 1, d)
        self.e_s2 = nn.Embedding(dims["sid_2_y"] + 1, d)
        self.e_s3 = nn.Embedding(dims["sid_3_y"] + 1, d)
        # Дополнительная фича — корзина популярности (квантильная,
        # 128 бинов).  Без неё модель плохо отличает топ-объявления от
        # "длинного хвоста" с одинаковыми content-фичами.
        self.e_pop = nn.Embedding(n_pop_buckets, d)

        for emb in (self.e_vert, self.e_cat, self.e_reg, self.e_loc,
                    self.e_s0, self.e_s1, self.e_s2, self.e_s3, self.e_pop):
            nn.init.normal_(emb.weight, std=0.02)

        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, d),
        )

    def lookup(self, feats: dict) -> torch.Tensor:
        """
        Сумма всех категориальных эмбеддингов БЕЗ нелинейности — это то,
        что использует `UserTower` как входы Transformer'а.
        Возвращает тензор той же формы, что и любой из feats['...'],
        с добавленной размерностью d в конце.
        """
        return (
            self.e_vert(feats["vert"])
            + self.e_cat(feats["cat"])
            + self.e_reg(feats["reg"])
            + self.e_loc(feats["loc"])
            + self.e_s0(feats["s0"])
            + self.e_s1(feats["s1"])
            + self.e_s2(feats["s2"])
            + self.e_s3(feats["s3"])
            + self.e_pop(feats["pop"])
        )

    def forward(self, feats: dict) -> torch.Tensor:
        """
        Полный item-эмбеддинг: lookup → LayerNorm → +MLP(residual).
        Используется для целевых/негативных объявлений в train и для
        всех объявлений в каталоге при инференсе.
        """
        x = self.lookup(feats)
        x = self.norm(x)
        # Резидуал-связь: x + MLP(x).  Без неё иногда модель ломается
        # на начальной фазе (LayerNorm + чистый MLP даёт нестабильные
        # активации); резидуал делает обучение значительно стабильнее.
        return x + self.mlp(x)


class UserTower(nn.Module):
    """
    Кодировщик пользователя: последние K событий → вектор размерности d.

    Архитектура SASRec-style:
      1) item_emb (без LayerNorm, чтобы не дублировать норм) + positional
         emb + event-type emb — складываем сразу как трёхмерный тензор
         (B, H, d).
      2) Transformer encoder с padding-маской.
      3) Берём ВЫХОД ПОСЛЕДНЕЙ непустой позиции (классический SASRec
         next-item prediction).
      4) Linear-проекция в d, без bias.

    Почему берём только последнюю позицию, а не mean-pool?
    - Эксперимент v11 показал, что concat(last, mean) даёт 0.026 vs
      0.031 у чистого last.  Mean-pool слишком сильно "размазывает"
      сигнал по старым событиям, теряя последний intent пользователя.

    Почему `norm_first=True` в TransformerEncoderLayer?
    - Это Pre-LN вариант (как в GPT-3+, в отличие от Post-LN у
      оригинального Vaswani et al).  Pre-LN стабильнее для длинных
      последовательностей и не требует прогрева LR.

    Почему bf16, а не fp16 при обучении?
    - Pre-LN с bf16 — стандарт для больших моделей.  Главное — bf16
      имеет полный fp32-диапазон экспоненты, поэтому НЕ нужен
      `torch.amp.GradScaler` (мы экспериментально проверили: GradScaler
      с bf16 даёт катастрофическое расхождение лосса на границах
      эпох).
    """

    def __init__(self, d: int, n_heads: int, n_layers: int,
                 max_hist: int, n_eids: int, dropout: float):
        super().__init__()
        self.pos = nn.Embedding(max_hist, d)
        # eid — это тип события (просмотр, контакт, и т.д.).  Эта
        # таблица позволяет Transformer'у "видеть", что недавний
        # КОНТАКТ важнее старого просмотра.
        self.eid = nn.Embedding(n_eids + 1, d)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=4 * d,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,   # Pre-LN — см. docstring
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, item_seq_emb: torch.Tensor,
                eid_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        item_seq_emb: (B, H, d)  — батч последовательностей объявлений
        eid_ids:      (B, H)     — id события на каждой позиции
        mask:         (B, H)     — 1 на валидных позициях, 0 на паддинге

        Возвращает: (B, d) — user-эмбеддинг для скорирования.
        """
        B, H, _ = item_seq_emb.shape
        # Берём первые H позиционных эмбеддингов (max_hist в модели
        # может быть >= H, если предсказываем на более коротких
        # историях, чем обучали — но обычно равно).
        pos = self.pos.weight[:H].unsqueeze(0)
        x = item_seq_emb + pos + self.eid(eid_ids)

        # Маска для attention: True означает PADDING (PyTorch-конвенция).
        key_padding_mask = (mask < 0.5)
        x = self.tr(x, src_key_padding_mask=key_padding_mask)

        # Индекс последней непустой позиции для каждого пользователя.
        # clamp(min=0) на случай пустых историй (теоретически не
        # должно случаться — мы их фильтруем в build_user_seq).
        last_idx = (mask.sum(dim=1).long() - 1).clamp(min=0)
        last_x = x[torch.arange(B, device=x.device), last_idx]
        return self.proj(last_x)


class TwoTower(nn.Module):
    """
    Объединяет item- и user-башни.

    Метод `item_embed(feats)` нужен для инференса — даёт полный
    item-эмбеддинг (с LayerNorm + MLP).

    Метод `user_embed(hist_feats, hist_eids, mask)` для входа в
    user-башню берёт ТОЛЬКО `lookup` (без MLP), потому что Transformer
    уже сам нормализует и трансформирует представления — добавление MLP
    перед ним избыточно и замедляет сходимость (проверено).
    """

    def __init__(self, dims: dict, d: int, max_hist: int,
                 n_heads: int, n_layers: int, dropout: float, n_eids: int):
        super().__init__()
        self.item = ItemTower(dims, d)
        self.user = UserTower(d, n_heads, n_layers, max_hist, n_eids, dropout)

    def item_embed(self, feats: dict) -> torch.Tensor:
        return self.item(feats)

    def user_embed(self, hist_feats: dict, hist_eids: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        item_seq = self.item.lookup(hist_feats)
        return self.user(item_seq, hist_eids, mask)
