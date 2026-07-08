"""Manifest loading and pair construction for video creatives.

Two manifest formats are accepted:

1. Pair format (mirrors data/tweet_pairs.json / linkedin_pairs.json):
   {
     "train": [{"creative_a": {...}, "creative_b": {...}, "label": 0}, ...],
     "test":  [...]
   }

2. Flat format: {"creatives": [{...}, ...]} or a bare list. Pairs are built
   here: grouped by `campaign` when present, labeled by `metric`, near-ties
   dropped, and split train/test at the *creative* level so no video leaks
   from train into test.

A creative dict needs `video_path` and either raw counts (impressions +
clicks / conversions) or a precomputed metric field. Optional: id, campaign,
brand, platform.
"""

import hashlib
import json
import random
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path


@dataclass
class Creative:
    id: str
    video_path: str
    metrics: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)  # brand, campaign, platform, ...

    def metric_value(self, metric: str) -> float | None:
        """Resolve a metric, deriving rates from raw counts when needed."""
        m = self.metrics
        if metric in m and m[metric] is not None:
            return float(m[metric])
        impressions = m.get("impressions")
        if not impressions:
            return None
        if metric == "ctr" and m.get("clicks") is not None:
            # Laplace smoothing keeps tiny-denominator creatives comparable
            return (m["clicks"] + 0.5) / (impressions + 1.0)
        if metric == "conversion_rate" and m.get("conversions") is not None:
            return (m["conversions"] + 0.5) / (impressions + 1.0)
        return None


@dataclass
class CreativePair:
    creative_a: Creative
    creative_b: Creative
    label: int          # 0 = A wins, 1 = B wins
    ratio: float = 1.0  # winner metric / loser metric

    @property
    def winner(self) -> Creative:
        return self.creative_a if self.label == 0 else self.creative_b

    @property
    def loser(self) -> Creative:
        return self.creative_b if self.label == 0 else self.creative_a


def _parse_creative(d: dict, manifest_dir: Path, fallback_id: str) -> Creative:
    video_path = d.get("video_path") or d.get("video") or d.get("path")
    if video_path is None:
        raise ValueError(f"Creative missing video_path: {d}")
    p = Path(video_path)
    if not p.is_absolute():
        p = manifest_dir / p
    metric_keys = {"impressions", "clicks", "conversions", "spend", "views",
                   "ctr", "conversion_rate", "cpa", "roas", "thumbstop",
                   "watch_time", "completion_rate"}
    metrics = {k: v for k, v in d.items() if k in metric_keys}
    meta = {k: v for k, v in d.items()
            if k not in metric_keys and k not in ("id", "video_path", "video", "path")}
    cid = str(d.get("id") or fallback_id)
    return Creative(id=cid, video_path=str(p), metrics=metrics, meta=meta)


def _parse_pair(d: dict, manifest_dir: Path, idx: int) -> CreativePair:
    a = _parse_creative(d["creative_a"], manifest_dir, f"pair{idx}_a")
    b = _parse_creative(d["creative_b"], manifest_dir, f"pair{idx}_b")
    return CreativePair(a, b, int(d["label"]), float(d.get("ratio", 1.0)))


def build_pairs(
    creatives: list[Creative],
    metric: str = "ctr",
    min_impressions: int = 500,
    min_ratio: float = 1.15,
    group_by: str = "campaign",
    max_pairs_per_creative: int = 6,
    seed: int = 42,
) -> list[CreativePair]:
    """Build labeled pairs from a flat creative list.

    Pairs are formed within the same `group_by` value when available (so we
    compare like with like); creatives lacking the field fall into one pool.
    """
    rng = random.Random(seed)
    usable = []
    for c in creatives:
        if c.metrics.get("impressions", min_impressions) < min_impressions:
            continue
        if c.metric_value(metric) is None:
            continue
        usable.append(c)

    groups: dict[str, list[Creative]] = {}
    for c in usable:
        groups.setdefault(str(c.meta.get(group_by, "_all")), []).append(c)

    pairs = []
    use_count: dict[str, int] = {}
    for group in groups.values():
        candidates = list(combinations(group, 2))
        rng.shuffle(candidates)
        for c1, c2 in candidates:
            if use_count.get(c1.id, 0) >= max_pairs_per_creative:
                continue
            if use_count.get(c2.id, 0) >= max_pairs_per_creative:
                continue
            v1, v2 = c1.metric_value(metric), c2.metric_value(metric)
            hi, lo = max(v1, v2), min(v1, v2)
            if lo <= 0 or hi / lo < min_ratio:
                continue  # too close to call — same tie filtering as tweets
            # randomize A/B position so position never correlates with label
            if rng.random() < 0.5:
                c1, c2, v1, v2 = c2, c1, v2, v1
            pairs.append(CreativePair(c1, c2, label=0 if v1 > v2 else 1,
                                      ratio=hi / lo))
            use_count[c1.id] = use_count.get(c1.id, 0) + 1
            use_count[c2.id] = use_count.get(c2.id, 0) + 1
    rng.shuffle(pairs)
    return pairs


def _split_flat(
    creatives: list[Creative], n_test_pairs: int, cfg
) -> tuple[list[CreativePair], list[CreativePair]]:
    """Split creatives (not pairs) into train/test, then pair within splits."""
    rng = random.Random(cfg.seed)
    shuffled = list(creatives)
    rng.shuffle(shuffled)
    # Heuristic: k creatives yield ~k*max_pairs/2 pairs; reserve enough for test
    n_test_creatives = max(4, min(len(shuffled) - 4,
                                  int(len(shuffled) * 0.35)))
    test_c = shuffled[:n_test_creatives]
    train_c = shuffled[n_test_creatives:]
    kw = dict(metric=cfg.metric, min_impressions=cfg.min_impressions,
              min_ratio=cfg.min_ratio, seed=cfg.seed)
    test_pairs = build_pairs(test_c, **kw)[:n_test_pairs]
    train_pairs = build_pairs(train_c, **kw)
    return train_pairs, test_pairs


def load_manifest(cfg) -> tuple[list[CreativePair], list[CreativePair]]:
    """Return (train_pairs, test_pairs) according to cfg."""
    path = Path(cfg.manifest)
    with open(path) as f:
        data = json.load(f)
    manifest_dir = path.parent

    if isinstance(data, dict) and "train" in data and "test" in data:
        train = [_parse_pair(p, manifest_dir, i) for i, p in enumerate(data["train"])]
        test = [_parse_pair(p, manifest_dir, i) for i, p in enumerate(data["test"])]
    else:
        raw = data["creatives"] if isinstance(data, dict) else data
        creatives = [_parse_creative(d, manifest_dir, f"creative{i}")
                     for i, d in enumerate(raw)]
        train, test = _split_flat(creatives, cfg.n_test, cfg)

    rng = random.Random(cfg.seed)
    rng.shuffle(train)
    rng.shuffle(test)
    if cfg.n_train:
        train = train[: cfg.n_train]
    test = test[: cfg.n_test]

    train_ids = {c.id for p in train for c in (p.creative_a, p.creative_b)}
    test_ids = {c.id for p in test for c in (p.creative_a, p.creative_b)}
    overlap = train_ids & test_ids
    if overlap:
        print(f"WARNING: {len(overlap)} creatives appear in both train and "
              f"test pairs — metric leakage risk (e.g. {sorted(overlap)[:3]})")

    print(f"Loaded {len(train)} train / {len(test)} test pairs "
          f"({len(train_ids)} / {len(test_ids)} unique creatives)")
    return train, test


def unique_creatives(pair_lists: list[list[CreativePair]]) -> list[Creative]:
    """All distinct creatives across pair lists (stable order)."""
    seen: dict[str, Creative] = {}
    for pairs in pair_lists:
        for p in pairs:
            for c in (p.creative_a, p.creative_b):
                seen.setdefault(c.id, c)
    return list(seen.values())


def video_cache_key(creative: Creative) -> str:
    """Stable cache key: content hash of the first MB + size, else the id."""
    p = Path(creative.video_path)
    try:
        h = hashlib.sha1()
        with open(p, "rb") as f:
            h.update(f.read(1 << 20))
        h.update(str(p.stat().st_size).encode())
        return h.hexdigest()[:16]
    except OSError:
        return hashlib.sha1(creative.id.encode()).hexdigest()[:16]
