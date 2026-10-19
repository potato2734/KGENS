"""MRR and Hits@K metrics split by relation cardinality.

Relation categories are computed from the training split using the convention
introduced for the link-prediction benchmarks: a relation is considered
"many" on a side when its average number of entities on that side is at least
1.5. Evaluation results are then split by relation category and prediction
direction.
"""

from collections import defaultdict
from functools import lru_cache
from pathlib import Path


RELATION_CATEGORIES = ("1-to-1", "1-to-N", "N-to-1", "N-to-N")
PREDICTION_SIDES = ("head", "tail")


def _read_id_map(path):
    """Read a ``*.dict`` file and return its name-to-integer-ID mapping."""
    mapping = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            try:
                integer_id, name = line.split("\t", 1)
                mapping[name] = int(integer_id)
            except ValueError as error:
                raise ValueError(
                    "Malformed mapping at {}:{}".format(path, line_number)
                ) from error
    return mapping


@lru_cache(maxsize=None)
def _relation_categories(data_path, threshold):
    """Return ``{relation_id: category}`` derived only from ``train.txt``."""
    data_path = Path(data_path).expanduser().resolve()
    entity_to_id = _read_id_map(data_path / "entities.dict")
    relation_to_id = _read_id_map(data_path / "relations.dict")

    triple_counts = defaultdict(int)
    heads = defaultdict(set)
    tails = defaultdict(set)

    train_path = data_path / "train.txt"
    with train_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            try:
                head, relation, tail = line.split("\t")
                head_id = entity_to_id[head]
                relation_id = relation_to_id[relation]
                tail_id = entity_to_id[tail]
            except (KeyError, ValueError) as error:
                raise ValueError(
                    "Malformed or unmapped triple at {}:{}".format(
                        train_path, line_number
                    )
                ) from error

            triple_counts[relation_id] += 1
            heads[relation_id].add(head_id)
            tails[relation_id].add(tail_id)

    categories = {}
    for relation_id, triple_count in triple_counts.items():
        tails_per_head = triple_count / len(heads[relation_id])
        heads_per_tail = triple_count / len(tails[relation_id])

        one_to_many = tails_per_head >= threshold
        many_to_one = heads_per_tail >= threshold

        if one_to_many and many_to_one:
            category = "N-to-N"
        elif one_to_many:
            category = "1-to-N"
        elif many_to_one:
            category = "N-to-1"
        else:
            category = "1-to-1"
        categories[relation_id] = category

    return categories


def relation_cardinality_statistics(data_path, threshold=1.5):
    """Count relations and train/test triples by relation cardinality."""
    data_path = Path(data_path).expanduser().resolve()
    relation_to_id = _read_id_map(data_path / "relations.dict")
    categories = _relation_categories(str(data_path), float(threshold))

    statistics = {
        "relation": {category: 0 for category in RELATION_CATEGORIES},
        "train": {category: 0 for category in RELATION_CATEGORIES},
        "test": {category: 0 for category in RELATION_CATEGORIES},
    }
    for category in categories.values():
        statistics["relation"][category] += 1

    for split in ("train", "test"):
        split_path = data_path / "{}.txt".format(split)
        with split_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.rstrip("\r\n")
                if not line:
                    continue
                try:
                    _, relation, _ = line.split("\t")
                    relation_id = relation_to_id[relation]
                    category = categories[relation_id]
                except (KeyError, ValueError) as error:
                    raise ValueError(
                        "Malformed, unmapped, or unclassified triple at "
                        "{}:{}".format(split_path, line_number)
                    ) from error
                statistics[split][category] += 1

    return statistics


def calculate_relation_cardinality_metrics(
    data_path, evaluation_records, hits_at=10, threshold=1.5
):
    """Calculate directional MRR and Hits@K by relation-cardinality category.

    ``evaluation_records`` is the existing ``info4eval`` sequence produced by
    each model's ``test_step``. Every record starts with
    ``[(head, relation, tail), prediction_side, rank, ...]``, where the side is
    ``"h"``/``"head"`` or ``"t"``/``"tail"``.

    A category with no evaluated queries is reported as NaN so all 16 metric keys
    remain present in the metrics dictionary without implying a score of zero.
    """
    categories = _relation_categories(str(Path(data_path)), float(threshold))
    hits = defaultdict(float)
    reciprocal_ranks = defaultdict(float)
    counts = defaultdict(int)

    for record in evaluation_records:
        triple, prediction_side, rank = record[:3]
        relation_id = int(triple[1])

        if prediction_side in ("h", "head", "head-batch"):
            side = "head"
        elif prediction_side in ("t", "tail", "tail-batch"):
            side = "tail"
        else:
            raise ValueError(
                "Unknown prediction side {!r}".format(prediction_side)
            )

        try:
            category = categories[relation_id]
        except KeyError as error:
            raise ValueError(
                "Relation ID {} does not occur in {}".format(
                    relation_id, Path(data_path) / "train.txt"
                )
            ) from error

        key = (side, category)
        counts[key] += 1
        hits[key] += float(rank <= hits_at)
        reciprocal_ranks[key] += 1.0 / rank

    metrics = {}
    for side in PREDICTION_SIDES:
        for category in RELATION_CATEGORIES:
            key = (side, category)
            mrr_name = "MRR_{}_{}".format(side, category)
            hits_name = "HITS@{}_{}_{}".format(hits_at, side, category)
            metrics[mrr_name] = (
                reciprocal_ranks[key] / counts[key]
                if counts[key]
                else float("nan")
            )
            metrics[hits_name] = (
                hits[key] / counts[key] if counts[key] else float("nan")
            )
    return metrics
