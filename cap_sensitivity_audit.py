"""Cap-sensitivity audit for the degree-channel AUC.

The paper's large-graph attacker evaluations use capped positive sets. This
lightweight audit checks the rank-statistic stability for the degree-product
score on the locally cached real edge lists.
"""

import csv
import gzip
import os
import random


DATASETS = {
    "email-Eu-core": "email-Eu-core.txt.gz",
    "ego-Facebook": "facebook_combined.txt.gz",
    "ca-GrQc": "ca-GrQc.txt.gz",
}
CAPS = (500, 1000, 2000, 5000, 10000)
SEEDS = range(10)


def load_edges(path):
    raw_edges = []
    nodes = set()
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u == v:
                continue
            raw_edges.append((u, v))
            nodes.update((u, v))

    remap = {node: idx for idx, node in enumerate(sorted(nodes))}
    edges = set()
    for u, v in raw_edges:
        a, b = remap[u], remap[v]
        if a > b:
            a, b = b, a
        edges.add((a, b))
    return len(remap), sorted(edges)


def mann_whitney_auc(pos_scores, neg_scores):
    values = [(score, 1) for score in pos_scores]
    values.extend((score, 0) for score in neg_scores)
    values.sort(key=lambda item: item[0])

    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[j][0] == values[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    rank_pos = sum(rank for rank, (_, label) in zip(ranks, values) if label == 1)
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    return (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def sample_auc(node_count, edges, cap, seed):
    rng = random.Random(9100 + seed)
    edge_set = set(edges)
    degrees = [0] * node_count
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1

    positives = rng.sample(edges, cap)
    negatives = []
    seen = set()
    while len(negatives) < cap:
        u = rng.randrange(node_count)
        v = rng.randrange(node_count)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) in edge_set or (u, v) in seen:
            continue
        seen.add((u, v))
        negatives.append((u, v))

    pos_scores = [degrees[u] * degrees[v] for u, v in positives]
    neg_scores = [degrees[u] * degrees[v] for u, v in negatives]
    return mann_whitney_auc(pos_scores, neg_scores)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "realdata_kit", "data")
    rows = []
    for dataset, filename in DATASETS.items():
        node_count, edges = load_edges(os.path.join(data_dir, filename))
        for cap in CAPS:
            if cap > len(edges):
                continue
            aucs = [sample_auc(node_count, edges, cap, seed) for seed in SEEDS]
            mean = sum(aucs) / len(aucs)
            sd = (sum((x - mean) ** 2 for x in aucs) / (len(aucs) - 1)) ** 0.5
            rows.append(
                {
                    "dataset": dataset,
                    "nodes": node_count,
                    "edges": len(edges),
                    "cap": cap,
                    "seeds": len(aucs),
                    "auc_mean": mean,
                    "auc_95ci": 1.96 * sd / (len(aucs) ** 0.5),
                }
            )

    out_path = os.path.join(here, "attacker_sample_sensitivity.csv")
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "nodes", "edges", "cap", "seeds", "auc_mean", "auc_95ci"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)


if __name__ == "__main__":
    main()
