#!/usr/bin/env python3
"""
loaders.py — download + load real networks with ground-truth communities.
Run where network egress is available (the sandbox used for the synthetic study
had none). Datasets are cached under ./data/.

SNAP   : email-Eu-core (+dept labels), ca-GrQc, ego-Facebook
OGB    : ogbl-collab (optional; requires `pip install ogb`)
SNAP-GT: com-Amazon, com-DBLP (large, with top-5000 ground-truth communities)
"""
import os, gzip, io, urllib.request
import numpy as np
import networkx as nx

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)
SNAP = "https://snap.stanford.edu/data/"
SNAP_COMM = SNAP + "bigdata/communities/"


def _download(url, fn):
    path = os.path.join(DATA, fn)
    if not os.path.exists(path):
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def _read_edges_gz(path):
    G = nx.Graph()
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            a, b = line.split()[:2]
            G.add_edge(int(a), int(b))
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.convert_node_labels_to_integers(G, label_attribute="orig")


def load_email_eu():
    e = _download(SNAP + "email-Eu-core.txt.gz", "email-Eu-core.txt.gz")
    l = _download(SNAP + "email-Eu-core-department-labels.txt.gz", "email-labels.txt.gz")
    G = nx.Graph()
    with gzip.open(e, "rt") as f:
        for line in f:
            a, b = line.split(); G.add_edge(int(a), int(b))
    G.remove_edges_from(nx.selfloop_edges(G))
    lab = {}
    with gzip.open(l, "rt") as f:
        for line in f:
            v, d = line.split(); lab[int(v)] = int(d)
    G2 = nx.convert_node_labels_to_integers(G, label_attribute="orig")
    labels = {i: lab[G2.nodes[i]["orig"]] for i in G2.nodes() if G2.nodes[i]["orig"] in lab}
    return G2, labels, "email-Eu-core"


def load_snap_plain(name, url_fn):
    G = _read_edges_gz(_download(SNAP + url_fn, url_fn))
    return G, None, name


def load_ca_grqc():
    return load_snap_plain("ca-GrQc", "ca-GrQc.txt.gz")


def load_facebook():
    return load_snap_plain("ego-Facebook", "facebook_combined.txt.gz")


def _load_snap_communities(name, graph_fn, comm_fn, top=5000):
    G = _read_edges_gz(_download(SNAP_COMM + graph_fn, graph_fn))
    orig2new = {G.nodes[i]["orig"]: i for i in G.nodes()}
    cpath = _download(SNAP_COMM + comm_fn, comm_fn)
    labels = {}
    with gzip.open(cpath, "rt") as f:
        for ci, line in enumerate(f):
            if ci >= top: break
            for tok in line.split():
                v = int(tok)
                if v in orig2new:           # last community wins (overlap-flattened)
                    labels[orig2new[v]] = ci
    return G, labels, name


def load_com_amazon():
    return _load_snap_communities("com-Amazon", "com-amazon.ungraph.txt.gz",
                                  "com-amazon.top5000.cmty.txt.gz")


def load_com_dblp():
    return _load_snap_communities("com-DBLP", "com-dblp.ungraph.txt.gz",
                                  "com-dblp.top5000.cmty.txt.gz")


def load_ogbl_collab():
    import torch
    torch_load = torch.load

    def trusted_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return torch_load(*args, **kwargs)

    torch.load = trusted_torch_load
    from ogb.linkproppred import LinkPropPredDataset
    ds = LinkPropPredDataset(name="ogbl-collab", root=DATA)
    g = ds[0]; ei = g["edge_index"]
    G = nx.Graph(); G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.convert_node_labels_to_integers(G), None, "ogbl-collab"


REGISTRY = {
    "email": load_email_eu, "grqc": load_ca_grqc, "facebook": load_facebook,
    "amazon": load_com_amazon, "dblp": load_com_dblp, "collab": load_ogbl_collab,
}

if __name__ == "__main__":
    import sys
    G, lab, nm = REGISTRY[sys.argv[1]]()
    print(f"{nm}: n={G.number_of_nodes()} m={G.number_of_edges()} "
          f"labels={'yes('+str(len(set(lab.values())))+' comms)' if lab else 'no'}")
