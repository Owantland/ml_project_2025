import ast, argparse, json, os
from pathlib import Path
import networkx as nx
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from networkx.exception import PowerIterationFailedConvergence


def safe_eigenvector_centrality(G, max_iter=1000, tol=1e-06):
    try:
        return nx.eigenvector_centrality(G, max_iter=max_iter, tol=tol)
    except PowerIterationFailedConvergence:
        return nx.eigenvector_centrality_numpy(G)

CENTRALITY_FUNCS = {
    "deg_cent":   nx.degree_centrality,
    "harm_cent":  nx.harmonic_centrality,
    "btwn_cent":  nx.betweenness_centrality,
    "pgrnk_cent": nx.pagerank,
    "close_cent": nx.closeness_centrality,
    "eig_cent":   safe_eigenvector_centrality,
    "load_cent":  nx.load_centrality,
}


def centrality_dict(edgelist):
    G = nx.from_edgelist(edgelist)
    node_feats = {v: {} for v in G}
    for name, fn in CENTRALITY_FUNCS.items():
        scores = fn(G)
        for v, val in scores.items():
            node_feats[v][name] = val
    return node_feats


def expand_df(df):
    rows = []
    for _, row in df.iterrows():
        lang      = row["language"]
        sent_id   = row["sentence"]
        n_nodes   = row["n"]
        root_node = row.get("root", np.nan)
        edges     = ast.literal_eval(row["edgelist"])
        cent      = centrality_dict(edges)

        for v, feats in cent.items():
            rows.append(
                {
                    "language": lang,
                    "sentence": sent_id,
                    "n": n_nodes,
                    "v": v,
                    **feats,
                    "is_root": int(v == root_node) if not np.isnan(root_node) else np.nan,
                }
            )
    return pd.DataFrame(rows)


NUM_FEATS = list(CENTRALITY_FUNCS.keys()) + ["n"]

def within_sentence_scale(df):
    df_scaled = df.copy()
    grp_cols  = ["language", "sentence"]

    means = df_scaled.groupby(grp_cols)[NUM_FEATS].transform("mean")
    stds  = df_scaled.groupby(grp_cols)[NUM_FEATS].transform("std")

    stds = stds.fillna(0.0).where(stds > 0, 1.0)

    df_scaled[NUM_FEATS] = (df_scaled[NUM_FEATS] - means) / stds
    return df_scaled

def make_pipeline(df):
    categ_cols = ["language"]
    num_cols   = NUM_FEATS
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categ_cols),
            ("num", StandardScaler(with_mean=False),       num_cols),
        ]
    )
    clf = GradientBoostingClassifier(random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe

def cross_val_report(pipe, X, y, groups):
    cv = GroupKFold(n_splits=5)
    scores = cross_validate(
        pipe, X, y, cv=cv, groups=groups,
        scoring=["accuracy", "balanced_accuracy", "roc_auc"],
        n_jobs=-1, return_train_score=False
    )
    print(pd.DataFrame(scores).mean())


def choose_root(df_node_probs):
    return df_node_probs.loc[df_node_probs["proba_root"].idxmax(), "v"]

def generate_submission(train_csv, test_csv, out_csv="submission.csv"):
    train = within_sentence_scale(expand_df(pd.read_csv(train_csv)))
    test  = within_sentence_scale(expand_df(pd.read_csv(test_csv)))

    pipe = make_pipeline(train)
    y    = train["is_root"]
    X    = train.drop(columns=["is_root", "v"])
    groups = train["sentence"]
    cross_val_report(pipe, X, y, groups)
    pipe.fit(X, y)

    X_test = test.drop(columns=["is_root", "v"])
    test["proba_root"] = pipe.predict_proba(X_test)[:, 1]

    root_per_sentence = (
        test.groupby("sentence")
            .apply(choose_root)
            .reset_index()
            .rename(columns={0: "root"})
    )
    if "id" in pd.read_csv(test_csv).columns:
        ids = pd.read_csv(test_csv)[["sentence", "id"]].drop_duplicates()
        root_per_sentence = ids.merge(root_per_sentence, on="sentence")[["id", "root"]]
    else:
        root_per_sentence = root_per_sentence.rename(columns={"sentence": "id"})

    root_per_sentence.to_csv(out_csv, index=False)
    print(f"Submission saved to {out_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="datasets/train.csv")
    p.add_argument("--test",  default="datasets/test.csv")
    p.add_argument("--out",   default="submission.csv")
    args = p.parse_args()

    generate_submission(args.train, args.test, args.out)
