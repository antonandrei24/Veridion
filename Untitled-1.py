import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

# ===================== your existing loader =====================
REQ_COMPANY = ["description", "business_tags", "sector", "category", "niche"]
REQ_INSURANCE  = ["label"]

def load_simple(company_csv: str, policy_csv: str, encoding: str | None = None):
    companies = pd.read_csv(company_csv, dtype=str, keep_default_na=False, encoding=encoding)
    insurance  = pd.read_csv(policy_csv,  dtype=str, keep_default_na=False, encoding=encoding)

    miss_c = [c for c in REQ_COMPANY if c not in companies.columns]
    miss_p = [c for c in REQ_INSURANCE  if c not in insurance.columns]
    if miss_c: raise ValueError(f"Company CSV missing: {miss_c}")
    if miss_p: raise ValueError(f"Policy CSV missing: {miss_p}")

    def norm(s: pd.Series) -> pd.Series:
        return (s.fillna("")
                 .str.lower()
                 .str.replace(r"[^\w\s]", " ", regex=True)
                 .str.replace(r"\s+", " ", regex=True)
                 .str.strip())

    for col in REQ_COMPANY:
        companies[col] = norm(companies[col])

    companies["text"] = (
        companies["description"] + " | " +
        companies["business_tags"] + " | " +
        companies["sector"] + " | " +
        companies["category"] + " | " +
        companies["niche"]
    ).str.replace(r"\s+\|\s+", " | ", regex=True).str.strip(" |")

    companies = (companies[companies["text"].str.len() > 0]
                 .drop_duplicates(subset=["text"])
                 .reset_index(drop=True))

    insurance["label"] = norm(insurance["label"])
    insurance = (insurance[insurance["label"].str.len() > 0]
                .drop_duplicates(subset=["label"])
                .reset_index(drop=True))
    insurance["text"] = insurance["label"]
    return companies, insurance

# ===================== sentence embedder =====================
# Good defaults:
#   English: "sentence-transformers/all-MiniLM-L6-v2"
#   Multilingual (incl. Romanian): "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # change to multilingual if needed

def load_st_model(name=MODEL_NAME) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(name, device=device)
    return model

def encode_texts(model: SentenceTransformer, texts: pd.Series, batch_size: int = 256) -> np.ndarray:
    # returns float32 numpy array
    embs = model.encode(
        texts.tolist(),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine-ready
        show_progress_bar=True
    )
    return embs.astype(np.float32, copy=False)

def maybe_save(path: str | None, arr: np.ndarray):
    if path:
        np.save(path, arr)

def maybe_load(path: str | None) -> np.ndarray | None:
    if path and Path(path).exists():
        return np.load(path)
    return None

# ===================== retrieval & pretty print =====================
def topk_sentence_matches(
    companies_df: pd.DataFrame,
    insurance_df: pd.DataFrame,
    model: SentenceTransformer,
    k: int = 10,
    sample_n: int = 20,
    seed: int = 42,
    cache_comp_path: str | None = None,
    cache_ins_path: str | None = None,
):
    # sample 20 companies to inspect
    sample = companies_df.sample(n=sample_n, random_state=seed).reset_index(drop=False)
    sample.rename(columns={"index": "company_row_index_in_original"}, inplace=True)

    comp_emb = maybe_load(cache_comp_path)
    if comp_emb is None:
        comp_emb = encode_texts(model, companies_df["text"])
        maybe_save(cache_comp_path, comp_emb)

    ins_emb = maybe_load(cache_ins_path)
    if ins_emb is None:
        ins_emb = encode_texts(model, insurance_df["text"])
        maybe_save(cache_ins_path, ins_emb)

    # cosine because normalized
    # restrict to sampled companies
    comp_idx = sample["company_row_index_in_original"].to_numpy()
    comp_emb_sample = comp_emb[comp_idx]  # (20, d)

    sims = comp_emb_sample @ ins_emb.T  # (20, num_insurance)

    # top-k per row
    k_eff = min(k, sims.shape[1])
    topk_idx = np.argpartition(-sims, kth=k_eff-1, axis=1)[:, :k_eff]
    row_idx = np.arange(topk_idx.shape[0])[:, None]
    topk_sorted = topk_idx[row_idx, np.argsort(-sims[row_idx, topk_idx])]

    # assemble results (readable)
    rows = []
    for i, r in sample.iterrows():
        orig_idx = int(r["company_row_index_in_original"])
        company_text = companies_df.iloc[orig_idx]["text"]
        matches = []
        for j, ins_i in enumerate(topk_sorted[i], start=1):
            matches.append({
                "rank": j,
                "insurance_label": insurance_df.iloc[ins_i]["label"],
                "score": float(sims[i, ins_i])
            })
        rows.append({
            "company_row_index_in_original": orig_idx,
            "company_text": company_text,
            "matches": matches
        })
    return rows, sample  # rows is a list of dicts suited for printing; sample keeps full company rows

def pretty_print(rows, top_k=10):
    for idx, item in enumerate(rows, start=1):
        print("=" * 90)
        print(f"[{idx:02d}/20] Company row #{item['company_row_index_in_original']} — TEXT:")
        print(item["company_text"])
        print("-" * 90)
        print("Top matches (label • confidence):")
        for m in item["matches"][:top_k]:
            print(f"{m['rank']:2d}. {m['insurance_label']} • {m['score']:.3f}")
        print()

def to_long_df(rows) -> pd.DataFrame:
    long_rows = []
    for item in rows:
        for m in item["matches"]:
            long_rows.append({
                "company_row_index_in_original": item["company_row_index_in_original"],
                "company_text": item["company_text"],
                "rank": m["rank"],
                "insurance_label": m["insurance_label"],
                "score": m["score"],
            })
    return pd.DataFrame(long_rows)

# ===================== RUN =====================
companies_df, insurance_df = load_simple("companies.csv", "insurance.csv")
model = load_st_model(MODEL_NAME)

# Optional: turn on caching by setting these file paths
COMP_EMB_NPY = None  # e.g., "companies_sent_emb.npy"
INS_EMB_NPY  = None  # e.g., "insurance_sent_emb.npy"

rows, sampled_companies = topk_sentence_matches(
    companies_df, insurance_df, model,
    k=10, sample_n=20, seed=22,
    cache_comp_path=COMP_EMB_NPY,
    cache_ins_path=INS_EMB_NPY,
)

pretty_print(rows, top_k=10)

# Save Excel-friendly outputs
pd.DataFrame(sampled_companies).to_csv("sampled_company_rows.csv", index=False)
to_long_df(rows).to_csv("insurance_top10_readable_list.csv", index=False)
print("\nSaved:")
print(" - sampled_company_rows.csv  (full sampled company rows)")
print(" - insurance_top10_readable_list.csv  (long list: company_text, top10 labels, scores)")
