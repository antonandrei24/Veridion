import pandas as pd

REQ_COMPANY = ["description", "business_tags", "sector", "category", "niche"]
REQ_INSURANCE  = ["label"]

def load_simple(company_csv: str, policy_csv: str, encoding: str | None = None):
    # load
    companies = pd.read_csv(company_csv, dtype=str, keep_default_na=False, encoding=encoding)
    insurance  = pd.read_csv(policy_csv,  dtype=str, keep_default_na=False, encoding=encoding)

    # quick schema check
    miss_c = [c for c in REQ_COMPANY if c not in companies.columns]
    miss_p = [c for c in REQ_INSURANCE  if c not in insurance.columns]
    if miss_c: raise ValueError(f"Company CSV missing: {miss_c}")
    if miss_p: raise ValueError(f"Policy CSV missing: {miss_p}")

    # tiny normalizer: lowercase, trim, collapse whitespace
    def norm(s: pd.Series) -> pd.Series:
        return (s.fillna("")
                 .str.lower()
                 .str.strip()
                 .str.replace(r"[^\w\s]", " ", regex=True)
                 .str.replace(r"\s+", " ", regex=True))

    # clean companies
    for col in REQ_COMPANY:
        companies[col] = norm(companies[col])

    companies["text"] = (
        companies["description"] + " | " +
        companies["business_tags"] + " | " +
        companies["sector"] + " | " +
        companies["category"] + " | " +
        companies["niche"]
    ).str.replace(r"\s+\|\s+", " | ", regex=True).str.strip(" |")

    # drop empties + dupes
    companies = (companies[companies["text"].str.len() > 0]
                 .drop_duplicates(subset=["text"])
                 .reset_index(drop=True))

    # clean insurance
    insurance["label"] = norm(insurance["label"])
    insurance = (insurance[insurance["label"].str.len() > 0]
                .drop_duplicates(subset=["label"])
                .reset_index(drop=True))
    insurance["text"] = insurance["label"]  # for symmetry with companies

    return companies, insurance

# Example:
companies_df, insurance_df = load_simple("companies.csv", "insurance.csv")
print("\nCompanies:")
print(companies_df.head(), "\n")
print("Insurance:")
print(insurance_df.head())