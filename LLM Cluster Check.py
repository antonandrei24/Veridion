import pandas as pd
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================== CONFIG ==================
CSV_PATH = "insurance_clustered.csv"
TARGET_CLUSTER = 29      # <-- change this when you want to audit cluster 1, 2, 3, ...
MAX_EXAMPLES = 15       # how many items we feed from that cluster to the model
MODEL_NAME = "google/gemma-3-12b-it"  # change to whatever Gemma 3 instruct checkpoint you have locally
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
# ============================================


def load_clustered_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    if "cluster_hdbscan" not in df.columns:
        raise ValueError("Expected 'cluster_hdbscan' column.")
    if "label" not in df.columns:
        raise ValueError("Expected 'label' column.")
    df["cluster_hdbscan"] = df["cluster_hdbscan"].astype(int)
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"] != ""].reset_index(drop=True)
    return df


def build_cluster_prompt(cluster_id: int, examples: list[str]) -> str:
    # We design the prompt to force JSON output and keep it deterministic.
    prompt = f"""
You are reviewing an insurance product taxonomy.

Below are insurance product / coverage names that were clustered together
by an unsupervised model. The cluster ID is {cluster_id}.

Cluster {cluster_id} items:
{chr(10).join(f"- {t}" for t in examples)}

You have one single job. Understand each item in the cluster and highlight the one or the ones that stand out as not fitting well with the rest.

Return ONLY valid JSON. Use this exact schema:
{{
  "outliers": [
    {{"item": "<string from list>", "reason": "<short reason>"}}
  ]
}}

If there are no outliers, use "outliers": [].
Do NOT add any extra keys, text, commentary, markdown, or explanations.
"""
    return prompt.strip()


def load_model(model_name: str):
    print(f"Loading model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None
    )
    if DEVICE != "cuda":
        model.to(DEVICE)
    return tokenizer, model


@torch.inference_mode()
def call_llm(tokenizer, model, prompt: str) -> str:
    # Gemma-style chat / instruct models generally behave best if we just feed the prompt
    # as a single turn. If the model you're using expects special chat formatting,
    # wrap it accordingly here.
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.2,
        do_sample=False  # deterministic so we get stable JSON
    )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Sometimes the model will echo the prompt. We try to remove it.
    if full_text.startswith(prompt):
        full_text = full_text[len(prompt):].strip()

    return full_text.strip()


def audit_single_cluster(df: pd.DataFrame,
                         target_cluster: int,
                         max_examples: int,
                         tokenizer,
                         model):
    # get all labels in that cluster
    items = df[df["cluster_hdbscan"] == target_cluster]["label"].tolist()

    if not items:
        print(f"[!] No items found for cluster {target_cluster}")
        return None

    # trim to avoid huge prompt
    sample_items = items[:max_examples]

    # build prompt for LLM
    prompt = build_cluster_prompt(target_cluster, sample_items)

    # ask model
    raw_response = call_llm(tokenizer, model, prompt)
    print("---- RAW MODEL RESPONSE ----")
    print(raw_response)
    print("----------------------------")

    # try parse json
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        # model might add junk around JSON (like text before/after)
        # try to salvage by extracting first {...} block
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(raw_response[start:end+1])
            except Exception:
                parsed = {
                    "cluster_name": None,
                    "coherence_score": None,
                    "outliers": [],
                    "raw_response": raw_response
                }
        else:
            parsed = {
                "cluster_name": None,
                "coherence_score": None,
                "outliers": [],
                "raw_response": raw_response
            }

    print("\n---- PARSED RESULT ----")
    print(json.dumps(parsed, indent=2))
    print("-----------------------")

    # return structured info so you can save/log it
    return {
        "cluster_id": target_cluster,
        "num_items": len(items),
        "sampled_items": sample_items,
        "llm_cluster_name": parsed.get("cluster_name"),
        "llm_coherence": parsed.get("coherence_score"),
        "llm_outliers": parsed.get("outliers"),
        "raw_response": raw_response
    }


if __name__ == "__main__":
    df = load_clustered_csv(CSV_PATH)
    tokenizer, model = load_model(MODEL_NAME)

    result = audit_single_cluster(
        df,
        target_cluster=TARGET_CLUSTER,
        max_examples=MAX_EXAMPLES,
        tokenizer=tokenizer,
        model=model
    )

    # optional: save to a quick CSV/log once you're happy
    if result is not None:
        out_df = pd.DataFrame([{
            "cluster_id": result["cluster_id"],
            "num_items": result["num_items"],
            "llm_cluster_name": result["llm_cluster_name"],
            "llm_coherence": result["llm_coherence"],
            "llm_outliers": json.dumps(result["llm_outliers"]),
            "sampled_items": json.dumps(result["sampled_items"]),
            "raw_response": result["raw_response"]
        }])
        out_df.to_csv(f"cluster_{TARGET_CLUSTER}_llm_audit.csv", index=False)
        print(f"\nSaved cluster_{TARGET_CLUSTER}_llm_audit.csv")
