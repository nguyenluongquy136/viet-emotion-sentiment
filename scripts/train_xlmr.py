
import os, re, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader

DATA_PATH = "data/vlsp2016_test1.txt"
OUT_DIR = "models/transformers"
CKPT = "xlm-roberta-base"
MAX_LEN = 160
EPOCHS = 3
BATCH = 32
LR = 2e-5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def normalize_text(s: str) -> str:
    t = str(s).strip().lower()
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH, sep=None, engine="python", on_bad_lines="skip", encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"class":"sentiment","data":"text"})
    assert "text" in df.columns and "sentiment" in df.columns
    lab_map = {"negative":0,"neg":0,"-1":0,"-":0,"neutral":1,"neu":1,"0":1,"positive":2,"pos":2,"1":2,"+":2}
    if df["sentiment"].dtype == object:
        df["sentiment"] = df["sentiment"].astype(str).str.lower().map(lambda x: lab_map.get(x, x))
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").astype("Int64")
    df = df[df["sentiment"].isin([0,1,2])].dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).apply(normalize_text)

    train_df, valid_df = train_test_split(df, test_size=0.1, random_state=SEED, stratify=df["sentiment"])
    def to_hfds(d): return Dataset.from_pandas(d[["text","sentiment"]], preserve_index=False)
    raw = DatasetDict({"train": to_hfds(train_df), "validation": to_hfds(valid_df)})

    tok = AutoTokenizer.from_pretrained(CKPT, use_fast=True)
    def f(b): return tok(b["text"], truncation=True, max_length=MAX_LEN)
    tokd = raw.map(f, batched=True, remove_columns=["text"])
    tokd = DatasetDict({k: v.rename_column("sentiment","labels") for k,v in tokd.items()})
    collator = DataCollatorWithPadding(tokenizer=tok)
    train_loader = DataLoader(tokd["train"], batch_size=BATCH, shuffle=True, collate_fn=collator)
    valid_loader = DataLoader(tokd["validation"], batch_size=BATCH, shuffle=False, collate_fn=collator)

    model = AutoModelForSequenceClassification.from_pretrained(CKPT, num_labels=3).to(DEVICE)
    cls_w = compute_class_weight('balanced', classes=np.array([0,1,2]), y=train_df["sentiment"].values).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cls_w, dtype=torch.float32, device=DEVICE))
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = get_linear_schedule_with_warmup(optim, int(0.06*EPOCHS*len(train_loader)), EPOCHS*len(train_loader))

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        tot = 0.0
        for batch in loader:
            ids, mask, y = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["labels"].to(DEVICE)
            with torch.set_grad_enabled(train):
                out = model(input_ids=ids, attention_mask=mask)
                loss = criterion(out.logits, y)
                if train:
                    optim.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step(); sched.step()
            tot += loss.item() * ids.size(0)
        return tot/len(loader.dataset)

    for ep in range(1, EPOCHS+1):
        tr = run_epoch(train_loader, True)
        va = run_epoch(valid_loader, False)
        print(f"[{ep}] train_loss={tr:.4f} | valid_loss={va:.4f}")

    model.save_pretrained(OUT_DIR); tok.save_pretrained(OUT_DIR)
    with open(os.path.join(OUT_DIR, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("Negative\nNeutral\nPositive\n")
    print("Saved to", OUT_DIR)

if __name__ == "__main__":
    main()
