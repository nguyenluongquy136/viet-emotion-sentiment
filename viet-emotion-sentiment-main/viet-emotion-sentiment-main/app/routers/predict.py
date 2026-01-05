# app/routers/predict.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from typing import List, Dict
import csv
import io

import torch
from app.schemas.payloads import (
    PredictIn, PredictOut, PredictBatchIn, PredictBatchOut
)
from app.services.model_manager import model_manager
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from app.core.config import settings

router = APIRouter(prefix="/api", tags=["predict"])

@router.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@router.get("/models")
def list_models():
    return {"models": ["transformers", "lstm", "gru"]}

@router.post("/predict", response_model=PredictOut)
def predict(req: PredictIn, model: str = Query("transformers", alias="model_name")):
    model_inst = model_manager.get_model(model)
    if not model_inst:
        raise HTTPException(status_code=404, detail="Model not found")
    exps, label, probs = model_inst.predict([req.text], return_probs=req.return_probs)
    return PredictOut(expanded=exps[0], label=label[0], probs=(probs[0] if probs else None))

@router.post("/predict_batch")
def predict_batch(req: PredictBatchIn, model: str = Query("transformers", alias="model_name")):
    model_inst = model_manager.get_model(model)
    if not model_inst:
        raise HTTPException(status_code=404, detail="Model not found")

    exps, labels, probs = model_inst.predict(req.texts, return_probs=True)

    predictions: List[Dict[str, object]] = []
    for i in range(len(labels)):
        row_prob = probs[i] if probs and i < len(probs) else None
        label = labels[i]
        score = None
        if isinstance(row_prob, dict) and row_prob:
            score = float(row_prob.get(label) or row_prob.get(str(label).lower()) or max(row_prob.values()))
        predictions.append({
            "text": exps[i] if exps and i < len(exps) else req.texts[i],
            "sentiment": label,
            "score": score
        })

    return PredictBatchOut(expanded=exps, labels=labels, probs=probs)


@router.post("/predict_file")
def predict_file(
    file: UploadFile = File(...),
    model: str = Form("lstm", alias="model_name"),
    column: str = Form("text"),
    delimiter: str = Form(","),
):
    model_inst = model_manager.get_model(model)
    if not model_inst:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        raw_bytes = file.file.read()
        text = raw_bytes.decode("utf-8-sig")
    finally:
        file.file.close()

    texts: List[str] = []
    delim = (delimiter[:1] if delimiter else ',')
    if (file.filename or "").lower().endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text), delimiter=delim)
        if reader.fieldnames and column in reader.fieldnames:
            for row in reader:
                value = row.get(column)
                if value is not None:
                    value_str = str(value).strip()
                    if value_str:
                        texts.append(value_str)
        else:
            rdr = csv.reader(io.StringIO(text), delimiter=delim)
            for row in rdr:
                if not row:
                    continue
                first = next((str(c).strip() for c in row if str(c).strip()), '')
                if first:
                    texts.append(first)
            if not texts:
                for line in io.StringIO(text):
                    ln = line.strip()
                    if not ln:
                        continue
                    if delim and delim in ln:
                        parts = [p.strip() for p in ln.split(delim)]
                        texts.extend([p for p in parts if p])
                    else:
                        texts.append(ln)
    else:
        for line in io.StringIO(text):
            ln = line.strip()
            if not ln:
                continue
            if delim and delim in ln:
                parts = [p.strip() for p in ln.split(delim)]
                texts.extend([p for p in parts if p])
            else:
                texts.append(ln)

    if not texts:
        raise HTTPException(status_code=400, detail="No texts found in file")

    exps, labels, probs = model_inst.predict(texts, return_probs=True)

    predictions: List[Dict[str, object]] = []
    for i in range(len(labels)):
        row_prob = probs[i] if probs and i < len(probs) else None
        label = labels[i]
        score = None
        if isinstance(row_prob, dict) and row_prob:
            score = float(row_prob.get(label) or row_prob.get(str(label).lower()) or max(row_prob.values()))
        predictions.append({
            "text": exps[i] if exps and i < len(exps) else texts[i],
            "sentiment": label,
            "score": score
        })

    return PredictBatchOut(expanded=exps, labels=labels, probs=probs)
