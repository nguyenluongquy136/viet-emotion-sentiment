import pathlib
from huggingface_hub import snapshot_download

# Định nghĩa các mô hình và các tệp yêu cầu cụ thể cho từng mô hình
MODELS = {
    "gru": {
        "repo": "nguyenhoaday/VNsentiment_GRU",
        "dir": pathlib.Path("models/gru"),
        "required_files": [
            "config.json",
            "pytorch_model.bin",
            "labels.txt",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "modeling_rnn.py",
        ],
    },
    "lstm": {
        "repo": "nguyenhoaday/VNsentiment_LSTM",
        "dir": pathlib.Path("models/lstm"),
        "required_files": [
            "config.json",
            "pytorch_model.bin",
            "labels.txt",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "modeling_rnn.py",
        ],
    },
    "transformer": {
        "repo": "Khanh456/VNsentiment_TRANFORMERS",
        "dir": pathlib.Path("models/transformers"),
        "required_files": [
            "config.json",
            "model.safetensors",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "labels.txt",
        ],
    },
}

def _has_files(p: pathlib.Path, required_files: list) -> bool:
    """Kiểm tra các tệp đã được tải"""
    return all((p / f).exists() for f in required_files)

def download_model(name: str, repo_id: str, target_dir: pathlib.Path, required_files: list, token=None):
    """Tải mô hình từ Hugging Face Hub """
    target_dir.mkdir(parents=True, exist_ok=True)
    if _has_files(target_dir, required_files):
        print(f"[startup] {name} already cached at {target_dir}")
        return
    print(f"[startup] downloading {name} from {repo_id} -> {target_dir}")
    snapshot_download(
        repo_id,
        local_dir=str(target_dir),
        token=token,
        allow_patterns=required_files
    )
    print(f"[startup] {name} ready.")

if __name__ == "__main__":
    for name, cfg in MODELS.items():
        download_model(name, cfg["repo"], cfg["dir"], cfg["required_files"])