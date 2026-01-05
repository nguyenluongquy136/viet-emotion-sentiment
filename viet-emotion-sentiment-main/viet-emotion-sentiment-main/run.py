from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/api/models")
def get_models():
    # Trả về danh sách mô hình hoặc dữ liệu mẫu
    return {"models": ["LSTM", "GRU", "Transformer"]}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=9000, reload=True)
