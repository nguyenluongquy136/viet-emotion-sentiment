# Viet Emotion Sentiment 🇻🇳🧠

API và demo web phân tích cảm xúc tiếng Việt. Xây dựng với FastAPI, tối ưu cho inference nhanh, hỗ trợ nhiều mô hình (Transformers, LSTM, GRU) và pipeline tiền xử lý tiếng Việt đầy đủ.

— Demo trực tuyến: https://huggingface.co/spaces/nnguyenluongquy136/VNsentiment

— Hugging Face Space: https://huggingface.co/spaces/nnguyenluongquy136/VNsentiment

Lưu ý: Notebook/script huấn luyện nằm trong thư mục `train_models/`.

## Mục lục
- Giới thiệu nhanh
- Nguồn dữ liệu & Tài nguyên
- Tính năng nổi bật
- Kiến trúc & Thư mục
- Cài đặt nhanh (Local & Docker)
- API Endpoints
- Sử dụng API (ví dụ cURL)
- Giao diện demo
- Huấn luyện mô hình (train_models)
- Giấy phép & Liên hệ

## Giới thiệu nhanh
- Ngôn ngữ: Tiếng Việt, tập trung social/e-commerce comments.
- Mô hình: `transformers`, `lstm`, `gru` (cache tại `models/`).
- Tiền xử lý: chuẩn hóa Unicode, teencode, emoji, tokenize PyVi, nối phủ định, loại stopwords (tùy chọn).

## Nguồn dữ liệu & Tài nguyên
### Dữ liệu huấn luyện
- **Dataset chính**: [Vietnamese Sentiment Analysis Dataset](https://www.kaggle.com/datasets/linhlpv/vietnamese-sentimentanalyst/data)
  - Nguồn: Kaggle dataset với dữ liệu tiếng Việt được gán nhãn cảm xúc
  - Sử dụng để huấn luyện các mô hình LSTM, GRU và Transformers

### Tài nguyên hỗ trợ
- **Teencode mapping**: [Vietnamese Teencode Dictionary](https://gist.github.com/behitek/7d9441c10b3c2739499fc5a4d9ea06fb)
  - File: `app/utils/teencode.txt`
  - Mục đích: Chuẩn hóa teencode tiếng Việt trong quá trình tiền xử lý văn bản

- **Stopwords**: [Vietnamese Stopwords](https://www.kaggle.com/datasets/linhlpv/vietnamese-stopwords)
  - File: `app/utils/vietnamese-stopwords.txt`
  - Mục đích: Loại bỏ các từ dừng không có ý nghĩa trong phân tích cảm xúc

### Pipeline tiền xử lý
Các tài nguyên trên được tích hợp vào pipeline tiền xử lý trong `app/utils/text.py`:
1. Chuẩn hóa Unicode và emoji
2. Áp dụng teencode mapping
3. Loại bỏ stopwords (tùy chọn)
4. Tokenization với PyVi
5. Xử lý phủ định

## Tính năng nổi bật
- Endpoint rõ ràng: `GET /api/health`, `GET /api/models`, `POST /api/predict`, `POST /api/predict_batch`, `POST /api/predict_file`.
- Tùy chọn mô hình qua `model_name` (`transformers|lstm|gru`).
- Trả về xác suất và văn bản đã chuẩn hóa (giúp debug/giải thích).
- Giao diện web trực quan (Bootstrap) cho thử nhanh.

## Kiến trúc & Thư mục
```
app/
  core/config.py        # cấu hình
  main.py               # FastAPI app, CORS, static/templates
  routers/predict.py    # các endpoint inference
  services/             # wrapper cho từng mô hình
  utils/text.py         # pipeline tiền xử lý tiếng Việt
models/                 # cache mô hình (transformers/lstm/gru)
startup_models.py       # tải model từ Hugging Face Hub
templates/              # trang web demo (home.html, index.html)
static/                 # css/js
train_models/           # notebook huấn luyện
```

## Cài đặt nhanh (Local)
1) Cài phụ thuộc
```
pip install --upgrade -r requirements.txt
```
2) Tải/caching mô hình (lần đầu)
```
python startup_models.py
```
3) Chạy API
```
uvicorn app.main:app --host 0.0.0.0 --port 7860
```
Truy cập: `http://localhost:7860` (UI) — `http://localhost:7860/docs` (Swagger).

## Chạy bằng Docker
```
docker build -t viet-emotion-sentiment .
docker run --rm -p 7860:7860 viet-emotion-sentiment
```
Image sẽ tự gọi `startup_models.py` để bảo đảm model sẵn sàng.

## API Endpoints
- GET `/api/health`
  - Trả về trạng thái dịch vụ và thiết bị sử dụng (`cpu`/`cuda`).
  - Response ví dụ:
    ```json
    { "status": "ok", "device": "cpu" }
    ```

- GET `/api/models`
  - Danh sách mô hình khả dụng.
  - Response ví dụ:
    ```json
    { "models": ["transformers", "lstm", "gru"] }
    ```

- POST `/api/predict?model_name=transformers`
  - Body (JSON):
    ```json
    { "text": "Sản phẩm này rất tốt!", "return_probs": true }
    ```
  - Response:
    ```json
    {
      "expanded": "van_ban_sau_chuan_hoa ...",
      "label": "positive",
      "probs": {"negative": 0.02, "neutral": 0.08, "positive": 0.90}
    }
    ```

- POST `/api/predict_batch?model_name=transformers`
  - Body (JSON):
    ```json
    { "texts": ["Câu 1", "Câu 2", "Câu 3"] }
    ```
  - Response: trả về `expanded`, `labels` và `probs` theo thứ tự input.

- POST `/api/predict_file`
  - Form-data: `file` (.csv/.txt), `model_name` (mặc định: `lstm`), `column` (mặc định: `text`), `delimiter` (mặc định: `,`).
  - Tự động suy luận văn bản từ file và dự đoán hàng loạt.

## Sử dụng API (cURL ví dụ)
Kiểm tra trạng thái dịch vụ:
```
curl http://localhost:7860/api/health
```
Danh sách mô hình:
```
curl http://localhost:7860/api/models
```
Dự đoán 1 câu (trả xác suất):
```
curl -X POST "http://localhost:7860/api/predict?model_name=transformers" \
  -H "Content-Type: application/json" \
  -d '{"text":"Sản phẩm này rất tốt!","return_probs":true}'
```
Batch nhiều câu:
```
curl -X POST "http://localhost:7860/api/predict_batch?model_name=lstm" \
  -H "Content-Type: application/json" \
  -d '{"texts":["Câu 1","Câu 2","Câu 3"]}'
```
Tải file CSV/TXT:
```
curl -X POST "http://localhost:7860/api/predict_file" \
  -F "file=@samples.csv" \
  -F "model_name=gru" -F "column=text" -F "delimiter=,"
```

## Giao diện demo
- Trang chủ: `GET /`
- Trang thử nghiệm: `GET /index`
  - Thử nhanh 1 câu, nhiều câu, hoặc từ tệp CSV/TXT

## Huấn luyện mô hình (train_models)
- `train_models/lstm_gru_train.ipynb`
- `train_models/transformers_train.ipynb`
Nội dung: tiền xử lý, cấu hình, huấn luyện, đánh giá, lưu artifact cho inference (`models/`) và được triển khai trên colab.

## Giấy phép
Phát hành theo tệp `LICENSE`.

## Liên hệ
Mở issue hoặc thảo luận nếu bạn gặp lỗi/đề xuất cải tiến. Đóng góp PR rất hoan nghênh!
