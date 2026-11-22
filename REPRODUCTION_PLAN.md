# Kế hoạch Dựng lại & Cải tiến Phần ML từ Bài báo `2611.04660v1`

Tài liệu này lập kế hoạch chi tiết để **dựng lại (reproduce)** phần Machine Learning dữ liệu bảng từ bài báo [2611.04660v1.pdf](./2611.04660v1.pdf) theo quy tắc trong [AGENTS.md](./AGENTS.md) (**chỉ giữ mô hình ML, bỏ qua toàn bộ mô hình đồ thị/GNN**), và chuẩn bị nền tảng để **cải tiến** các điểm yếu của ML được chỉ ra trong bài.

---

## 1. Phạm vi & Bối cảnh hạ tầng

- **Phạm vi mô hình dựng lại:** `XGBoost`, `LightGBM`, `Random Forest` (bỏ `R-GCN`, `GCN`, `GraphSAGE`, `GAT`, `E-GraphSAGE`, và các ablation đồ thị).
- **Bộ dữ liệu:** `CICIDS2017` (8 file flow gốc) và `NF-UNSW-NB15-v2`.
- **Hạ tầng thực thi hiện tại (WSL Ubuntu-24.04):**
  - CPU: `16` luồng | GPU: `RTX 3050 6GB` | RAM: `11 GiB` (+ `4 GiB` swap) | Ổ cứng trống: `915 GB` | `Python 3.12.3`.
  - **Chiến lược quản lý bộ nhớ:** Toàn bộ đặc trưng ép kiểu `float32` (CICIDS2017 chỉ còn ~680 MB thay vì ~1.37 GB), xử lý từng dataset và từng seed tuần tự, chủ động giải phóng RAM giữa các lần chạy.

---

## 2. Cấu trúc thư mục dự án đề xuất

```text
project/
├── 2611.04660v1.pdf
├── AGENTS.md                         # Quy tắc dự án (ML-only, leakage-free)
├── REPRODUCTION_PLAN.md              # Tài liệu kế hoạch này
├── .agents/skills/leakage-free-nids-ml/SKILL.md
├── pyproject.toml / requirements.txt # Môi trường Python cố định phiên bản
├── data/
│   ├── raw/                          # Chứa CICIDS2017/ và NF-UNSW-NB15-v2/ (chỉ đọc)
│   └── processed/                    # Bản đã clean + dedup dạng Parquet/float32
├── src/
│   ├── data/
│   │   ├── clean_cicids2017.py       # Đọc 8 CSV, map 8 nhãn, label-aware dedup
│   │   ├── clean_nf_unsw.py          # Đọc NetFlow v2, gộp Rare (<500), dedup
│   │   └── split_preprocess.py       # Split 60/20/20 theo seed, log1p, z-score, clip
│   ├── models/
│   │   └── tree_baselines.py         # XGBoost, LightGBM, Random Forest (Bảng 4)
│   ├── evaluation/
│   │   ├── metrics.py                # F1, MCC, PR-AUC, Recall@0.1%/1% FPR, Macro-F1
│   │   └── stats.py                  # Paired diff, 95% t-CI, Holm-corrected t-test
│   └── experiments/
│       ├── run_main.py               # Chạy Binary & Multiclass (2 datasets x 3 seeds, có cờ lưu model)
│       ├── run_leave_one_out.py      # Stress test giấu từng họ tấn công (CICIDS2017) + lưu model
│       ├── run_noise_test.py         # Stress test nhiễu Gaussian (nạp model đã lưu hoặc train+lưu)
│       └── demo_evaluate.py          # Nạp model từ saved_models/ để chấm điểm live khi demo/bảo vệ
├── saved_models/                     # Đóng gói model + preprocessor (.joblib) trước khi del RAM
│   ├── nf_unsw/                      # Model + preprocessor cho NF-UNSW-NB15-v2
│   └── cicids2017/                   # Model Binary, Multiclass & Leave-one-family-out
└── results/
    ├── tables/                       # File CSV/JSON tổng hợp mean ± SD
    └── figures/                      # Biểu đồ so sánh & confusion matrix
```

---

## 3. Lộ trình thực hiện từng giai đoạn

### Giai đoạn 0 — Thiết lập môi trường & Dữ liệu đầu vào
- **Công việc:**
  1. Tạo `.venv` Python 3.12 và cài các thư viện ML bảng (`numpy`, `pandas`, `pyarrow`, `scikit-learn`, `xgboost`, `lightgbm`, `scipy`, `matplotlib`, `seaborn`).
  2. Tải hoặc liên kết 2 bộ dữ liệu gốc vào `data/raw/cicids2017/` và `data/raw/nf_unsw_nb15_v2/`.
- **Tiêu chí nghiệm thu (Acceptance Criteria):**
  - Import thành công `xgboost`, `lightgbm`, `sklearn`; xác nhận GPU CUDA sẵn sàng cho `xgboost` nếu dùng.
  - Có đủ 8 file CSV của CICIDS2017 (`2,830,743` dòng gốc) và file CSV/Parquet của NF-UNSW-NB15-v2 (`2,390,275` dòng gốc).

### Giai đoạn 1 — Pipeline Làm sạch & Kiểm chứng phân bố lớp
- **Công việc:**
  1. Viết script làm sạch tách riêng cho `CICIDS2017` và `NF-UNSW-NB15-v2`, xuất ra `.parquet` (`float32` + nhãn số nguyên).
  2. Kiểm tra tên cột chính xác trong CSV gốc (đặc biệt cột header trùng lặp của CICIDS2017 và cột label/IP của NF-UNSW-NB15-v2).
- **Tiêu chí nghiệm thu:**
  - Khớp **tuyệt đối** số lượng dòng & cột với Bảng 2 và Bảng 3 của bài báo:
    - `CICIDS2017`: `2,520,798` dòng, `68` features (sau khi bỏ `Destination Port`), đúng số lượng từng lớp (`Benign: 2,095,057`, `DoS: 193,745`, ..., `Rare: 47`).
    - `NF-UNSW-NB15-v2`: `731,810` dòng, `38` features (sau khi bỏ `Destination Port`), đúng số lượng từng lớp (`Benign: 690,559`, ..., `Rare: 69`).

### Giai đoạn 2 — Tiền xử lý Leakage-Free & Huấn luyện Baseline ML (Main Benchmark)
- **Công việc:**
  1. Cài đặt `split_preprocess.py` cho seeds `0, 1, 2` (fit hoàn toàn trên train 60%, transform val 20% và test 20%).
  2. Chạy `XGBoost`, `LightGBM`, `Random Forest` cho cả Binary và Multiclass trên 2 datasets (`3 seeds × 3 models × 2 tasks × 2 datasets = 36 runs` chính — lưu ý bài báo không báo cáo Random Forest ở Bảng 9 multiclass NF-UNSW-NB15-v2, có thể chạy bổ sung để đầy đủ).
  3. Xuất bảng kết quả `mean ± SD` và kiểm định thống kê paired Holm.
- **Mốc đối chiếu với bài báo (Target Benchmarks):**

| Dataset | Task | Metric | XGBoost (Bài báo) | LightGBM (Bài báo) | Random Forest (Bài báo) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| CICIDS2017 | Binary | F1 (%) | 99.67 ± 0.02 | 99.66 ± 0.01 | 99.59 ± 0.01 |
| CICIDS2017 | Binary | Recall @ 0.1% FPR | 99.88 ± 0.01 | 99.85 ± 0.01 | 99.70 ± 0.01 |
| CICIDS2017 | Multiclass | Macro-F1 (%) | 93.08 ± 1.22 | 81.57 ± 0.75 | 88.37 ± 2.11 |
| NF-UNSW-NB15-v2 | Binary | F1 (%) | 91.92 ± 0.20 | 91.79 ± 0.23 | 93.38 ± 0.18 |
| NF-UNSW-NB15-v2 | Binary | PR-AUC (%) | 96.94 ± 0.13 | 96.81 ± 0.05 | 96.42 ± 0.13 |
| NF-UNSW-NB15-v2 | Binary | Recall @ 0.1% FPR | 56.95 ± 3.09 | 55.72 ± 3.04 | 50.66 ± 4.37 |
| NF-UNSW-NB15-v2 | Multiclass | Macro-F1 (%) | 55.98 ± 0.88 | 53.82 ± 1.04 | *(chưa báo cáo)* |

### Giai đoạn 3 — Đóng gói mô hình, 2 Stress Tests & Script Demo kiểm chứng live
- **Công việc:**
  1. **Nâng cấp đóng gói mô hình (`saved_models/`):** Thêm `joblib.dump(..., compress=3)` lưu `model + preprocessor + metadata` trước khi `del model` cho `run_main.py`, `run_noise_test.py` và `run_leave_one_out.py` (hỗ trợ nạp lại model đã lưu nếu đã tồn tại trên đĩa để không phải train lại).
  2. **`run_noise_test.py` (Bảng 12):** Chạy trên `CICIDS2017` với $\sigma \in \{0, 0.1, 0.25, 0.5\}$ trên 100,000 test flows cho `XGBoost` (và `LightGBM`/`Random Forest`), đồng thời **lưu luôn bộ model Binary `CICIDS2017`** ra `saved_models/cicids2017/binary/`. Ghi nhận `F1`, `PR-AUC` và `Recall` để kiểm chứng cú sập ở Bảng 12 (`99.69% -> 5.84% -> 1.84% -> 0.67%`).
  3. **`run_leave_one_out.py` (Bảng 11):** Train 6 bộ Binary detector riêng biệt (mỗi bộ giấu 1 họ: `DoS`, `DDoS`, `PortScan`, `BruteForce`, `Web`, `Bot`) × 3 seeds, **lưu model vào `saved_models/cicids2017/leave_one_out/`** và đối chiếu Bảng 11.
  4. **Đóng gói bản demo & viết `demo_evaluate.py`:** Lưu model đại diện (`seed=0` hoặc cả 3 seeds) cho `NF-UNSW-NB15-v2` và `CICIDS2017 Multiclass` nếu cần demo live, kèm script `demo_evaluate.py` nạp thẳng `.joblib` từ `saved_models/` chạy chấm điểm trực tiếp trên tập test trong vài giây để đối chiếu trước giảng viên.

---

## 4. Lộ trình Cải tiến sau khi Dựng lại (Phase 4 — Research Extensions)

Sau khi Giai đoạn 1–3 khớp chuẩn baseline của bài báo, tập trung giải quyết **4 điểm yếu của ML** dưới cùng giao thức leakage-free:

1. **Chống nhiễu cho GBDT (Khắc phục Bảng 12):**
   - Thử nghiệm *Gaussian noise data augmentation* trong tập train, hiệu chỉnh ngưỡng/xác suất (*calibration + validation threshold tuning*), và giảm độ nhạy của *binning* (`max_bin`) hoặc kết hợp với mô hình bảng biên mượt (như regularized MLP / Tabular ResNet).
2. **Tăng Recall @ 0.1% FPR & Macro-F1 lớp hiếm (Khắc phục Bảng 6 & 9):**
   - Thay thế/tinh chỉnh công thức trọng số $w_c$, thử *Focal Loss* cho XGBoost/LightGBM, tối ưu ngưỡng theo từng lớp trên validation, hoặc thiết kế bộ phân loại 2 tầng (*Two-stage cascade*).
3. **Phát hiện Zero-day / Họ tấn công chưa biết (Khắc phục Bảng 11):**
   - Kết hợp mô hình có giám sát (`XGBoost`/`LightGBM`) với một bộ phát hiện bất thường chỉ học trên lớp `Benign` (`Isolation Forest`, `Autoencoder`, hoặc ước lượng mật độ chuẩn hóa).
4. **Mở rộng đánh giá thực tế hơn:**
   - Bổ sung kiểm thử tách tập theo phiên/thời gian (*Session/Temporal split* trên 8 file ngày của CICIDS2017) để kiểm chứng độ bền của các cải tiến ML khi triển khai thực tế.
