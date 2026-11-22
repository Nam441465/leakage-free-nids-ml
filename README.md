# Leakage-Free Tabular ML for Network Intrusion Detection (NIDS)

Tái hiện và kiểm chứng toàn bộ nhánh **Tabular Machine Learning (`XGBoost`, `LightGBM`, `Random Forest`)** từ bài báo:
> *"Leakage-free evaluation of residual graph convolutional networks for network intrusion detection"* ([`2611.04660v1.pdf`](2611.04660v1.pdf), Zahid Iqbal).

---

## 1. Điểm nổi bật của Dự án
- **Giao thức chống rò rỉ dữ liệu nghiêm ngặt (Leakage-Free Protocol):**
  - Loại bỏ trùng lặp có nhận thức nhãn (*label-aware exact de-duplication*) trên toàn bộ tập dữ liệu **trước khi** chia tập, khớp tuyệt đối **100% (`0` dòng sai lệch)** với **Bảng 2 (`CICIDS2017` — `2,520,798` dòng, 8 lớp)** và **Bảng 3 (`NF-UNSW-NB15-v2` — `731,810` dòng, 10 lớp)**.
  - Loại bỏ hoàn toàn các cột định danh (`Flow ID`, `Source/Destination IP`, `Source Port`, `Timestamp`, `Protocol`, `DNS_QUERY_ID`) và mặc định loại bỏ `Destination Port` / `L4_DST_PORT` (`68` đặc trưng đầu vào cho `CICIDS2017`, `38` đặc trưng đầu vào cho `NF-UNSW-NB15-v2`).
  - Chia tập phân tầng `60% / 20% / 20%` (`train / validation / test`) độc lập qua 3 hạt giống ngẫu nhiên $s \in \{0, 1, 2\}$.
  - Mọi tham số tiền xử lý — chọn cột biến đổi $\text{sign}(x)\log(1+|x|)$ theo Công thức (1), trung bình (`mean`), độ lệch chuẩn (`std`), cắt ngưỡng `[-10, 10]`, và trọng số lớp cân bằng có cắt trần $w_c = \min(N/(C \cdot n_c), 100)$ (tách riêng $C=2$ cho Binary và $C=K$ cho Multiclass) — **chỉ được fit trên tập Train** của từng seed.
- **Đóng gói toàn bộ 54 mô hình đã huấn luyện (`saved_models/`):**
  - Lưu trữ đầy đủ `36` mô hình benchmark chính (`2 datasets × 2 tasks × 3 seeds × 3 models`) và `18` mô hình `Leave-One-Attack-Family-Out` (`6 families × 3 seeds`), kèm `preprocessor.joblib` và `metadata.json`.
  - Hỗ trợ chấm điểm và demo trực tiếp trên tập test chỉ trong `0.04s – 2.3s` qua script [`src/experiments/demo_evaluate.py`](src/experiments/demo_evaluate.py) mà không cần huấn luyện lại.

---

## 2. Cấu trúc Thư mục
```text
├── 2611.04660v1.pdf                # Bài báo gốc đối chiếu
├── AGENTS.md                       # Quy tắc dự án & giao thức chống rò rỉ dữ liệu
├── REPRODUCTION_PLAN.md            # Kế hoạch tái hiện 4 giai đoạn
├── requirements.txt                # Danh sách thư viện Python
├── data/
│   └── processed/                  # Metadata JSON khớp Bảng 2 & Bảng 3 (Parquet lưu cục bộ)
├── src/
│   ├── data/
│   │   ├── clean_cicids2017.py     # Làm sạch & lọc trùng CICIDS2017 (2,520,798 dòng)
│   │   ├── clean_nf_unsw.py        # Làm sạch & lọc trùng NF-UNSW-NB15-v2 (731,810 dòng)
│   │   └── split_preprocess.py     # Chia 60/20/20, Eq.(1) signed-log, Z-score, trọng số w_c
│   ├── models/
│   │   ├── tree_baselines.py       # XGBoost, LightGBM, Random Forest (Bảng 4)
│   │   └── persistence.py          # Đóng gói & nạp lại bundle (model + preprocessor + metadata)
│   ├── evaluation/
│   │   ├── metrics.py              # F1, MCC, PR-AUC, FPR, R@0.1%FPR, R@1%FPR, Macro/Weighted-F1, CM
│   │   ├── stats.py                # Mean ± SD, 95% Student-t CI, Holm-Bonferroni paired t-test
│   │   └── plot_figures.py         # Vẽ 5 biểu đồ bài báo ở độ phân giải 300 DPI
│   └── experiments/
│       ├── run_main.py             # Chạy benchmark chính (Bảng 5, 6, 7, 8, 9)
│       ├── run_leave_one_out.py    # Stress test 1: Leave-One-Attack-Family-Out (Bảng 11)
│       ├── run_noise_test.py       # Stress test 2: Nhiễu Gaussian (Bảng 12)
│       └── demo_evaluate.py        # Demo nạp mô hình từ saved_models/ và chấm điểm trực tiếp
├── saved_models/                   # 54 bộ mô hình đã đóng gói (114 files)
├── results/
│   ├── tables/                     # Kết quả JSON tổng hợp & CSV chi tiết từng seed
│   └── figures/                    # Hình 4, 6, 7b, 9, 10 (300 DPI PNG)
└── tests/                          # 52 unit tests kiểm tra toàn bộ pipeline & chống rò rỉ
```

---

## 3. Tóm tắt Kết quả Tái hiện so với Bài báo

| Bảng / Thí nghiệm | Chỉ số tiêu biểu (`XGBoost` / `Random Forest`) | Bài báo (`2611.04660v1`) | Dự án tái tạo |
| :--- | :--- | :---: | :---: |
| **Bảng 2 (`CICIDS2017`)** | Tổng số dòng sạch / Số lớp | `2,520,798` / `8 lớp` | **`2,520,798` / `8 lớp`** (`0` lệch) |
| **Bảng 3 (`NF-UNSW-NB15-v2`)** | Tổng số dòng sạch / Số lớp | `731,810` / `10 lớp` | **`731,810` / `10 lớp`** (`0` lệch) |
| **Bảng 5 (`CICIDS2017` Binary)** | `XGBoost` F1 / MCC / PR-AUC / R@1%FPR | `99.67%` / `0.996` / `99.99%` / `99.99%` | **`99.68%` / `0.996` / `99.99%` / `99.99%`** |
| **Bảng 6 (`NF-UNSW` Binary)** | `Random Forest` F1 / `XGBoost` F1 | `93.38 ± 0.18%` / `91.92 ± 0.20%` | **`93.23 ± 0.22%` / `91.68 ± 0.21%`** |
| **Bảng 8 (`CICIDS2017` Multiclass)** | `XGBoost` Macro-F1 / Weighted-F1 / Acc | `93.08%` / `99.85%` / `99.84%` | **`94.10%` / `99.84%` / `99.83%`** |
| **Bảng 9 (`NF-UNSW` Multiclass)** | `XGBoost` Macro-F1 / Weighted-F1 / Acc | `55.98%` / `97.90%` / `97.48%` | **`56.84%` / `97.89%` / `97.48%`** |
| **Bảng 11 (`LOFO` — `CICIDS2017`)** | Overall Seen Recall / Benign FPR | `99.95 ± 0.02%` / `0.10 ± 0.02%` | **`99.95 ± 0.01%` / `0.10 ± 0.02%`** |
| **Bảng 12 (`Gaussian Noise` $\sigma=0.1$)** | `XGBoost` F1 / Precision / PR-AUC | `5.84%` / `95.95%` / `80.93%` | **`9.84%` / `96.59%` / `79.20%`** |

---

## 4. Hướng dẫn Chạy Nhanh

### Chạy bộ kiểm thử tự động (52 unit tests)
```bash
.venv/bin/python -m pytest -q
```

### Demo chấm điểm trực tiếp từ mô hình đã đóng gói (`saved_models/`)
```bash
# Chấm điểm một mô hình cụ thể (không cần train lại)
.venv/bin/python -m src.experiments.demo_evaluate --dataset cicids2017 --task binary --model xgboost --seed 0

# Đối chiếu toàn bộ 36 mô hình đã lưu với bảng kết quả CSV
.venv/bin/python -m src.experiments.demo_evaluate --all
```
