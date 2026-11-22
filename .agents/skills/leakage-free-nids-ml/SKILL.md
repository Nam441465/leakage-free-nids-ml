---
name: leakage-free-nids-ml
description: >-
  Reproduces the leakage-free tabular machine learning pipeline (XGBoost, LightGBM,
  Random Forest) on CICIDS2017 and NF-UNSW-NB15-v2 from 2611.04660v1.pdf without
  any graph neural networks. Use this skill when preparing datasets, running
  label-aware deduplication, training or evaluating the tabular ML baselines, or
  running the leave-one-family-out and Gaussian noise stress tests.
---

# Leakage-Free NIDS Tabular ML Pipeline

Quy trình chuẩn hóa để dựng lại các mô hình Machine Learning dữ liệu bảng (`XGBoost`, `LightGBM`, `Random Forest`) từ bài báo `2611.04660v1.pdf`. Tuyệt đối không xây dựng đồ thị hay chạy mô hình GNN.

## 1. Chuẩn bị & Làm sạch dữ liệu

### CICIDS2017
1. Đọc 8 file CSV theo phiên thu thập của bộ `CICIDS2017`.
2. Loại bỏ các cột định danh (`Flow ID`, `Source IP`, `Destination IP`, `Source Port`, `Timestamp`) và cột header trùng lặp.
3. Chuẩn hóa nhãn về đúng 8 họ:
   - `Benign`, `DoS`, `DDoS`, `PortScan`, `BruteForce` (gộp FTP/SSH Patator), `Web`, `Bot`, `Rare` (`Infiltration`, `Heartbleed` và nhãn khác).
4. Làm sạch: chuyển `inf`/`-inf` thành `NaN`, xóa dòng có giá trị thiếu, thực hiện **label-aware exact de-duplication** (xóa các dòng trùng cả toàn bộ đặc trưng và nhãn), bỏ cột hằng số.
5. **Kiểm tra khớp số liệu:** Sau khi bỏ `Destination Port`, tổng số dòng phải là `2,520,798` với `68` cột đầu vào (nhị phân/đa lớp).

### NF-UNSW-NB15-v2
1. Đọc file NetFlow v2 (`43` đặc trưng gốc).
2. Loại bỏ `source port`, `DNS query identifier`, hai cột nhãn thừa khi tách feature, và loại bỏ luôn cột IP nguồn/đích (chỉ dùng cho host-graph đã bị loại bỏ).
3. Gộp các lớp tấn công có `< 500` flows thành lớp `Rare` (tại đây là `Worms`).
4. Làm sạch tương tự (xử lý `inf`/missing, label-aware de-duplication, bỏ cột hằng số).
5. **Kiểm tra khớp số liệu:** Sau khi bỏ `Destination Port`, tổng số dòng phải là `731,810` với `38` cột đầu vào.

> **Lưu ý bộ nhớ:** Ép toàn bộ cột đặc trưng sang `float32` ngay sau khi làm sạch để giữ mức tiêu thụ RAM an toàn trong giới hạn 11 GiB.

## 2. Chia tập & Tiền xử lý không rò rỉ (Per Seed)

Với mỗi seed $s \in \{0, 1, 2\}$:
1. Chia stratified `60%` train / `20%` validation / `20%` test theo nhãn đa lớp (cũng dùng chung split này cho bài toán nhị phân để đảm bảo công bằng giữa các lớp tấn công).
   - Kiểm tra kích thước tập test: `504,160` (CICIDS2017) và `146,362` (NF-UNSW-NB15-v2).
2. **Chỉ dùng tập train** để xác định:
   - Danh sách cột $j$ thỏa mãn: không âm trên toàn bộ tập train (`min >= 0`) VÀ phân vị thứ 99 trên tập train $> 10$.
   - Với các cột $j$ đó, áp dụng biến đổi trên cả 3 tập:
     $$\tilde{x}_j = \text{sign}(x_j) \log(1 + |x_j|)$$
   - Tính `mean` và `std` trên tập train (nếu `std == 0` thay bằng `1` để tránh chia 0), chuẩn hóa z-score cả 3 tập rồi clip toàn bộ giá trị về `[-10, 10]`.
3. Tính trọng số lớp trên tập train:
   $$w_c = \min\left(\frac{N}{C \cdot n_c},\; 100\right)$$
   với $N$ là số dòng train, $C$ là số lớp, $n_c$ là số dòng train của lớp $c$.

## 3. Huấn luyện các mô hình ML (Bảng 4)

Chạy tuần tự từng mô hình trên bài toán **Binary** (mọi lớp tấn công = 1, Benign = 0) và **Multiclass**:

- **XGBoost:**
  - `n_estimators=400`, `max_depth=8`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method="hist"`, `early_stopping_rounds=20` theo validation, truyền `sample_weight` bằng $w_c$.
- **LightGBM:**
  - `n_estimators=400`, `num_leaves=63`, `max_depth=8`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `early_stopping(20)` theo validation, truyền `sample_weight` bằng $w_c$.
- **Random Forest:**
  - `n_estimators=100`, `max_samples=0.3`, `class_weight="balanced_subsample"`, `random_state=s`, `n_jobs=-1`.
- **Đóng gói mô hình (Bắt buộc trước khi `del model`):**
  - Lưu `model`, `preprocessor` (fitted trên train) và metadata (`seed`, `feature_names`, `label_mapping`) bằng `joblib.dump(..., compress=3)` vào `saved_models/<dataset>/<task>/seed_<s>/` để tái sử dụng cho noise test và demo inference trực tiếp không cần train lại.

## 4. Đánh giá & Kiểm thử áp lực (Stress Tests)

1. **Binary metrics:** `F1`, `MCC`, `PR-AUC`, `FPR` (tại ngưỡng `0.5`), `Recall @ 0.1% FPR`, `Recall @ 1% FPR`.
2. **Multiclass metrics:** `Macro-F1` (trung bình trên các lớp có mặt trong tập test), `Weighted F1`, `Accuracy`, `Per-class F1`, và ma trận nhầm lẫn chuẩn hóa theo hàng.
3. **Thống kê:** Tổng hợp `mean ± SD` qua 3 seeds; tính `paired difference`, khoảng tin cậy 95% (t-distribution) và `paired t-test` hiệu chỉnh Holm.
4. **Leave-one-family-out (CICIDS2017):**
   - Lần lượt loại bỏ hoàn toàn 1 họ (`DoS`, `DDoS`, `PortScan`, `BruteForce`, `Web`, `Bot`) khỏi tập train và validation; giữ nguyên tập test.
   - Đánh giá Binary detector (ngưỡng `0.5`) theo: recall trên họ bị giấu, recall trên các họ đã học, và FPR trên Benign.
5. **Gaussian feature noise (CICIDS2017):**
   - Lấy mẫu tối đa `100,000` test flows đã chuẩn hóa, cộng nhiễu $\mathcal{N}(0, \sigma^2)$ với $\sigma \in \{0, 0.1, 0.25, 0.5\}$, clip lại về `[-10, 10]`, đo Binary `F1` (kèm `PR-AUC` và `Recall` để theo dõi hiện tượng lệch ngưỡng).

## 5. Bước kiểm tra hoàn thành (Validation)

- Đối chiếu số lượng dòng sau làm sạch và phân bố lớp với Bảng 2 & Bảng 3 của bài báo.
- Đối chiếu kết quả tái lập với mốc tham chiếu của bài báo:
  - CICIDS2017 Binary F1: XGBoost ~`99.67%`, LightGBM ~`99.66%`, RF ~`99.59%`.
  - NF-UNSW-NB15-v2 Binary F1: XGBoost ~`91.92%`, LightGBM ~`91.79%`, RF ~`93.38%`.
  - CICIDS2017 Multiclass Macro-F1: XGBoost ~`93.08%`, LightGBM ~`81.57%`, RF ~`88.37%`.
  - NF-UNSW-NB15-v2 Multiclass Macro-F1: XGBoost ~`55.98%`, LightGBM ~`53.82%`.
