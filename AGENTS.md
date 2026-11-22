# Quy tắc Dự án: NIDS ML Không Rò Rỉ Dữ Liệu (Leakage-Free ML Only)

Dự án này dựng lại và phát triển phần **Machine Learning (Tabular ML)** từ bài báo *"Leakage-free evaluation of residual graph convolutional networks for network intrusion detection"* (`2611.04660v1.pdf`). Mọi tác vụ lập trình, thiết kế thí nghiệm và đánh giá phải tuân thủ nghiêm ngặt các quy tắc dưới đây.

## 1. Quy tắc phạm vi mô hình (Bắt buộc)
- **CHỈ sử dụng các mô hình ML phi-đồ-thị (Tabular ML):**
  - Trọng tâm chính: `XGBoost`, `LightGBM`, `Random Forest`.
  - Chỉ thêm baseline đối chiếu phi-đồ-thị (như `MLP` hoặc mô hình Anomaly Detection bảng thuần) nếu người dùng yêu cầu rõ.
- **TUYỆT ĐỐI BỎ QUA mọi mô hình đồ thị (GNN / Graph-based):**
  - Không triển khai, không cài đặt thư viện phục vụ đồ thị (như PyTorch Geometric) và không viết code cho `R-GCN`, `GCN`, `GraphSAGE`, `GAT`, `E-GraphSAGE`, hay bất kỳ biến thể host-graph / kNN-graph message passing nào.
  - Bỏ qua toàn bộ các ablation dành cho đồ thị (số láng giềng `k`, số tầng GNN, residual skip của GNN, transductive graph).

## 2. Quy tắc chống rò rỉ dữ liệu (Leakage-Free Protocol)
- **Loại bỏ trùng lặp trước khi chia tập:** Thực hiện *label-aware exact de-duplication* (chỉ xóa các dòng trùng khớp hoàn toàn cả mọi cột đặc trưng lẫn nhãn) **trước** khi tách train/validation/test.
- **Không dùng đặc trưng định danh / lối tắt:**
  - Loại bỏ IP nguồn/đích, port nguồn, Flow ID, timestamp, DNS query ID.
  - Mặc định loại bỏ `Destination Port` khỏi đặc trưng huấn luyện.
- **Fit tiền xử lý chỉ trên tập Train:**
  - Điều kiện chọn cột áp dụng *signed log transform* ($\text{sign}(x)\log(1+|x|)$), giá trị trung bình (`mean`), độ lệch chuẩn (`std`) và trọng số lớp $w_c$ **chỉ được tính từ tập train** của từng seed, sau đó áp dụng cố định sang validation và test.
- **Tách tập & hạt giống ngẫu nhiên:**
  - Dùng stratified split `60% / 20% / 20%` (train / validation / test) cho mỗi seed $s \in \{0, 1, 2\}$ (mỗi seed tạo một split độc lập).
  - Không tinh chỉnh siêu tham số trên tập test; tập test chỉ được chấm điểm một lần sau khi hoàn tất huấn luyện và chọn mô hình trên validation.

## 3. Quy tắc đánh giá & báo cáo
- Mọi chỉ số chính phải báo cáo dưới dạng `mean ± SD` qua đủ 3 seeds `{0, 1, 2}`.
- Khi so sánh mô hình (hoặc phiên bản cải tiến so với baseline), dùng sai khác theo cặp theo từng seed (`paired difference`), khoảng tin cậy 95% theo phân phối $t$, và kiểm định paired $t$-test có hiệu chỉnh Holm.
- Bắt buộc tách riêng 2 kịch bản stress test: `leave-one-family-out` (trên CICIDS2017) và `Gaussian feature noise` ($\sigma \in \{0, 0.1, 0.25, 0.5\}$).

## 4. Quy tắc môi trường, đóng gói mô hình & tối ưu tài nguyên cục bộ
- Làm việc trong virtual environment Python riêng của dự án (không `pip install` global).
- **Bắt buộc đóng gói & lưu mô hình trước khi giải phóng RAM (Model Persistence):**
  - Mọi script huấn luyện (Giai đoạn 3, Giai đoạn 4 và các lần chạy benchmark có bật lưu) **phải đóng gói lưu `model` + `preprocessor` (fitted trên train) + `metadata` (seed, tên cột, `label_mapping`)** bằng `joblib.dump(..., compress=3)` vào `saved_models/<dataset>/<task>/seed_<s_id>/` **trước khi** gọi `del model`.
  - Các bài test không đổi tập train (như đánh giá lại trên test sạch, test nhiễu Gaussian, hoặc demo live trước hội đồng/giảng viên) phải ưu tiên **nạp lại model đã lưu từ `saved_models/`** thay vì train lại từ đầu.
- RAM hệ thống trong WSL giới hạn (~11 GiB):
  - Ép kiểu dữ liệu đặc trưng số sang `float32` cho lưu trữ Parquet và huấn luyện (riêng bước lọc trùng `drop_duplicates` trên CICIDS2017 giữ `float64` làm tròn `round(10)` trước khi ép `float32` để khớp tuyệt đối Bảng 2).
  - Không giữ đồng thời nhiều bản sao dữ liệu lớn trong bộ nhớ; chủ động `del` và `gc.collect()` sau mỗi bước xử lý và chạy tuần tự từng mô hình/seed.
  - Giữ dữ liệu thô (`data/raw/`) ở trạng thái chỉ đọc, không ghi đè file gốc.

## 5. Quy tắc điều phối nhiều Agent (Swarm / Subagents)
- **Tách rạch ròi khâu viết code và khâu chạy dữ liệu/train:**
  - **Được phép chạy song song:** Chia nhỏ việc viết code ở các file/module độc lập (ví dụ: script clean từng dataset, module metric/thống kê, viết unit test) và kiểm tra chéo (verifier) bằng việc đọc code hoặc chạy test nhỏ trên mẫu dữ liệu giả lập.
  - **TUYỆT ĐỐI KHÔNG chạy song song các tác vụ nặng về dữ liệu/train:** Không để 2 agent cùng lúc đọc/làm sạch full dataset (~2.52M dòng), không chạy song song nhiều lượt `XGBoost` / `LightGBM` / `Random Forest` trên 11 GiB RAM & 6 GB VRAM. Mọi bước clean dữ liệu thật và huấn luyện/đánh giá đều phải điều phối chạy **tuần tự**.
- **Phân quyền sở hữu file (Zero Collision):** Mỗi worker chỉ sửa đúng danh sách file được giao; không để hai agent sửa cùng một file đồng thời (dùng `workspace:"share"` nếu cần tách nhánh làm việc cho tác giả sửa code cùng lúc).
- **Giữ đội hình gọn (Lean Swarm):** Chỉ gọi thêm worker khi có đầu việc độc lập thực sự, kèm `verifier`; giữ `coordinator` ở vai trò lên kế hoạch, điều phối, kiểm tra cổng cuối và chốt kết quả, không tự ôm việc code dài khi đang điều phối swarm.
- **Lựa chọn model cho từng vai trò (Cố định theo yêu cầu người dùng):**
  - **Kiểm tra tổng quát (`verifier`):** Dùng **Gemini 4 Neon (High)** (`gemini-4-neon-high`) để tiết kiệm quota Argon mà vẫn đảm bảo kiểm tra sâu.
  - **Các Worker thực thi công việc:** Dùng **Gemini 4 Neon (Medium)** (`gemini-4-neon-medium`).

## 6. Quy trình làm việc 5 bước & Cổng kiểm tra bắt buộc (Closed-Loop Quality Gate)
Mọi lượt giao việc (dù làm một mình hay dùng Swarm/Subagent) đều phải đi qua đủ 5 bước sau, **cấm bỏ bước**:

1. **Bước 1 — Phân tích & Chốt hợp đồng kỹ thuật (Coordinator):**
   - Trước khi giao việc, Coordinator phải chốt rõ trong brief: input/output giữa các module, công thức toán học chính xác (ví dụ: tách $w_c$ cho Binary $C=2$ vs Multiclass $C=K$), thứ tự xử lý (`float32` trước dedup), và các bẫy rò rỉ dữ liệu cần tránh.
2. **Bước 2 — Thực thi & Tự kiểm tra có test phản biện (Workers):**
   - Worker không chỉ viết test "chạy thông" (*happy-path*), mà phải viết unit test kiểm tra đúng các bẫy logic/biên (ví dụ: test chống rò rỉ train/val/test, test phân biệt trọng số Binary vs Multiclass, test `float32`).
   - Tự chạy `pytest` cho phần của mình trước khi đẩy lên `verifier`.
3. **Bước 3 — Kiểm tra chéo vòng 1 (`Verifier`):**
   - `Verifier` **không được chỉ chạy `pytest` rồi ký duyệt**. Phải đọc đối chiếu từng công thức, siêu tham số và **nơi gọi hàm (callers)** giữa các file với `2611.04660v1.pdf` và `AGENTS.md`.
   - Nếu phát hiện sai lệch, trả thẳng về cho worker sở hữu sửa và chạy lại test tới khi đạt.
4. **Bước 4 — Cổng kiểm tra cuối của Agent chính (Coordinator Final Gate — Bắt buộc):**
   - **Coordinator TUYỆT ĐỐI KHÔNG được tin hoàn toàn báo cáo `[done]` của `verifier`/worker để báo xong cho người dùng.**
   - Trước khi trả lời người dùng là "hoàn tất", Coordinator bắt buộc tự thực hiện:
     1. Tự đọc trực tiếp các hàm cốt lõi / diff quan trọng (tiền xử lý, tính loss/weights, vòng lặp train/eval, stress tests) để đối chiếu lần cuối với bài báo.
     2. Tự chạy lại toàn bộ bộ kiểm thử (`pytest`) và kiểm tra không còn cảnh báo/lỗi ẩn.
     3. Nếu thấy sai sót: **chưa được báo xong**, phải đẩy ngược lại cho worker sửa (hoặc xử lý dứt điểm) và kiểm tra lại đạt chuẩn rồi mới báo người dùng.
5. **Bước 5 — Báo cáo minh bạch & Chốt điểm dừng:**
   - Báo cáo rõ ràng phần nào đã kiểm chứng bằng dữ liệu thật, phần nào mới kiểm tra trên dữ liệu mẫu, và **dừng chờ người dùng xác nhận** trước khi bước sang giai đoạn tiếp theo (trừ khi được yêu cầu chạy liên tục).
