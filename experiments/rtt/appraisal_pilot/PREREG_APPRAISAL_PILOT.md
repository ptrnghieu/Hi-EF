# Đăng ký trước — pilot khả năng quan sát appraisal trên Hi-EF

Trạng thái: **khóa phần phương pháp**. Hai mục đánh dấu [CHƯA XÁC NHẬN] phải được điền và commit **trước khi**
người đánh giá nhận gói chính. Mọi thay đổi sau thời điểm đó được ghi là sai lệch so với đăng ký.

## 1. Câu hỏi
Trong cửa sổ clip I–III của Hi-EF, những thành phần appraisal theo góc nhìn người nói kế tiếp (B) có thể được suy ra
đáng tin cậy **trước** clip IV hay không, và ở tỷ lệ mẫu nào. Đây là pilot quyết định đầu tư, không phải kiểm định
xác nhận lý thuyết.

## 2. Tập mẫu
- Nguồn: 2.421 MCIS thuộc split train/val (`source_folder_split_seed42.csv`). Không dùng test.
- Phân tầng (metadata thiết kế, từ `g16_per_mcis.csv`, notebook G16):
  - *B nói ở clip k*: cosine ECAPA giữa audio clip k và audio clip IV ≥ 0,35. **Dùng clip IV**, nên chỉ phục vụ
    thiết kế mẫu và không bao giờ đưa vào gói đánh giá. Clip không có giọng được tính là "B không nói".
  - *Có nhãn*: cột cảm xúc (cột 7) của `annotation.csv` có giá trị cho clip k.
  - *Thấy người nghe*: người xuất hiện nhiều thứ hai ở clip III (gom cụm ArcFace) có ít nhất một khuôn mặt trong I–III.
  - S1: B nói ở I hoặc II và clip đó có nhãn (654). S2: B nói ở I/II, không clip nào của B có nhãn (562).
    S3: B không nói ở I/II, thấy người nghe (737). S4: còn lại (468). Bốn tầng loại trừ nhau và phủ hết 2.421 mẫu
    (kiểm tra bằng assert trong `make_sample.py`).
- Mẫu chính: 50 mẫu mỗi tầng, seed 20261002. Lượt hiệu chỉnh: 15 mẫu khác, không tính vào kết quả.
- Trọng số tầng: N_h / N với số thực tế ở trên.
- Phạm vi suy rộng: độ phủ ước lượng là của **train/val đủ điều kiện**, chưa phải toàn Hi-EF.
- Đơn vị cụm cho bootstrap: `source_folder`. Việc mỗi folder tương ứng một tập phim **chưa được xác minh độc lập**;
  báo cáo kết quả bootstrap với ghi chú này.

## 3. Người đánh giá  [CHƯA XÁC NHẬN]
- Kế hoạch: 3 người, mỗi người đánh giá toàn bộ 200 mẫu, độc lập trước khi trao đổi. Ưu tiên người chưa xem phim.
- Danh sách người đánh giá: **[CHƯA XÁC NHẬN]**
- Mức quen House of Cards của từng người (chưa xem / xem một phần / đã xem): **[CHƯA XÁC NHẬN]**
- Kết quả tách theo mức quen phim chỉ báo cáo mô tả, không kiểm định.

## 4. Quy trình
1. Lượt hiệu chỉnh trên `calibration.csv` (15 mẫu); chỉnh câu chữ codebook nếu cần; đo thời gian thực tế.
   Sau lượt này codebook bị khóa.
2. Gói chính `rater_package.csv`: chỉ ba video đổi tên và phụ đề I–III. `item_map.csv` và `design_strata.csv`
   không được chia sẻ cho người đánh giá. Người đánh giá không nhận thư mục dataset.
3. Thu nhãn độc lập của cả 3 người; sau đó mới phân xử bất đồng (phân xử không dùng cho α).

## 5. Đại lượng và ngưỡng (khóa)
- Một mẫu **đạt** khi ít nhất 2/3 người cùng lúc:
  (a) chọn cùng phương án Q1 khác `khong_xac_dinh`;
  (b) chọn cùng một quan hệ cụ thể ở Q3 thuộc {`ho_tro`, `can_tro`, `hon_hop`};
  (c) ghi mức bằng chứng Q3 là `truc_tiep` hoặc `suy_ra`, có vị trí bằng chứng ở Q4.
- ĉ = Σ_h (N_h/N) · (tỷ lệ mẫu đạt trong tầng h). Khoảng bất định: bootstrap lấy lại mẫu trong từng tầng (2.000 lần).
  ĉ là tỷ lệ **được người đánh giá đồng thuận**, không phải tỷ lệ mục tiêu thật của B đã xác minh.
- α_Q3: Krippendorff's α, mức đo **nominal**, năm loại của Q3 (gồm `chua_du`), trên nhãn độc lập trước phân xử,
  200 mẫu × 3 người. Khoảng bất định bằng bootstrap theo mẫu. Nếu α không xác định (nhãn không biến thiên), báo cáo
  "không xác định", không gán bằng 1. Báo cáo thêm α cho biến nhị phân "đủ / chưa đủ bằng chứng" và α cho Q1.
- Quyết định sau 200 mẫu:
  - **Tiếp tục sang bước B**: ĉ ≥ 30% **và** α_Q3 ≥ 0,67.
  - **Dừng hướng này trên giao thức hiện tại**: ĉ < 15% **hoặc** α_Q3 < 0,40.
  - **Chưa rõ**: còn lại → thêm đúng 200 mẫu (cùng phân tầng, seed khác), tổng tối đa 400.
  - Sau 400 mẫu: chỉ tiếp tục nếu đạt điều kiện tiếp tục; nếu không, ghi "chưa đủ cơ sở đầu tư tiếp".
- Quyết định dựa trên ước lượng điểm theo quy tắc trên; khoảng bất định được báo cáo kèm.

## 6. Bước B (chỉ khi bước A đạt điều kiện tiếp tục; chi tiết khóa trước khi chạy B)
- Chỉ dùng **text**: phụ đề I–III. Mô tả appraisal và bản tóm tắt đối chứng đều phải được tạo **chỉ từ phụ đề**
  (một lượt đánh giá text-only riêng), không dùng lại mô tả đa phương thức của bước A.
- Ba điều kiện trên cùng mẫu: (1) phụ đề; (2) phụ đề + tóm tắt thông thường cùng ngân sách độ dài;
  (3) phụ đề + appraisal.
- Mô hình zero-shot, không huấn luyện trên mẫu pilot. Model, prompt, cách lấy phân phối trên 7 nhãn và xử lý đầu
  ra lỗi: **chưa chọn**, phải khóa trước khi chấm.
- Thước đo: Δ_i = log p_(3)(y_i) − log p_(baseline)(y_i), gọi là "mức cải thiện log-score". Bootstrap giữ cặp và
  trọng số tầng. Kết quả là pilot, không phải xác nhận.

## 7. Bước C (chỉ khi bước B cho tín hiệu)
Biến thể chỉ sửa phụ đề, nên phép thử chỉ áp dụng cho đầu vào text; không ghép với video/âm thanh gốc. Biến thể được
người đánh giá kiểm tra tính hợp lý; không gán nhãn clip IV gốc cho tình huống đã sửa.

## 8. File
- `make_sample.py` — tạo mẫu (seed cố định).
- `design_strata.csv`, `item_map.csv` — chỉ cho thiết kế và phân tích.
- `rater_package.csv`, `calibration.csv` — gói cho người đánh giá.
- `CODEBOOK.md` — hướng dẫn đánh giá.
