# Codebook — pilot khả năng quan sát appraisal (bước A)

Người đánh giá chỉ nhận **ba video clip I–III** (đã đổi tên `<item>_1/2/3.mp4`) và phụ đề tương ứng của mỗi mục.
Không nhận clip IV, mã mẫu, mã clip, thư mục dataset hay bất kỳ thông tin phân tầng nào.

**Nhiệm vụ chung.** Ngay sau clip III, một người (gọi là B) sẽ nói lượt tiếp theo. Bạn không biết B là ai và không
biết B sẽ nói gì. Chỉ dựa vào **những gì nghe/thấy trong clip I–III**, trả lời bốn câu hỏi dưới đây. Không dùng
hiểu biết về cốt truyện phim. Nếu nhận ra nhân vật hoặc tình tiết từ trước, đánh dấu ở ô Q0 của mục đó.

Làm độc lập, không trao đổi với người đánh giá khác trước khi nộp toàn bộ.

## Q0. Nhận ra từ ngoài clip (mỗi mục)
- `khong` — không nhận ra nhân vật/tình tiết.
- `co` — có nhận ra; câu trả lời có thể bị ảnh hưởng bởi hiểu biết ngoài clip.

## Q1. B là ai trong phần đã quan sát? (chọn một)
- `noi_I` — người đang nói ở clip I.
- `noi_II` — người đang nói ở clip II.
- `nghe_III` — một người nghe xuất hiện trên hình ở clip III (ghi thêm mô tả ngắn, ví dụ "người đàn ông bên trái").
- `khac` — một người khác có xuất hiện (ghi mô tả).
- `khong_xac_dinh` — không đủ cơ sở để đoán ai sẽ nói tiếp.

## Q2. Điều B mong muốn, tránh né hoặc quan tâm
- Ghi ngắn gọn bằng lời (ví dụ "muốn tiếp tục dự án").
- Mức bằng chứng: `truc_tiep` (được nói/thể hiện rõ) · `suy_ra` (suy ra có căn cứ từ chi tiết cụ thể) ·
  `chua_du` (chưa đủ thông tin). Nếu Q1 là `khong_xac_dinh`, Q2 tự động là `chua_du`.

## Q3. Sự kiện ở clip III quan hệ thế nào với điều ở Q2? (biến định danh, không có thứ bậc)
- `ho_tro` — giúp hoặc ủng hộ điều B muốn.
- `can_tro` — ngăn cản hoặc đi ngược điều B muốn.
- `hon_hop` — vừa giúp vừa cản trở.
- `khong_lien_quan` — không ảnh hưởng đến điều B muốn.
- `chua_du` — chưa đủ thông tin. **Đây là câu trả lời hợp lệ, không phải ô bỏ trống.**

Kèm mức bằng chứng cho Q3: `truc_tiep` · `suy_ra` · `chua_du`.

## Q4. Vị trí bằng chứng
Với mỗi câu trả lời Q2/Q3 không phải `chua_du`, ghi: clip (I/II/III), kênh (lời nói/giọng/hình ảnh) và trích dẫn
phụ đề hoặc mốc thời gian (giây).

## Thời gian
Ghi tổng thời gian làm cho mỗi mục (phút), để ước tính khối lượng ở lượt hiệu chỉnh.
