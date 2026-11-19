# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603252 - Phan Đình Giáp

> **Chỉ viết phần thay đổi so với v1.0** (v1.0 giữ nguyên tại `guideline/guideline_v1.0_label_contract.md`).
> Mỗi ca khó có ảnh ví dụ (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 4 | Người ngồi trong xe ba bánh có mui, chỉ thấy thấp thoáng: ngưỡng "còn thấy ≥ 20%" có hợp lý? | Thay ước lượng 20% diện tích (khó đo bằng mắt) bằng tiêu chí kiểm được: gán `person` khi thấy **đầu và ít nhất một phần thân** (vai/ngực); chỉ thấy tay, chân hoặc mảng áo → không gán | img_013 (khách ngồi sau tuk-tuk), img_010 (khách trong xích lô xếp hàng) | CAL-03 |
| 5 | Kính chắn gió / cửa sổ thuộc mask xe? | **Có**: kính và khung mui thuộc mask xe. **Người** nhìn thấy qua kính hoặc khoang hở **không** thuộc mask xe, mỗi người một mask riêng (theo 4.2.4) | img_013, img_002 (`masks/tuktuk_img013.jpg`, `masks/tuktuk_img002.jpg`) | CAL-05 |
| 6 | Box máy chồng nhau nhiều trong ảnh đêm: gộp hay giữ? | Một vật chỉ giữ **1 box** ôm sát, xoá box trùng. Vệt sáng nhòe không nhận ra hình dạng vật → không gán; cụm ≥ 5 vật nhòe sát nhau → một `ignore_region` | img_024 (phơi sáng dài, box máy #283, #290, #294, #302 nằm trên vệt nhòe) | CAL-04 |
| 1, 2, 3, 7 | Xe đạp điện; khối xe máy đỗ dày gần camera; hàng nhô ra; conf tối thiểu | **Chưa chốt** trong v1.1: không gặp đủ ca rõ ràng trong 4 ảnh gold + 24 ảnh batch để ra quy tắc có ảnh ví dụ. Giữ nguyên v1.0; với câu 7, **không** đặt conf tối thiểu để tự chấp nhận box (lần export đầu chấp nhận 100% box máy nhưng chỉ bắt 1/12 bẫy, xem CAL-06) | — | — |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| img_011, img_013, img_002 | Tuk-tuk / xe ba bánh | YOLO26m gọi `truck` 0.77 (img_011), `car` 0.41 (img_013), bỏ sót (img_002); YOLOE bắt đúng img_002 (0.93) | Luôn `three_wheeler` (lỗi máy, v1.0 mục 2 đã rõ) | **2.1** Khi duyệt: lọc riêng box `car`/`truck`, đếm số bánh nhìn thấy; 3 bánh → `three_wheeler` (CAL-01) |
| img_002, img_024 | Xe bán tải; xe van / minivan | YOLO26m gọi `truck` (0.73 bán tải img_002; 0.53 xe van img_024) | `car` (lỗi máy, v1.0 mục 2 đã rõ) | **2.2** `truck` chỉ khi buồng lái **tách riêng** thùng hàng; thân liền khối (van, minivan) hoặc bán tải → `car` (CAL-02) |
| img_024 | Ảnh đêm phơi sáng dài, xe chuyển động thành vệt sáng | YOLO26m vẽ `car`/`person` lên vệt nhòe, box chồng nhau | Không gán vệt nhòe; cụm → `ignore_region` | **4.1.8** Ảnh đêm / nhòe chuyển động (CAL-04) |
| img_013, img_002 | Mask SAM tuk-tuk có người lái + khách | SAM 2.1 gộp người vào mask xe (1 click âm không đủ) | Người không thuộc mask xe; kính thuộc xe | **4.2.5** Kính thuộc xe, người sau kính không thuộc xe (CAL-05) |
| img_014, img_026 | Bẫy lọt: box thừa không ứng với vật nào, nằm giữa vùng đông vật thật | (bẫy nhân tạo) | **Lỗi người duyệt**, quy tắc v1.0 đã đủ rõ → không đổi luật nhãn, sửa nhãn | **8** (mới) Checklist duyệt bắt buộc (CAL-06) |

### Mục 8 (mới) — Checklist duyệt bắt buộc

1. Ẩn toàn bộ nhãn, quét ảnh trái → phải, **đếm** số vật theo lớp trước khi xem nhãn máy.
2. Hiện nhãn: so số đếm với số box; thêm box sót, xoá box thừa / trùng.
3. Soi **lớp** và **4 cạnh** từng box (sai lệch ≤ 2 px).
4. Sau khi xong cả lô: dùng Filters lọc từng lớp `three_wheeler`, `truck`, `car` rà lại một lượt.

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt câu hỏi mở 4, 5, 6; thêm quy tắc 2.1 (đếm bánh khi máy gọi car/truck), 2.2 (truck vs van/bán tải), 4.1.2b (người trong xe có mui: đầu + thân), 4.1.8 (ảnh đêm / nhòe), 4.2.5 (kính thuộc xe, người không), mục 8 checklist duyệt. Câu 1, 2, 3, 7 chưa chốt | CAL-01 … CAL-06 |
