# Thông tin học viên

- Họ và tên: Phan Đình Giáp
- MSHV: 2A202603252
- Lớp (L2 / L3 / khác): L2B
- Tài khoản GitHub: giappd-ai (giappdhe140952@gmail.com)
- Repo bài nộp (URL): https://github.com/giappd-ai/K5-L2B-Day08-2A202603252-PhanDinhGiap-AI-Assisted-Labeling.git
- Commit chốt bài (hash): 994f968
- Máy: Windows 11 Home, NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA, torch 2.6.0+cu124, ultralytics 8.4.215) — chạy model trên máy local, không dùng Colab
- CVAT (Docker): v2.81.0 (cvat/server:v2.81.0, http://localhost:8080)

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 475 / 7 / 24 (mẫu số: 506 box máy = 93.9% / 1.4% / 4.7%); added: 7 (mẫu số: 489 đối tượng sau duyệt = 1.4%)
- So với gold: precision / recall của **máy** 54.0% / 65.4% (34 TP; 63 box máy / 52 đối tượng gold)
- Bắt bẫy: 10/12 = 83.3% (lượt duyệt đầy đủ; trước đó 2 lần export dở: 1/12, 3/12, xem REPORT mục 3)
- Thời gian: gán tay 2.0 phút/ảnh (8 phút / 4 ảnh); duyệt có pre-label 2.1 phút/ảnh (50 phút / 24 ảnh)
- Lỗi QC: 50.0% theo ảnh (4/8), 5.9% theo đối tượng (12/205)
- **Quyết định:** Rework — lý do một dòng: ảnh lỗi 4/8 = 50% > 2%, lỗi hệ thống "thừa" 9/12 (vệt nhòe ban đêm, YOLOE gọi quang gánh là three_wheeler).
