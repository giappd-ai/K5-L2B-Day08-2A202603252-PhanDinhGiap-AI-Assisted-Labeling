# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 506; đối tượng sau duyệt: 489; ngưỡng accepted: IoU >= 0.9
- Đã **loại 26 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 475 | 475/506 = 93.9% | 506 box máy |
| edited (sửa) | 7 | 7/506 = 1.4% | 506 box máy |
| deleted (xoá) | 24 | 24/506 = 4.7% | 506 box máy |
| added (vẽ thêm) | 7 | 7/489 = 1.4% | 489 đối tượng sau duyệt |

Trong số deleted + added, **2 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 5.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 265 | 1 | 4 | 1 |
| bicycle | 4 | 0 | 3 | 1 |
| motorbike | 130 | 0 | 7 | 1 |
| car | 53 | 0 | 3 | 0 |
| bus | 3 | 0 | 0 | 0 |
| truck | 6 | 0 | 1 | 0 |
| three_wheeler | 14 | 6 | 6 | 4 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 52 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 34 | 29 | 18 | 54.0% (KTC 95%: 42–66%) | 65.4% (KTC 95%: 52–77%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 10/12 = 83.3% (KTC 95%: 55–95%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - wrong_class: 3/3
  - extra: 1/3
  - missing: 3/3
  - loose: 3/3
  - lọt: img_014.jpg (extra), img_026.jpg (extra)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
