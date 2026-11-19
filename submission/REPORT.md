# Báo cáo Day 08 — 2A202603252 - Phan Đình Giáp

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m (COCO) | mặc định 0.3 cho mọi lớp (`--thr ""`) | r03/r04: 0.5 làm mất xe máy và người ở xa (img_007: motorbike 11→5); r07 + `threshold_sweep`: không lớp nào đạt recall 90% ở ngưỡng ≤ 0.5 (person tối đa 67% trên 27 vật gold, motorbike 78%/9, car 88%/8); bicycle (1 vật), bus, truck (0 vật) < 5 mẫu → không chọn ngưỡng riêng |
| three_wheeler | YOLOE-26s, prompt: tuk-tuk, cyclo, cargo tricycle | mặc định 0.3 | r05: bắt được tuk-tuk img_002 (0.93) mà YOLO26 bỏ sót, xích lô img_032 (0.92); r08: recall tối đa 86% trên 7 vật gold, chưa đạt 90% → giữ mặc định. Đặt YOLOE **trước** YOLO26 khi gộp để box three_wheeler thắng box car/truck trùng chỗ |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "" --conf 0.3 --iou 0.6 --split batch,gold --out work/prelabel/merged_review/pre.coco.json` → 591 box (person 311, motorbike 154, car 68, three_wheeler 32, bicycle 14, truck 8, bus 4); loại 8 car + 5 truck + 2 bus của YOLO26 vì trùng box three_wheeler ưu tiên cao hơn.
- **Phiên bản SAM bạn chọn**: SAM 2.1 base (`sam2.1_b.pt`). Lý do: chạy được trên GPU RTX 3050 6GB local (r06: 2.27 s/ảnh gồm cả detect + mask), mask bám biên mui/khung xe tốt; đã so sánh với SAM 1 (`mobile_sam.pt`) trên img_013 cùng 3 click (s01 vs s07): SAM 1 tách người lái tốt hơn nhưng biên bánh xe thô hơn; vẫn chọn SAM 2.1 làm bản chính vì biên mask sát hơn và GPU đủ chạy, chấp nhận thêm click âm. SAM 3 không thử (cần quyền truy cập riêng).
- Lỗi nào **không sửa được bằng đổi ngưỡng**: tuk-tuk bị gọi sai lớp — img_011 `truck 0.77`, img_013 `car 0.41` — vẫn sai ở cả 0.25 và 0.5 (r03, r04, r06); xích lô img_007 không có box nào ở mọi ngưỡng; xe bán tải img_002 bị gọi `truck 0.73` trong khi guideline xếp bán tải là `car`. Sửa bằng cách đổi **model/prompt** (YOLOE với prompt tuk-tuk/cyclo cho three_wheeler, gộp với ưu tiên YOLOE) và **người duyệt** đổi lớp bán tải → car; SAM không sửa được lớp (r06).
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng**: điểm cao mà sai — img_011 tuk-tuk `truck 0.77`, img_002 bán tải `truck 0.73`; điểm thấp mà đúng — img_002 người đi bộ hậu cảnh `person 0.27–0.36` (bị loại ở 0.5), img_032 YOLOE `three_wheeler 0.27` đúng là xích lô (nhưng box lỏng trùm 2 xe).
- Giới hạn: ngưỡng được chọn và chấm trên **cùng** 4 ảnh gold (56 đối tượng, phần lớn là person 27) nên precision/recall của máy trên gold là con số lạc quan; bicycle/bus/truck gần như không có mẫu gold nên không có bằng chứng nào cho ngưỡng của các lớp này.

## 2. AI error profile (từ `qc/review_stats.md`)

Số của **lượt duyệt bằng mắt đầy đủ** (bản `batch_reviewed.json` cuối: 24 ảnh, 503 đối tượng). 26 đối tượng quanh vùng bẫy đã bị script loại khỏi bảng vì đó là lỗi nhân tạo, không phải lỗi của máy.

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 475 | 93.9% | 506 box máy |
| edited | 7 | 1.4% | 506 box máy |
| deleted | 24 | 4.7% | 506 box máy |
| added | 7 | 1.4% | 489 đối tượng sau duyệt |

- Trong deleted + added có 2 ca là **đổi lớp** (box đúng chỗ, máy gán sai lớp), nên số vật máy **sót thật** ≈ 7 − 2 = 5.
- Lớp nào máy tệ nhất? **three_wheeler**: chỉ 14/26 box máy được giữ nguyên (6 sửa, 6 xoá) và phải vẽ thêm 4. Box YOLOE lỏng hoặc trùm 2 xe (img_032 ở pilot), và YOLO26 gọi tuk-tuk là car/truck (img_011, img_013). Lớp tốt nhất là person (265/270 giữ nguyên), motorbike (130/137) và car (53/56). bicycle có tỉ lệ xoá cao (3/7), nhưng mẫu quá nhỏ để kết luận.
- So với gold: precision / recall của **máy** = 54.0% (TP 34 / 63 box máy) / 65.4% (TP 34 / **52 đối tượng gold**); KTC 95%: precision 42–66%, recall 52–77%. Máy sót 18/52 vật gold, phần lớn là person nhỏ ở hậu cảnh.
- **Mâu thuẫn cần đọc đúng:** trên batch chỉ phải vẽ thêm 7/489 (1.4%), nhưng trên gold máy sót 34.6%. Vì vậy accepted 93.9% **không** có nghĩa máy tốt 93.9%. Có thể (a) ảnh batch dễ hơn gold, hoặc (b) khi duyệt tôi vẫn bỏ qua vật nhỏ máy không vẽ, mà bẫy `missing` (vật rõ, bị xoá box) không đo được loại sót này. Số trên gold thì lạc quan vì ngưỡng được chọn trên chính 4 ảnh này.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: **10/12 = 83.3%** (KTC 95%: 55–95%), vừa đạt ngưỡng 80% (≥ 10/12).
  - wrong_class 3/3 · missing 3/3 · loose 3/3 · **extra 1/3**.
  - Bẫy lọt: 2 box thừa ở img_014 và img_026.
- Quá trình: số trên là của **một lượt duyệt bằng mắt đầy đủ** cả 24 ảnh. Trước lượt này có 2 lần export khi chưa duyệt xong (lần 1 chưa sửa box nào: accepted 100%, 1/12; lần 2 sửa dở: 3/12), và `review_stats` của 2 lần đó đã in tên các ảnh có bẫy lọt. Vì vậy 10/12 có thể lạc quan hơn một lượt duyệt mù.
- Vì sao lọt 2 box thừa: box thừa nằm giữa vùng có nhiều vật thật nên trông "hợp lý". Tôi đã bỏ bước 2 của checklist, tức đối chiếu **số box với số vật đếm được**.
- **Accepted cao mà bắt bẫy thấp** (lần export 1: chấp nhận 100% box máy, chỉ bắt 1/12) là dấu hiệu điển hình của thiên kiến tự động hóa. Khi làm đủ checklist, tỉ lệ bắt tăng lên 10/12. Như vậy chất lượng lô phụ thuộc vào việc người duyệt có làm đủ các bước hay không, chứ không phụ thuộc vào model.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay **8 phút / 4 ảnh = 2.0 phút/ảnh** (56 đối tượng ≈ 8.6 giây/đối tượng); duyệt có pre-label **50 phút / 24 ảnh ≈ 2.1 phút/ảnh** (503 đối tượng ≈ 6.0 giây/đối tượng).
- Thiên lệch khi so: gold làm trước nên khi duyệt batch tôi đã quen cảnh và quen CVAT. Ảnh batch đông hơn (≈ 21 so với 14 đối tượng/ảnh). Thời gian duyệt cũng không tính các lần export dở trước đó.
- Kết luận: **có, nhưng có điều kiện**. Máy đáng dùng cho person / motorbike / car (accepted ≥ 94% theo lớp, nhanh hơn ≈ 30% theo giây/đối tượng). **Không** tin cho three_wheeler (14/26 giữ nguyên) và không tin rằng máy không bỏ sót (recall gold 65.4%). Pre-label chỉ có lợi khi người duyệt làm đủ checklist; lần 1 (chấp nhận 100%) cho thấy nếu bỏ checklist, lô sẽ mang nguyên lỗi của máy.
- **Ba quy tắc giảm thiên kiến của tôi** (mỗi quy tắc kiểm tra được):
  1. **Đếm trước khi nhìn:** mỗi ảnh ghi số vật đếm được khi ẩn nhãn, rồi so với số box sau duyệt. Kiểm bằng số: |đếm − box| > 2 thì duyệt lại ảnh đó. Mục tiêu: bẫy `extra` và `missing` đều 3/3.
  2. **Lượt lọc theo lớp:** sau khi duyệt xong lô, dùng Filters rà riêng `three_wheeler`, `truck`, `car`, ghi số box đổi lớp vào note `time_log`. Kiểm bằng số: bẫy `wrong_class` 3/3.
  3. **Không export / chạy số đo khi chưa duyệt xong cả 24 ảnh**, và mọi ảnh được duyệt như nhau, kể cả ảnh "không bị nghi". Kiểm bằng việc làm: chỉ 1 lần chạy `review_stats` trên lô (khác bài này: 3 lần), thời gian `batch_review` ≥ 2 phút/ảnh với ảnh > 15 vật.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_013 (tuk-tuk hồng, có tài xế + khách ngồi sau quầy) | SAM 2.1 b | 3 → 6 → 8 (3 lượt) | Lượt 1 (2+, 1−): **có**, dính tài xế và người phía sau. Lượt 3 (5+, 3−): gần như không | Lượt 1: 45.4% ảnh, nuốt tài xế. Lượt 2 thêm click âm → co lại chỉ còn mui (10.5%, 2 polygon). Lượt 3 thêm click dương ở thân/bánh → đủ xe nhưng **lan sang xe máy đỗ phía sau** (x≈880–970) và còn lốm đốm trên mặt/tay tài xế | `tuktuk_img013.jpg`, `tuktuk_img013_v2.jpg`, `tuktuk_img013_v3.jpg` |
| img_013 (cùng xe, so sánh SAM 1) | MobileSAM (SAM 1) | 3 (2+, 1−) — cùng click với lượt 1 SAM 2.1 | Gần như không: loại được thân tài xế và người phía sau, chỉ còn dính mặt + bàn chân | 35.2% ảnh, 2 polygon; sàn xe dưới chân tài xế bị khuyết lốm đốm; biên bánh xe thô hơn SAM 2.1 | `tuktuk_img013_mobilesam.jpg` |
| img_002 (tuk-tuk xanh giữa phố) | SAM 2.1 b | 3 (2+, 1−) | Không dính tài xế (nhờ click âm trên người) | Mask lấp cả khoảng hở khoang khách, gồm người đứng phía sau nhìn xuyên qua (x≈400–560) — lẽ ra là lỗ | `tuktuk_img002.jpg` |
| img_011 (xe tay ga xanh, người lái ngồi trên) | SAM 2.1 b | 2 → 6 (2 lượt) | Không (click âm trên thân + 2 chân người lái) | Lượt 1 (1+, 1−): chỉ ra chắn bùn trước (0.4% ảnh). Lượt 2 (3+, 3−): đủ thân xe, gương/tay lái bị tay người che chỉ lấy được một phần | `scooter_img011.jpg`, `scooter_img011_v2.jpg` |

- Đối chiếu guideline mục 4.2: 4.2.4 (hai vật chạm nhau, mỗi vật một polygon) là quy tắc bị SAM vi phạm nhiều nhất — tuk-tuk ↔ tài xế (img_013 lượt 1), tuk-tuk ↔ xe máy đỗ sau (img_013 lượt 3). 4.2.2 (lỗ > 5% bỏ ra) giúp quyết img_002: khoang khách hở lớn hơn 5% diện tích xe nên phải bỏ, SAM không tự làm. Guideline **thiếu**: (a) người ngồi trong xe ba bánh có thuộc mask xe không (chỉ có quy tắc cho box), (b) gương/tay lái bị tay người che thì polygon xe cắt theo tay hay không.
- So sánh SAM 2.1 vs SAM 1 trên img_013 với **cùng 3 click**: SAM 2.1 nuốt cả tài xế (45.4% ảnh) còn MobileSAM tách được thân tài xế (35.2%), nhưng biên bánh xe của MobileSAM thô hơn. Không có model nào thắng tuyệt đối; với 1 click âm thì cả hai vẫn dính một phần người lái.
- Quy tắc rút ra: click âm phải đặt trên **từng** vật kề (người lái, xe bên cạnh) nhưng phải có click dương ở **mỗi phần** của vật (mui, thân, bánh) — chỉ thêm click âm sẽ làm mask co (img_013 lượt 2); 1 click dương là không đủ cho xe nhỏ có người ngồi che (img_011 lượt 1). Mask SAM là điểm khởi đầu, vẫn phải sửa tay chỗ chạm vật khác.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: không thử; dùng `sam2_points.py` offline trên GPU local.

## 6. Calibration (Part 6)

- Số quyết định: **6** (`calibration_decisions.csv`): 2 `loi_may` (CAL-01, CAL-02), 3 `cau_hoi_mo` (CAL-03 câu 4, CAL-04 câu 6, CAL-05 câu 5), 1 `bay_lot` (CAL-06: 2 box thừa lọt ở img_014, img_026). Quan trọng nhất là **CAL-01**: tuk-tuk bị YOLO26 gọi car/truck là lỗi không sửa được bằng ngưỡng, và lớp three_wheeler cũng là lớp máy tệ nhất trong review (12/26 accepted).
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: 2.1 (đếm bánh khi máy gọi car/truck), 2.2 (truck vs van/bán tải), 4.1.2b (người trong xe có mui: đầu + thân), 4.1.8 (ảnh đêm / nhòe), 4.2.5 (kính thuộc xe, người không), mục 8 (checklist duyệt). Câu hỏi mở 1, 2, 3, 7 chưa chốt vì không có ca đủ rõ.
- Nguyên nhân gốc, tách rõ ba loại:
  - **Máy sai, guideline đã rõ** (CAL-01, CAL-02): sửa nhãn, không đổi luật.
  - **Guideline mơ hồ** (CAL-03, CAL-04, CAL-05): bổ sung quy tắc.
  - **Tôi sai, guideline đã rõ** (CAL-06, 2 bẫy box thừa lọt): không đổi luật để hợp thức hoá, mà siết quy trình duyệt.
- Lưu ý mâu thuẫn cần xử lý: trong `batch_reviewed.json`, img_024 #308 (xe van trắng, box [701,198,136,105]) vẫn đang là `truck` + `can_review`, trái với quy tắc 2.2. Cần sửa thành `car` trong CVAT và export `qc_fixed.json` (xem mục 7).

## 7. QC và quyết định (Part 6)

- Mẫu: **8 ảnh** (3 ảnh có `can_review`: img_010, img_017, img_024 + 5 ảnh ngẫu nhiên seed 42: img_003, img_006, img_014, img_018, img_035), **205 đối tượng**. Người kiểm: AI (Claude) soi ảnh `work/qc/review/`, tôi xác nhận lại từng dòng. **Hạn chế:** người kiểm không độc lập, vì tôi vừa là người duyệt lô.
- Lỗi theo **đối tượng**: **12/205 = 5.9%**. Theo **ảnh**: **4/8 = 50.0%** (ngưỡng Release ≤ 2%).
- Top lỗi:
  - **thừa** 9: img_024 có 5 box car/truck trên vệt sáng nhòe (#274, #282, #286, #294, #306); img_014 có box car trong cửa hàng (#147) và 2 box three_wheeler trùm quang gánh (#141, #148); img_010 có box bicycle trên bánh sau xích lô (#115).
  - **sai lớp** 2: img_024 #308 xe van gán truck (đúng là car); img_018 #175 xe khách nhỏ gán three_wheeler (đúng là bus).
  - **bỏ sót** 1: img_010, xích lô thứ 2 trong hàng không có box.
  - 3 ảnh có `can_review` thì 2 ảnh có lỗi (img_010, img_024). Ca tôi đánh dấu "không chắc" đúng là nơi lỗi tập trung.
- **Giới hạn mẫu nhỏ:** 4/8 ảnh lỗi, cận trên 95% (Wilson) của tỉ lệ ảnh lỗi thật là **78%** (`qc_report_auto.md`). Mẫu 8 ảnh chỉ đủ phát hiện lỗi lớn. Ngay cả khi 0 lỗi cũng không chứng minh được lô dưới 2%.
- Script kiểm được / không kiểm được:
  - **Kiểm được:** cấu trúc COCO, đủ 24 ảnh, box < 20 px, tỉ lệ bắt bẫy, so sánh với nhãn máy.
  - **Không kiểm được:** box có ôm sát không; vật mà máy và người **cùng bỏ sót** (bẫy chỉ đo vật bị xoá box có chủ đích); lớp sai mà máy và người cùng đồng ý (xe van = truck); người duyệt có thật sự làm theo checklist không.

**Quyết định: Rework**, ngày 2026-11-19. Lý do (đi theo Hold → Rework → Release):
- **Hold — không:** các câu hỏi mở gặp trong dữ liệu (4, 5, 6) đã chốt ở v1.1; các câu 1, 2, 3, 7 không có ca trong lô; ảnh không hỏng.
- **Rework — có:** ảnh lỗi 4/8 = 50% > 2%. Lỗi hệ thống: `thua` chiếm 9/12 lỗi, đều là box máy không bị xoá: vệt nhòe ban đêm, đồ vật bị YOLOE gọi là three_wheeler. Bắt bẫy 10/12 vừa đạt nhưng con số này lạc quan (mục 3), và chính 2 bẫy lọt cũng là box thừa. Hai nguồn bằng chứng cùng chỉ ra một điểm yếu: không xoá box thừa.

Việc cần làm tiếp:
1. Sửa trong CVAT 12 lỗi ở mẫu QC, rồi rà **toàn bộ 24 ảnh** theo 2 quy tắc mới: v1.1 mục 4.1.8 (vệt nhòe) và 2.2 (van/bán tải = car). Rà riêng mọi box `three_wheeler` điểm < 0.6 của YOLOE, vì đây là nguồn chính của box thừa.
2. Export `submission/labels/qc_fixed.json`, chạy lại `build_release.py`, `qc_sample.py --force` (mẫu mới), điền phiếu, chạy `qc_report.py`.
3. Recipe: nâng ngưỡng YOLOE three_wheeler hoặc thêm prompt phủ định (quang gánh, xe đẩy tay). Kiểm lại trên ảnh pilot trước khi gộp.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài**: Claude Code (Claude Opus) — chạy các script `validate_labels`, `threshold_sweep`, `merge_prelabels`, `inject_traps`, `sam2_points`; 
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? Không. Bằng chứng thời gian: `gold_manual.json` được export lúc 12:30 ngày 17/11, còn lượt chạy model đầu tiên (r01 trong `work/runs.csv`) là 18:40 cùng ngày. Gold còn 2 box person nhỏ hơn 20 px (img_001 #12: 14.5 px, img_036 #47: 19.6 px), đáng lẽ phải bỏ hoặc gộp vào `ignore_region` theo v1.0 mục 4.1.6.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: Windows 11, RTX 3050 6GB Laptop (CUDA), chạy model local thay cho Colab; CVAT v2.81.0 (Docker); 
- Dùng cache `data/cache/` thay vì tự chạy? Không, tự chạy YOLO26m, YOLOE-26s, SAM 2.1 và MobileSAM trên GPU local (xem `work/runs.csv`).
- Lỗi / bug script đã gặp (nếu có): không gặp bug script. Các lỗi đều do thao tác của tôi: có lúc đặt nhầm bản sao `gold_manual.json` thành `batch_reviewed.json`, nên `build_release.py` báo "Thiếu ảnh img_003.jpg"; `validate_labels --all` báo 4 ảnh thay vì 24 nên tôi đã export lại đúng Task.
- Một điều bạn làm khác lần sau: không export và không chạy `review_stats` khi chưa duyệt xong cả 24 ảnh, vì chạy sớm làm lộ vị trí bẫy. Ngoài ra, ẩn nhãn để đếm vật ngay từ ảnh đầu tiên.