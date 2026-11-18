# Báo cáo Day 09 — Trương Duy Anh, 2A202603027

## 1. Kết luận trước
Pipeline kết luận **HOLD** (mã thoát 10). Vào **224** ảnh, bước nhận loại **22** ảnh, xuất **202** ảnh dưới dạng **bản nháp** (`manifest.json` ghi `decision: Hold`, chưa phải bản phát hành).
Không còn lỗi bắt buộc nào sau sửa tự động và người duyệt (`07_error_list_v3`: 0 lỗi `error`), nhưng **38/202 ảnh (18,8%)** còn cảnh báo chưa được người xem, vượt ngưỡng `warn_ratio_hold` = 10%. Nguyên nhân: hàng đợi có 125 ảnh mà ngân sách duyệt chỉ 80 ảnh.

## 2. Lỗi và phân việc
Số liệu từ `01_intake` + `02_error_list_v1` (trước khi sửa). Mẫu số: 224 ảnh ở bước nhận, 202 ảnh ở QC.

| Mã lỗi | Số ảnh/dòng | Ai sửa | Vì sao |
|---|---:|---|---|
| `DUP_IMAGE_EXACT` | 8 ảnh / 224 | máy (loại ở cửa vào) | trùng md5, chắc chắn 100%; giữ ảnh đầu tiên |
| `DUP_IMAGE_NEAR` | 4 ảnh / 224 | máy (loại) | phash ≤ 12, đã mở xem từng cặp: đều là trùng thật |
| `LEAK_TEST` | 5 ảnh / 224 | máy (loại) | trùng ảnh test/val: giữ lại là metric đẹp giả |
| `CORRUPT_IMAGE` | 2 ảnh / 224 | máy (loại) | không `load()` được |
| `BAD_SIZE` | 3 ảnh / 224 | máy (loại) | cạnh ngắn < 250 px |
| `BAD_CLASS_ID` | 6 dòng | máy (`drop_line`) | id 93, 4, −1: không biết lớp đúng nên xoá dòng |
| `BAD_FORMAT` | 3 dòng | máy (`drop_line`) | thiếu cột / `nan` |
| `BOX_ZERO` | 4 dòng | máy (`drop_line`) | box rộng hoặc cao bằng 0 |
| `DUP_BOX` | 7 dòng | máy (`drop_line`) | cùng lớp, IoU > 0,9 với dòng trước |
| `BOX_OUT_OF_IMAGE` | 8 dòng | máy (`clip_box` ×8) | cắt về [0,1], giữ lớp |
| `BAD_CLASS_NAME` | 2 tên (`Person`, `Bus`) | máy (`rename_classes`) | ghi lại `names` theo danh sách dự án |
| `MISSING_LABEL_FILE` | 6 ảnh | **người** | máy không được bịa nhãn (`needs_label`, không tạo file rỗng) |
| `POSSIBLE_MISSING` | 53 | **người** | chỉ là nghi ngờ từ yolo26l |
| `POSSIBLE_WRONG_CLASS` | 18 | **người** | lỗi ngữ nghĩa (car↔truck↔bus) |
| `POSSIBLE_LOOSE_BOX` | 28 | **người** | cần mắt người chỉnh box |
| `POSSIBLE_FALSE_POS` | 29 | **người** | model có thể bỏ sót, không xoá tự động |
| `EMPTY_LABEL` / `BOX_TINY` / `BOX_HUGE` | 12 / 6 / 3 | **người** | ảnh nền hay thiếu nhãn? box nhỏ/to bất thường |
| `CLASS_RARE` | 4 (cấp dataset) | — | do id lạ (93, 4, −1) trước khi sửa; hết sau sửa |

Máy tự sửa 35 thay đổi (`03_fixes.csv`: 20 `drop_line`, 8 `clip_box`, 6 `needs_label`, 1 `rename_classes`), sau đó chỉ còn 6 lỗi bắt buộc (`MISSING_LABEL_FILE`), đều chuyển người.

**Hàng đợi** (`05_review_queue.csv`, 125 ảnh): điểm = tổng trọng số mã × (1 + conf của model). Đầu hàng là `img_0020` (11,5: vừa bị xoá dòng id 93, vừa có `POSSIBLE_MISSING` + `POSSIBLE_WRONG_CLASS`), sau đó 6 ảnh `NEEDS_LABEL` (trọng số 10) để chắc chắn ảnh chưa có nhãn được người gán trong ngân sách — nếu không sẽ thành Rework. Tiếp theo là ảnh có nhiều cờ ngữ nghĩa với conf cao (`img_0160`, `img_0127`, `img_0004`…).

Với ngân sách 80 ảnh, người duyệt 80 ảnh (`06_review_log.csv`). Cảnh báo ngữ nghĩa giảm từ **123 → 57** (v2 → v3): `POSSIBLE_MISSING` 53 → 13, `WRONG_CLASS` 18 → 3, `LOOSE_BOX` 28 → 20, `FALSE_POS` 24 → 21. 45 ảnh cuối hàng không ai xem.

**3 lỗi tự nhìn thấy bằng `viz_labels.py`** (trên `data/raw`):
- `img_0160`: bãi xe tải rất đông nhưng chỉ có ~5 box; dòng 1 gán **bus** cho một xe tải (model: truck, IoU 0,95 → `POSSIBLE_WRONG_CLASS`).
- `img_0127`: box person dòng 1 phình to, bao cả con bò phía sau (`POSSIBLE_LOOSE_BOX`, IoU 0,44).
- `img_0004`: file nhãn rỗng nhưng góc phải có ô tô đỗ (`EMPTY_LABEL` + 2 `POSSIBLE_MISSING` conf 0,89 / 0,79).

## 3. Calibration (2 ngưỡng)
Số liệu từ `submission/calibration/calibration_report.md` (calib 39 ảnh, mỗi ô là precision / recall).

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | **12** | `DUP_IMAGE_NEAR` 1,00 / 0,50 → 1,00 / 0,67; `DUP_IMAGE_EXACT` giữ 1,00 / 1,00 | Ảnh bị cờ là **bị xoá** nên ưu tiên precision. Trên calib, 4–14 cho kết quả như nhau. Trên raw, cặp trùng thật xa nhất có phash 10 (`img_0186`/`img_0061`), còn từ 16 bắt đầu gộp nhầm ảnh khác nhau → chọn 12, có khoảng an toàn cả hai phía. |
| `second_opinion.conf` | 0,25 | **0,6** | `WRONG_CLASS` 0,56 / 0,83 → 0,62 / 0,83; `MISSING` 0,50 / 1,00 → 0,57 / 1,00; `LOOSE_BOX` 0,29 / 0,40 → 0,20 / 0,20; `FALSE_POS` 0,31 / 1,00 (không đổi, dùng `fp_conf`) | Đây là **cảnh báo cho người**, FN (lỗi lọt vào dataset) đắt hơn FP (người mở xem uổng). 0,6 là mức cao nhất vẫn giữ recall `MISSING` = 1,00 và tăng precision. 0,8 cho `WRONG_CLASS` 1,00 / 0,83 và ra **Release**, nhưng recall `MISSING` tụt còn 0,75 — hạ recall chỉ để hàng đợi vừa ngân sách là Release giả, nên không chọn. Cái giá: `LOOSE_BOX` recall giảm từ 0,40 xuống 0,20. |

Ảnh hưởng trên raw (hàng đợi / tỉ lệ cảnh báo chưa duyệt): conf 0,25 → 143 ảnh / 28,7%; 0,6 → 125 ảnh / 18,8%; 0,8 → 102 ảnh / 7,9% (Release).

**Chỗ calibration có thể đánh lừa:** trên calib, `phash_dist = 16` cho F1 cao nhất (`DUP_IMAGE_NEAR` 1,00 / 0,83 / 0,91) và precision vẫn 1,00 — nhìn bảng thì tưởng là lựa chọn tốt nhất. Nhưng chạy trên raw, số ảnh bị loại nhảy từ 17 lên 26 (`LEAK_TEST` 5 → 11, `DUP_IMAGE_NEAR` 4 → 7). Mở xem các cặp mới: `img_0011`/`img_0008` (tennis vs trượt tuyết) và `img_0177`/`img_0113` (lướt sóng vs người ngồi bàn) là ảnh **khác hẳn nhau** bị coi là trùng. Ở 20 thậm chí 105 ảnh bị coi là rò rỉ test. Calib chỉ có 6 ảnh gần trùng (mỗi ảnh = 17% recall) và không có ca `LEAK_TEST`, nên không đo được rủi ro gộp nhầm này.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
**(a) Gate chặn đúng.** Chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` (không có người duyệt):
kết luận **REWORK** — "còn 6 lỗi bắt buộc (MISSING_LABEL_FILE): phải sửa trước khi xuất", **mã thoát 20**, log in "KHÔNG xuất: còn lỗi bắt buộc (Rework)", và thư mục `work/exp_nb/clean_v1` **không được tạo**. Gate đếm mọi lỗi `error` kể cả ở ảnh đã duyệt, nên người duyệt không thể "che" lỗi bắt buộc.

**(b) Điểm mù có bằng chứng:**
1. **Ảnh trùng biến đổi mạnh:** `img_0084`/`img_0074` là cùng một cảnh (mở xem bằng mắt) nhưng phash = 16 > 12 nên lọt qua — cả hai vào dataset. Tăng ngưỡng để bắt thì lại gộp nhầm ảnh khác (mục 3).
2. **Box lỏng nhẹ:** recall `POSSIBLE_LOOSE_BOX` trên calib chỉ 0,20 ở `conf = 0,6` (1/5 lỗi chèn); sau người duyệt vẫn còn 20 cờ `LOOSE_BOX` chưa ai xem. Box phình ~1,5 lần mà IoU với dự đoán vẫn ≥ 0,5 thì không bị cờ.
3. **Vật rất nhỏ / ảnh đông:** `img_0160` có hàng trăm xe tải nhỏ nhưng chỉ ~5 box; dự đoán < 600 px² bị bỏ qua (`min_area_px`) nên phần lớn vật thiếu nhãn không bị cờ.
4. **Model lớn cũng nhầm:** `second_opinion` chỉ nghi ngờ khi yolo26l khác nhãn; nếu cả nhãn và model cùng gán truck thành car thì không có cờ nào. Trên calib vẫn sót 1/6 `WRONG_CLASS`.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
**Chưa chạy.** Chưa train trên Colab ở thời điểm nộp phần lõi (`make check-core`). Khi train, cần so chênh mAP50 (clean − raw) với độ lệch chuẩn giữa 2 seed, và lưu ý `raw` chứa 5 ảnh rò rỉ test (`LEAK_TEST`) có thể thổi phồng điểm của `raw`.

## 6. Quyết định và ghi chú AI
**(a) Quyết định cuối: HOLD.** Dataset `clean_v1` chỉ là bản nháp để thử nghiệm, chưa phát hành. Điều kiện để thành **Release**: người duyệt thêm tối thiểu ~18 ảnh trong số 38 ảnh còn cảnh báo (để tỉ lệ < 10%, tức ≤ 20/202 ảnh), ưu tiên theo thứ tự `05_review_queue.csv` — tức tăng ngân sách duyệt lên khoảng 100 ảnh, hoặc chạy vòng duyệt thứ hai. Không hạ `second_opinion.conf` chỉ để đạt Release. Nếu một vòng duyệt sau để lại lỗi bắt buộc thì quyết định chuyển thành Rework.

**(b) Dùng AI viết code:** tham khảo, cho 4 module (`label_format`, `gate`, `autofix`, `second_opinion`).
- *Prompt chính:* dán docstring của từng stub (đã mô tả vào/ra, mã lỗi, ca biên) + yêu cầu ngưỡng đọc từ `cfg`, không sửa `data/raw`, dòng hỏng không được làm script chết.
- *Cách kiểm:* chạy unit test từng module (79/79 xanh), chạy `label_format` + `autofix` trên **bản sao** của `data/raw` (sau sửa không còn lỗi bắt buộc, chạy lần 2 không đổi gì), chạy calib để xem precision/recall, và mở ảnh bằng `viz_labels.py` để đối chiếu cờ với mắt.
- *Một lần AI sai/thiếu:* khi chứng minh gate ở mục 4a, AI chạy `python -m pipeline.run ... | grep ...; echo $?` và báo "mã thoát 0" — thực ra đó là mã thoát của `grep`, không phải pipeline. Chạy lại không qua pipe thì đúng là **20** (Rework). Bài học: kiểm mã thoát trực tiếp, không tin số in ra qua pipe.
