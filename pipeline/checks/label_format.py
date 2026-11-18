"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401


def run(ds, cfg):
    c = cfg["label_format"]
    tol, min_px, max_frac, dup_iou = c["overflow_tol"], c["min_box_px"], c["max_box_frac"], c["dup_box_iou"]
    n_cls = len(ds.classes)
    issues = []

    def add(sev, code, stem, msg, line, **extra):
        issues.append(Issue("label_format", sev, code, stem, msg, line, extra))

    for img in ds.image_files():
        stem = img.stem
        lines = ds.read_label_lines(stem)          # giữ số dòng gốc, đã bỏ dòng trống
        if lines is None:                          # không có file nhãn: missing_labels báo
            continue
        size = ds.image_size(stem)                 # (W, H) hoặc None nếu ảnh hỏng
        kept = []                                  # box hợp lệ ở các dòng TRƯỚC (để kiểm DUP_BOX)
        for no, text in lines:
            # 1) định dạng: dòng hỏng thì báo rồi bỏ qua, không làm chết script
            try:
                cls, x, y, w, h = parse_yolo_line(text)
            except ValueError as e:
                add(ERROR, "BAD_FORMAT", stem, f"dòng không đúng 5 cột số: {e}", no)
                continue
            # 2) id lớp
            if not 0 <= cls < n_cls:
                add(ERROR, "BAD_CLASS_ID", stem, f"id lớp {cls} ngoài 0..{n_cls - 1}", no, id=cls)
                continue
            # 3) box rỗng
            if w <= 0 or h <= 0:
                add(ERROR, "BOX_ZERO", stem, f"box rộng/cao bằng 0 (w={w}, h={h})", no)
                continue
            # 4) tràn ảnh: mức tràn lớn nhất trong 4 mép
            overflow = max(0.0, -(x - w / 2), (x + w / 2) - 1, -(y - h / 2), (y + h / 2) - 1)
            if overflow > tol:
                add(ERROR, "BOX_OUT_OF_IMAGE", stem, f"box tràn ra ngoài ảnh {overflow:.4f}", no, overflow=round(overflow, 6))
            # 5) box lặp: cùng lớp với một dòng trước đó và IoU cao -> báo ở dòng SAU
            box = (x, y, w, h)
            prev = next(((pno, iou_xywh(box, pb)) for pno, pc, pb in kept if pc == cls and iou_xywh(box, pb) > dup_iou), None)
            if prev:
                add(ERROR, "DUP_BOX", stem, f"lặp box ở dòng {prev[0]} (IoU {prev[1]:.2f})", no, of_line=prev[0], iou=round(prev[1], 4))
            kept.append((no, cls, box))
            # 6) cảnh báo kích thước
            if size:
                W, H = size
                short = min(w * W, h * H)
                if short < min_px:
                    add(WARN, "BOX_TINY", stem, f"cạnh ngắn {short:.1f}px < {min_px}px", no, short_px=round(short, 2))
            if w * h > max_frac:
                add(WARN, "BOX_HUGE", stem, f"box phủ {w * h:.0%} ảnh", no, frac=round(w * h, 4))
    return issues
