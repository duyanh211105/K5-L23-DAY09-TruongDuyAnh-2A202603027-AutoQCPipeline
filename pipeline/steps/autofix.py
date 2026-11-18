"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
CLIP_CODES = {"BOX_OUT_OF_IMAGE"}
REMOVE_CODES = {"DUP_IMAGE_EXACT", "DUP_IMAGE_NEAR", "LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}


def _fmt(v):
    """Số gọn: 6 chữ số thập phân, bỏ số 0 thừa."""
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def _clip(text):
    """Cắt box về trong [0,1]. Trả về dòng mới, hoặc None nếu box không còn diện tích."""
    c, x, y, w, h = parse_yolo_line(text)
    x1, x2 = max(0.0, x - w / 2), min(1.0, x + w / 2)
    y1, y2 = max(0.0, y - h / 2), min(1.0, y + h / 2)
    if x2 <= x1 or y2 <= y1:
        return None
    return " ".join([str(c)] + [_fmt(v) for v in ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)])


def fix(ds, issues, cfg):
    errors = [i for i in issues if i.severity == ERROR]          # cảnh báo là việc của người: không động tới
    fixes = []

    # 1) xoá ảnh (trùng / rò rỉ test / hỏng / nhỏ) — làm TRƯỚC để bỏ qua issue dòng của ảnh đã xoá
    removed = set()
    for stem in sorted({i.image for i in errors if i.code in REMOVE_CODES}):
        img = ds.image_path(stem)
        lab = ds.label_path(stem)
        if img is None and not lab.exists():
            continue                                             # đã xoá từ lần trước
        codes = ",".join(sorted({i.code for i in errors if i.image == stem and i.code in REMOVE_CODES}))
        if img is not None:
            img.unlink()
        if lab.exists():
            lab.unlink()
        removed.add(stem)
        fixes.append(dict(image=stem, line=None, action="remove_image",
                          before=img.name if img is not None else lab.name, after=f"(đã xoá: {codes})"))

    # 2) ảnh chưa có file nhãn: KHÔNG bịa nhãn (kể cả file rỗng), chuyển người gán
    for stem in sorted({i.image for i in errors if i.code == "MISSING_LABEL_FILE"} - removed):
        if ds.image_path(stem) is not None and not ds.label_path(stem).exists():
            fixes.append(dict(image=stem, line=None, action="needs_label", before="(không có file)", after="chuyển người gán"))

    # 3) tên lớp trong data.yaml
    if any(i.code == "BAD_CLASS_NAME" for i in errors):
        d = {}
        if ds.yaml_path.exists():
            d = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
        old = d.get("names")
        new = dict(enumerate(cfg["classes"]))
        if old != new:
            d["names"] = new
            ds.yaml_path.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
            fixes.append(dict(image="data.yaml", line=None, action="rename_classes", before=str(old), after=str(new)))

    # 4) lỗi theo dòng: gom theo file, quyết định cho từng dòng GỐC, rồi ghi lại file MỘT lần
    drop, clip = defaultdict(set), defaultdict(set)
    for i in errors:
        if i.line is None or i.image in removed:
            continue
        if i.code in DROP_CODES:
            drop[i.image].add(i.line)
        elif i.code in CLIP_CODES:
            clip[i.image].add(i.line)

    for stem in sorted(set(drop) | set(clip)):
        p = ds.label_path(stem)
        if not p.exists():
            continue                                             # ảnh/nhãn đã biến mất: bỏ qua êm
        raw_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()   # số dòng gốc tính cả dòng trống
        out, changed = [], False
        for no, text in enumerate(raw_lines, 1):
            if no in drop[stem]:                                 # xoá dòng thắng: không ghi thêm clip_box
                fixes.append(dict(image=stem, line=no, action="drop_line", before=text, after=""))
                changed = True
                continue
            if no in clip[stem]:
                try:
                    new = _clip(text)
                except ValueError:                               # dòng hỏng mà chỉ bị báo tràn: an toàn nhất là xoá
                    new = None
                if new is None:                                  # nằm hẳn ngoài ảnh: xoá thay vì ghi box rỗng
                    fixes.append(dict(image=stem, line=no, action="drop_line", before=text, after=""))
                    changed = True
                    continue
                if new != text:
                    fixes.append(dict(image=stem, line=no, action="clip_box", before=text, after=new))
                    changed = True
                out.append(new)
                continue
            out.append(text)
        if changed:
            p.write_text("".join(l + "\n" for l in out), encoding="utf-8")

    return fixes
