"""
Azrael lock code solver
-----------------------
Watches the clipboard. Screenshot the lock (PrtScn for the whole screen, or
Win+Shift+S to snip the decode table and the code line separately, in either
order) and the decoded digits appear in a small always-on-top window and are
read out loud.

Setup (once):   pip install -r requirements.txt
Run:            python lock_solver.py          (English voice)
                python lock_solver.py --ru     (Russian voice, if installed)
Self-test:      python lock_solver.py --test screenshot1.png [screenshot2.png ...]
"""
import cv2, numpy as np

SYMS = list("TAHVGKSYBJ@<>$%?")
DIGS = list("0123456789")
N = 24  # template size


def to_gray(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def text_mask(gray, dark_text):
    """Return binary mask (255 = text) robust to the dimmed band / overlay."""
    g = gray.astype(np.uint8)
    h, w = g.shape
    scale = 900.0 / max(h, w) if max(h, w) < 900 else 1.0
    if h * scale < 60:  # very short code strip
        scale = max(scale, 60.0 / h)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if not dark_text:
        g = 255 - g
    # remove things brighter than local background (overlay hint text)
    k = max(31, (min(g.shape) // 8) | 1)
    med = cv2.medianBlur(g, min(k, 255) if k % 2 else k + 1)
    overlay = (g.astype(np.int16) > med.astype(np.int16) + 25).astype(np.uint8) * 255
    overlay = cv2.dilate(overlay, np.ones((3, 3), np.uint8))
    g = np.minimum(g.astype(np.int16), med.astype(np.int16) + 12).astype(np.uint8)
    # flatten background
    ks = max(15, int(g.shape[0] / 12))
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks)))
    ratio = g.astype(np.float32) / (bg.astype(np.float32) + 1)
    mask = (ratio < 0.62).astype(np.uint8) * 255
    if overlay.any():
        k = max(5, int(g.shape[0] / 90)) | 1
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(closed, overlay))
    if dark_text:  # drop 1-3px hairlines left by the dimmed band edges
        v = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4)))
        hz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1)))
        thin_h = cv2.subtract(mask, v)  # pixels in thin horizontal structures
        mask = cv2.subtract(mask, cv2.subtract(thin_h, cv2.subtract(mask, hz)))
    return mask


def glyphs(mask, drop_border=True):
    L = max(25, mask.shape[1] // 6)
    lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    lines = cv2.dilate(lines, np.ones((3, 1), np.uint8))
    mask = cv2.subtract(mask, lines)
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    H, W = mask.shape
    comps = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 6:
            continue
        if h > H * 0.5 or w > W * 0.3:
            continue
        if drop_border and (x == 0 or y == 0 or x + w >= W or y + h >= H):
            continue
        comps.append([x, y, x + w, y + h, a, [i]])
    mh = 1e9
    if comps:
        big = sorted(comps, key=lambda c: -c[4])[: max(3, len(comps) // 3)]
        mh = float(np.median([c[3] - c[1] for c in big]))
        comps = [c for c in comps if (c[3] - c[1]) < 2.2 * mh and (c[2] - c[0]) < 3 * mh]
    # merge components overlapping in x and close in y (%, ?, :, $ pieces)
    comps.sort(key=lambda c: c[0])
    merged = True
    while merged:
        merged = False
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                a, b = comps[i], comps[j]
                ox = min(a[2], b[2]) - max(a[0], b[0])
                minw = min(a[2] - a[0], b[2] - b[0])
                hh = max(a[3] - a[1], b[3] - b[1])
                gapy = max(a[1], b[1]) - min(a[3], b[3])
                comb = max(a[3], b[3]) - min(a[1], b[1])
                if ox > 0.5 * minw and gapy < 0.6 * hh and comb < 1.35 * mh:
                    comps[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), a[4] + b[4], a[5] + b[5]]
                    del comps[j]
                    merged = True
                    break
            if merged:
                break
    out = []
    for x0, y0, x1, y1, a, ids in comps:
        m = np.isin(lab[y0:y1, x0:x1], ids).astype(np.uint8) * 255
        out.append(dict(x0=x0, y0=y0, x1=x1, y1=y1, area=a, img=m))
    return out


def rows_of(gl):
    """Cluster glyphs into text rows."""
    gl = [g for g in gl if g["y1"] - g["y0"] >= 3]
    gl.sort(key=lambda g: (g["y0"] + g["y1"]) / 2)
    rows = []
    for g in gl:
        cy = (g["y0"] + g["y1"]) / 2
        hgt = g["y1"] - g["y0"]
        if rows and abs(cy - rows[-1]["cy"]) < max(rows[-1]["h"], hgt) * 0.55:
            r = rows[-1]
            r["g"].append(g)
            r["cy"] = np.mean([(q["y0"] + q["y1"]) / 2 for q in r["g"]])
            r["h"] = max(r["h"], hgt)
        else:
            rows.append(dict(cy=cy, h=hgt, g=[g]))
    for r in rows:
        r["g"].sort(key=lambda g: g["x0"])
    return rows


def norm(g, rowh):
    m = g["img"]
    h, w = m.shape
    s = max(h, w)
    pad = np.zeros((s, s), np.uint8)
    pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = m
    v = (cv2.resize(pad, (N, N), interpolation=cv2.INTER_AREA) > 127).astype(np.float32)
    feat = np.concatenate([v.ravel(), [h / rowh * 3, w / rowh * 3, (w / max(h, 1)) * 1.0]])
    return feat


def is_dash(g, rowh):
    h = g["y1"] - g["y0"]
    w = g["x1"] - g["x0"]
    return w > 1.6 * h and h < 0.35 * rowh


class Model:
    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y

    def predict(self, f, allowed):
        idx = np.array([c in allowed for c in self.y])
        X = self.X[idx]
        y = self.y[idx]
        d = ((X - f) ** 2).sum(1)
        o = np.argsort(d)
        return y[o[0]], float(d[o[0]])


def find_entries(mask):
    gl = glyphs(mask)
    if not gl:
        return []
    areas = np.array([g["area"] for g in gl])
    big = [g for g in gl if g["area"] >= np.percentile(areas, 70)]
    mh = float(np.median([g["y1"] - g["y0"] for g in big]))
    def cy(g):
        return (g["y0"] + g["y1"]) / 2
    cands = []
    for g in gl:
        h = g["y1"] - g["y0"]; w = g["x1"] - g["x0"]
        if w > 1.6 * h and 0.2 * mh < w < 0.8 * mh and h < 0.3 * mh:
            cands.append(g)
    ents = []
    def union(parts):
        x0 = min(p["x0"] for p in parts); x1 = max(p["x1"] for p in parts)
        y0 = min(p["y0"] for p in parts); y1 = max(p["y1"] for p in parts)
        canvas = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for p in parts:
            sub = canvas[p["y0"] - y0:p["y1"] - y0, p["x0"] - x0:p["x1"] - x0]
            np.maximum(sub, p["img"], out=sub)
        return dict(x0=x0, x1=x1, y0=y0, y1=y1, area=sum(p["area"] for p in parts), img=canvas)
    for dsh in cands:
        c = cy(dsh)
        near = [g for g in gl if g is not dsh and abs(cy(g) - c) < 0.45 * mh]
        left = [g for g in near if g["x1"] <= dsh["x0"] + 2 and dsh["x0"] - g["x1"] < 1.0 * mh]
        right = [g for g in near if g["x0"] >= dsh["x1"] - 2 and g["x0"] - dsh["x1"] < 1.0 * mh]
        if not left or not right:
            continue
        s0 = max(left, key=lambda g: g["x1"])
        d0 = min(right, key=lambda g: g["x0"])
        # gather every piece belonging to the symbol / digit (%, broken glyphs)
        sp = [g for g in near if g["x1"] <= dsh["x0"] + 2 and g["x0"] >= s0["x1"] - 1.3 * mh
              and dsh["x0"] - g["x1"] < 1.4 * mh]
        dp = [g for g in near if g["x0"] >= dsh["x1"] - 2 and g["x1"] <= d0["x0"] + 0.95 * mh]
        if s0 not in sp: sp.append(s0)
        if d0 not in dp: dp.append(d0)
        s = union(sp); d = union(dp)
        hs, hd = s["y1"] - s["y0"], d["y1"] - d["y0"]
        if hd < 0.7 * mh or hs < 0.45 * mh:
            continue
        rh = max(hs, hd)
        ents.append(dict(cy=c, x=dsh["x0"], h=rh, s=s, d=d))
    if not ents:
        return []
    md = float(np.median([e["h"] for e in ents]))
    ents = [e for e in ents if 0.75 * md < e["h"] < 1.3 * md]
    ents.sort(key=lambda e: e["cy"])
    rows = []
    for e in ents:
        if rows and abs(e["cy"] - rows[-1][0]["cy"]) < 0.5 * md:
            rows[-1].append(e)
        else:
            rows.append([e])
    out = []
    for r in rows:
        out += sorted(r, key=lambda e: e["x"])
    return out


def parse_table(img, model):
    gray = to_gray(img)
    mask = text_mask(gray, dark_text=True)
    preds = []
    for e in find_entries(mask):
        fs, fd = norm(e["s"], e["h"]), norm(e["d"], e["h"])
        sym, sd = model.predict(fs, SYMS)
        dig, dd = model.predict(fd, DIGS)
        preds.append([sym, dig, sd, dd, fs])
    # each symbol appears once per table: fix duplicates using the missing symbols
    for _ in range(3):
        seen = {}
        for p in preds:
            seen.setdefault(p[0], []).append(p)
        missing = [c for c in SYMS if c not in seen]
        dups = [p for v in seen.values() if len(v) > 1 for p in sorted(v, key=lambda p: p[2])[1:]]
        if not dups or not missing:
            break
        for p in dups:
            p[0], p[2] = model.predict(p[4], missing)
    table = {p[0]: p[1] for p in preds}
    return table, [(p[0], p[1], p[2], p[3]) for p in preds]


def split_wide(gs, rh):
    out = []
    for g in gs:
        m = g["img"]
        h, w = m.shape
        if w > 1.2 * rh:
            proj = (m > 0).sum(0)
            a, b = int(w * 0.3), int(w * 0.7)
            cut = a + int(np.argmin(proj[a:b]))
            for x0, x1 in ((0, cut), (cut, w)):
                sub = m[:, x0:x1]
                ys = np.where(sub.any(1))[0]
                if len(ys) == 0:
                    continue
                out += split_wide([dict(x0=g["x0"] + x0, x1=g["x0"] + x1, y0=g["y0"] + ys[0], y1=g["y0"] + ys[-1] + 1,
                                        area=int((sub > 0).sum()), img=sub[ys[0]:ys[-1] + 1])], rh)
        else:
            out.append(g)
    return out


def code_glyphs(img):
    gray = to_gray(img)
    mask = text_mask(gray, dark_text=False)
    rows = rows_of(glyphs(mask, drop_border=False))
    if not rows:
        return [], 1
    r = max(rows, key=lambda r: sum(g["area"] for g in r["g"]))
    rh = float(np.percentile([g["y1"] - g["y0"] for g in r["g"]], 75))
    gs = [g for g in r["g"] if (g["y1"] - g["y0"]) > 0.3 * rh]
    return split_wide(gs, rh), rh


def parse_code(img, model, want_word=False):
    gs, rh = code_glyphs(img)
    rows = [dict(g=gs, h=rh)] if gs else []
    if not rows:
        return ("", False) if want_word else ""
    out = []
    for g in gs:
        lab, dist = model.predict(norm(g, rh), SYMS + list("code"))
        out.append((lab, dist, g))
    labs = [o[0] for o in out]
    had_word = sum(1 for l in labs if l in "code") >= 3
    for k in range(len(labs) - 1, -1, -1):  # drop everything up to the "code" word
        if labs[k] in "code":
            out = out[k + 1:]
            break
    out = [o for o in out if o[0] in SYMS]
    if len(out) > 10 and out[0][2]["x0"] <= 1:
        out = out[1:]
    if len(out) > 10:
        out = out[-10:]
    code = "".join(o[0] for o in out)
    return (code, had_word) if want_word else code


def decode(table, code):
    return "".join(table.get(c, "?") for c in code)


# ---------- full-screenshot support ----------
def find_paper(img):
    """Bounding box of the beige decode-table paper in a big screenshot, or None."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] < 70) & (hsv[..., 2] > 150)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    H, W = m.shape
    best = None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 0.03 * H * W or h < 0.2 * H or w > 0.8 * W:
            continue
        fill = a / float(w * h)
        if fill < 0.5:
            continue
        if best is None or a > best[4]:
            best = (x, y, w, h, a)
    if best is None:
        return None
    x, y, w, h, _ = best
    # the dimmed hint band can split the paper: extend with aligned pieces
    for i in range(1, n):
        x2, y2, w2, h2, a2 = st[i]
        if a2 > 0.01 * H * W and abs(x2 - x) < 0.1 * w and abs(w2 - w) < 0.2 * w:
            nx0, ny0 = min(x, x2), min(y, y2)
            nx1, ny1 = max(x + w, x2 + w2), max(y + h, y2 + h2)
            x, y, w, h = nx0, ny0, nx1 - nx0, ny1 - ny0
    return x, y, w, h


def find_code_strip(img, model):
    """Locate the 'code: XXXXXXXXXX' label in a big screenshot and decode its symbols."""
    gray = to_gray(img)
    H, W = gray.shape
    light = (gray > 185).astype(np.uint8) * 255
    n, lab, st, _ = cv2.connectedComponentsWithStats(light, 8)
    comps = [st[i] for i in range(1, n) if 0.008 * H < st[i][3] < 0.06 * H and st[i][2] < 0.06 * W]
    comps.sort(key=lambda s: s[1] + s[3] / 2)
    best = ""
    tried = set()
    for c in comps:
        cyc = c[1] + c[3] / 2
        row = [s for s in comps if abs((s[1] + s[3] / 2) - cyc) < 0.6 * c[3]]
        if len(row) < 10:
            continue
        x0 = min(s[0] for s in row); x1 = max(s[0] + s[2] for s in row)
        y0 = min(s[1] for s in row); y1 = max(s[1] + s[3] for s in row)
        key = (x0 // 10, y0 // 10)
        if key in tried:
            continue
        tried.add(key)
        pad = int((y1 - y0) * 0.6)
        crop = img[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)]
        code, had_word = parse_code(crop, model, want_word=True)
        if had_word and len(code) == 10:
            return code
    return best


# one averaged 24x24 template per character: (char, bitmap hex, rel. height, rel. width, aspect)
TEMPLATES = [
    ('$', "001800001800001800003c0001ff8003ffc003988003180003180003980003f80001fe00003f80001fc00018c00018c00018c00319c007ffc001ff00003c00001800001800001800", 3.25, 1.75, 0.54),
    ('%', "0000001f00207f80707bc0f0e1e0e0e0e1e0e0e3c0e0e380e1e70071cf003f9e001f1c000038f8007bfc00f38e00e78701e70703c7070787070f07070f038e0e01fe0c00f8000000", 3.00, 3.12, 1.04),
    ('0', "00400001ff0003ff8007ffc00f83e00f01f01f00f01e00f81e00781e00781e00781e00781e00781e00781e00781e00781e00f81e00f00f01f00f81f007c7e007ffc001ff80007c00", 2.93, 2.30, 0.79),
    ('1', "000600000f80001fc0003fc0007fc000fbc003fbc003e3c003c3c00183c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c0", 2.98, 1.59, 0.53),
    ('2', "00400003ff0007ffc00fffe00783e00201f00001f00001f00001f00001f00001e00003e00007c0000f80001f00007e0000fc0001f80003f00007e0000fe0000ffff00ffff00ffff0", 2.97, 2.08, 0.70),
    ('3', "00400003ff0007ff8007ffc00703e00001f00001f00001f00001e00003e00007c0007f80007f800007e00001e00000f00000f00000f00401f00e01f00f87e00fffc003ff8000fe00", 2.92, 2.03, 0.69),
    ('4', "000800001fc0001fe0003fe0007fe0007be000fbe001f3e001e3e003e3e007c3e00783e00f03e01f03e01f03e01ffff81ffffc1ffff80003e00003e00003e00003e00003e00003c0", 2.85, 2.28, 0.80),
    ('5', "0800000fffe00fffe00fffc00f00000f00000f00000f00000f10000f7f800fffc00fefe00701f00000f00000f00000f00000f00000f00401f00e01f01fc7e00fffc003ff8000fc00", 2.92, 2.10, 0.72),
    ('6', "001800007f8001ffe003ffe007c0600780000f80000f00000f00000f1e001f7fc01fffe01fc1e01f80f01f00f00f00f80f00f80f00f80f00f00781f007c3e003ffe001ff80007e00", 2.93, 2.11, 0.72),
    ('7', "0800001fffe01ffff00ffff00000f00001f00003e00003e00007c00007c0000f80000f80001f00001e00003e00003c00007c0000f80000f80001f00001f00003e00003c00000c000", 2.93, 2.10, 0.72),
    ('8', "00400001ff0007ffc007ffe00f81e00f00f00f00f00f00f00f00e00781e003c3c001ff8001ff800fc3e00f00f01f00f01e00f81e00f81f00f80f01f00fc3f007ffe003ffc000fe00", 3.00, 2.21, 0.74),
    ('9', "00000001ff0007ff8007ffc00f81e01f00f01f00f01e00f01e00f01e00f80f01f80f01f807c7f803fef001fcf00000f00000f00001f00001e00003e0070fc00fff8007ff0001fc00", 2.98, 2.18, 0.73),
    ('<', "00000c00003c0000fc0001fc0007f8003ff0007fc001fe0007fc001fe0003fc0007f00007e00003f00001fc0000ff00003fc0000ff00003fc0000ff00003fc0000fc00007c00001c", 2.26, 1.97, 0.87),
    ('>', "3000003800007e00003f80001fe00007f80001fe00007f80001fe00007f80001fc0000fe0000fe0001fc0003f8000fe0007fc001ff0003fc001ff0003fc0007f00003e0000300000", 2.28, 2.04, 0.90),
    ('?', "007c0003ff8007ffc007ffe00783e00201e00001e00001e00001e00003e00003c0000f80000f00001e00003c00003c00003c00003800000000000000000000003c00003c00003c00", 2.94, 1.80, 0.61),
    ('@', "00000000fc0007fe000fcf801e03803c03807803c0703fc660ffc6e3e3c7e383c7e383c3e383c7e387c7e3ffc7e1fcce70f00e70001c38003c1e00f80fc7e007ffc000fe00000000", 3.02, 3.36, 1.11),
    ('A', "005800007e0000fe0000ff0000ff0001e70001e78001e78003c3c003c3c00781c00781e00781e00f00f00f00f00ffff01ffff81ffff83e00783c003c3c003c78003c78001e180018", 2.95, 2.65, 0.90),
    ('B', "1ffe003fffc03fffe03ffff03e00f03e00f03c00703c00f03c00f03c01f03e3fe03fff803fffc03e03e03c00f03c00783c00783c007c3c007c3c00f83e00f83ffff03fffe03fff80", 2.90, 2.32, 0.80),
    ('G', "003f0001fff003e1f80780780f80381f00001e00003c00003c00003c00003c00003c00003c00083c003c3c003c3c003c1c003c1e003c1e003c0f803c07e0fc03fff801ffe0003f80", 2.99, 2.55, 0.85),
    ('H', "3000183c003c3c003c3c003c3c003c3c003c3c003c3c003c3c003c3c003c3ffffc3ffffc3ffffc3e007c3c003c3c003c3c003c3c003c3c003c3c003c3c003c3c003c3c003c3c003c", 2.89, 2.42, 0.84),
    ('J', "0002000003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00007c00107c0018f8003ff8003ff0000fc00", 2.93, 1.44, 0.49),
    ('K', "0000601e00f01e01f01e01f01e07e01e0fc01e0f801e1f001e7e001e7c001efc001ffc001ffc001ffe001f9f001f0f801e07801e07c01e03e01e01f01e00f81e00f81e0078000030", 2.93, 2.19, 0.75),
    ('S', "003c0001ff8007ffe00ffff00f80e01f00201e00001e00001f00000f80000ff00007ff8001ffe0001ff00001f00000f80000780000780800f81c00f01f83f01fffe007ffc000fe00", 3.00, 2.26, 0.75),
    ('T', "0ffff01ffff81ffff81ffff0003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00003c00", 2.91, 2.14, 0.74),
    ('V', "10000878001e78001e3c003e3c003c1c003c1e00781e00780f00780f00f00700f00781e00781e00381e003c3c003c3c001c3c001e78000e78000ef8000ff00007f00007e00003e00", 2.88, 2.62, 0.91),
    ('Y', "1800103c00381c00381e007c1e00f80f00f80f81f00781e003c3e001c3c001e78000f78000ff00007e00003e00003c00003c00003c00003c00003c00003c00003c00003c00003c00", 2.90, 2.39, 0.82),
    ('c', "001f0000ffe001fff003fff80780380f00000f00001e00001e00001e00001e00001e00001e00001e00001e00001e00001e00000f00000f000007803807f1f803fff800fff0001f80", 2.29, 1.77, 0.77),
    ('d', "000020000070000070000070000070000070007c7001fff003fff00700f00700f00e00700e00700e00700e00700e00700e00700e00700e00700700f00781f003e7f001fff0007c00", 3.01, 2.02, 0.67),
    ('e', "003c0001ff8003ffc007c1f00f00781e00781c00383c001c38001c3e007c3ffffc3ffffc3c00f03800003800003800003c00001c00001e00000f00700fe3f007fff001ffc0003e00", 2.09, 1.72, 0.80),
    ('o', "003c0001ff8003ffc007fff00f00f01e00783c003c3c003c3c001c78001c78001c78001c78001c78001c78001c7c001c3c001c3c003c1e00781f00f80fe7f007ffe001ffc0003e00", 2.30, 2.05, 0.89),
]


# ======================================================================
#  App: watch the clipboard, decode, show + speak the answer
# ======================================================================
import hashlib, queue, subprocess, sys, threading, time


def load_model():
    X, y = [], []
    for ch, hx, *ex in TEMPLATES:
        bits = np.unpackbits(np.frombuffer(bytes.fromhex(hx), np.uint8))[: N * N]
        X.append(np.concatenate([bits.astype(np.float32), np.array(ex, np.float32)]))
        y.append(ch)
    return Model(np.array(X, np.float32), np.array(y))


# ---------------- speech ----------------
class Speaker:
    def __init__(self, russian=False):
        self.russian = russian
        self.voice = None
        try:
            import win32com.client  # pywin32: instant, async SAPI
            self.voice = win32com.client.Dispatch("SAPI.SpVoice")
            if russian:
                for v in self.voice.GetVoices():
                    if "ru" in v.GetAttribute("Language").lower() or "419" in v.GetAttribute("Language"):
                        self.voice.Voice = v
                        break
            self.voice.Rate = 1
        except Exception:
            self.voice = None

    def say(self, digits):
        # "5 9 2 5 0 ... 7 5 7 2 5" with a short pause in the middle
        half = len(digits) // 2
        text = " ".join(digits[:half]) + ", ,  " + " ".join(digits[half:])
        if self.voice is not None:
            try:
                self.voice.Speak(text, 1 | 2)  # async + purge previous
                return
            except Exception:
                pass
        if sys.platform == "win32":  # fallback: PowerShell (slower to start)
            ps = ("Add-Type -AssemblyName System.Speech;"
                  "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                  + ("try{$s.SelectVoiceByHints('NotSet','NotSet',0,[Globalization.CultureInfo]'ru-RU')}catch{};" if self.russian else "")
                  + f"$s.Speak('{text}')")
            subprocess.Popen(["powershell", "-NoProfile", "-Command", ps],
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


# ---------------- clipboard ----------------
def clipboard_image():
    from PIL import ImageGrab, Image
    try:
        im = ImageGrab.grabclipboard()
    except Exception:
        return None
    if isinstance(im, list):  # copied file(s)
        for p in im:
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                pass
        return None
    if im is None or not hasattr(im, "convert"):
        return None
    return im.convert("RGB")


def to_bgr(pil):
    a = np.array(pil)
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


# ---------------- solver state ----------------
class Solver:
    def __init__(self, model):
        self.m = model
        self.table = None
        self.code = None

    def feed(self, img):
        """Returns (status_text, answer_or_None)."""
        H, W = img.shape[:2]
        # 1) whole screenshot with both table and code?
        if H > 350 and W > 700:
            bb = find_paper(img)
            code = find_code_strip(img, self.m)
            if bb is not None:
                x, y, w, h = bb
                t, _ = parse_table(img[y:y + h, x:x + w], self.m)
                if len(t) >= 12:
                    self.table = t
            if len(code) == 10:
                self.code = code
        else:
            # 2) a cropped table or a cropped code strip
            gray = to_gray(img)
            bright_paper = gray.mean() > 120
            t, _ = parse_table(img, self.m) if bright_paper else ({}, [])
            if len(t) >= 12:
                self.table = t
            else:
                c = parse_code(img, self.m)
                if len(c) >= 8:
                    self.code = c
                elif not t:
                    return "Couldn't read that screenshot - try again", None
        if self.table and self.code:
            ans = decode(self.table, self.code)
            note = ""
            if len(self.table) < 16:
                note = f"  (table: only {len(self.table)}/16 read)"
            if len(self.code) != 10:
                note += f"  (code: {len(self.code)} symbols)"
            status = f"{self.code}{note}"
            self.table, self.code = None, None
            return status, ans
        if self.table and not self.code:
            return f"Table read ({len(self.table)}/16). Now screenshot the code.", None
        if self.code and not self.table:
            return f"Code read: {self.code}. Now screenshot the table.", None
        return "Couldn't read that screenshot - try again", None


# ---------------- UI ----------------
def run_app(russian=False):
    import tkinter as tk
    model = load_model()
    solver = Solver(model)
    speaker = Speaker(russian)
    q = queue.Queue()

    def watcher():
        last = None
        first = True
        while True:
            im = clipboard_image()
            if im is not None:
                h = hashlib.md5(im.tobytes()).hexdigest()
                if h != last:
                    last = h
                    if first:  # ignore whatever was already on the clipboard at start
                        first = False
                    else:
                        try:
                            q.put(("busy", None))
                            q.put(solver.feed(to_bgr(im)))
                        except Exception as e:
                            q.put((f"Error: {e}", None))
                first = False
            else:
                first = False
            time.sleep(0.25)

    root = tk.Tk()
    root.title("Code solver")
    root.attributes("-topmost", True)
    root.configure(bg="#111")
    root.geometry("+30+30")
    big = tk.Label(root, text="----------", font=("Consolas", 40, "bold"), fg="#e8e83a", bg="#111", padx=16)
    big.pack()
    small = tk.Label(root, text="Screenshot the lock (Win+Shift+S or PrtScn)", font=("Segoe UI", 11), fg="#ccc", bg="#111", padx=10, pady=6)
    small.pack()

    def poll():
        try:
            while True:
                status, ans = q.get_nowait()
                if status == "busy":
                    small.config(text="Reading...")
                    continue
                small.config(text=status)
                if ans:
                    big.config(text=ans)
                    speaker.say(ans)
                    print(time.strftime("%H:%M:%S"), ans, "<-", status, flush=True)
        except queue.Empty:
            pass
        root.after(100, poll)

    threading.Thread(target=watcher, daemon=True).start()
    poll()
    root.mainloop()


def cli_test(paths):
    model = load_model()
    s = Solver(model)
    for p in paths:
        img = cv2.imread(p)
        print(p, *s.feed(img))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--test":
        cli_test(args[1:])
    else:
        run_app(russian="--ru" in args)
