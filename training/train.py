"""
Retrain the glyph templates embedded in ../lock_solver.py.

1. Put screenshots in training/samples/ (cropped tables, cropped code lines).
2. Add their labels to TABLES / CODES below (key = file name without .png).
   Tables: entries in reading order (left, right, next row...). Codes: the text as
   shown, including any visible "code" prefix letters (colon left out).
3. python train.py            -> leave-one-out accuracy check
   python train.py --write    -> also rewrites TEMPLATES in lock_solver.py
"""
import os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from lock_solver import *  # noqa: F401,F403  (recognition code lives in the solver)

SAMPLES = os.path.join(HERE, "samples")

TABLES = {
    "b93f7c82": "Y9 A1 H1 S0 B7 >5 T4 %5 J8 G3 V9 ?6 K2 $7 @0 <2",
    "e645355e": "V9 S7 ?0 G6 K2 A0 >8 B5 @7 H1 <2 T3 $4 J1 %5 Y9",
    "a74dbfce": ">1 <9 A8 H5 V0 $3 S1 J5 T9 @2 G7 Y6 K2 B4 ?0 %7",
    "c24babf9": "%2 J1 T9 V1 G4 @2 ?0 H7 B9 $0 >3 A5 K7 Y8 S5 <6",
    "a3ce5d80": "V2 K9 @7 H3 B7 G5 $5 T8 >1 S6 Y0 A0 ?4 %1 <9 J2",
    "5864438d": "<5 Y2 ?6 S9 K7 A8 V2 J4 @3 T1 %1 >0 B9 H7 $0 G5",
    "f8dcce32": "%6 S7 H1 <2 K3 ?2 J4 Y0 B9 @8 T7 V1 G9 $5 A5 >0",
    "6f0688b2": "G2 <9 H9 >7 T7 V4 ?5 J2 A8 B5 $3 @0 S1 %1 K6 Y0",
    "30d5310b": "K0 @1 $6 A7 H2 <0 J4 T8 V9 G3 S7 Y1 B9 >2 ?5 %5",
    "56954247": ">8 B1 G4 %3 $6 T7 <0 ?2 H7 V9 S9 A5 J2 Y5 @0 K1",
    "00dfc01c": "K8 H5 Y3 ?4 S0 @7 $0 >1 B1 J6 G9 V9 %7 <2 T5 A2",
    "ef1f3d99": "$4 K0 G6 >2 B2 V7 %3 T0 <9 Y7 H5 ?1 A1 J5 @9 S8",
    "86adcef4": "K3 J8 G7 %2 H7 Y1 >4 ?6 B0 <2 A9 T0 @9 S1 $5 V5",
    "81bd8b94": "$9 J7 A2 Y0 T5 ?7 <2 H1 %1 K5 @3 G6 B4 S8 V0 >9",
    "189d9f05": "%2 >1 G5 T1 <6 Y2 $0 ?8 J9 H3 A4 @5 S7 V7 K0 B9",
}
CODES = {
    "4da933ef": "code:GYHYTG$YBT",
    "de115e4a": "code:%STH@?$J@A",
    "eb965b39": "code:YSST>G<J?>",
    "39439fec": "code:TT?$B$$%SV",
    "5f3f7568": "code:V?GGAVBS>B",
    "80bce34c": "VA%@@?>AHK",
    "b144362d": "$Y?$KGVH$V",
    "d4c77680": "HBBGGBTK@G",
    "28bc721b": "TG@H>AHB?>",
    "f36c7ab4": ":V@?%T>$VAT",
    "bcfc5d44": "e:%?<AH?$@T%",
    "817e89ab": "HVGTYHYS@S",
    "8253669a": ":AVYGY>H?JS",
    "fa768f3f": "YGYG@%<?%K",
}


def path(key):
    return os.path.join(SAMPLES, key + ".png")


def table_samples(key):
    ents = find_entries(text_mask(to_gray(cv2.imread(path(key))), True))
    labels = TABLES[key].split()
    if len(ents) != 16:
        print("WARN table", key, "found", len(ents), "entries")
        return []
    out = []
    for e, lab in zip(ents, labels):
        out.append((norm(e["s"], e["h"]), lab[0]))
        out.append((norm(e["d"], e["h"]), lab[1]))
    return out


def code_samples(key):
    gs, rh = code_glyphs(cv2.imread(path(key)))
    lab = CODES[key].replace(":", "")
    if len(gs) > len(lab):
        gs = gs[-len(lab):]  # partial glyphs cut at the left edge
    elif len(gs) < len(lab):
        print("WARN code", key, len(gs), "glyphs for", len(lab), "labels")
        return []
    return [(norm(g, rh), c) for g, c in zip(gs, lab)]


def prototypes(exclude=None):
    """Average all samples of each character into one binarized template."""
    feats = {}
    for k in TABLES:
        if k != exclude and os.path.exists(path(k)):
            for f, l in table_samples(k):
                feats.setdefault(l, []).append(f)
    for k in CODES:
        if k != exclude and os.path.exists(path(k)):
            for f, l in code_samples(k):
                feats.setdefault(l, []).append(f)
    X, y = [], []
    for l in sorted(feats):
        m = np.mean(feats[l], axis=0)
        m[: N * N] = m[: N * N] > 0.5
        m[N * N:] = np.round(m[N * N:], 2)
        X.append(m); y.append(l)
    return Model(np.array(X, np.float32), np.array(y))


def evaluate():
    errs = 0
    for k, v in TABLES.items():
        if not os.path.exists(path(k)):
            continue
        t, _ = parse_table(cv2.imread(path(k)), prototypes(exclude=k))
        exp = {e[0]: e[1] for e in v.split()}
        if t != exp:
            errs += 1
            print("TABLE FAIL", k, {s: (t.get(s), exp[s]) for s in exp if t.get(s) != exp[s]})
    for k, v in CODES.items():
        if not os.path.exists(path(k)):
            continue
        c = parse_code(cv2.imread(path(k)), prototypes(exclude=k))
        exp = "".join(ch for ch in v if ch in SYMS)
        if c != exp:
            errs += 1
            print("CODE FAIL", k, c, exp)
    print("leave-one-out errors:", errs)


def write_templates():
    m = prototypes()
    rows = []
    for x, lab in zip(m.X, m.y):
        hx = np.packbits(x[: N * N].astype(np.uint8)).tobytes().hex()
        rows.append(f"    ({str(lab)!r}, \"{hx}\", " + ", ".join(f"{v:.2f}" for v in x[N * N:]) + "),")
    block = "TEMPLATES = [\n" + "\n".join(rows) + "\n]\n"
    p = os.path.join(HERE, "..", "lock_solver.py")
    src = open(p, encoding="utf-8").read()
    src = re.sub(r"TEMPLATES = \[\n.*?\n\]\n", lambda _: block, src, flags=re.S)
    open(p, "w", encoding="utf-8").write(src)
    print("wrote", len(rows), "templates to", os.path.normpath(p))


if __name__ == "__main__":
    evaluate()
    if "--write" in sys.argv:
        write_templates()
