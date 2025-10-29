# app.py — version safe-boot : /docs démarre même sans dépendances d'imagerie
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="DM Backend", version="0.3.0")

@app.get("/")
def root():
    return {"ok": True, "msg": "Service up. See /docs"}

@app.get("/health")
def health():
    return {"ok": True}

def _lazy_imports():
    """
    Importe à la demande les dépendances lourdes.
    Si une lib manque, on lève une HTTPException 501 (Not Implemented)
    au lieu de faire crasher l'app au démarrage.
    """
    try:
        from io import BytesIO
        import os, time
        import numpy as np  # type: ignore
        from PIL import Image, ImageOps, ImageFilter  # type: ignore
        from pylibdmtx.pylibdmtx import decode as dm_decode  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=501, detail=f"Scan not available: missing libs: {e}")

    # OpenCV facultatif
    try:
        import cv2  # type: ignore
        OPENCV_AVAILABLE = True
    except Exception:
        cv2 = None  # type: ignore
        OPENCV_AVAILABLE = False

    return {
        "BytesIO": BytesIO,
        "os": os,
        "time": time,
        "np": np,
        "Image": Image,
        "ImageOps": ImageOps,
        "ImageFilter": ImageFilter,
        "dm_decode": dm_decode,
        "cv2": cv2,
        "OPENCV_AVAILABLE": OPENCV_AVAILABLE,
    }

# ---- Logic de décodage, identique à ta version mais encapsulée et lazy ----

def _get_settings(env):
    MAX_UPLOAD_MB = float(env["os"].getenv("MAX_UPLOAD_MB", "8"))
    return {
        "MAX_UPLOAD_MB": MAX_UPLOAD_MB,
        "MAX_UPLOAD_BYTES": int(MAX_UPLOAD_MB * 1024 * 1024),
        "MAX_SIDE_PX": int(env["os"].getenv("MAX_SIDE_PX", "1600")),
        "TIME_BUDGET_MS": int(env["os"].getenv("TIME_BUDGET_MS", "20000")),
        "ATTEMPT_TIMEOUT_MS": int(env["os"].getenv("ATTEMPT_TIMEOUT_MS", "800")),
        "SHRINKS": tuple(int(x) for x in env["os"].getenv("SHRINKS", "3,2,1").split(",")),
        "TRY_THRESHOLDS": (None, 20),
    }

def _resize_cap(Image, im, max_side=1600):
    w, h = im.size
    m = max(w, h)
    if m <= max_side:
        return im
    r = max_side / m
    return im.resize((int(w*r), int(h*r)), Image.LANCZOS)

def _try_variants(env, im, attempts, label_prefix, t0, settings):
    ImageOps, ImageFilter, np, dm_decode = env["ImageOps"], env["ImageFilter"], env["np"], env["dm_decode"]
    TIME_BUDGET_MS, ATTEMPT_TIMEOUT_MS, SHRINKS, TRY_THRESHOLDS = (
        settings["TIME_BUDGET_MS"], settings["ATTEMPT_TIMEOUT_MS"], settings["SHRINKS"], settings["TRY_THRESHOLDS"]
    )

    gray = ImageOps.grayscale(im)
    gray_ac = ImageOps.autocontrast(gray)
    inv = ImageOps.invert(gray_ac)
    bw = gray_ac.point(lambda p: 0 if p < 128 else 255, mode='1').convert("L")
    sharp = ImageOps.autocontrast(
        env["Image"].fromarray(np.array(gray_ac))
    ).filter(ImageFilter.UnsharpMask(radius=1.5, percent=100, threshold=2))

    variants = [("rgb", im), ("gray", gray_ac), ("inv", inv), ("bw", bw), ("sharp", sharp)]

    for vname, vimg in variants:
        for angle in (0, 90, 180, 270):
            if (env["time"].perf_counter() - t0) * 1000 > TIME_BUDGET_MS:
                break
            rot = vimg if angle == 0 else vimg.rotate(angle, expand=True)
            for sh in SHRINKS:
                thresholds = TRY_THRESHOLDS if vname in ("gray", "inv", "bw") else (None,)
                for thr in thresholds:
                    if (env["time"].perf_counter() - t0) * 1000 > TIME_BUDGET_MS:
                        break
                    if thr is None:
                        res = dm_decode(rot, timeout=ATTEMPT_TIMEOUT_MS, max_count=1, shrink=sh)
                        step = f"{label_prefix}_{vname}_rot{angle}_sh{sh}"
                    else:
                        res = dm_decode(rot, timeout=ATTEMPT_TIMEOUT_MS, max_count=1, shrink=sh, threshold=thr)
                        step = f"{label_prefix}_{vname}_rot{angle}_sh{sh}_th{thr}"

                    out = []
                    for r in res or []:
                        out.append({"text": r.data.decode("utf-8", errors="ignore"),
                                    "rect": getattr(r, "rect", None)})
                    attempts.append({"step": step, "found": len(out), "size": rot.size})
                    if out:
                        for o in out:
                            o["pretty"] = o["text"].replace("\x1D", "|")
                        return {"ok": True, "codes": out}
    return {"ok": True, "codes": []}

def _auto_crop_dm(env, im):
    if not env["OPENCV_AVAILABLE"]:
        return None
    cv2 = env["cv2"]
    Image = env["Image"]
    import numpy as np
    img_cv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    g = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    g = cv2.bilateralFilter(g, d=7, sigmaColor=50, sigmaSpace=50)
    edges = cv2.Canny(g, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = g.shape
    best, best_score = None, 0.0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < (W * H * 0.01):
            continue
        ar = w / float(h)
        squareness = 1.0 - abs(1.0 - ar)
        roi = g[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
        if roi.size == 0: continue
        texture = float(roi.var())
        score = squareness * 0.6 + min(texture / 5000.0, 1.0) * 0.4
        if score > best_score:
            best_score, best = score, (x, y, w, h)
    if not best:
        return None
    x, y, w, h = best
    pad = int(max(w, h) * 0.15)
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    crop = img_cv[y0:y1, x0:x1]
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return crop_pil

@app.post("/decode/file")
async def decode_file(file: UploadFile = File(...)):
    # Imports à la demande (ne casse pas le démarrage si libs absentes)
    env = _lazy_imports()
    settings = _get_settings(env)
    BytesIO = env["BytesIO"]; Image = env["Image"]; ImageOps = env["ImageOps"]

    t0 = env["time"].perf_counter()
    try:
        buf = BytesIO()
        total = 0
        while True:
            chunk = await file.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if total > settings["MAX_UPLOAD_BYTES"]:
                raise HTTPException(status_code=413, detail=f"File too large (> {settings['MAX_UPLOAD_MB']} MB)")
            buf.write(chunk)
        raw = buf.getvalue()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty upload")

        try:
            img = Image.open(BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
        img = ImageOps.exif_transpose(img)
        img = _resize_cap(Image, img, settings["MAX_SIDE_PX"])

        attempts = []
        out = _try_variants(env, img, attempts, "full", t0, settings)
        if out["codes"]:
            return {"ok": True, "found": len(out["codes"]), "codes": out["codes"], "debug": {"attempts": attempts}}

        for scale in (0.75, 0.6, 0.5, 0.4, 0.33):
            if (env["time"].perf_counter() - t0) * 1000 > settings["TIME_BUDGET_MS"]:
                break
            w, h = img.size
            ds = img.resize((max(64, int(w * scale)), max(64, int(h * scale))), Image.LANCZOS)
            out = _try_variants(env, ds, attempts, f"down{scale}", t0, settings)
            if out["codes"]:
                return {"ok": True, "found": len(out["codes"]), "codes": out["codes"], "debug": {"attempts": attempts}}

        try:
            roi = _auto_crop_dm(env, img)
        except Exception:
            roi = None
        if roi is not None and (env["time"].perf_counter() - t0) * 1000 <= settings["TIME_BUDGET_MS"]:
            out = _try_variants(env, roi, attempts, "roi", t0, settings)
            if out["codes"]:
                return {"ok": True, "found": len(out["codes"]), "codes": out["codes"], "debug": {"attempts": attempts}}

        return {"ok": True, "found": 0, "codes": [], "debug": {"attempts": attempts}}

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
