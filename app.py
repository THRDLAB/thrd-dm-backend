from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter
import numpy as np, time, os
from pylibdmtx.pylibdmtx import decode as dm_decode

# OpenCV est optionnel : on démarre même s'il n'est pas installé
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

# Limites configurables via variables d'env (Dockerfile les définit aussi)
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "8"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
MAX_SIDE_PX = int(os.getenv("MAX_SIDE_PX", "1600"))      # cap résolution pour CPU
TIME_BUDGET_MS = int(os.getenv("TIME_BUDGET_MS", "40000"))  # budget temps par requête

def _resize_cap(im: Image.Image, max_side=1600) -> Image.Image:
    w, h = im.size
    m = max(w, h)
    if m <= max_side:
        return im
    r = max_side / m
    return im.resize((int(w*r), int(h*r)), Image.LANCZOS)

def _decode_once(im: Image.Image):
    res = dm_decode(im)
    out = []
    for r in res or []:
        out.append({
            "text": r.data.decode("utf-8", errors="ignore"),
            "rect": getattr(r, "rect", None),
        })
    return out

def _try_variants(im: Image.Image, attempts: list, label_prefix: str, t0: float):
    """Essais rapides: couleur/gris/invert/binaire/sharp + rotations."""
    gray = ImageOps.grayscale(im)
    gray_ac = ImageOps.autocontrast(gray)
    inv = ImageOps.invert(gray_ac)
    # Binarisation simple pour rattraper les impressions pâles
    bw = gray_ac.point(lambda p: 0 if p < 128 else 255, mode='1').convert("L")
    # Lissage léger puis autocontrast pour nettoyer le grain
    sharp = ImageOps.autocontrast(
        Image.fromarray(np.array(gray_ac))  # PIL → np
    ).filter(ImageFilter.UnsharpMask(radius=1.5, percent=100, threshold=2))

    for vname, vimg in [
        ("rgb", im),
        ("gray", gray_ac),
        ("inv", inv),
        ("bw", bw),
        ("sharp", sharp),
    ]:
        for angle in (0, 90, 180, 270):
            if (time.perf_counter() - t0) * 1000 > TIME_BUDGET_MS:
                break
            rot = vimg if angle == 0 else vimg.rotate(angle, expand=True)
            out = _decode_once(rot)
            attempts.append({"step": f"{label_prefix}_{vname}_rot{angle}",
                             "found": len(out), "size": rot.size})
            if out:
                for o in out:
                    o["pretty"] = o["text"].replace("\x1D", "|")
                return {"ok": True, "codes": out}
    return {"ok": True, "codes": []}

def _auto_crop_dm(im: Image.Image):
    """
    Heuristique OpenCV : trouve un gros contour quasi-carré 'à damier'
    pour cadrer le DataMatrix; renvoie un crop (avec padding) ou None.
    """
    if not OPENCV_AVAILABLE:
        return None

    img_cv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    g = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    g = cv2.bilateralFilter(g, d=7, sigmaColor=50, sigmaSpace=50)
    edges = cv2.Canny(g, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = g.shape
    best = None
    best_score = 0.0

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < (W * H * 0.01):  # ignore trop petit
            continue
        # Proximité du carré
        ar = w / float(h)
        squareness = 1.0 - abs(1.0 - ar)
        # 'Texture' (variance): plus élevée ≈ damier
        roi = g[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
        if roi.size == 0:
            continue
        texture = float(roi.var())
        score = squareness * 0.6 + min(texture / 5000.0, 1.0) * 0.4
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if not best:
        return None

    x, y, w, h = best
    pad = int(max(w, h) * 0.15)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    crop = img_cv[y0:y1, x0:x1]
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return crop_pil

@app.post("/decode/file")
async def decode_file(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    try:
        # Lecture streamée (évite OOM) + limite de taille
        buf = BytesIO()
        total = 0
        while True:
            chunk = await file.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413,
                                    detail=f"File too large (> {MAX_UPLOAD_MB} MB)")
            buf.write(chunk)
        raw = buf.getvalue()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty upload")

        # Ouverture + orientation EXIF + cap de taille
        try:
            img = Image.open(BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
        img = ImageOps.exif_transpose(img)
        img = _resize_cap(img, MAX_SIDE_PX)

        attempts = []

        # A) Image complète (variants)
        out = _try_variants(img, attempts, "full", t0)
        if out["codes"]:
            return {"ok": True, "found": len(out["codes"]), "codes": out["codes"],
                    "debug": {"attempts": attempts}}

        # B) Downscale pyramidal (cas "trop près")
        for scale in (0.75, 0.6, 0.5, 0.4, 0.33):
            if (time.perf_counter() - t0) * 1000 > TIME_BUDGET_MS:
                break
            w, h = img.size
            ds = img.resize((max(64, int(w * scale)),
                             max(64, int(h * scale))), Image.LANCZOS)
            out = _try_variants(ds, attempts, f"down{scale}", t0)
            if out["codes"]:
                return {"ok": True, "found": len(out["codes"]), "codes": out["codes"],
                        "debug": {"attempts": attempts}}

        # C) Auto-crop ROI + variants (si OpenCV dispo)
        try:
            roi = _auto_crop_dm(img)
        except Exception:
            roi = None
        if roi is not None and (time.perf_counter() - t0) * 1000 <= TIME_BUDGET_MS:
            out = _try_variants(roi, attempts, "roi", t0)
            if out["codes"]:
                return {"ok": True, "found": len(out["codes"]), "codes": out["codes"],
                        "debug": {"attempts": attempts}}

        # Rien trouvé dans le budget de temps
        return {"ok": True, "found": 0, "codes": [], "debug": {"attempts": attempts}}

    except HTTPException:
        raise
    except Exception as e:
        # Ne pas laisser crasher le worker → 500 JSON propre
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


