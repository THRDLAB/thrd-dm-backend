from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter
import numpy as np, cv2, time, os
from pylibdmtx.pylibdmtx import decode as dm_decode

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "8"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
MAX_SIDE_PX = int(os.getenv("MAX_SIDE_PX", "1600"))
TIME_BUDGET_MS = int(os.getenv("TIME_BUDGET_MS", "12000"))

def _pillow_to_cv(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def _cv_to_pillow(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def _resize_cap(im: Image.Image, max_side=1600):
    w, h = im.size
    m = max(w, h)
    if m <= max_side: return im
    r = max_side / m
    return im.resize((int(w*r), int(h*r)), Image.LANCZOS)

def _decode_once(im: Image.Image):
    res = dm_decode(im)
    out = []
    for r in res or []:
        out.append({
            "text": r.data.decode("utf-8", errors="ignore"),
            "rect": getattr(r, "rect", None)
        })
    return out

def _try_variants(im: Image.Image, attempts, label_prefix, time_start):
    # variantes rapides : couleur/gris/invert/binaire + rotations
    variants = []
    gray = ImageOps.grayscale(im)
    gray_ac = ImageOps.autocontrast(gray)
    inv = ImageOps.invert(gray_ac)
    # binarisation simple (évite Otsu coûteux) + petite netteté
    bw = gray_ac.point(lambda p: 0 if p < 128 else 255, mode='1').convert("L")
    sharp = ImageOps.autocontrast(Image.fromarray(
        cv2.GaussianBlur(np.array(gray_ac), (0,0), 0.8)
    ))
    for vname, vimg in [
        ("rgb", im),
        ("gray", gray_ac),
        ("inv", inv),
        ("bw", bw),
        ("sharp", sharp)
    ]:
        for angle in (0, 90, 180, 270):
            if (time.perf_counter() - time_start)*1000 > TIME_BUDGET_MS: break
            rot = vimg if angle == 0 else vimg.rotate(angle, expand=True)
            out = _decode_once(rot)
            attempts.append({"step": f"{label_prefix}_{vname}_rot{angle}", "found": len(out), "size": rot.size})
            if out:
                for o in out: o["pretty"] = o["text"].replace("\x1D", "|")
                return {"ok": True, "codes": out}
    return {"ok": True, "codes": []}

def _auto_crop_dm(im: Image.Image):
    """
    Heuristique OpenCV : cherche le plus grand contour quasi-carré à damier,
    retourne un crop (avec padding) si trouvé, sinon None.
    """
    img_cv = _pillow_to_cv(im)
    g = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    # légers filtres pour nettoyer le papier texturé
    g = cv2.bilateralFilter(g, d=7, sigmaColor=50, sigmaSpace=50)
    # edges
    e = cv2.Canny(g, 50, 150)
    # dilate un peu pour fermer les petits trous
    e = cv2.dilate(e, np.ones((3,3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = g.shape
    best = None
    best_score = 0.0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w*h
        if area < (W*H*0.01):  # ignore tout petit
            continue
        # ratio ~ carré
        ar = w / float(h)
        squareness = 1.0 - abs(1.0 - ar)
        # texture: variance locale élevée ~ damier
        roi = g[max(0,y):min(H,y+h), max(0,x):min(W,x+w)]
        if roi.size == 0: 
            continue
        texture = float(roi.var())
        score = squareness * 0.6 + min(texture/5000.0, 1.0) * 0.4
        if score > best_score:
            best_score, best = (score, (x,y,w,h))
        if not best:
        return None
-   x,y,w,h = best[1]
+   x, y, w, h = best
    pad = int(max(w,h)*0.15)
    x0 = max(0, x-pad); y0 = max(0, y-pad)
    x1 = min(W, x+w+pad); y1 = min(H, y+h+pad)
    crop = img_cv[y0:y1, x0:x1]
    return _cv_to_pillow(crop)

@app.post("/decode/file")
async def decode_file(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    try:
        # lecture sécurisée
        buf = BytesIO(); total = 0
        while True:
            chunk = await file.read(65536)
            if not chunk: break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_MB} MB)")
            buf.write(chunk)
        raw = buf.getvalue()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty upload")

        # ouverture + orientation + cap taille
        try:
            img = Image.open(BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
        img = ImageOps.exif_transpose(img)
        img = _resize_cap(img, MAX_SIDE_PX)

        attempts = []

        # 0) essai direct + variantes
        out = _try_variants(img, attempts, "full", t0)
        if out["codes"]:
            return {"ok": True, "found": len(out["codes"]), "codes": out["codes"], "debug": {"attempts": attempts}}

        # 1) downscale pyramide (trop près)
        for scale in (0.75, 0.6, 0.5, 0.4, 0.33):
            if (time.perf_counter() - t0)*1000 > TIME_BUDGET_MS: break
            w,h = img.size
            ds = img.resize((max(64,int(w*scale)), max(64,int(h*scale))), Image.LANCZOS)
            out = _try_variants(ds, attempts, f"down{scale}", t0)
            if out["codes"]:
                return {"ok": True, "found": len(out["codes"]), "codes": out["codes"], "debug": {"attempts": attempts}}

        # 2) auto-crop ROI + variantes
        roi = _auto_crop_dm(img)
        if roi is not None and (time.perf_counter() - t0)*1000 <= TIME_BUDGET_MS:
            out = _try_variants(roi, attempts, "roi", t0)
            if out["codes"]:
                return {"ok": True, "found": len(out["codes"]), "codes": out["codes"], "debug": {"attempts": attempts}}

        return {"ok": True, "found": 0, "codes": [], "debug": {"attempts": attempts}}

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

