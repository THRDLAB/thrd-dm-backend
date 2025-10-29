# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, ImageFilter
from io import BytesIO
import os, time
import numpy as np
from pylibdmtx.pylibdmtx import decode as dm_decode

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "8"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
MAX_SIDE_PX = int(os.getenv("MAX_SIDE_PX", "1600"))   # borne dure pour temps cpu
TIME_BUDGET_MS = int(os.getenv("TIME_BUDGET_MS", "12000"))

def _decode_once(im: Image.Image):
    res = dm_decode(im)
    out = []
    for r in res or []:
        out.append({
            "text": r.data.decode("utf-8", errors="ignore"),
            "rect": getattr(r, "rect", None)
        })
    return out

def _resize_cap(im: Image.Image, max_side=1600):
    w, h = im.size
    m = max(w, h)
    if m <= max_side:
        return im
    ratio = max_side / m
    return im.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)

@app.post("/decode/file")
async def decode_file(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    try:
        # lecture en chunks (évite OOM)
        buf = BytesIO()
        total = 0
        while True:
            chunk = await file.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_MB} MB)")
            buf.write(chunk)
        raw = buf.getvalue()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty upload")

        # ouverture + orientation + cap de taille
        try:
            img = Image.open(BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
        img = ImageOps.exif_transpose(img)
        img = _resize_cap(img, MAX_SIDE_PX)

        attempts = []

        def attempt(label, im):
            nonlocal attempts
            out = _decode_once(im)
            attempts.append({"step": label, "found": len(out), "size": im.size})
            return out

        # 1) Original + 4 rotations
        for angle in (0, 90, 180, 270):
            if (time.perf_counter() - t0)*1000 > TIME_BUDGET_MS: break
            im = img if angle == 0 else img.rotate(angle, expand=True)
            out = attempt(f"rgb_rot{angle}", im)
            if out: 
                for o in out: o["pretty"] = o["text"].replace("\x1D", "|")
                return {"ok": True, "found": len(out), "codes": out, "debug": {"attempts": attempts}}

        # 2) Grayscale + autocontrast + rotations
        gray = ImageOps.autocontrast(ImageOps.grayscale(img))
        for angle in (0, 90, 180, 270):
            if (time.perf_counter() - t0)*1000 > TIME_BUDGET_MS: break
            im = gray if angle == 0 else gray.rotate(angle, expand=True)
            out = attempt(f"gray_ac_rot{angle}", im)
            if out:
                for o in out: o["pretty"] = o["text"].replace("\x1D", "|")
                return {"ok": True, "found": len(out), "codes": out, "debug": {"attempts": attempts}}

        # 3) Downscale pyramide (cas 'trop près')
        for scale in (0.75, 0.6, 0.5):
            if (time.perf_counter() - t0)*1000 > TIME_BUDGET_MS: break
            w, h = img.size
            ds = img.resize((max(64,int(w*scale)), max(64,int(h*scale))), Image.LANCZOS)
            for angle in (0, 90, 180, 270):
                if (time.perf_counter() - t0)*1000 > TIME_BUDGET_MS: break
                im = ds if angle == 0 else ds.rotate(angle, expand=True)
                out = attempt(f"down_{scale}_rot{angle}", im)
                if out:
                    for o in out: o["pretty"] = o["text"].replace("\x1D", "|")
                    return {"ok": True, "found": len(out), "codes": out, "debug": {"attempts": attempts}}

        # 4) Sharpen léger + gray (rapide)
        sharp = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=100, threshold=2))
        g2 = ImageOps.autocontrast(ImageOps.grayscale(sharp))
        for angle in (0, 90, 180, 270):
            if (time.perf_counter() - t0)*1000 > TIME_BUDGET_MS: break
            im = g2 if angle == 0 else g2.rotate(angle, expand=True)
            out = attempt(f"sharp_gray_rot{angle}", im)
            if out:
                for o in out: o["pretty"] = o["text"].replace("\x1D", "|")
                return {"ok": True, "found": len(out), "codes": out, "debug": {"attempts": attempts}}

        # Pas trouvé dans le budget temps
        return {"ok": True, "found": 0, "codes": [], "debug": {"attempts": attempts}}

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
