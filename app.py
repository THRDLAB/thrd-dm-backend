# app.py
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from typing import List, Dict, Any
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from pylibdmtx.pylibdmtx import decode as dm_decode

app = FastAPI()

def _decode_once(img: Image.Image):
    """Retourne liste de dicts {text, rect} via pylibdmtx."""
    res = dm_decode(img)
    out = []
    for r in res or []:
        try:
            txt = r.data.decode("utf-8")
        except Exception:
            txt = r.data.decode("latin-1", errors="ignore")
        # rect: (left, top, width, height) dispo selon version
        rect = getattr(r, "rect", None)
        out.append({"text": txt, "rect": rect})
    return out

def _variance_of_laplacian(img_gray: Image.Image) -> float:
    """Mesure simple de netteté (plus haut = plus net)."""
    arr = np.array(img_gray, dtype=np.float32)
    # kernel Laplacien 3x3
    lap = (
        -1*arr[:-2,1:-1] + -1*arr[1:-1,:-2] + 4*arr[1:-1,1:-1] + -1*arr[1:-1,2:] + -1*arr[2:,1:-1]
    )
    return float(lap.var())

def _try_pipeline(img: Image.Image) -> Dict[str, Any]:
    """
    Multi-essais:
      - original, rotations
      - grayscale + autocontrast
      - downscale (pour 'trop près')
      - upscale (pour 'trop loin / petit')
      - sharpen léger
    On s'arrête au premier succès.
    """
    debug = {"attempts": []}

    def attempt(label, im):
        out = _decode_once(im)
        debug["attempts"].append({"step": label, "found": len(out), "size": im.size})
        return out

    # Toujours corriger orientation EXIF
    img = ImageOps.exif_transpose(img)

    # 0) Nettoyage léger
    base = img.convert("RGB")

    # 1) Original + rotations
    for angle in (0, 90, 180, 270):
        im = base if angle == 0 else base.rotate(angle, expand=True)
        out = attempt(f"rgb_rot{angle}", im)
        if out:
            return {"codes": out, "debug": debug}

    # 2) Grayscale + autocontrast + rotations
    gray = ImageOps.grayscale(base)
    gray_ac = ImageOps.autocontrast(gray)
    for angle in (0, 90, 180, 270):
        im = gray_ac if angle == 0 else gray_ac.rotate(angle, expand=True)
        out = attempt(f"gray_ac_rot{angle}", im)
        if out:
            return {"codes": out, "debug": debug}

    # 3) Downscale pyramide (utile si le code remplit trop l'image)
    for scale in (0.85, 0.7, 0.5, 0.35):
        w, h = base.size
        nw, nh = max(64, int(w*scale)), max(64, int(h*scale))
        ds = base.resize((nw, nh), Image.LANCZOS)
        for angle in (0, 90, 180, 270):
            im = ds if angle == 0 else ds.rotate(angle, expand=True)
            out = attempt(f"down_{scale}_rot{angle}", im)
            if out:
                return {"codes": out, "debug": debug}

    # 4) Upscale (utile si le code est petit/lointain)
    if max(base.size) < 1200:
        for scale in (1.25, 1.5, 2.0):
            w, h = base.size
            us = base.resize((int(w*scale), int(h*scale)), Image.NEAREST)  # préserve modules
            for angle in (0, 90, 180, 270):
                im = us if angle == 0 else us.rotate(angle, expand=True)
                out = attempt(f"up_{scale}_rot{angle}", im)
                if out:
                    return {"codes": out, "debug": debug}

    # 5) Sharpen léger + grayscale
    sharp = base.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))
    g2 = ImageOps.autocontrast(ImageOps.grayscale(sharp))
    for angle in (0, 90, 180, 270):
        im = g2 if angle == 0 else g2.rotate(angle, expand=True)
        out = attempt(f"sharp_gray_rot{angle}", im)
        if out:
            return {"codes": out, "debug": debug}

    return {"codes": [], "debug": debug}

def _distance_hint(img: Image.Image, codes: List[Dict[str, Any]]) -> str:
    """
    Si rect dispo, estime la 'taille relative' du code.
    > ~0.6 du grand côté = trop près ; < ~0.15 = trop loin.
    """
    if not codes:
        return ""
    rects = [c.get("rect") for c in codes if c.get("rect")]
    if not rects:
        return ""
    W, H = img.size
    max_side = max(W, H)
    # rect = (left, top, width, height)
    ratios = []
    for r in rects:
        try:
            _, _, rw, rh = r
            ratios.append(max(rw, rh) / max_side)
        except Exception:
            pass
    if not ratios:
        return ""
    r = max(ratios)
    if r >= 0.65:
        return "Le code remplit trop l'image : reculez légèrement."
    if r <= 0.12:
        return "Le code est trop petit : rapprochez-vous un peu."
    return ""

@app.post("/decode/file")
async def decode_file(file: UploadFile = File(...)):
    raw = await file.read()
    img = Image.open(BytesIO(raw)).convert("RGB")

    # Mesure de netteté (avant traitement) pour donner un message utile
    sharp_score = _variance_of_laplacian(ImageOps.grayscale(img))
    blur_hint = ""
    if sharp_score < 30:   # seuil empirique; ajuste selon tes retours
        blur_hint = "Photo probablement floue : stabilisez et faites la mise au point."

    result = _try_pipeline(img)
    found = len(result["codes"])
    hint = ""
    if found:
        # Ajoute un conseil 'distance' si on a un rect
        dh = _distance_hint(img, result["codes"])
        if dh:
            hint = dh
    else:
        hint = blur_hint or "Aucun code lu. Essayez de vous reculer un peu et évitez les reflets."

    # Option : pretty print GS1 si connu (ex: (01)(17)(21)(10))
    def pretty_gs1(t: str) -> str:
        s = t.replace("\x1D", "|")  # FNC1 visible
        return s

    codes_out = []
    for c in result["codes"]:
        txt = c["text"]
        codes_out.append({
            "text": txt,
            "pretty": pretty_gs1(txt),
            "rect": c.get("rect"),
        })

    return {
        "ok": True,
        "found": found,
        "codes": codes_out,
        "debug": {
            "sharp_score": sharp_score,
            "attempts": result["debug"]["attempts"],
        },
        "hint": hint
    }
