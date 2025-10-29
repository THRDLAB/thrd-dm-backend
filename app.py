from fastapi import FastAPI, File, UploadFile, HTTPException, Query
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

# --- Imports ajoutés pour l'étape 2 (lookup CIP13 -> infos médicament)
from resolver import (
    parse_datamatrix_to_cip13,
    extract_conditionnement,
    extract_dosage_from_compo,
    fallback_dosage_from_text,
)
from index_builder import CipIndexManager

app = FastAPI()

# ----------------------------
# Santé / état de l'index
# ----------------------------

# Config via variables d'env
API_BASE_URL = os.getenv("API_BASE_URL", "https://medicaments-api.giygas.dev")
CIP_INDEX_CACHE_PATH = os.getenv("CIP_INDEX_CACHE_PATH", "cip_index.json")

CIP_MGR = CipIndexManager(cache_path=CIP_INDEX_CACHE_PATH)

@app.on_event("startup")
def on_startup():
    """
    Au boot :
    - tente un warm-start depuis cip_index.json si présent
    - sinon construit l'index local depuis l'API (pagination /database/{page})
    """
    CIP_MGR.warm_start_or_build(base_url=API_BASE_URL)

@app.get("/health")
def health():
    return {
        "ok": True,
        "index_size": CIP_MGR.size,
        "last_refresh_ts": CIP_MGR.last_refresh_ts
    }

# ----------------------------
# Paramètres scan image (conservés)
# ----------------------------

# Limites configurables via variables d'env (Dockerfile les définit aussi)
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "8"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
MAX_SIDE_PX = int(os.getenv("MAX_SIDE_PX", "1600"))            # cap résolution pour CPU
TIME_BUDGET_MS = int(os.getenv("TIME_BUDGET_MS", "20000"))     # budget temps par requête
ATTEMPT_TIMEOUT_MS = int(os.getenv("ATTEMPT_TIMEOUT_MS", "800"))  # max par appel pylibdmtx
SHRINKS = tuple(int(x) for x in os.getenv("SHRINKS", "3,2,1").split(","))  # du plus rapide au plus précis
TRY_THRESHOLDS = (None, 20)  # None = défaut pylibdmtx, puis un seuil un peu plus agressif

# ----------------------------
# Outils scan image (conservés)
# ----------------------------

def _resize_cap(im: Image.Image, max_side=1600) -> Image.Image:
    w, h = im.size
    m = max(w, h)
    if m <= max_side:
        return im
    r = max_side / m
    return im.resize((int(w*r), int(h*r)), Image.LANCZOS)

def _decode_once(im: Image.Image, timeout_ms=800, shrink=2, max_count=1):
    res = dm_decode(im, timeout=timeout_ms, max_count=max_count, shrink=shrink)
    out = []
    for r in res or []:
        out.append({
            "text": r.data.decode("utf-8", errors="ignore"),
            "rect": getattr(r, "rect", None),
        })
    return out

def _try_variants(im: Image.Image, attempts: list, label_prefix: str, t0: float):
    """Essais rapides: couleur/gris/invert/binaire/sharp + rotations,
    avec timebox (timeout) + shrink progressif (3→2→1) par tentative."""
    gray = ImageOps.grayscale(im)
    gray_ac = ImageOps.autocontrast(gray)
    inv = ImageOps.invert(gray_ac)
    # Binarisation simple pour rattraper les impressions pâles
    bw = gray_ac.point(lambda p: 0 if p < 128 else 255, mode='1').convert("L")
    # Lissage léger puis autocontrast pour nettoyer le grain
    sharp = ImageOps.autocontrast(
        Image.fromarray(np.array(gray_ac))
    ).filter(ImageFilter.UnsharpMask(radius=1.5, percent=100, threshold=2))

    # ordre des variantes
    variants = [
        ("rgb", im),
        ("gray", gray_ac),
        ("inv", inv),
        ("bw", bw),
        ("sharp", sharp),
    ]

    for vname, vimg in variants:
        for angle in (0, 90, 180, 270):
            if (time.perf_counter() - t0) * 1000 > TIME_BUDGET_MS:
                break
            rot = vimg if angle == 0 else vimg.rotate(angle, expand=True)

            # boucle shrink (3→2→1) avec timeout borné par essai
            for sh in SHRINKS:
                # pour les variantes non N&B, on garde threshold par défaut, puis 20
                thresholds = TRY_THRESHOLDS if vname in ("gray", "inv", "bw") else (None,)
                for thr in thresholds:
                    if (time.perf_counter() - t0) * 1000 > TIME_BUDGET_MS:
                        break

                    # appel pylibdmtx "timeboxé" et échantillonné
                    if thr is None:
                        res = dm_decode(rot, timeout=ATTEMPT_TIMEOUT_MS, max_count=1, shrink=sh)
                        step = f"{label_prefix}_{vname}_rot{angle}_sh{sh}"
                    else:
                        res = dm_decode(rot, timeout=ATTEMPT_TIMEOUT_MS, max_count=1, shrink=sh, threshold=thr)
                        step = f"{label_prefix}_{vname}_rot{angle}_sh{sh}_th{thr}"

                    out = []
                    for r in res or []:
                        out.append({
                            "text": r.data.decode("utf-8", errors="ignore"),
                            "rect": getattr(r, "rect", None)
                        })

                    attempts.append({"step": step, "found": len(out), "size": rot.size})

                    if out:
                        # pretty: rendre FNC1 visible
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

# ----------------------------
# Endpoint scan image (conservé)
# ----------------------------

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
        if out["codes"]:\n            return {\"ok\": True, \"found\": len(out[\"codes\"]), \"codes\": out[\"codes\"],\n                    \"debug\": {\"attempts\": attempts}}\n\n        # B) Downscale pyramidal (cas \"trop près\")\n        for scale in (0.75, 0.6, 0.5, 0.4, 0.33):\n            if (time.perf_counter() - t0) * 1000 > TIME_BUDGET_MS:\n                break\n            w, h = img.size\n            ds = img.resize((max(64, int(w * scale)),\n                             max(64, int(h * scale))), Image.LANCZOS)\n            out = _try_variants(ds, attempts, f\"down{scale}\", t0)\n            if out[\"codes\"]:\n                return {\"ok\": True, \"found\": len(out[\"codes\"]), \"codes\": out[\"codes\"],\n                        \"debug\": {\"attempts\": attempts}}\n\n        # C) Auto-crop ROI + variants (si OpenCV dispo)\n        try:\n            roi = _auto_crop_dm(img)\n        except Exception:\n            roi = None\n        if roi is not None and (time.perf_counter() - t0) * 1000 <= TIME_BUDGET_MS:\n            out = _try_variants(roi, attempts, \"roi\", t0)\n            if out[\"codes\"]:\n                return {\"ok\": True, \"found\": len(out[\"codes\"]), \"codes\": out[\"codes\"],\n                        \"debug\": {\"attempts\": attempts}}\n\n        # Rien trouvé dans le budget de temps\n        return {\"ok\": True, \"found\": 0, \"codes\": [], \"debug\": {\"attempts\": attempts}}\n\n    except HTTPException:\n        raise\n    except Exception as e:\n        # Ne pas laisser crasher le worker → 500 JSON propre\n        return JSONResponse(status_code=500, content={\"ok\": False, \"error\": str(e)})\n
# ----------------------------
# Nouveaux endpoints: lookup CIP13
# ----------------------------

@app.get("/lookup/cip/{cip13}")
def lookup_cip(cip13: str):
    """
    Retourne les infos médicament à partir d'un CIP13.
    Réponse:
    {
      "ok": true,
      "data": {
        "cip13": "...",
        "nom": "...",
        "forme": "...",
        "dosage": "...",
        "conditionnement": {"valeur": 30, "unite": "comprimés"},
        "libelle": "Boîte de 30 comprimés sécables"
      }
    }
    """
    item = CIP_MGR.get(cip13)
    if not item:
        raise HTTPException(status_code=404, detail=f"CIP13 {cip13} introuvable.")

    cond = extract_conditionnement(item.get("libelle"))
    dosage = extract_dosage_from_compo(item.get("composition")) or \
             fallback_dosage_from_text(item.get("nom"), item.get("libelle"))

    return {"ok": True, "data": {
        "cip13": cip13,
        "nom": item.get("nom"),
        "forme": item.get("forme"),
        "dosage": dosage,
        "conditionnement": cond,
        "libelle": item.get("libelle"),
    }}

@app.get("/lookup/from-dm")
def lookup_from_dm(gs1: str = Query(..., description="Chaîne GS1 brute (DataMatrix)")):
    """
    Prend une chaîne GS1 (issue de /decode/file ou autre), extrait le CIP13 (NTIN '03400' -> '3400…'),
    puis fait le lookup local.
    """
    cip13 = parse_datamatrix_to_cip13(gs1)
    if not cip13:
        raise HTTPException(status_code=422, detail="CIP13 non dérivable (GTIN sans préfixe 03400).")
    # Réutilise la logique ci-dessus
    return lookup_cip(cip13)
