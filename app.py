import io
import os
import time
from typing import List, Optional, Dict

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from PIL import Image, ImageOps
from pylibdmtx.pylibdmtx import decode as dmtx_decode


# =========================
# Config / constantes
# =========================
API_KEY = os.getenv("API_KEY", "").strip() or None

# CORS : mets plusieurs origines séparées par des virgules dans ALLOW_ORIGINS,
# ex: "https://tonapp.bubbleapps.io,https://thyropartner.com"
ALLOW_ORIGINS = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "*").split(",")]

# Téléchargement d'image par URL
URL_TIMEOUT = float(os.getenv("URL_TIMEOUT", "10"))  # secondes
MAX_DOWNLOAD_MB = float(os.getenv("MAX_DOWNLOAD_MB", "8"))  # taille max acceptée
MAX_DOWNLOAD_BYTES = int(MAX_DOWNLOAD_MB * 1024 * 1024)

# Pré-traitement image avant décodage
MAX_SIDE = int(os.getenv("MAX_SIDE", "1400"))  # 1200–1600 conseillé


# =========================
# App FastAPI + CORS
# =========================
app = FastAPI(title="DataMatrix Scanner", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOW_ORIGINS == ["*"] else ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Modèles
# =========================
class URLBody(BaseModel):
    url: HttpUrl


# =========================
# Utilitaires
# =========================
def _check_api_key(x_api_key: Optional[str]):
    if API_KEY and (x_api_key or "") != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _prepare_image(img: Image.Image) -> Image.Image:
    """Corrige orientation EXIF, réduit la taille et convertit en niveaux de gris."""
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        # pas d'EXIF, on ignore
        pass

    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

    # Niveaux de gris → plus rapide/robuste pour DataMatrix
    return img.convert("L")


def _decode_pylibdmtx(img: Image.Image) -> List[Dict]:
    """Appelle pylibdmtx et renvoie une liste de dicts normalisés."""
    # dmtx_decode attend une image PIL (mode 'L' ou 'RGB' ok)
    results = dmtx_decode(img)

    out: List[Dict] = []
    for r in results or []:
        # r.data est un bytes; on tente utf-8 puis on nettoie
        try:
            text = r.data.decode("utf-8", errors="ignore")
        except Exception:
            text = ""

        # r.rect ~ {'left':..., 'top':..., 'width':..., 'height':...}
        bbox = None
        if hasattr(r, "rect") and r.rect:
            # Option simple : left, top, width, height
            rect = r.rect
            bbox = [rect.get("left"), rect.get("top"), rect.get("width"), rect.get("height")]

        out.append(
            {
                "type": "DataMatrix",
                "text": text,
                "bbox": bbox,
                "polygon": None,  # pylibdmtx ne renvoie pas de polygone détaillé
            }
        )
    return out


def _decode_image_pipeline(img: Image.Image) -> List[Dict]:
    """Chaîne complète de pré-traitement + décodage."""
    t0 = time.time()
    prepared = _prepare_image(img)
    t1 = time.time()
    results = _decode_pylibdmtx(prepared)
    t2 = time.time()
    # Logs simples en stdout (visibles sur Northflank)
    print(
        f"[decode] original={img.size} prepared={prepared.size} "
        f"prep_ms={int((t1 - t0)*1000)} dmtx_ms={int((t2 - t1)*1000)} "
        f"found={len(results)}",
        flush=True,
    )
    return results


def _download_image_bytes(url: str) -> bytes:
    """Télécharge une image avec limite de taille."""
    with requests.get(url, stream=True, timeout=URL_TIMEOUT) as resp:
        try:
            resp.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Download error: {e}")

        # Taille max : on lit par morceaux
        total = 0
        chunks = []
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if chunk:
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image too large (> {MAX_DOWNLOAD_MB} MB)",
                    )
                chunks.append(chunk)
        return b"".join(chunks)


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/decode/file")
async def decode_file(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None),
):
    """
    Envoi direct du fichier (multipart/form-data).
    + Rapide (pas de re-téléchargement) et RGPD-friendly.
    """
    _check_api_key(x_api_key)

    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file")
        img = Image.open(io.BytesIO(raw))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    results = _decode_image_pipeline(img)
    return {"ok": True, "found": len(results), "barcodes": results}


@app.post("/decode/url")
def decode_url(
    body: URLBody,
    x_api_key: Optional[str] = Header(default=None),
):
    """
    Décodage à partir d'une URL (image publique).
    Utile si l'image est déjà hébergée; plus lent (download).
    """
    _check_api_key(x_api_key)

    t0 = time.time()
    img_bytes = _download_image_bytes(str(body.url))
    t1 = time.time()

    try:
        img = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid downloaded image: {e}")

    results = _decode_image_pipeline(img)
    t2 = time.time()

    print(
        f"[decode_url] download_ms={int((t1 - t0)*1000)} total_ms={int((t2 - t0)*1000)}",
        flush=True,
    )

    return {"ok": True, "found": len(results), "barcodes": results}
