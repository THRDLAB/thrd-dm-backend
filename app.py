import io, os
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import requests
from pylibdmtx.pylibdmtx import decode as dmtx_decode

API_KEY = os.getenv("API_KEY")  # définis-la dans Northflank
PORT = int(os.getenv("PORT", "8080"))

app = FastAPI(title="DataMatrix Decoder", version="1.0.0")

class UrlBody(BaseModel):
    url: str

def check_key(x_api_key: Optional[str]):
    if API_KEY and (x_api_key != API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")

def decode_image(img: Image.Image):
    # Astuce robustesse : convert en niveaux de gris
    gray = img.convert("L")
    results = dmtx_decode(gray)  # renvoie une liste
    out = []
    for r in results:
        text = r.data.decode("utf-8", errors="replace")
        # r.rect est (x, y, w, h) si dispo ; sinon polygon via r.polygon
        bbox = getattr(r, "rect", None)
        poly = getattr(r, "polygon", None)
        out.append({
            "type": "DataMatrix",
            "text": text,
            "bbox": bbox,
            "polygon": poly
        })
    return out

@app.post("/decode/url")
def decode_from_url(body: UrlBody, x_api_key: Optional[str] = Header(default=None)):
    check_key(x_api_key)
    try:
        resp = requests.get(body.url, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        results = decode_image(img)
        if not results:
            return JSONResponse({"ok": True, "found": 0, "barcodes": []})
        return {"ok": True, "found": len(results), "barcodes": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/decode/file")
async def decode_from_file(file: UploadFile = File(...), x_api_key: Optional[str] = Header(default=None)):
    check_key(x_api_key)
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
        results = decode_image(img)
        if not results:
            return JSONResponse({"ok": True, "found": 0, "barcodes": []})
        return {"ok": True, "found": len(results), "barcodes": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

# Lancement local: uvicorn app:app --host 0.0.0.0 --port 8080
