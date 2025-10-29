# app.py — DM Backend (scan DataMatrix + lookup CIP13)
# - Safe-boot : /, /docs, /health dispo même sans libs d’imagerie
# - Index médicaments persistant (cache disque) + build en arrière-plan
# - Refresh de l’index piloté par variable d’env (INDEX_REFRESH_HOURS)
# - Endpoints: /decode/file, /decode/url, /lookup/cip/{cip13}, /lookup/from-dm
# - Admin/Diag: /admin/rebuild-index, /admin/rebuild-index-throttled, /net/ping, /net/meds-page

from __future__ import annotations
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os, threading, time as _time
from typing import Any, Dict
from fastapi import Query
import traceback, json, os

# ==== Lookup / Index manager (stdlib only) ====
from resolver import (
    parse_datamatrix_to_cip13,
    extract_conditionnement,
    extract_dosage_from_compo,
    fallback_dosage_from_text,
)
from index_builder import CipIndexManager, build_cip_index_from_api, save_index_to_disk, merge_indexes

app = FastAPI(title="DM Backend", version="1.1.0")

# ==== CORS (ajuste si besoin) ====
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOW_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Config ====
API_BASE_URL = os.getenv("API_BASE_URL", "https://medicaments-api.giygas.dev")
CIP_INDEX_CACHE_PATH = os.getenv("CIP_INDEX_CACHE_PATH", "cip_index.json")
INDEX_REFRESH_HOURS = int(os.getenv("INDEX_REFRESH_HOURS", "24"))  # fréquence de refresh

# Index en mémoire + cache disque
CIP_MGR = CipIndexManager(cache_path=CIP_INDEX_CACHE_PATH)
INDEX_READY = False  # flag simple pour l'état de l'index

def _build_index_job():
    """Construit/charge l'index au boot puis le refresh périodiquement, en arrière-plan."""
    global INDEX_READY
    try:
        print(f"[INDEX] warm_start_or_build from {API_BASE_URL}", flush=True)
        CIP_MGR.warm_start_or_build(base_url=API_BASE_URL)
        INDEX_READY = (CIP_MGR.size > 0)
        print(f"[INDEX] ready={INDEX_READY} size={CIP_MGR.size}", flush=True)

        while True:
            _time.sleep(max(1, INDEX_REFRESH_HOURS) * 3600)
            try:
                print("[INDEX] periodic refresh…", flush=True)
                CIP_MGR.refresh(base_url=API_BASE_URL)
            finally:
                INDEX_READY = (CIP_MGR.size > 0)
                print(f"[INDEX] after refresh size={CIP_MGR.size}", flush=True)
    except Exception as e:
        print(f"[INDEX] background job error: {e}", flush=True)
        INDEX_READY = (CIP_MGR.size > 0)

@app.on_event("startup")
def _start_background_jobs():
    # Build en arrière-plan pour démarrage instantané
    if os.getenv("SKIP_INDEX_BUILD", "0") != "1":
        t = threading.Thread(target=_build_index_job, daemon=True)
        t.start()

# ==== Endpoints de base ====

@app.get("/")
def root():
    return {"ok": True, "msg": "Service up. See /docs"}

@app.get("/health")
def health():
    return {"ok": True, "index_size": CIP_MGR.size, "index_ready": INDEX_READY}

@app.get("/admin/rebuild-index")
def admin_rebuild_index():
    """Relance le job de build/refresh en arrière-plan (non bloquant)."""
    t = threading.Thread(target=_build_index_job, daemon=True)
    t.start()
    return {"ok": True, "msg": "Background index (re)build triggered."}

from fastapi import Query

@app.get("/admin/rebuild-index-throttled")
def admin_rebuild_index_throttled(
    pages: int = Query(5, ge=1, le=500, description="Nombre de pages BDPM à traiter dans CET appel")
):
    try:
        # (si tu n’as PAS modifié index_builder pour accepter pages_per_run,
        # on ne passe rien ici : il lira INDEX_PAGES_PER_RUN depuis l'env)
        # Si tu l’as modifié, dé-commente cette ligne et commente la suivante :
        # tmp = build_cip_index_from_api(base_url=API_BASE_URL, max_pages=None, pages_per_run=pages)
        tmp = build_cip_index_from_api(base_url=API_BASE_URL, max_pages=None)

        added = len(tmp)
        if added == 0:
            return {"ok": True, "indexed": 0, "index_size": CIP_MGR.size, "note": "no new pages (or rate-limited)"}

        existing = []
        if CIP_INDEX_CACHE_PATH and os.path.exists(CIP_INDEX_CACHE_PATH):
            with open(CIP_INDEX_CACHE_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)

        merged = merge_indexes(existing, tmp)
        save_index_to_disk(merged, CIP_INDEX_CACHE_PATH)
        CIP_MGR._index = merged
        return {"ok": True, "indexed": added, "index_size": len(merged), "cache_path": CIP_INDEX_CACHE_PATH}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e), "trace": traceback.format_exc()}
        )
        
# ==== Diag egress ====
import socket, json
from urllib.request import Request, urlopen

def _fetch_json(url: str, timeout: int = 10):
    req = Request(url, headers={"User-Agent": "egress-check/1.0", "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8", errors="ignore")
        ctype = r.headers.get("Content-Type", "")
        try:
            return r.status, ctype, json.loads(body)
        except Exception:
            return r.status, ctype, body

@app.get("/net/ping")
def net_ping():
    out = {}
    try:
        out["dns_medicaments_api"] = socket.gethostbyname("medicaments-api.giygas.dev")
    except Exception as e:
        out["dns_medicaments_api_error"] = str(e)
    for name, url in [("ifconfig", "https://ifconfig.me/ip"), ("ipify", "https://api.ipify.org?format=json")]:
        try:
            status, ctype, data = _fetch_json(url, timeout=8)
            out[f"egress_{name}_status"] = status
            out[f"egress_{name}_data"] = data
        except Exception as e:
            out[f"egress_{name}_error"] = str(e)
    return {"ok": True, "env_http_proxy": os.getenv("HTTP_PROXY"), "env_https_proxy": os.getenv("HTTPS_PROXY"), "diag": out}

@app.get("/net/meds-page")
def net_meds_page(page: int = 1):
    url = f"{API_BASE_URL.rstrip('/')}/database/{page}"
    try:
        status, ctype, data = _fetch_json(url, timeout=12)
        return {"ok": True, "url": url, "status": status, "content_type": ctype, "sample": (data if isinstance(data, dict) else str(data)[:400])}
    except Exception as e:
        return JSONResponse(status_code=502, content={"ok": False, "url": url, "error": str(e)})

# ==== Lazy imports (scan) : ne bloquent pas le boot ====

def _lazy_imports() -> Dict[str, Any]:
    try:
        from io import BytesIO
        import numpy as np  # type: ignore
        from PIL import Image, ImageOps, ImageFilter  # type: ignore
        from pylibdmtx.pylibdmtx import decode as dm_decode  # type: ignore
        import time
        import urllib.request as urlreq
    except Exception as e:
        raise HTTPException(status_code=501, detail=f"Scan not available: missing libs: {e}")
    try:
        import cv2  # type: ignore
        OPENCV_AVAILABLE = True
    except Exception:
        cv2 = None  # type: ignore
        OPENCV_AVAILABLE = False
    return {
        "BytesIO": BytesIO, "np": np,
        "Image": Image, "ImageOps": ImageOps, "ImageFilter": ImageFilter,
        "dm_decode": dm_decode, "time": time, "urlreq": urlreq,
        "cv2": cv2, "OPENCV_AVAILABLE": OPENCV_AVAILABLE, "os": os,
    }

def _get_settings(env: Dict[str, Any]) -> Dict[str, Any]:
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

# ==== Outils image (identiques à ta version safe) ====

def _resize_cap(Image, im, max_side=1600):
    w, h = im.size
    m = max(w, h)
    if m <= max_side:
        return im
    r = max_side / m
    return im.resize((int(w*r), int(h*r)), Image.LANCZOS)

def _try_variants(env: Dict[str, Any], im, attempts: list, label_prefix: str, t0: float, settings: Dict[str, Any]):
    ImageOps, ImageFilter, np, dm_decode = env["ImageOps"], env["ImageFilter"], env["np"], env["dm_decode"]
    TIME_BUDGET_MS, ATTEMPT_TIMEOUT_MS, SHRINKS, TRY_THRESHOLDS = (
        settings["TIME_BUDGET_MS"], settings["ATTEMPT_TIMEOUT_MS"], settings["SHRINKS"], settings["TRY_THRESHOLDS"]
    )
    gray = ImageOps.grayscale(im)
    gray_ac = ImageOps.autocontrast(gray)
    inv = ImageOps.invert(gray_ac)
    bw = gray_ac.point(lambda p: 0 if p < 128 else 255, mode='1').convert("L")
    sharp = ImageOps.autocontrast(env["Image"].fromarray(np.array(gray_ac))).filter(
        ImageFilter.UnsharpMask(radius=1.5, percent=100, threshold=2)
    )
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

def _auto_crop_dm(env: Dict[str, Any], im):
    if not env["OPENCV_AVAILABLE"]:
        return None
    cv2 = env["cv2"]; Image = env["Image"]; import numpy as _np
    img_cv = cv2.cvtColor(_np.array(im), cv2.COLOR_RGB2BGR)
    g = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    g = cv2.bilateralFilter(g, d=7, sigmaColor=50, sigmaSpace=50)
    edges = cv2.Canny(g, 50, 150)
    edges = cv2.dilate(edges, _np.ones((3, 3), _np.uint8), iterations=1)
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
        if roi.size == 0:
            continue
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

# ==== Endpoints scan ====

@app.post("/decode/file")
async def decode_file(file: UploadFile = File(...)):
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

@app.get("/decode/url")
def decode_url(url: str = Query(..., description="URL publique d'une image (jpg/png)")):
    env = _lazy_imports()
    settings = _get_settings(env)
    BytesIO = env["BytesIO"]; Image = env["Image"]; ImageOps = env["ImageOps"]; urlreq = env["urlreq"]
    try:
        req = urlreq.Request(url, headers={"User-Agent": "dm-backend/1.0"})
        with urlreq.urlopen(req, timeout=10) as r:
            raw = r.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de télécharger l'image: {e}")
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    img = ImageOps.exif_transpose(img)
    img = _resize_cap(Image, img, settings["MAX_SIDE_PX"])
    t0 = env["time"].perf_counter()
    attempts = []
    out = _try_variants(env, img, attempts, "url_full", t0, settings)
    if out["codes"]:
        return {"ok": True, "found": len(out["codes"]), "codes": out["codes"], "debug": {"attempts": attempts}}
    for scale in (0.75, 0.6, 0.5, 0.4, 0.33):
        if (env["time"].perf_counter() - t0) * 1000 > settings["TIME_BUDGET_MS"]:
            break
        w, h = img.size
        ds = img.resize((max(64, int(w * scale)), max(64, int(h * scale))), Image.LANCZOS)
        out = _try_variants(env, ds, attempts, f"url_down{scale}", t0, settings)
        if out["codes"]:
            return {"ok": True, "found": len(out["codes"]), "codes": out["codes"], "debug": {"attempts": attempts}}
    try:
        roi = _auto_crop_dm(env, img)
    except Exception:
        roi = None
    if roi is not None and (env["time"].perf_counter() - t0) * 1000 <= settings["TIME_BUDGET_MS"]:
        out = _try_variants(env, roi, attempts, "url_roi", t0, settings)
        if out["codes"]:
            return {"ok": True, "found": len(out["codes"]), "codes": out["codes"], "debug": {"attempts": attempts}}
    return {"ok": True, "found": 0, "codes": [], "debug": {"attempts": attempts}}

@app.get("/admin/progress")
def admin_progress():
    path = os.getenv("INDEX_PROGRESS_PATH", "")
    if not path:
        return {"ok": False, "msg": "INDEX_PROGRESS_PATH not set"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            js = json.load(f)
        return {"ok": True, "progress": js, "path": path}
    except Exception as e:
        return {"ok": False, "error": str(e), "path": path}

@app.post("/admin/progress/reset")
def admin_progress_reset(next_page: int = 1):
    path = os.getenv("INDEX_PROGRESS_PATH", "")
    if not path:
        raise HTTPException(status_code=400, detail="INDEX_PROGRESS_PATH not set")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"next_page": int(next_page), "ts": _time.time()}, f)
    return {"ok": True, "path": path, "next_page": next_page}


# ==== Endpoints lookup (utilisent l’index local persistant) ====

@app.get("/lookup/cip/{cip13}")
def lookup_cip(cip13: str):
    if not INDEX_READY or CIP_MGR.size == 0:
        raise HTTPException(status_code=503, detail="INDEX_NOT_READY")
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
    if not INDEX_READY or CIP_MGR.size == 0:
        raise HTTPException(status_code=503, detail="INDEX_NOT_READY")
    cip13 = parse_datamatrix_to_cip13(gs1)
    if not cip13:
        raise HTTPException(status_code=422, detail="CIP13 non dérivable (GTIN sans préfixe 03400).")
    return lookup_cip(cip13)





