# index_builder.py
# Construction & gestion d'un index local CIP13 -> {nom, forme, composition, libelle}
# Source: API "médicaments" (pagination /database/{page})
# Stdlib only (urllib), robustesse réseau, cache disque (JSON), hot-swap en mémoire.

from __future__ import annotations
import json, time, gzip, io
from typing import Any, Dict, List, Optional, Callable, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

DEFAULT_BASE_URL = "https://medicaments-api.giygas.dev"  # paramétrable
DEFAULT_UA = "dm-backend/1.0 (+index_builder)"
DEFAULT_TIMEOUT = 20
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_S = 0.8

# ----------- HTTP utils (stdlib) -----------

def http_get_json(url: str, etag: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT) -> Tuple[Optional[dict], Optional[str], int]:
    """
    GET JSON (gzip accepté). Retourne (json_or_none, etag, status_code).
    Si 304 Not Modified -> (None, etag, 304)
    """
    headers = {
        "User-Agent": DEFAULT_UA,
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "Connection": "close",
    }
    if etag:
        headers["If-None-Match"] = etag

    req = Request(url, headers=headers, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        status = resp.status
        new_etag = resp.headers.get("ETag")
        if status == 304:
            return None, new_etag or etag, 304

        data = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
                data = gz.read()

        if not data:
            return {}, new_etag, status

        try:
            js = json.loads(data.decode("utf-8"))
        except Exception:
            # Certains endpoints renvoient déjà des bytes UTF-8 valides
            js = json.loads(data)
        return js, new_etag, status


def get_with_retry(url: str, etag: Optional[str] = None, retries: int = DEFAULT_RETRIES) -> Tuple[Optional[dict], Optional[str], int]:
    attempt = 0
    last_err: Optional[Exception] = None
    while attempt <= retries:
        try:
            return http_get_json(url, etag=etag)
        except (HTTPError, URLError, TimeoutError) as e:
            last_err = e
            # backoff
            if attempt == retries:
                raise
            time.sleep((attempt + 1) * DEFAULT_BACKOFF_S)
            attempt += 1
    # Should not reach here
    raise last_err if last_err else RuntimeError("Unknown HTTP error")

# ----------- Index build -----------

def _normalize_presentation_libelle(libelle: Optional[str]) -> str:
    if not libelle:
        return ""
    return " ".join(str(libelle).split())  # trim + collapse spaces

def _extract_nom(med: dict) -> str:
    # différentes implémentations possibles selon la source
    return (
        med.get("elementPharmaceutique")
        or med.get("denomination")
        or med.get("nom")
        or ""
    ).strip()

def _extract_forme(med: dict) -> str:
    return (
        med.get("formePharmaceutique")
        or med.get("forme")
        or ""
    ).strip().capitalize()

def _extract_composition(med: dict) -> List[dict]:
    # on normalise quelques clés possibles
    comp = med.get("composition") or []
    norm: List[dict] = []
    for c in comp:
        norm.append({
            "denominationSubstance": (c.get("denominationSubstance") or c.get("substance") or "").strip(),
            "dosage": (c.get("dosage") or "").strip()
        })
    return norm

def build_cip_index_from_api(
    base_url: str = DEFAULT_BASE_URL,
    progress: Optional[Callable[[int, int], None]] = None,
    max_pages: Optional[int] = None,
) -> Dict[str, dict]:
    """
    Parcourt /database/{page} et construit un dict:
      index[cip13] = { "nom": str, "forme": str, "composition": [...], "libelle": str }
    - progress(page, maxPage) si fourni
    - max_pages pour limiter durant les tests
    """
    index: Dict[str, dict] = {}
    page = 1
    etag: Optional[str] = None

    while True:
        url = f"{base_url.rstrip('/')}/database/{page}"
        js, etag, status = get_with_retry(url, etag=None)  # pas de cache conditionnel page à page

        if not js or "data" not in js:
            # format inattendu : on sort proprement si l’API change
            break

        cur = int(js.get("page") or page)
        maxp = int(js.get("maxPage") or cur)

        if progress:
            try:
                progress(cur, maxp)
            except Exception:
                pass

        # Chaque "med" contient souvent "presentation": list avec cip13/libelle
        for med in js["data"]:
            nom = _extract_nom(med)
            forme = _extract_forme(med)
            compo = _extract_composition(med)
            for pres in (med.get("presentation") or []):
                cip13 = str(pres.get("cip13") or "").strip()
                if len(cip13) == 13 and cip13.isdigit():
                    index[cip13] = {
                        "nom": nom,
                        "forme": forme,
                        "composition": compo,
                        "libelle": _normalize_presentation_libelle(pres.get("libelle")),
                    }

        if max_pages and page >= max_pages:
            break
        if cur >= maxp:
            break
        page += 1

    return index

# ----------- Cache disque (warm start) -----------

def save_index_to_disk(index: Dict[str, dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, separators=(",", ":"))

def load_index_from_disk(path: str) -> Optional[Dict[str, dict]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# ----------- Hot-swap manager (optionnel) -----------

class CipIndexManager:
    """
    Gère un index en mémoire + refresh hot-swap.
    Usage:
        mgr = CipIndexManager(cache_path="cip_index.json")
        mgr.warm_start_or_build(base_url=...)
        item = mgr.get("3400...")
        mgr.refresh(base_url=...)  # à appeler périodiquement (cron / background task)
    """
    def __init__(self, cache_path: Optional[str] = None):
        self._index: Dict[str, dict] = {}
        self._last_refresh_ts: float = 0.0
        self._cache_path = cache_path

    @property
    def size(self) -> int:
        return len(self._index)

    @property
    def last_refresh_ts(self) -> float:
        return self._last_refresh_ts

    def get(self, cip13: str) -> Optional[dict]:
        return self._index.get(cip13)

    def warm_start_or_build(self, base_url: str = DEFAULT_BASE_URL, progress: Optional[Callable[[int,int],None]] = None) -> None:
        if self._cache_path:
            disk = load_index_from_disk(self._cache_path)
            if isinstance(disk, dict) and disk:
                self._index = disk
                self._last_refresh_ts = time.time()
                return
        # sinon on construit
        self.refresh(base_url=base_url, progress=progress)

    def refresh(self, base_url: str = DEFAULT_BASE_URL, progress: Optional[Callable[[int,int],None]] = None) -> None:
        new_index = build_cip_index_from_api(base_url=base_url, progress=progress)
        if new_index:
            self._index = new_index  # hot-swap
            self._last_refresh_ts = time.time()
            if self._cache_path:
                try:
                    save_index_to_disk(new_index, self._cache_path)
                except Exception:
                    pass
