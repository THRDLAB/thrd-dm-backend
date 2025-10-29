# index_builder.py — construction & gestion de l’index médicaments (BDPM)
# - Throttle & retry/backoff (429/5xx)
# - Build par tranches (INDEX_PAGES_PER_RUN)
# - Reprise sur crash via fichier de progression
# - Cache disque cip_index.json
# - CipIndexManager pour usage direct dans app.py

from __future__ import annotations
import os, time, json
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# =========================
# Paramètres via variables d'env (avec défauts safe)
# =========================
INDEX_MAX_PAGES        = int(os.getenv("INDEX_MAX_PAGES", "0"))            # 0 = pas de borne (toutes les pages)
INDEX_PAGES_PER_RUN    = int(os.getenv("INDEX_PAGES_PER_RUN", "250"))      # pages max par build/passe (éviter rate limit)
INDEX_RATE_LIMIT_QPS   = float(os.getenv("INDEX_RATE_LIMIT_QPS", "2.0"))   # requêtes/s (2 par défaut)
INDEX_RETRY_MAX        = int(os.getenv("INDEX_RETRY_MAX", "6"))            # nb max de retries
INDEX_BACKOFF_BASE_MS  = float(os.getenv("INDEX_BACKOFF_BASE_MS", "500"))  # base backoff en ms
INDEX_PROGRESS_PATH    = os.getenv("INDEX_PROGRESS_PATH", "")              # ex. /data/cip_index.progress (facultatif)

# =========================
# Utilitaires HTTP
# =========================

def _throttle(last_call_ts: List[float]) -> None:
    """Respecte INDEX_RATE_LIMIT_QPS en espaçant les requêtes HTTP."""
    qps = max(0.1, INDEX_RATE_LIMIT_QPS)
    min_interval = 1.0 / qps
    now = time.perf_counter()
    if last_call_ts and (now - last_call_ts[0]) < min_interval:
        time.sleep(min_interval - (now - last_call_ts[0]))
    last_call_ts[:] = [time.perf_counter()]

def _api_get_json(url: str) -> Dict[str, Any]:
    """
    GET JSON avec:
    - throttle (QPS)
    - retry/backoff sur 429/5xx (respecte Retry-After si fourni)
    """
    if not hasattr(_api_get_json, "_last"):
        _api_get_json._last = []  # type: ignore[attr-defined]
    last_call_ts: List[float] = getattr(_api_get_json, "_last")  # type: ignore[attr-defined]
    _throttle(last_call_ts)

    attempt = 0
    while True:
        attempt += 1
        try:
            req = Request(url, headers={
                "User-Agent": "dm-index-builder/1.0",
                "Accept": "application/json",
            })
            with urlopen(req, timeout=20) as r:
                body = r.read().decode("utf-8", errors="ignore")
                try:
                    return json.loads(body)
                except Exception:
                    # Certaines pages pourraient ne pas être strictement JSON → lève une erreur claire
                    raise ValueError(f"Invalid JSON from {url!r} (len={len(body)})")
        except HTTPError as e:
            # Respecte Retry-After si présent
            retry_after_s: Optional[float] = None
            try:
                if e.headers and "Retry-After" in e.headers:
                    retry_after_s = float(e.headers.get("Retry-After"))
            except Exception:
                retry_after_s = None

            if e.code in (429, 500, 502, 503, 504) and attempt <= INDEX_RETRY_MAX:
                sleep_s = retry_after_s if retry_after_s is not None else (INDEX_BACKOFF_BASE_MS / 1000.0) * (2 ** (attempt - 1))
                time.sleep(min(sleep_s, 30.0))
                continue
            raise
        except URLError as e:
            # réseau capricieux → retry avec backoff
            if attempt <= INDEX_RETRY_MAX:
                sleep_s = (INDEX_BACKOFF_BASE_MS / 1000.0) * (2 ** (attempt - 1))
                time.sleep(min(sleep_s, 30.0))
                continue
            raise

# =========================
# Build & I/O index
# =========================

def _normalize_item(med: Dict[str, Any], pres: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalise une entrée médicament + présentation BDPM en notre format index."""
    cip13 = str(pres.get("cip13") or "").strip()
    if not cip13:
        return None
    nom = (med.get("elementPharmaceutique") or med.get("denomination") or med.get("nom") or "").strip()
    forme = (med.get("formePharmaceutique") or med.get("forme") or "").strip()
    compo = med.get("composition") or []
    libelle = " ".join(str(pres.get("libelle") or "").split())
    return {
        "cip13": cip13,
        "nom": nom,
        "forme": forme.capitalize() if forme else "",
        "composition": [
            {
                "denominationSubstance": (c.get("denominationSubstance") or c.get("substance") or "").strip(),
                "dosage": (c.get("dosage") or "").strip()
            }
            for c in compo
        ],
        "libelle": libelle,
    }

def _max_page(base_url: str) -> int:
    """Tente de lire la page d'index pour découvrir le nombre maximum de pages."""
    try:
        idx = _api_get_json(f"{base_url.rstrip('/')}/database/index")
        mp = int(idx.get("maxPage") or 0) if isinstance(idx, dict) else 0
        return mp if mp > 0 else 999_999
    except Exception:
        return 999_999

def _load_progress(default_start: int = 1) -> int:
    """Charge la dernière page traitée (progress) si INDEX_PROGRESS_PATH est défini."""
    if not INDEX_PROGRESS_PATH:
        return default_start
    try:
        with open(INDEX_PROGRESS_PATH, "r", encoding="utf-8") as f:
            js = json.load(f)
        return int(js.get("next_page") or default_start)
    except Exception:
        return default_start

def _save_progress(next_page: int) -> None:
    """Sauvegarde la progression (page suivante à traiter)."""
    if not INDEX_PROGRESS_PATH:
        return
    try:
        os.makedirs(os.path.dirname(INDEX_PROGRESS_PATH), exist_ok=True)
    except Exception:
        pass
    try:
        with open(INDEX_PROGRESS_PATH, "w", encoding="utf-8") as f:
            json.dump({"next_page": next_page, "ts": time.time()}, f)
    except Exception:
        pass

def build_cip_index_from_api(base_url: str, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Télécharge la base paginée de l'API médicaments et retourne une liste d'items normalisés.
    - Respecte INDEX_MAX_PAGES (>0) et/ou max_pages (param).
    - Limite à INDEX_PAGES_PER_RUN par passe.
    - Reprend à partir de INDEX_PROGRESS_PATH si défini.
    """
    items: List[Dict[str, Any]] = []

    hard_cap_env = INDEX_MAX_PAGES if INDEX_MAX_PAGES > 0 else None
    hard_cap = max_pages if max_pages is not None else hard_cap_env

    max_page_api = _max_page(base_url)
    if hard_cap is None or hard_cap <= 0:
        hard_cap = max_page_api

    # Reprise si progress
    start_page = _load_progress(default_start=1)
    page = max(1, start_page)
    fetched = 0
    per_run = max(1, INDEX_PAGES_PER_RUN)

    while page <= hard_cap:
        if fetched >= per_run:
            break  # tranche terminée
        js = _api_get_json(f"{base_url.rstrip('/')}/database/{page}")
        data = js.get("data", []) if isinstance(js, dict) else []
        if not data:
            # fin des pages
            _save_progress(page)  # on mémorise quand même
            break

        for med in data:
            pres_list = med.get("presentation") or []
            for pres in pres_list:
                it = _normalize_item(med, pres)
                if it:
                    items.append(it)

        page += 1
        fetched += 1
        _save_progress(page)

        # borne de sécurité si l'API ne donne pas d'index clair
        if page > max_page_api:
            break

    return items

def save_index_to_disk(items: List[Dict[str, Any]], path: str) -> None:
    """Écrit l’index fusionné sur disque (JSON)."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)

def load_index_from_disk(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def merge_indexes(existing, new_items):
    """Fusion robuste : ignore ce qui n'est pas un dict avec 'cip13'."""
    base = {}
    for e in (existing or []):
        if isinstance(e, dict) and "cip13" in e:
            base[e["cip13"]] = e
    for it in (new_items or []):
        if isinstance(it, dict) and "cip13" in it:
            base[it["cip13"]] = it
    return list(base.values())

# =========================
# Manager mémoire + disque
# =========================

class CipIndexManager:
    """
    Petit gestionnaire d'index en mémoire + disque, utilisé par app.py :
    - warm_start_or_build(base_url)
    - refresh(base_url)
    - get(cip13) → dict|None
    - size (property)
    - last_refresh_ts (property)
    """
    def __init__(self, cache_path: str = "cip_index.json") -> None:
        self._cache_path = cache_path
        self._index: List[Dict[str, Any]] = []
        self._last_ts: float = 0.0

    # Properties
    @property
    def size(self) -> int:
        return len(self._index)

    @property
    def last_refresh_ts(self) -> float:
        return self._last_ts

    # Core ops
    def warm_start_or_build(self, base_url: str) -> None:
        """
        - Si un cache disque existe → charge immédiatement (démarrage instantané).
        - Sinon → construit une première tranche, écrit le cache, charge en mémoire.
        """
        try:
            if self._cache_path and os.path.exists(self._cache_path):
                self._index = load_index_from_disk(self._cache_path)
                self._last_ts = time.time()
                return
        except Exception:
            # continue sur un premier build si le cache est corrompu
            pass

        # Pas de cache → première tranche
        new_items = build_cip_index_from_api(base_url=base_url, max_pages=None)
        if new_items:
            save_index_to_disk(new_items, self._cache_path)
            self._index = new_items
            self._last_ts = time.time()

    def refresh(self, base_url: str, progress: Optional[Dict[str, Any]] = None) -> None:
        """
        Reconstruit (ou complète) l’index :
        - télécharge une tranche (INDEX_PAGES_PER_RUN),
        - fusionne avec l’existant,
        - écrit le cache,
        - met à jour l’index en mémoire.
        """
        try:
            new_items = build_cip_index_from_api(base_url=base_url, max_pages=None)
            if not new_items and self._index:
                # rien de nouveau (rate-limit temporaire, ou arrêt en fin d'index)
                self._last_ts = time.time()
                return

            existing = []
            if self._cache_path and os.path.exists(self._cache_path):
                try:
                    existing = load_index_from_disk(self._cache_path)
                except Exception:
                    existing = self._index or []

            merged = merge_indexes(existing, new_items)
            save_index_to_disk(merged, self._cache_path)
            self._index = merged
            self._last_ts = time.time()
        except Exception:
            # On ne casse pas l'app : on garde l'index actuel
            pass

    # Query
    def get(self, cip13: str) -> Optional[Dict[str, Any]]:
        """Retourne l’entrée pour un CIP13 exact, sinon None."""
        if not cip13:
            return None
        # petit dict d’accès rapide (peut être remplacé par un index hashmap si tu veux optimiser)
        for it in self._index:
            if it.get("cip13") == cip13:
                return it
        return None
