# resolver.py
# Utilitaires "purs" pour traiter une chaîne GS1/Datamatrix et extraire les infos utiles.
# Aucune dépendance externe : uniquement la stdlib.

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

# --- Constantes & regex

# AI(01) = GTIN-14 : 14 chiffres
_AI01_RE = re.compile(r'01(\d{14})')

# FNC1 (ASCII 29) peut apparaître entre des AIs : on le remplace par un séparateur visuel.
_FNC1 = "\x1d"

# Pour extraire un "conditionnement" depuis un libellé de présentation (texte ANSM/API)
# Exemples: "Boîte de 30 comprimés", "plaquettes de 28 gélules", "flacon 20 mL", "(16)" en fin de libellé…
_COND_RE_NUMBER_PARENS_END = re.compile(r'\((\d+)\)\s*$', re.IGNORECASE)
_COND_RE_NUMBER_AND_UNIT = re.compile(
    r'(\d+)\s*(comprim(?:é|e|es|és)?|gélule(?:s)?|capsule(?:s)?|sachet(?:s)?|unidose(?:s)?|ampoule(?:s)?|suppositoire(?:s)?|ml|mL|unit(?:é|e|és|es)?)',
    re.IGNORECASE
)

# Pour identifier un dosage "lisible" si jamais on doit le dériver d’un texte brut.
# (En pratique on privilégiera la composition structurée renvoyée par l’API.)
_DOSAGE_INLINE_RE = re.compile(
    r'(\d+(?:[.,]\d+)?)\s*(µg|mcg|microgrammes?|mg|g|kg|ui|u.i\.?|iu|mL|ml|L)\b',
    re.IGNORECASE
)


# --- Normalisation & extraction GS1 / CIP13

def normalize_gs1(raw: str) -> str:
    """
    Nettoie la chaîne GS1/Datamatrix :
    - remplace FNC1 (ASCII 29) par '|'
    - supprime espaces et retours chariot
    """
    if not isinstance(raw, str):
        raw = str(raw)
    s = raw.replace(_FNC1, '|')
    s = s.replace(' ', '').replace('\n', '').replace('\r', '')
    return s


def extract_gtin14(gs1: str) -> Optional[str]:
    """
    Trouve l'AI(01) = GTIN-14 dans une chaîne GS1 normalisée ou brute.
    Retourne le GTIN-14 (14 chiffres) ou None.
    """
    s = normalize_gs1(gs1)
    m = _AI01_RE.search(s)
    return m.group(1) if m else None


def gtin14_to_cip13(gtin14: Optional[str]) -> Optional[str]:
    """
    Règle FR (NTIN/CIP) :
    - si GTIN-14 commence par '03400', alors CIP13 = GTIN-14 sans le premier '0'
      (le CIP13 résultant commence par '3400' et contient 13 chiffres).
    - sinon, on ne peut pas directement déduire un CIP13 -> retourne None.
    """
    if not gtin14:
        return None
    if len(gtin14) == 14 and gtin14.startswith('03400'):
        return gtin14[1:]
    return None


def parse_datamatrix_to_cip13(gs1: str) -> Optional[str]:
    """
    Helper tout-en-un :
    - Extrait le GTIN-14 via AI(01)
    - Applique la règle NTIN FR pour obtenir un CIP13
    """
    gtin = extract_gtin14(gs1)
    return gtin14_to_cip13(gtin)


# --- Extraction "métier" depuis les champs API (libellé / composition)

def extract_conditionnement(libelle: Optional[str]) -> Dict[str, Optional[Any]]:
    """
    Essaye d'extraire un 'nombre d'unités' + 'unité' depuis un libellé de présentation.
    Renvoie un dict: {"valeur": int|None, "unite": str|None}
    Stratégie:
      1) Nombre entre parenthèses en fin de chaîne: "... (16)"
      2) Motif "<nombre> <unité>" (comprimé(s), gélule(s), ml/mL, etc.)
      3) Fallback -> None
    """
    if not libelle:
        return {"valeur": None, "unite": None}

    # 1) (16) en fin de libellé
    m = _COND_RE_NUMBER_PARENS_END.search(libelle)
    if m:
        try:
            return {"valeur": int(m.group(1)), "unite": "unité(s)"}
        except ValueError:
            pass

    # 2) "<nombre> <unité>"
    m = _COND_RE_NUMBER_AND_UNIT.search(libelle)
    if m:
        try:
            val = int(m.group(1))
        except ValueError:
            val = None
        unit = (m.group(2) or "").lower()
        # Normalise quelques variantes pour homogénéiser l'affichage
        unit = unit.replace('ml', 'mL')
        unit = 'comprimés' if unit.startswith('comprim') else unit
        unit = 'gélules' if unit.startswith('gélule') else unit
        unit = 'capsules' if unit.startswith('capsule') else unit
        unit = 'sachets' if unit.startswith('sachet') else unit
        unit = 'ampoules' if unit.startswith('ampoule') else unit
        unit = 'suppositoires' if unit.startswith('suppositoire') else unit
        unit = 'unités' if unit.startswith('unit') else unit
        return {"valeur": val, "unite": unit}

    # 3) Rien trouvé
    return {"valeur": None, "unite": None}


def extract_dosage_from_compo(composition: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """
    Construit un dosage lisible à partir de la composition structurée de l'API.
    Exemple de composition (typique):
      [
        {"denominationSubstance": "Levothyroxine sodique", "dosage": "100 µg"},
        {"denominationSubstance": "Excipient", "dosage": "..."}  # on peut filtrer si besoin
      ]
    Stratégie:
      - concatène "Substance Dosage" pour chaque entrée où les 2 existent
      - joint avec ' + '
      - si rien d'exploitable, retourne None
    """
    if not composition:
        return None

    parts: List[str] = []
    for c in composition:
        sub = (c.get("denominationSubstance")
               or c.get("substance")
               or "").strip()
        dos = (c.get("dosage") or "").strip()
        if sub and dos:
            parts.append(f"{sub} {dos}")

    return " + ".join(parts) if parts else None


def fallback_dosage_from_text(*texts: Optional[str]) -> Optional[str]:
    """
    En dernier recours (rare), tente d'extraire un dosage "XX unité" depuis
    un texte libre (nom commercial, libellé…).
    """
    for t in texts:
        if not t:
            continue
        m = _DOSAGE_INLINE_RE.search(t)
        if m:
            # Renvoie la sous-chaîne exacte repérée pour garder l'unité d'origine
            start, end = m.span()
            return t[start:end].strip()
    return None
