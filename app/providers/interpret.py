"""
Vigia entiende lo que el vecino dice (por voz o texto) y lo convierte en datos.

    interpret_report : "robaron un celular en la Avenida Santander"
                       -> tipo de incidente, titulo, descripcion y lugar.
    interpret_route  : "estoy en la universidad, necesito ir al centro"
                       -> barrio de origen y de destino.

Con Gemini (u otro LLM) se le pide un JSON. Si no hay modelo o falla, se
responde con reglas (clasificador + busqueda de nombres de barrio en el texto),
asi el vecino nunca se queda sin respuesta.
"""

import json
import logging
import re
import unicodedata
from typing import Optional, Tuple

from .summary import LLMProvider, SummaryProvider, provider_label

log = logging.getLogger("alertabarrio.ai.interpret")

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def parse_json(text: str) -> dict:
    """Saca el primer objeto JSON de la respuesta del modelo (a veces viene con ```json)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("El modelo no devolvio JSON")
    return json.loads(match.group(0))


def find_place(text: str, places: list[dict]) -> Tuple[Optional[int], int]:
    """Busca el nombre de un barrio dentro del texto. Devuelve (id, posicion)."""
    norm = normalize(text)
    best: Tuple[Optional[int], int, int] = (None, -1, 0)
    for p in places:
        name = normalize(p["name"])
        pos = norm.find(name)
        # nos quedamos con el nombre mas largo encontrado ("San Juan de Dios" antes que "San Juan")
        if pos >= 0 and len(name) > best[2]:
            best = (p["id"], pos, len(name))
    return best[0], best[1]


def _place_text(text: str) -> Optional[str]:
    m = re.search(r"\b(?:en|por)\s+(?:el\s+|la\s+|los\s+|las\s+|barrio\s+)?([^,.;!?]{3,60})", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _title(type_name: str, place: Optional[str]) -> str:
    title = type_name.capitalize()
    if place:
        title += f" en {place}"
    return title[:150]


# ------------------------------------------------------------------ reportes
def _report_prompt(data: dict) -> str:
    types = "\n".join(f"- {t['code']}: {t['name']}" for t in data["incident_types"])
    places = ", ".join(p["name"] for p in data["neighborhoods"])
    return "\n".join(
        [
            f"Eres Vigia, el asistente de seguridad de AlertaBarrio en {data.get('city') or 'la ciudad'}, Colombia.",
            "Un vecino te cuenta un incidente (puede venir de un audio, con errores de transcripcion).",
            "Conviertelo en un reporte. Responde SOLO un JSON con estas claves:",
            '{"incident_type_code": "<uno de los codigos de abajo>", "title": "<titulo corto, max 80 letras>",',
            ' "description": "<lo que paso, en tercera persona, claro, max 300 letras>",',
            ' "place": "<lugar que menciona (calle, avenida, sitio) o null>",',
            ' "neighborhood": "<barrio de la lista que corresponde a ese lugar, o null si no se sabe>"}',
            "No inventes detalles que el vecino no dijo.",
            "",
            "Tipos de incidente:",
            types,
            f"Barrios de la app: {places}",
            "",
            f"Lo que dijo el vecino: {data['text']}",
        ]
    )


def _report_rules(classifier, data: dict) -> dict:
    text = data["text"].strip()
    names = {t["code"]: t["name"] for t in data["incident_types"]}
    predictions = [(c, p) for c, p in classifier.predict(text) if c in names] or [(data["incident_types"][0]["code"], 0.0)]
    code, confidence = predictions[0]
    place = _place_text(text)
    hood_id, _ = find_place(text, data["neighborhoods"])
    description = text[0].upper() + text[1:] if text else text
    return {
        "incident_type_code": code,
        "confidence": round(confidence, 3),
        "title": _title(names[code], place),
        "description": description,
        "place": place,
        "neighborhood_id": hood_id,
    }


def interpret_report(provider: SummaryProvider, classifier, data: dict) -> dict:
    rules = _report_rules(classifier, data)
    if not isinstance(provider, LLMProvider):
        return {**rules, "model": f"reglas ({classifier.name})"}
    try:
        raw = parse_json(provider.generate(_report_prompt(data)))
        codes = {t["code"] for t in data["incident_types"]}
        code = raw.get("incident_type_code")
        hood_id = None
        if raw.get("neighborhood"):
            hood_id, _ = find_place(str(raw["neighborhood"]), data["neighborhoods"])
        place = raw.get("place") or rules["place"]
        return {
            "incident_type_code": code if code in codes else rules["incident_type_code"],
            "confidence": 0.9 if code in codes else rules["confidence"],
            "title": (raw.get("title") or rules["title"])[:150],
            "description": (raw.get("description") or rules["description"])[:2000],
            "place": place if place and str(place).lower() != "null" else None,
            "neighborhood_id": hood_id or rules["neighborhood_id"],
            "model": provider_label(provider),
        }
    except Exception as exc:
        log.warning("Interpretar reporte con %s fallo (%s); usando reglas", provider.name, exc)
        return {**rules, "model": f"reglas (respaldo de {provider_label(provider)})"}


# ------------------------------------------------------------------ rutas
def _route_prompt(data: dict) -> str:
    places = ", ".join(p["name"] for p in data["neighborhoods"])
    return "\n".join(
        [
            f"Eres Vigia, el asistente de seguridad de AlertaBarrio en {data.get('city') or 'la ciudad'}, Colombia.",
            "El vecino pide una ruta segura (puede venir de un audio). Conoces la ciudad: relaciona cada lugar",
            "que nombre (universidad, centro comercial, parque, avenida) con el barrio de la lista mas cercano.",
            'Responde SOLO un JSON: {"origin": "<barrio de la lista o null>", "destination": "<barrio de la lista o null>"}.',
            'Si el vecino sale de donde esta ("estoy aqui", "desde mi ubicacion") o no dice el origen, origin es null.',
            "",
            f"Barrios de la app: {places}",
            "",
            f"Lo que dijo el vecino: {data['text']}",
        ]
    )


DEST_BEFORE = re.compile(r"\b(hasta|hacia|para|ir a|ir al|llegar a|llegar al|voy a|voy al)\s+(el\s+|la\s+|barrio\s+)?$")
ORIGIN_BEFORE = re.compile(r"\b(desde|estoy en|salgo de|salgo del)\s+(el\s+|la\s+|barrio\s+)?$")


def _route_rules(data: dict) -> dict:
    text = normalize(data["text"])
    found = sorted((text.find(normalize(p["name"])), p["id"]) for p in data["neighborhoods"] if normalize(p["name"]) in text)
    origin = destination = None
    for pos, pid in found:
        before = text[:pos]
        if DEST_BEFORE.search(before):
            destination = pid
        elif ORIGIN_BEFORE.search(before):
            origin = pid
    rest = [pid for _, pid in found if pid not in (origin, destination)]
    if destination is None and rest:
        destination = rest.pop()
    if origin is None and rest:
        origin = rest.pop(0)
    return {"origin_id": origin, "destination_id": destination}


def interpret_route(provider: SummaryProvider, data: dict) -> dict:
    rules = _route_rules(data)
    if not isinstance(provider, LLMProvider):
        return {**rules, "model": "reglas"}
    try:
        raw = parse_json(provider.generate(_route_prompt(data)))
        origin = find_place(str(raw["origin"]), data["neighborhoods"])[0] if raw.get("origin") else None
        destination = find_place(str(raw["destination"]), data["neighborhoods"])[0] if raw.get("destination") else None
        return {
            "origin_id": origin,
            "destination_id": destination or rules["destination_id"],
            "model": provider_label(provider),
        }
    except Exception as exc:
        log.warning("Interpretar ruta con %s fallo (%s); usando reglas", provider.name, exc)
        return {**rules, "model": f"reglas (respaldo de {provider_label(provider)})"}
