"""
Componente de IA de AlertaBarrio (servicio independiente del backend).

    POST /api/v1/analyze-report  clasifica el reporte y calcula riesgo de la zona
    POST /api/v1/heatmap         mapa de calor predictivo + zonas/horas calientes
    POST /api/v1/summary         resumen diario en lenguaje natural
    GET  /api/v1/models          modelos activos
    POST /api/v1/models/switch   cambiar de modelo en caliente

Arranque local: uvicorn app.main:app --reload --port 8001
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Header, HTTPException

from .config import get_settings
from .models.classifier import CLASSIFIERS
from .models.risk import BLOCKS_ES, DAYS_ES, HEATMAPS, bbox_for, parse_dt, temporal_matrix, zone_risk
from .providers.summary import build_provider, provider_label, summarize_with_fallback
from .schemas import AnalyzeIn, HeatmapIn, SummaryIn, SwitchIn

TYPE_NAMES = {
    "armed_robbery": "robo armado", "assault": "agresion o rina", "home_burglary": "robo a vivienda",
    "vehicle_theft": "robo de vehiculo", "theft": "hurto", "drug_dealing": "expendio de drogas",
    "vandalism": "vandalismo", "suspicious_activity": "actividad sospechosa",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="AlertaBarrio IA", version="1.0.0", description="Componente de inteligencia artificial de AlertaBarrio")


class Registry:
    """Modelos activos. Cambiar el 'switch' aqui no afecta al backend ni al frontend."""

    def __init__(self) -> None:
        self.settings = get_settings().model_copy()
        self.reload()

    def reload(self) -> None:
        s = self.settings
        if s.classifier_model not in CLASSIFIERS:
            raise ValueError(f"Clasificador desconocido: {s.classifier_model}")
        if s.heatmap_model not in HEATMAPS:
            raise ValueError(f"Modelo de mapa de calor desconocido: {s.heatmap_model}")
        self.classifier = CLASSIFIERS[s.classifier_model]()
        self.heatmap = HEATMAPS[s.heatmap_model]()
        self.summary = build_provider(s)

    def describe(self) -> dict:
        return {
            "classifier": self.classifier.name,
            "heatmap": self.heatmap.name,
            "summary": provider_label(self.summary),
            "available": {
                "classifier": list(CLASSIFIERS),
                "heatmap": list(HEATMAPS),
                "summary": ["template", "ollama", "openai_compatible", "anthropic"],
            },
        }


registry = Registry()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/hello")
def hello():
    return {"message": "Hola mundo desde el componente de IA de AlertaBarrio", "models": registry.describe()}


@app.get("/api/v1/models")
def models():
    return registry.describe()


@app.post("/api/v1/models/switch")
def switch(data: SwitchIn, x_admin_token: Optional[str] = Header(None)):
    expected = get_settings().admin_token
    if expected and x_admin_token != expected:
        raise HTTPException(401, "Token de administrador invalido")
    previous = registry.settings.model_copy()
    for field, value in data.model_dump(exclude_none=True).items():
        setattr(registry.settings, field, value)
    try:
        registry.reload()
    except ValueError as exc:
        registry.settings = previous
        registry.reload()
        raise HTTPException(400, str(exc))
    return registry.describe()


@app.post("/api/v1/analyze-report")
def analyze_report(data: AnalyzeIn):
    predictions = registry.classifier.predict(data.description)
    best_code, confidence = predictions[0]
    occurred = parse_dt(data.occurred_at)
    risk, explanation = zone_risk(
        [h.model_dump() for h in data.history], occurred, now_utc(), registry.settings.risk_half_life_days
    )
    return {
        "model": f"{registry.classifier.name} + temporal_risk",
        "suggested_type_code": best_code,
        "confidence": round(confidence, 3),
        "top_predictions": [{"code": c, "probability": round(p, 3)} for c, p in predictions[:3]],
        "matches_user_choice": best_code == data.incident_type_code,
        "zone_risk": risk,
        "risk_explanation": explanation,
    }


@app.post("/api/v1/heatmap")
def heatmap(data: HeatmapIn):
    reports = [r.model_dump() for r in data.reports]
    neighborhoods = [n.model_dump() for n in data.neighborhoods]
    names = {n["id"]: n["name"] for n in neighborhoods}
    now = now_utc()
    half_life = registry.settings.risk_half_life_days

    bbox = bbox_for(reports, neighborhoods)
    cells = registry.heatmap.cells(reports, bbox, now, half_life) if bbox and reports else []

    matrix = temporal_matrix(reports, now, half_life)
    peak = max(matrix.values(), default=0) or 1.0
    hotspots = sorted(
        (
            {"neighborhood_id": nid, "day_of_week": d, "hour_block": b, "risk_score": round(v / peak, 3)}
            for (nid, d, b), v in matrix.items()
            if v / peak >= 0.15
        ),
        key=lambda h: -h["risk_score"],
    )

    by_zone = defaultdict(float)
    for (nid, _, _), v in matrix.items():
        by_zone[nid] += v
    zone_peak = max(by_zone.values(), default=0) or 1.0
    neighborhood_risk = sorted(
        ({"neighborhood_id": nid, "risk_score": round(v / zone_peak, 3)} for nid, v in by_zone.items()),
        key=lambda x: -x["risk_score"],
    )

    insights = []
    for h in hotspots[:3]:
        insights.append(
            f"Pico de riesgo en {names.get(h['neighborhood_id'], 'zona ' + str(h['neighborhood_id']))} los "
            f"{DAYS_ES[h['day_of_week']]} {BLOCKS_ES[h['hour_block']]} (indice {round(h['risk_score'] * 100)}%)."
        )
    if reports:
        common = Counter(r["incident_type_code"] for r in reports).most_common(1)[0]
        hours = Counter(parse_dt(r["occurred_at"]).hour for r in reports).most_common(1)[0][0]
        insights.append(f"Incidente mas frecuente: {TYPE_NAMES.get(common[0], common[0])} ({common[1]} casos). Hora con mas reportes: {hours}:00.")

    return {
        "model": f"{registry.heatmap.name} + temporal_decay",
        "cells": cells,
        "hotspots": hotspots,
        "neighborhood_risk": neighborhood_risk,
        "insights": insights,
    }


@app.post("/api/v1/summary")
def summary(data: SummaryIn):
    text, model = summarize_with_fallback(registry.summary, data.model_dump())
    return {"model": model, "summary": text}
