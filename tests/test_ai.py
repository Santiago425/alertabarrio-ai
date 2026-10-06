from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.models.classifier import KeywordClassifier, NaiveBayesClassifier, tokenize

client = TestClient(app)


def _iso(days_ago: float, hour: int, weekday_target: int | None = None) -> str:
    tz = timezone(timedelta(hours=-5))
    d = datetime.now(tz) - timedelta(days=days_ago)
    if weekday_target is not None:
        while d.weekday() != weekday_target:
            d -= timedelta(days=1)
    return d.replace(hour=hour, minute=0, second=0, microsecond=0).isoformat()


def test_tokenize_removes_accents_and_plurals():
    assert tokenize("Robaron CELULARES en el parqué") == ["robaron", "celular", "parque"]


def test_naive_bayes_classifies():
    nb = NaiveBayesClassifier()
    assert nb.predict("me atracaron con una pistola")[0][0] == "armed_robbery"
    assert nb.predict("venden droga en el parque")[0][0] == "drug_dealing"
    assert nb.predict("se metieron a la casa por la ventana")[0][0] == "home_burglary"
    probs = nb.predict("raponazo de celular")
    assert abs(sum(p for _, p in probs) - 1) < 1e-6


def test_keyword_model_also_works():
    assert KeywordClassifier().predict("hombre sospechoso rondando")[0][0] == "suspicious_activity"


def test_analyze_report_risk_higher_in_pattern_slot():
    history = [{"occurred_at": _iso(7 * k, 21, weekday_target=3), "severity": 5} for k in range(1, 8)]
    history += [{"occurred_at": _iso(3, 9), "severity": 2}]
    thursday_night = _iso(0, 21, weekday_target=3)
    monday_morning = _iso(0, 9, weekday_target=0)
    hot = client.post("/api/v1/analyze-report", json={"description": "robo con cuchillo", "occurred_at": thursday_night, "history": history}).json()
    cold = client.post("/api/v1/analyze-report", json={"description": "robo con cuchillo", "occurred_at": monday_morning, "history": history}).json()
    assert hot["zone_risk"] > cold["zone_risk"]
    assert hot["suggested_type_code"] == "armed_robbery"


def test_heatmap_finds_hotspot():
    reports = [
        {"latitude": 4.65, "longitude": -74.06, "occurred_at": _iso(7 * k, 22, weekday_target=4), "severity": 5, "neighborhood_id": 1, "incident_type_code": "armed_robbery"}
        for k in range(1, 6)
    ] + [{"latitude": 4.60, "longitude": -74.15, "occurred_at": _iso(20, 10), "severity": 2, "neighborhood_id": 2, "incident_type_code": "vandalism"}]
    hoods = [{"id": 1, "name": "Chapinero", "latitude": 4.65, "longitude": -74.06}, {"id": 2, "name": "Kennedy", "latitude": 4.6, "longitude": -74.15}]
    for model in ("kde", "frequency"):
        client.post("/api/v1/models/switch", json={"heatmap_model": model})
        body = client.post("/api/v1/heatmap", json={"reports": reports, "neighborhoods": hoods}).json()
        assert body["model"].startswith(model)
        assert body["hotspots"][0] == {"neighborhood_id": 1, "day_of_week": 4, "hour_block": 3, "risk_score": 1.0}
        assert max(c["intensity"] for c in body["cells"]) == 1.0
    client.post("/api/v1/models/switch", json={"heatmap_model": "kde"})


def test_summary_template_and_fallback():
    payload = {
        "neighborhood_name": "Chapinero",
        "date": "2026-10-04",
        "reports": [{"title": "Atraco", "description": "con cuchillo", "incident_type": "Robo armado", "severity": 5, "status": "published", "occurred_at": "2026-10-04T21:10:00-05:00"}],
        "week_reports": [],
        "hotspots": [{"day_of_week": 3, "hour_block": 3, "risk_score": 1.0}],
    }
    body = client.post("/api/v1/summary", json=payload).json()
    assert body["model"] == "template" and "Chapinero" in body["summary"] and "jueves" in body["summary"]
    # Un proveedor externo caido no rompe el servicio: responde con la plantilla
    client.post("/api/v1/models/switch", json={"summary_provider": "ollama", "llm_base_url": "http://127.0.0.1:9"})
    body = client.post("/api/v1/summary", json=payload).json()
    assert body["model"].startswith("template (respaldo de ollama")
    client.post("/api/v1/models/switch", json={"summary_provider": "template"})


def test_switch_rejects_unknown_model():
    assert client.post("/api/v1/models/switch", json={"classifier_model": "gpt-99"}).status_code == 400
    assert client.get("/api/v1/models").json()["classifier"] == "naive_bayes"
