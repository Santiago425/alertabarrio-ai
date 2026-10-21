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
        {"latitude": 1.2136, "longitude": -77.2811, "occurred_at": _iso(7 * k, 22, weekday_target=4), "severity": 5, "neighborhood_id": 1, "incident_type_code": "armed_robbery"}
        for k in range(1, 6)
    ] + [{"latitude": 1.1968, "longitude": -77.2728, "occurred_at": _iso(20, 10), "severity": 2, "neighborhood_id": 2, "incident_type_code": "vandalism"}]
    hoods = [{"id": 1, "name": "Centro", "latitude": 1.2136, "longitude": -77.2811}, {"id": 2, "name": "Lorenzo de Aldana", "latitude": 1.1968, "longitude": -77.2728}]
    for model in ("kde", "frequency"):
        client.post("/api/v1/models/switch", json={"heatmap_model": model})
        body = client.post("/api/v1/heatmap", json={"reports": reports, "neighborhoods": hoods}).json()
        assert body["model"].startswith(model)
        assert body["hotspots"][0] == {"neighborhood_id": 1, "day_of_week": 4, "hour_block": 3, "risk_score": 1.0}
        assert max(c["intensity"] for c in body["cells"]) == 1.0
    client.post("/api/v1/models/switch", json={"heatmap_model": "kde"})


def test_summary_template_and_fallback():
    payload = {
        "neighborhood_name": "Centro",
        "date": "2026-10-04",
        "reports": [{"title": "Atraco", "description": "con cuchillo", "incident_type": "Robo armado", "severity": 5, "status": "published", "occurred_at": "2026-10-04T21:10:00-05:00"}],
        "week_reports": [],
        "hotspots": [{"day_of_week": 3, "hour_block": 3, "risk_score": 1.0}],
    }
    body = client.post("/api/v1/summary", json=payload).json()
    assert body["model"] == "template" and "Centro" in body["summary"] and "jueves" in body["summary"]
    # Un proveedor externo caido no rompe el servicio: responde con la plantilla
    client.post("/api/v1/models/switch", json={"summary_provider": "ollama", "llm_base_url": "http://127.0.0.1:9"})
    body = client.post("/api/v1/summary", json=payload).json()
    assert body["model"].startswith("template (respaldo de ollama")
    client.post("/api/v1/models/switch", json={"summary_provider": "template"})


def test_switch_rejects_unknown_model():
    assert client.post("/api/v1/models/switch", json={"classifier_model": "gpt-99"}).status_code == 400
    assert client.get("/api/v1/models").json()["classifier"] == "naive_bayes"


def test_kde_bandwidth_adapts_to_city_size():
    from app.models.risk import KDEHeatmap

    kde = KDEHeatmap()
    pasto = kde.bandwidth((1.19, -77.30, 1.24, -77.25))  # ~5 km
    bogota = kde.bandwidth((4.55, -74.22, 4.78, -74.00))  # ~25 km
    assert pasto < bogota and 0.0012 <= pasto <= 0.003
    assert KDEHeatmap(bandwidth_deg=0.005).bandwidth((0, 0, 1, 1)) == 0.005


def test_llm_prompt_mentions_city():
    from app.providers.summary import _prompt

    p = _prompt({"neighborhood_name": "Centro", "city": "Pasto", "date": "2026-10-20"})
    assert "Centro en Pasto" in p


CHAT_CONTEXT = {
    "city": "Pasto",
    "neighborhood": "Centro",
    "active_alerts": 40,
    "ranking": [
        {"neighborhood": "Centro", "active_alerts": 20, "active_severity": 70},
        {"neighborhood": "Potrerillo", "active_alerts": 10, "active_severity": 30},
    ],
    "hotspots": [{"neighborhood": "Centro", "day_of_week": 3, "hour_block": 3}],
    "recent": [{"incident_type": "Hurto", "neighborhood": "Centro", "when": "hace 1 h", "title": "Raponazo"}],
}


def test_chat_without_llm_uses_rules():
    body = client.post("/api/v1/chat", json={"question": "¿A qué hora es más peligroso?", "context": CHAT_CONTEXT}).json()
    assert body["model"] == "reglas" and "jueves" in body["answer"]


def test_chat_with_gemini(monkeypatch):
    from app.providers import summary as prov

    calls = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "Evita el Centro los jueves en la noche."}]}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.update(url=url, headers=headers, prompt=json["contents"][0]["parts"][0]["text"])
        return Resp()

    monkeypatch.setattr(prov.httpx, "post", fake_post)
    client.post("/api/v1/models/switch", json={"summary_provider": "gemini", "llm_api_key": "test-key"})
    try:
        body = client.post("/api/v1/chat", json={"question": "¿Es seguro el Centro?", "context": CHAT_CONTEXT}).json()
        assert body == {"model": "gemini:gemini-3.1-flash-lite", "answer": "Evita el Centro los jueves en la noche."}
        assert calls["url"].endswith("/models/gemini-3.1-flash-lite:generateContent")
        assert calls["headers"] == {"x-goog-api-key": "test-key"}
        assert "Pasto" in calls["prompt"] and "¿Es seguro el Centro?" in calls["prompt"]
    finally:
        client.post("/api/v1/models/switch", json={"summary_provider": "template", "llm_api_key": ""})


def test_gemini_down_falls_back_to_rules():
    client.post("/api/v1/models/switch", json={"summary_provider": "gemini", "llm_base_url": ""})
    from app.main import registry

    registry.summary.base = "http://127.0.0.1:9"
    try:
        body = client.post("/api/v1/chat", json={"question": "¿Qué barrio evitar?", "context": CHAT_CONTEXT}).json()
        assert body["model"].startswith("reglas (respaldo de gemini") and "Centro" in body["answer"]
    finally:
        client.post("/api/v1/models/switch", json={"summary_provider": "template"})


def test_auto_provider_picks_gemini_only_with_key():
    from app.config import Settings
    from app.providers.summary import GeminiProvider, TemplateProvider, build_provider

    assert isinstance(build_provider(Settings(summary_provider="auto", gemini_api_key=None)), TemplateProvider)
    assert isinstance(build_provider(Settings(summary_provider="auto", gemini_api_key="k")), GeminiProvider)


def test_chat_victim_without_llm_gives_steps():
    body = client.post(
        "/api/v1/chat",
        json={"question": "me robaron en bosques de la colina 2", "context": {**CHAT_CONTEXT, "neighborhoods": ["Centro", "Potrerillo"]}},
    ).json()
    assert body["model"] == "reglas"
    assert "bosques de la colina 2" in body["answer"] and "123" in body["answer"] and "Reportar" in body["answer"]


def test_chat_prompt_knows_pasto():
    from app.providers.summary import _chat_prompt

    prompt = _chat_prompt({"question": "me robaron en Bosques de la Colina 2", "context": {**CHAT_CONTEXT, "neighborhoods": ["Centro", "Potrerillo"]}})
    assert "geografia de Pasto" in prompt and "Centro, Potrerillo" in prompt and "123" in prompt
