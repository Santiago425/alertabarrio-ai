"""
Proveedores para el RESUMEN DIARIO en lenguaje natural (SUMMARY_PROVIDER).

    template          : sin modelo externo; arma el texto con reglas. Siempre
                        funciona (ideal para la sustentacion, no se "duerme").
    ollama            : modelo open source local (ej. llama3.2:3b, qwen2.5:3b).
    openai_compatible : cualquier API compatible con OpenAI que sirva modelos
                        open source (Groq, OpenRouter, LM Studio, vLLM...).
    anthropic         : API de Claude.

Si un proveedor externo falla, se usa `template` como respaldo para que el
sistema nunca se quede sin respuesta.
"""

import logging
from collections import Counter
from typing import Protocol, Tuple

import httpx

from ..config import Settings
from ..models.risk import BLOCKS_ES, DAYS_ES, parse_dt

log = logging.getLogger("alertabarrio.ai.summary")


class SummaryProvider(Protocol):
    name: str

    def summarize(self, payload: dict) -> str:
        ...


class TemplateProvider:
    name = "template"

    def summarize(self, payload: dict) -> str:
        hood = payload["neighborhood_name"]
        today = payload.get("reports", [])
        week = payload.get("week_reports", [])
        hotspots = payload.get("hotspots", [])
        parts = []
        if not today:
            parts.append(f"Hoy no se han registrado reportes de seguridad en {hood}.")
        else:
            types = Counter(r["incident_type"] for r in today)
            listed = ", ".join(f"{n} de {t.lower()}" for t, n in types.most_common())
            worst = max(today, key=lambda r: r["severity"])
            hour = parse_dt(worst["occurred_at"]).strftime("%H:%M")
            parts.append(f"Hoy se registraron {len(today)} reportes en {hood}: {listed}.")
            parts.append(f"El caso mas grave fue \"{worst['title']}\" ({worst['incident_type'].lower()}) cerca de las {hour}.")
            verified = sum(1 for r in today if r["status"] == "verified")
            if verified == 1:
                parts.append("Uno de ellos ya fue confirmado por otros vecinos.")
            elif verified > 1:
                parts.append(f"{verified} de ellos ya fueron confirmados por otros vecinos.")
        if week:
            avg = len(week) / 7
            trend = "por encima" if len(today) > avg * 1.3 else "por debajo" if len(today) < avg * 0.7 else "dentro"
            parts.append(f"En los ultimos 7 dias hubo {len(week)} reportes (promedio {avg:.1f} por dia), asi que hoy esta {trend} de lo normal.")
        if hotspots:
            h = hotspots[0]
            parts.append(
                f"Segun el historial, el mayor riesgo en la zona es los {DAYS_ES[h['day_of_week']]} "
                f"{BLOCKS_ES[h['hour_block']]}: evite transitar solo y prefiera vias iluminadas en ese horario."
            )
        elif today:
            parts.append("Recomendacion: este atento, no exhiba objetos de valor y reporte cualquier situacion sospechosa.")
        return " ".join(parts)


def _prompt(payload: dict) -> str:
    lines = [
        f"Eres el asistente de seguridad del barrio {payload['neighborhood_name']} en Bogota.",
        f"Escribe en espanol un resumen breve (maximo 5 oraciones) de la actividad del {payload['date']}",
        "para los vecinos: que paso, que tan grave fue, comparacion con la semana y una recomendacion.",
        "No inventes datos que no esten abajo.",
        "",
        "Reportes de hoy:",
    ]
    for r in payload.get("reports", []) or []:
        lines.append(f"- {r['occurred_at'][11:16]} | {r['incident_type']} (gravedad {r['severity']}/5) | {r['title']}: {r['description']}")
    if not payload.get("reports"):
        lines.append("- (ninguno)")
    lines.append(f"Total de reportes en los ultimos 7 dias: {len(payload.get('week_reports', []))}")
    for h in payload.get("hotspots", [])[:2]:
        lines.append(f"Franja de mayor riesgo historico: {DAYS_ES[h['day_of_week']]} {BLOCKS_ES[h['hour_block']]}")
    return "\n".join(lines)


class OllamaProvider:
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.base = (settings.llm_base_url or "http://localhost:11434").rstrip("/")
        self.model = settings.summary_model or "llama3.2:3b"
        self.timeout = settings.llm_timeout_seconds

    def summarize(self, payload: dict) -> str:
        r = httpx.post(
            f"{self.base}/api/generate",
            json={"model": self.model, "prompt": _prompt(payload), "stream": False},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["response"].strip()


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(self, settings: Settings) -> None:
        self.base = (settings.llm_base_url or "https://api.groq.com/openai/v1").rstrip("/")
        self.model = settings.summary_model or "llama-3.1-8b-instant"
        self.key = settings.llm_api_key
        self.timeout = settings.llm_timeout_seconds

    def summarize(self, payload: dict) -> str:
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        r = httpx.post(
            f"{self.base}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": _prompt(payload)}],
                "temperature": 0.3,
                "max_tokens": 350,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.base = (settings.llm_base_url or "https://api.anthropic.com").rstrip("/")
        self.model = settings.summary_model or "claude-haiku-4-5"
        self.key = settings.llm_api_key or ""
        self.timeout = settings.llm_timeout_seconds

    def summarize(self, payload: dict) -> str:
        r = httpx.post(
            f"{self.base}/v1/messages",
            headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"},
            json={
                "model": self.model,
                "max_tokens": 400,
                "messages": [{"role": "user", "content": _prompt(payload)}],
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return "".join(block.get("text", "") for block in r.json()["content"]).strip()


def build_provider(settings: Settings) -> SummaryProvider:
    name = settings.summary_provider
    if name == "ollama":
        return OllamaProvider(settings)
    if name == "openai_compatible":
        return OpenAICompatibleProvider(settings)
    if name == "anthropic":
        return AnthropicProvider(settings)
    return TemplateProvider()


def provider_label(p: SummaryProvider) -> str:
    model = getattr(p, "model", "")
    return f"{p.name}:{model}" if model else p.name


def summarize_with_fallback(provider: SummaryProvider, payload: dict) -> Tuple[str, str]:
    try:
        return provider.summarize(payload), provider_label(provider)
    except Exception as exc:  # cualquier error del proveedor externo
        if isinstance(provider, TemplateProvider):
            raise
        log.warning("Proveedor %s fallo (%s); usando template", provider.name, exc)
        return TemplateProvider().summarize(payload), f"template (respaldo de {provider_label(provider)})"
