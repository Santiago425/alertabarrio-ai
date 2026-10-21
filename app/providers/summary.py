"""
Proveedores para el RESUMEN DIARIO en lenguaje natural (SUMMARY_PROVIDER).

    template          : sin modelo externo; arma el texto con reglas. Siempre
                        funciona (ideal para la sustentacion, no se "duerme").
    ollama            : modelo open source local (ej. llama3.2:3b, qwen2.5:3b).
    openai_compatible : cualquier API compatible con OpenAI que sirva modelos
                        open source (Groq, OpenRouter, LM Studio, vLLM...).
    anthropic         : API de Anthropic.
    gemini            : Google Gemini (tiene plan gratis en Google AI Studio).
    auto              : Gemini si hay GEMINI_API_KEY; si no, template.

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


class LLMProvider:
    """Proveedor con un modelo de lenguaje: resume y tambien responde preguntas."""

    name = "llm"

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def summarize(self, payload: dict) -> str:
        return self.generate(_prompt(payload))


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
        f"Eres el asistente de seguridad del barrio {payload['neighborhood_name']} en {payload.get('city') or 'Pasto'}.",
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


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.base = (settings.llm_base_url or "http://localhost:11434").rstrip("/")
        self.model = settings.summary_model or "llama3.2:3b"
        self.timeout = settings.llm_timeout_seconds

    def generate(self, prompt: str) -> str:
        r = httpx.post(
            f"{self.base}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["response"].strip()


class OpenAICompatibleProvider(LLMProvider):
    name = "openai_compatible"

    def __init__(self, settings: Settings) -> None:
        self.base = (settings.llm_base_url or "https://api.groq.com/openai/v1").rstrip("/")
        self.model = settings.summary_model or "llama-3.1-8b-instant"
        self.key = settings.llm_api_key
        self.timeout = settings.llm_timeout_seconds

    def generate(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        r = httpx.post(
            f"{self.base}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 350,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.base = (settings.llm_base_url or "https://api.anthropic.com").rstrip("/")
        self.model = settings.summary_model or "claude-haiku-4-5"
        self.key = settings.llm_api_key or ""
        self.timeout = settings.llm_timeout_seconds

    def generate(self, prompt: str) -> str:
        r = httpx.post(
            f"{self.base}/v1/messages",
            headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"},
            json={
                "model": self.model,
                "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return "".join(block.get("text", "") for block in r.json()["content"]).strip()


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        self.base = (settings.gemini_base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self.model = settings.gemini_model or "gemini-2.5-flash"
        self.key = settings.gemini_api_key or settings.llm_api_key or ""
        self.timeout = settings.llm_timeout_seconds

    def generate(self, prompt: str) -> str:
        r = httpx.post(
            f"{self.base}/models/{self.model}:generateContent",
            headers={"x-goog-api-key": self.key},
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.4, "maxOutputTokens": 600},
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()


def build_provider(settings: Settings) -> SummaryProvider:
    name = settings.summary_provider
    if name == "auto":
        name = "gemini" if settings.gemini_api_key else "template"
    if name == "gemini":
        return GeminiProvider(settings)
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


# ------------------------------------------------------------------ asistente
def _chat_prompt(payload: dict) -> str:
    ctx = payload.get("context", {}) or {}
    lines = [
        f"Eres el asistente de seguridad de AlertaBarrio en {ctx.get('city', 'Pasto')}, Colombia.",
        "Responde en espanol, cercano y claro, en maximo 4 oraciones. Usa SOLO los datos de abajo;",
        "si no sabes algo dilo. Da consejos practicos de prevencion. No inventes cifras.",
        "",
        f"Barrio del usuario: {ctx.get('neighborhood') or 'no indicado'}",
        f"Alertas activas en la ciudad: {ctx.get('active_alerts', 'sin dato')}",
    ]
    for r in ctx.get("ranking", [])[:6]:
        lines.append(f"- Barrio {r['neighborhood']}: {r['active_alerts']} alertas activas (gravedad acumulada {r['active_severity']})")
    for h in ctx.get("hotspots", [])[:5]:
        lines.append(f"- Riesgo alto en {h['neighborhood']} los {DAYS_ES[h['day_of_week']]} {BLOCKS_ES[h['hour_block']]}")
    for r in ctx.get("recent", [])[:8]:
        lines.append(f"- Reciente: {r['incident_type']} en {r['neighborhood']} ({r['when']}): {r['title']}")
    lines += ["", f"Pregunta del vecino: {payload['question']}"]
    return "\n".join(lines)


def _rule_answer(payload: dict) -> str:
    """Respuesta sin modelo externo (cuando no hay Gemini u otro LLM configurado)."""
    ctx = payload.get("context", {}) or {}
    q = payload["question"].lower()
    ranking = [r for r in ctx.get("ranking", []) if r.get("active_alerts")]
    hotspots = ctx.get("hotspots", [])
    hood = ctx.get("neighborhood")
    parts = []
    if any(w in q for w in ("hora", "cuando", "noche", "dia", "horario")) and hotspots:
        h = hotspots[0]
        parts.append(f"Segun el historial, el momento mas riesgoso es en {h['neighborhood']} los {DAYS_ES[h['day_of_week']]} {BLOCKS_ES[h['hour_block']]}.")
    if any(w in q for w in ("barrio", "zona", "donde", "peligro", "seguro", "evitar")) and ranking:
        top = ", ".join(r["neighborhood"] for r in ranking[:3])
        safest = ranking[-1]["neighborhood"] if len(ranking) > 3 else None
        parts.append(f"Ahora mismo las zonas con mas alertas activas son {top}.")
        if safest:
            parts.append(f"Entre los barrios con alertas, {safest} es de los mas tranquilos.")
    if hood and not parts:
        mine = next((r for r in ranking if r["neighborhood"] == hood), None)
        if mine:
            parts.append(f"En {hood} hay {mine['active_alerts']} alertas activas.")
    if not parts:
        parts.append(f"Hay {ctx.get('active_alerts', 'varias')} alertas activas en {ctx.get('city', 'la ciudad')}.")
        if ranking:
            parts.append(f"El barrio con mas alertas es {ranking[0]['neighborhood']}.")
    parts.append("Consejo: evita caminar solo de noche por vias oscuras, no muestres el celular y reporta cualquier cosa sospechosa.")
    return " ".join(parts)


def answer_question(provider: SummaryProvider, payload: dict) -> Tuple[str, str]:
    if isinstance(provider, LLMProvider):
        try:
            return provider.generate(_chat_prompt(payload)), provider_label(provider)
        except Exception as exc:
            log.warning("Asistente con %s fallo (%s); respondiendo con reglas", provider.name, exc)
            return _rule_answer(payload), f"reglas (respaldo de {provider_label(provider)})"
    return _rule_answer(payload), "reglas"
