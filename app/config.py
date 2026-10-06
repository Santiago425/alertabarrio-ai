"""
Configuracion del componente de IA.

Para CAMBIAR DE MODELO no se toca codigo: solo se cambian estas variables
de entorno (o se usa POST /api/v1/models/switch en caliente).

    CLASSIFIER_MODEL   naive_bayes | keywords
    HEATMAP_MODEL      kde | frequency
    SUMMARY_PROVIDER   template | ollama | openai_compatible | anthropic
    SUMMARY_MODEL      nombre del modelo del proveedor (ej: llama3.2:3b, llama-3.1-8b-instant)
    LLM_BASE_URL       URL del proveedor (ej: http://localhost:11434, https://api.groq.com/openai/v1)
    LLM_API_KEY        llave del proveedor (si aplica)
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    classifier_model: str = "naive_bayes"
    heatmap_model: str = "kde"
    summary_provider: str = "template"
    summary_model: str = ""
    llm_base_url: str = ""
    llm_api_key: Optional[str] = None
    llm_timeout_seconds: float = 25.0

    # Si se define, POST /api/v1/models/switch exige el header X-Admin-Token
    admin_token: Optional[str] = None

    # Vida media (dias) para que los reportes viejos pesen menos en el riesgo
    risk_half_life_days: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
