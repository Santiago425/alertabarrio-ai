# AlertaBarrio · Componente de IA

Servicio **independiente** del backend (como recomendó el profe). El backend le pega por HTTP y no sabe qué modelo hay detrás.

| Endpoint | Qué hace | Modelos (variable) |
|---|---|---|
| `POST /api/v1/analyze-report` | Sugiere el tipo de incidente a partir del texto y calcula el riesgo de la zona en ese día/hora | `CLASSIFIER_MODEL`: `naive_bayes` (por defecto) · `keywords` |
| `POST /api/v1/heatmap` | Mapa de calor predictivo + zonas/franjas calientes con decaimiento temporal | `HEATMAP_MODEL`: `kde` (por defecto) · `frequency` |
| `POST /api/v1/summary` | Resumen diario en lenguaje natural | `SUMMARY_PROVIDER`: `template` · `ollama` · `openai_compatible` (Groq, OpenRouter, LM Studio…) · `anthropic` |
| `GET /api/v1/models` | Modelos activos | |
| `POST /api/v1/models/switch` | Cambiar de modelo en caliente (header `X-Admin-Token` si se configuró `ADMIN_TOKEN`) | |

## Cambiar de modelo ("el switch")

Solo variables de entorno, sin tocar backend ni frontend. Ejemplo con un modelo open source local:

```bash
ollama pull llama3.2:3b
SUMMARY_PROVIDER=ollama SUMMARY_MODEL=llama3.2:3b uvicorn app.main:app --port 8001
```

O en caliente:

```bash
curl -X POST http://localhost:8001/api/v1/models/switch -H "Content-Type: application/json" \
     -d '{"heatmap_model": "frequency", "classifier_model": "keywords"}'
```

Si un proveedor externo falla, el resumen se genera con la plantilla y la respuesta lo indica (`"model": "template (respaldo de ...)"`).

## Modelos

- **Naive Bayes multinomial** (`app/models/classifier.py`): implementado a mano con suavizado de Laplace, entrenado al arrancar con frases típicas de grupos de barrio (`training_data.py`).
- **Riesgo temporal** (`app/models/risk.py`): peso = gravedad/5 × 0.5^(edad/vida_media). Se acumula por barrio, día de la semana y franja de 6 horas.
- **KDE**: densidad por kernel gaussiano sobre una grilla de 36×36. **Frequency**: histograma 2D ponderado.

## Correr

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8001
pytest
```
