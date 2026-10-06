"""
Modelo temporal de riesgo y mapas de calor.

Idea: cada reporte historico "suma" riesgo a su barrio en su dia de la
semana y franja horaria. Los reportes mas graves suman mas y los mas viejos
suman menos (decaimiento exponencial con vida media configurable).

    peso = (gravedad / 5) * 0.5 ^ (edad_en_dias / vida_media)

Franjas horarias (hour_block): 0 = 00-06, 1 = 06-12, 2 = 12-18, 3 = 18-24

Mapas de calor espaciales intercambiables (HEATMAP_MODEL):
    kde       : estimacion de densidad por kernel gaussiano sobre una grilla
    frequency : conteo ponderado por celda de la grilla (histograma 2D)
"""

import math
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Protocol, Tuple

DAYS_ES = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabados", "domingos"]
BLOCKS_ES = ["entre 00:00 y 06:00", "entre 06:00 y 12:00", "entre 12:00 y 18:00", "entre 18:00 y 24:00"]


def hour_block(hour: int) -> int:
    return min(3, hour // 6)


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def decay_weight(severity: int, occurred: datetime, now: datetime, half_life: float) -> float:
    age_days = max(0.0, (now - occurred).total_seconds() / 86400)
    return (severity / 5.0) * (0.5 ** (age_days / half_life))


def temporal_matrix(reports: List[dict], now: datetime, half_life: float) -> Dict[Tuple[int, int, int], float]:
    """(barrio, dia, franja) -> peso acumulado."""
    m: Dict[Tuple[int, int, int], float] = defaultdict(float)
    for r in reports:
        dt = parse_dt(r["occurred_at"])
        key = (r.get("neighborhood_id", 0), dt.weekday(), hour_block(dt.hour))
        m[key] += decay_weight(r["severity"], dt, now, half_life)
    return m


def zone_risk(history: List[dict], occurred: datetime, now: datetime, half_life: float) -> Tuple[float, str]:
    """Riesgo 0..1 del barrio en el dia/franja del nuevo reporte."""
    if not history:
        return 0.0, "Sin historial en esta zona: riesgo base."
    m: Dict[Tuple[int, int], float] = defaultdict(float)
    total = 0.0
    for r in history:
        dt = parse_dt(r["occurred_at"])
        w = decay_weight(r["severity"], dt, now, half_life)
        m[(dt.weekday(), hour_block(dt.hour))] += w
        total += w
    slot = (occurred.weekday(), hour_block(occurred.hour))
    peak = max(m.values())
    slot_share = m.get(slot, 0.0) / peak if peak else 0.0
    volume = min(1.0, total / 12.0)  # 12 = "mucho" historial reciente
    risk = round(0.6 * slot_share + 0.4 * volume, 3)
    explanation = (
        f"Los {DAYS_ES[slot[0]]} {BLOCKS_ES[slot[1]]} esta zona tiene {round(slot_share * 100)}% "
        f"del riesgo de su peor franja; actividad reciente {round(volume * 100)}%."
    )
    return risk, explanation


# ------------------------------------------------------------ mapas de calor
class HeatmapModel(Protocol):
    name: str

    def cells(self, reports: List[dict], bbox: Tuple[float, float, float, float], now: datetime, half_life: float) -> List[dict]:
        ...


def _bbox_grid(bbox, size: int):
    min_lat, min_lon, max_lat, max_lon = bbox
    dlat = (max_lat - min_lat) / size
    dlon = (max_lon - min_lon) / size
    return min_lat, min_lon, dlat, dlon


class KDEHeatmap:
    name = "kde"

    def __init__(self, grid_size: int = 36, bandwidth_deg: float = 0.0055) -> None:
        self.grid_size = grid_size
        self.h = bandwidth_deg

    def cells(self, reports, bbox, now, half_life):
        min_lat, min_lon, dlat, dlon = _bbox_grid(bbox, self.grid_size)
        points = [
            (r["latitude"], r["longitude"], decay_weight(r["severity"], parse_dt(r["occurred_at"]), now, half_life))
            for r in reports
        ]
        two_h2 = 2 * self.h * self.h
        cutoff = (3 * self.h) ** 2
        grid = []
        for i in range(self.grid_size):
            lat = min_lat + (i + 0.5) * dlat
            for j in range(self.grid_size):
                lon = min_lon + (j + 0.5) * dlon
                value = 0.0
                for plat, plon, w in points:
                    d2 = (lat - plat) ** 2 + (lon - plon) ** 2
                    if d2 < cutoff:
                        value += w * math.exp(-d2 / two_h2)
                if value > 0:
                    grid.append((lat, lon, value))
        return _normalize(grid)


class FrequencyHeatmap:
    name = "frequency"

    def __init__(self, grid_size: int = 24) -> None:
        self.grid_size = grid_size

    def cells(self, reports, bbox, now, half_life):
        min_lat, min_lon, dlat, dlon = _bbox_grid(bbox, self.grid_size)
        bins: Dict[Tuple[int, int], float] = defaultdict(float)
        for r in reports:
            i = int((r["latitude"] - min_lat) / dlat) if dlat else 0
            j = int((r["longitude"] - min_lon) / dlon) if dlon else 0
            if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                bins[(i, j)] += decay_weight(r["severity"], parse_dt(r["occurred_at"]), now, half_life)
        grid = [(min_lat + (i + 0.5) * dlat, min_lon + (j + 0.5) * dlon, v) for (i, j), v in bins.items()]
        return _normalize(grid)


def _normalize(grid: List[Tuple[float, float, float]], threshold: float = 0.04) -> List[dict]:
    if not grid:
        return []
    peak = max(v for _, _, v in grid) or 1.0
    return [
        {"latitude": round(lat, 5), "longitude": round(lon, 5), "intensity": round(v / peak, 3)}
        for lat, lon, v in grid
        if v / peak >= threshold
    ]


HEATMAPS = {"kde": KDEHeatmap, "frequency": FrequencyHeatmap}


def bbox_for(reports: List[dict], neighborhoods: List[dict]) -> Optional[Tuple[float, float, float, float]]:
    lats = [p["latitude"] for p in reports + neighborhoods]
    lons = [p["longitude"] for p in reports + neighborhoods]
    if not lats:
        return None
    pad = 0.01
    return min(lats) - pad, min(lons) - pad, max(lats) + pad, max(lons) + pad
