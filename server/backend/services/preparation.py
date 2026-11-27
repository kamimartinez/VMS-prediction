"""Generic preparation of uploaded milk-production session workbooks."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

PRODUCTION_COLUMNS = [
    "hora_inicio", "numero_ordenio", "duracion", "produccion_kg",
    "rcs_celulas_ml", "intervalo_ordenio", "produccion_di",
    "produccion_dd", "produccion_ti", "produccion_td", "estado_md_ubre",
    "estado_md_pezon", "media_flujo_di", "media_flujo_dd",
    "media_flujo_ti", "media_flujo_td", "flujo_maximo_di",
    "flujo_maximo_dd", "flujo_maximo_ti", "flujo_maximo_td", "sangre_di",
    "sangre_dd", "sangre_ti", "sangre_td", "conductividad_di",
    "conductividad_dd", "conductividad_ti", "conductividad_td",
    "contador_ubre", "contador_di", "contador_dd", "contador_td",
    "contador_ti", "destino_leche", "razon_desviacion", "programa_lavado",
    "ms", "usuario", "patada", "incompleto", "pezones_no_encotnrados",
    "mdi",
]
TIME_STEP = pd.Timedelta(minutes=30)
SEASON_CATEGORIES = ["invierno", "primavera", "verano", "otoño"]
MACHINE_CATEGORIES = ["VMS 1", "VMS 2", "VMS 3"]
READ_BATCH_SIZE = 8
MONTHS_EXCLUDED = {"2022-07", "2026-01"}

MODEL_FEATURE_COLUMNS = [
    "ms", "bloque_tiempo", "total_ordenios", "produccion_promedio",
    "duracion_total_min", "patadas_totales", "desviaciones_leche_totales",
    "ordenios_con_lavado", "pct_primer_parto", "pct_fresca", "pct_alta",
    "pct_baja", "pct_extendida", "hubo_ordenios", "pct_ocupacion_robot",
    "hubo_lavado", "ordenios_otros_robots", "hora", "dia_semana",
    "temporada", "min_desde_alimento", "vacas_activas_dia", "sin_etapa",
    "pct_posible_cola",
]


def _batches(items: list[dict], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _minutes(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype("string").str.strip()
    text = text.where(text.str.count(":") != 1, "00:" + text)
    parsed = pd.to_timedelta(text, errors="coerce").dt.total_seconds() / 60
    numeric_minutes = numeric.where(numeric.abs() >= 1, numeric * 24 * 60)
    return numeric_minutes.where(numeric.notna(), parsed).astype(float)


def _machine_name(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    match = re.fullmatch(r"(?:VMS\s*)?([123])(?:\.0)?", text, flags=re.IGNORECASE)
    return f"VMS {match.group(1)}" if match else text


def _load_workbook(file_info: dict) -> pd.DataFrame:
    filename = str(file_info["filename"])
    path = Path(file_info["path"])
    if path.suffix.casefold() not in {".xlsx", ".xls"}:
        raise ValueError(f"Solo se aceptan archivos XLS/XLSX: {filename}")
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo subido: {filename}")

    try:
        frame = pd.read_excel(path, skiprows=2, names=PRODUCTION_COLUMNS)
    except Exception as exc:
        raise ValueError(f"No se pudo leer el Excel {filename}: {exc}") from exc
    frame = frame.dropna(how="all")
    if frame.empty:
        raise ValueError(f"El Excel no contiene sesiones: {filename}")

    digits = "".join(re.findall(r"\d+", Path(filename).stem))
    frame["id_vaca"] = digits or Path(filename).stem
    frame["origen_archivo"] = filename
    frame["hora_inicio"] = pd.to_datetime(frame["hora_inicio"], errors="coerce")
    frame["duracion_min"] = _minutes(frame["duracion"])
    frame["intervalo_min"] = _minutes(frame["intervalo_ordenio"])
    frame["produccion_kg"] = pd.to_numeric(
        frame["produccion_kg"], errors="coerce"
    )
    frame["ms"] = frame["ms"].map(_machine_name)

    for column in (
        "destino_leche", "programa_lavado", "patada", "incompleto",
        "pezones_no_encotnrados",
    ):
        frame[column] = frame[column].astype("string")

    return frame[
        [
            "id_vaca", "origen_archivo", "hora_inicio", "duracion_min",
            "intervalo_min", "produccion_kg", "ms", "destino_leche",
            "programa_lavado", "patada", "incompleto",
            "pezones_no_encotnrados",
        ]
    ]


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "invierno"
    if month in (3, 4, 5):
        return "primavera"
    if month in (6, 7, 8):
        return "verano"
    return "otoño"


def _minutes_since_feed(times: pd.Series) -> pd.Series:
    hour = times.dt.hour + times.dt.minute / 60
    last_feed = np.select(
        [hour >= 14.5, hour >= 7.5],
        [14.5, 7.5],
        default=14.5 - 24,
    )
    return pd.Series(((hour - last_feed) * 60).round(1), index=times.index)


def _event_count(values: pd.Series) -> pd.Series:
    def count(value) -> int:
        if pd.isna(value):
            return 0
        text = str(value).upper().strip()
        if text in {"", "NO", "FALSE", "0", "NONE", "N/A"}:
            return 0
        if "TODOS" in text:
            return 4
        quadrants = sum(part in text for part in ("TD", "DI", "TI", "DD"))
        return quadrants or 1

    return values.map(count).astype("int64")


def _add_lactation_features(sessions: pd.DataFrame) -> pd.DataFrame:
    frame = sessions.copy()
    frame = frame.sort_values(["id_vaca", "hora_inicio"], kind="stable")
    first_date = frame["hora_inicio"].min()
    previous_gap = frame.groupby("id_vaca")["hora_inicio"].diff().dt.total_seconds() / 86400
    first_session = previous_gap.isna()
    arrived_late = (
        frame.groupby("id_vaca")["hora_inicio"].transform("min")
        > first_date + pd.Timedelta(days=30)
    )
    new_segment = first_session | previous_gap.ge(40)
    frame["segmento"] = new_segment.groupby(frame["id_vaca"]).cumsum()
    known_start = frame["segmento"].gt(1) | arrived_late
    lactation_start = frame.groupby(
        ["id_vaca", "segmento"]
    )["hora_inicio"].transform("min")
    days_in_milk = (frame["hora_inicio"] - lactation_start).dt.days
    frame["dias_en_leche"] = days_in_milk.where(known_start)
    frame["etapa_lactancia"] = pd.cut(
        frame["dias_en_leche"],
        bins=[-1, 100, 200, 305, np.inf],
        labels=[
            "fresca (0-100)", "alta prod. (100-200)",
            "baja prod. (200-305)", "extendida (> 305)",
        ],
    )
    frame["es_primer_parto"] = np.where(
        known_start,
        frame["segmento"].eq(1).astype(float),
        np.nan,
    )
    for name, label in (
        ("fresca", "fresca (0-100)"),
        ("alta", "alta prod. (100-200)"),
        ("baja", "baja prod. (200-305)"),
        ("extendida", "extendida (> 305)"),
    ):
        frame[f"es_{name}"] = (
            frame["etapa_lactancia"].eq(label).astype(float)
            .where(frame["etapa_lactancia"].notna())
        )
    return frame


def _drop_overlapping_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    keep = pd.Series(True, index=sessions.index)
    for _, group in sessions.groupby("ms", sort=False):
        last_end = None
        for index, row in group.iterrows():
            end = row["hora_inicio"] + pd.to_timedelta(
                row["duracion_min"], unit="m"
            )
            if last_end is not None and row["hora_inicio"] < last_end:
                keep.at[index] = False
            else:
                last_end = end
    return sessions.loc[keep].copy()


def _build_model_frame(sessions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "id_vaca", "hora_inicio", "duracion_min", "produccion_kg", "ms",
        "intervalo_min", "destino_leche", "programa_lavado", "patada",
        "incompleto", "pezones_no_encotnrados",
    }
    missing = required - set(sessions.columns)
    if missing:
        raise ValueError(f"Faltan columnas de sesiones: {sorted(missing)}")
    if sessions.empty:
        raise ValueError("No hay sesiones de leche para preparar.")

    frame = sessions.dropna(
        subset=["id_vaca", "hora_inicio", "duracion_min", "produccion_kg", "ms"]
    ).copy()
    unknown_machines = set(frame["ms"].unique()) - set(MACHINE_CATEGORIES)
    if unknown_machines:
        raise ValueError(
            f"Máquinas no reconocidas por los modelos: {sorted(unknown_machines)}"
        )
    frame = frame[
        frame["duracion_min"].between(1, 30)
        & frame["produccion_kg"].gt(0)
        & frame["intervalo_min"].ne(0)
    ].copy()
    if frame.empty:
        raise ValueError("No quedaron sesiones válidas después de la limpieza.")

    frame = _add_lactation_features(frame)
    frame["fecha"] = frame["hora_inicio"].dt.normalize()
    daily_cows = frame.groupby("fecha")["id_vaca"].nunique()
    frame["vacas_activas_dia"] = frame["fecha"].map(daily_cows)
    frame["patadas_totales_fila"] = _event_count(frame["patada"])
    frame["programa_lavado"] = (
        frame["programa_lavado"].fillna("").eq("Local Aclarado").astype("int64")
    )
    destination = frame["destino_leche"]
    frame["desviacion_leche"] = np.where(
        destination.isna(),
        np.nan,
        destination.str.strip().ne("Tanque").astype(float),
    )

    frame = frame.sort_values(
        ["ms", "hora_inicio"], kind="stable"
    ).reset_index(drop=True)
    frame = _drop_overlapping_sessions(frame)
    frame = frame.sort_values(
        ["ms", "hora_inicio"], kind="stable"
    ).reset_index(drop=True)

    next_start = frame.groupby("ms")["hora_inicio"].shift(-1)
    free_minutes = (next_start - frame["hora_inicio"]).dt.total_seconds() / 60
    free_minutes = free_minutes - frame["duracion_min"]
    valid_next = free_minutes.between(0, 24 * 60)
    frame["posible_cola"] = np.nan
    frame.loc[valid_next, "posible_cola"] = (
        free_minutes.loc[valid_next] <= 2
    ).astype(float)
    frame["bloque_tiempo"] = frame["hora_inicio"].dt.floor("30min")
    frame = frame[
        ~frame["bloque_tiempo"].dt.to_period("M").astype(str).isin(
            MONTHS_EXCLUDED
        )
    ].copy()
    if frame.empty:
        raise ValueError("No hay sesiones dentro de los meses aceptados.")

    aggregated = frame.groupby(
        ["ms", "bloque_tiempo"], sort=False
    ).agg(
        pct_posible_cola=("posible_cola", "mean"),
        total_ordenios=("id_vaca", "count"),
        produccion_promedio=("produccion_kg", "mean"),
        duracion_total_min=("duracion_min", "sum"),
        patadas_totales=("patadas_totales_fila", "sum"),
        desviaciones_leche_totales=("desviacion_leche", "sum"),
        ordenios_con_lavado=("programa_lavado", "sum"),
        pct_primer_parto=("es_primer_parto", "mean"),
        pct_fresca=("es_fresca", "mean"),
        pct_alta=("es_alta", "mean"),
        pct_baja=("es_baja", "mean"),
        pct_extendida=("es_extendida", "mean"),
    )

    time_grid = pd.date_range(
        aggregated.index.get_level_values("bloque_tiempo").min(),
        aggregated.index.get_level_values("bloque_tiempo").max(),
        freq="30min",
    )
    time_grid = time_grid[
        ~time_grid.to_period("M").astype(str).isin(MONTHS_EXCLUDED)
    ]
    grid = pd.MultiIndex.from_product(
        [MACHINE_CATEGORIES, time_grid], names=["ms", "bloque_tiempo"]
    )
    model_frame = aggregated.reindex(grid).reset_index()
    no_sessions = model_frame["total_ordenios"].isna()
    zero_columns = [
        "total_ordenios", "duracion_total_min", "patadas_totales",
        "ordenios_con_lavado",
    ]
    model_frame.loc[no_sessions, zero_columns] = 0
    model_frame.loc[no_sessions, "pct_posible_cola"] = 0

    model_frame["pct_ocupacion_robot"] = (
        model_frame["duracion_total_min"] / 30
    ).clip(upper=1)
    model_frame["hora"] = model_frame["bloque_tiempo"].dt.hour
    model_frame["dia_semana"] = model_frame["bloque_tiempo"].dt.dayofweek
    model_frame["temporada"] = model_frame["bloque_tiempo"].dt.month.map(_season)
    model_frame["min_desde_alimento"] = _minutes_since_feed(
        model_frame["bloque_tiempo"]
    )
    model_frame["hubo_ordenios"] = (~no_sessions).astype("int64")
    model_frame["hubo_lavado"] = (
        model_frame["ordenios_con_lavado"] > 0
    ).astype("int64")
    cows_per_day = frame.groupby("fecha")["id_vaca"].nunique()
    model_frame["vacas_activas_dia"] = (
        model_frame["bloque_tiempo"].dt.normalize().map(cows_per_day).fillna(0)
    )
    model_frame["ordenios_otros_robots"] = (
        model_frame.groupby("bloque_tiempo")["total_ordenios"].transform("sum")
        - model_frame["total_ordenios"]
    )
    model_frame["sin_etapa"] = model_frame["pct_fresca"].isna().astype("int64")
    stage_columns = [
        "pct_primer_parto", "pct_fresca", "pct_alta", "pct_baja", "pct_extendida"
    ]
    model_frame[stage_columns] = model_frame[stage_columns].fillna(0)

    model_frame = model_frame[MODEL_FEATURE_COLUMNS].sort_values(
        ["bloque_tiempo", "ms"], kind="stable"
    ).reset_index(drop=True)
    if not np.isfinite(model_frame["total_ordenios"]).all():
        raise ValueError("La preparación produjo un conteo de ordeños inválido.")
    return model_frame


def prepare_uploaded_files(
    files: Iterable[dict],
    work_dir: Path,
    *,
    batch_size: int = READ_BATCH_SIZE,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Read XLS/XLSX inputs in bounded batches and build the shared model frame."""
    file_list = list(files)
    if not file_list:
        raise ValueError("Debes subir al menos un Excel de producción de leche.")
    if batch_size < 1:
        raise ValueError("batch_size debe ser mayor que cero.")

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for batch_number, batch in enumerate(_batches(file_list, batch_size)):
        frames = [_load_workbook(file_info) for file_info in batch]
        batch_frame = pd.concat(frames, ignore_index=True)
        batch_path = work_dir / f"production-{batch_number:05d}.parquet"
        batch_frame.to_parquet(batch_path, index=False)
        del frames, batch_frame
        if progress is not None:
            progress(min((batch_number + 1) * batch_size, len(file_list)), len(file_list))

    source = pd.read_parquet(work_dir)
    return _build_model_frame(source)
