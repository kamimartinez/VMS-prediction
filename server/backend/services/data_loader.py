
from pathlib import Path
from typing import Iterable
import pandas as pd


CSV_COLUMNS = [
    "hora_inicio", "accion", "duracion", "produccion_kg",
    "numero_ordeños", "rcs_celulas_ml", "patada", "incompleto",
    "pezones_no_encontrados", "estado_amd_ubre", "estado_amd_pezon",
    "media_flujos_di", "media_flujos_dd", "media_flujos_ti",
    "media_flujos_td", "sangre_di", "sangre_dd", "sangre_ti",
    "sangre_td", "conductividad_di", "conductividad_dd",
    "conductividad_ti", "conductividad_td", "eo_po", "usuario",
    "destino_leche", "razon_de_desviacion", "flujo_maximo_di",
    "flujo_maximo_dd", "flujo_maximo_ti", "flujo_maximo_td",
    "produccion_di", "produccion_dd", "produccion_ti", "produccion_td",
]

CONSUMPTION_COLUMNS = [
    "alimento", "hora_evento", "consumo", "consumido"
]

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


def load_uploaded_sources(files: Iterable[dict]) -> dict[str, pd.DataFrame]:
    """
    Reads all uploaded files and consolidates them into
    their corresponding DataFrames.

    Each item must contain:
        filename: Original filename
        path: Local path where the file was stored
    """

    csv_frames = []
    consumption_frames = []
    production_frames = []

    for file_info in files:

        filename = file_info["filename"]
        path = Path(file_info["path"])

        extension = path.suffix.lower()

        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        # CSV: Cow records
        if extension == ".csv":

            # Preserve the original notebook's naming convention
            if not filename[0].isdigit():
                continue

            df = pd.read_csv(
                path,
                skiprows=2,
                names=CSV_COLUMNS
            )

            df["origen_archivo"] = filename
            df["id_vaca"] = path.stem

            csv_frames.append(df)

        # Excel: Historical records
        elif extension in {".xlsx", ".xls"}:

            # Preserve the original notebook's ID extraction
            id_vaca = "".join(
                filter(str.isdigit, filename)
            ) or "Desconocido"

            if filename.casefold().startswith("concentrado consumido"):

                df = pd.read_excel(
                    path,
                    skiprows=1,
                    names=CONSUMPTION_COLUMNS
                )

                consumption_frames.append(
                    df.assign(
                        origen_archivo=filename,
                        id_vaca=id_vaca
                    )
                )

            else:

                df = pd.read_excel(
                    path,
                    skiprows=2,
                    names=PRODUCTION_COLUMNS
                )

                production_frames.append(
                    df.assign(
                        origen_archivo=filename,
                        id_vaca=id_vaca
                    )
                )

        else:
            raise ValueError(
                f"Unsupported file format: {filename}"
            )

    # Validate the batch
    if not csv_frames:
        raise ValueError("No valid cow CSV files were uploaded.")

    if not consumption_frames:
        raise ValueError(
            "No concentrate consumption Excel files were uploaded."
        )

    if not production_frames:
        raise ValueError(
            "No milk production Excel files were uploaded."
        )

    # Consolidate each source
    return {
        "df_consolidado": pd.concat(
            csv_frames,
            ignore_index=True
        ),

        "concentrado_consumido": pd.concat(
            consumption_frames,
            ignore_index=True
        ),

        "prod_leche_sesion": pd.concat(
            production_frames,
            ignore_index=True
        ),
    }
