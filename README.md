# DANA Predictor

Mapa de susceptibilidad a inundación urbana en Valencia ante un episodio tipo DANA, con datos 100 % reales.

> ⚠️ Prototipo académico informativo. No sustituye los avisos de las autoridades.

## Qué hace

Para cada celda de 250 m de Valencia calcula la probabilidad de que se inunde en un episodio como la DANA del
29-10-2024 y la clasifica como **Bajo, Medio o Alto**. Usa 4 variables de terreno (altitud, pendiente, % de suelo
sellado y peligrosidad PATRICOVA) y aprende de las zonas que se inundaron en la DANA según Copernicus EMS.

No usa la lluvia: no hay en abierto un histórico de lluvia horaria ni mapas de inundación de otros episodios en
Valencia. Es un mapa fijo de susceptibilidad, no una predicción en tiempo real.

## Requisitos

- Python 3.12 (o Docker)
- Unos 300 MB libres para los datos
- Los datos ya vienen en el repositorio: `data/raw/originales/` (descargas originales) y `data/raw/` (preparados por `ingest`). Fuentes en [docs/DATOS.md](docs/DATOS.md)

No hace falta ninguna variable de entorno. `.env.example` solo documenta `DATA_DIR`, opcional.

## Ejecución local

```bash
git clone https://github.com/Victorproyecto/IA_Project.git
cd IA_Project
python3 -m venv .venv && source .venv/bin/activate
pip install -r environment/requirements.txt

# Los datos ya están en data/raw/ y data/raw/originales/. En este orden:
python -m src.pipeline.ingest     # 1. originales → data/raw (recorta y unifica)
python -m src.pipeline.run        # 2. data/raw → data/processed (celdas, variables, etiquetas)
python -m src.pipeline.train      # 3. entrena, valida y genera predicciones en data/models y data/processed
streamlit run src/frontend/app.py # 4. aplicación en http://localhost:8501

python -m pytest                  # tests (se saltan los que necesitan datos si falta data/raw)
```

### Con Docker

```bash
docker compose -f environment/docker-compose.yml build ingest
docker compose -f environment/docker-compose.yml run --rm ingest
docker compose -f environment/docker-compose.yml run --rm pipeline
docker compose -f environment/docker-compose.yml run --rm train
docker compose -f environment/docker-compose.yml up app          # http://localhost:8501
```

## Arquitectura

```
data/raw/originales ──ingest──► data/raw ──run──► data/processed ──train──► data/models + predicciones
                                                                                   │
                                          frontend (Streamlit) ◄── backend (services.py)
```

- **Pipeline** (por lotes): prepara los datos, construye las celdas y entrena. Escribe Parquet / GeoParquet.
- **Backend**: única puerta de acceso a los resultados del pipeline, más la búsqueda de direcciones.
- **Frontend**: mapa, buscador, tabla y ficha del modelo. Solo lee a través del backend.

## Estructura

```
src/
├── config.py              # rutas, CRS, zona del piloto, parámetros del modelo
├── pipeline/
│   ├── ingest.py          # originales → data/raw: une el MDT, recorta, prepara PATRICOVA, barrios y la DANA
│   ├── grid.py            # cuadrícula de celdas de 250 m y pendiente
│   ├── sources.py         # lleva cada fuente de data/raw a la celda
│   ├── features.py        # variables derivadas (PATRICOVA invertido)
│   ├── blocks.py          # bloques espaciales de 1 km y grupos de validación
│   ├── checks.py          # comprobaciones de datos antes de escribir
│   ├── run.py             # orquesta la preparación: zonas, etiquetas y tabla de entrenamiento
│   ├── evaluation.py      # métricas, umbrales Bajo/Medio/Alto
│   ├── train.py           # validación cruzada por bloques, selección de modelo, ficha del modelo
│   └── predict.py         # predicción por celda
├── backend/
│   └── services.py        # lectura de resultados para el frontend y geocodificación (Nominatim)
└── frontend/
    └── app.py             # aplicación Streamlit
tests/                     # pytest: unitarios, datos, modelo, backend y frontend
environment/               # requirements.txt, Dockerfile, docker-compose.yml, entrypoint.sh
docs/                      # entregas, DATOS.md y ENTREGA_NF3.md
data/                      # raw/ y raw/originales/ versionados; processed/ y models/ se generan
```

## Datos

| Fichero en `data/raw/` | Fuente | Licencia |
|---|---|---|
| `mdt.tif` | IGN / CNIG, PNOA MDT05 | CC BY 4.0 |
| `imperviousness.tif` | Copernicus Land, Imperviousness Density 2024 | Copernicus data policy |
| `patricova.gpkg` | Generalitat Valenciana, PATRICOVA | Por verificar |
| `barrios.geojson` | Ayuntamiento de Valencia, datos abiertos | CC BY 4.0 |
| `flood_events.gpkg` | Copernicus EMS, activación EMSR773 | Copernicus data policy |

Cómo descargarlos y colocarlos: [docs/DATOS.md](docs/DATOS.md).

## Modelo

- Una fila por celda (1.397 celdas, 264 inundadas en la DANA).
- Regresión logística (base) y Random Forest (candidato), calibrados y con pesos de clase.
- Validación cruzada en 5 grupos de bloques espaciales de 1 km: cada celda se predice con un modelo que no ha visto su zona.
- Resultado: regresión logística, Recall 0,80, F1 0,39 (objetivo de la Fase 1: 0,70), frente a F1 0,32 de PATRICOVA.
- Ficha completa en `data/models/model_card.json` y en la pestaña *Modelo y datos* de la aplicación.

## Contribuir

Todo cambio entra por pull request a `develop`, con revisión de otro miembro del equipo. Los commits siguen [Conventional Commits](https://www.conventionalcommits.org/).

## Equipo

Grupo 10 · Proyecto Final de Big Data e IA

| Nombre | Rol |
|---|---|
| Dennis Bonilla | Data |
| Jaume Palau | ML |
| Víctor Martínez | Platform |
| Rafael Chambilla | BI |
