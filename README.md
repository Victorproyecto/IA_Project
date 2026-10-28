# DANA Predictor

Mapa de riesgo de inundación urbana en Valencia ante episodios de lluvia intensa.

> ⚠️ Prototipo académico informativo. No sustituye los avisos de las autoridades.

## Descripción

El modelo estima el riesgo de inundación (Bajo, Medio o Alto) de cada celda de 250 m de Valencia. Para ello combina lluvia, terreno, impermeabilidad, peligrosidad PATRICOVA e inundaciones históricas. El resultado se consulta en un mapa interactivo.

## Tecnologías

Python · Pandas · GeoPandas · scikit-learn · Streamlit · Docker

## Instalación

```bash
git clone https://github.com/Victorproyecto/IA_Project.git
cd IA_Project
```

### Con Docker (recomendado)

```bash
docker compose -f environment/docker-compose.yml build pipeline
docker compose -f environment/docker-compose.yml run --rm pipeline   # genera data/processed
docker compose -f environment/docker-compose.yml run --rm train      # entrena y genera predicciones
docker compose -f environment/docker-compose.yml up app              # http://localhost:8501
```

### En local (Python 3.12)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r environment/requirements.txt
cp .env.example .env          # opcional: AEMET_API_KEY para la previsión horaria
python -m src.pipeline.run
python -m src.models.train
streamlit run src/app/streamlit_app.py
python -m pytest              # tests unitarios, de datos, del modelo y de integración
```

## Datos reales y datos de muestra

Cada fuente se carga desde `data/raw/` si el fichero existe y, si no, desde una **muestra simulada**
(`src/pipeline/sample.py`) para que el sistema funcione de punta a punta. El estado de cada fuente
(`real` / `simulado`) queda en `data/processed/data_sources.csv` y se muestra en el dashboard.

| Fichero en `data/raw/` | Fuente |
|---|---|
| `mdt.tif` | CNIG/IGN, MDT (altitud) |
| `imperviousness.tif` | Copernicus HRL Imperviousness Density |
| `patricova.gpkg` (columna `nivel`) | PATRICOVA peligrosidad |
| `barrios.geojson` (`nombre`, `distrito`) | Ayuntamiento de Valencia |
| `stations.csv` / `aemet_observations.csv` | AEMET OpenData |
| `episodes.csv` | Catálogo de episodios (equipo Data) |
| `flood_events.gpkg` (`episode`, `severity`) | Copernicus EMS |

Con `AEMET_API_KEY` en `.env`, el pipeline añade la previsión horaria de AEMET para Valencia como episodio consultable.

## Arquitectura

```
Fuentes externas → pipeline (Pandas/GeoPandas) → Parquet/GeoParquet (data/processed)
                                     ↓                         ↑
                           modelo (scikit-learn) ── predicciones
                                                               ↓
                     data_access → aplicación Streamlit (mapa, buscador, episodios)
```

- **Pipeline** (`src/pipeline`): cuadrícula de 250 m en EPSG:25830, variables físicas, lluvia por celda
  (estación más cercana) con acumulados 1 h / 24 h, etiquetas y comprobaciones de datos.
- **Modelos** (`src/models`): regresión logística (base) y Random Forest (candidato), calibrados y con pesos de clase;
  validación temporal por episodio, umbrales Bajo/Medio/Alto fijados en validación, prueba retrospectiva con la DANA
  excluida del entrenamiento y comparación con las referencias de umbral de lluvia y PATRICOVA. La ficha del modelo
  queda en `data/models/model_card.json` y cada experimento en `data/models/experiments.jsonl`.
- **Acceso a datos** (`src/data_access`): única puerta del dashboard al almacenamiento.
- **Aplicación** (`src/app`): mapa, leyenda, selector de episodio/escenario, franja horaria, buscador, tabla accesible,
  consulta histórica y ficha del modelo, con aviso permanente de uso responsable.

## Estructura

```
├── docs/              # Entregas y diagramas
├── src/
│   ├── pipeline/      # Ingesta, limpieza, variables, etiquetas
│   ├── models/        # Entrenamiento, evaluación, predicción
│   ├── data_access/   # Lectura de datos para el dashboard
│   └── app/           # Streamlit
├── tests/
├── data/              # raw/ y processed/, no se versiona
├── environment/       # Dockerfile, docker-compose, requirements
└── .github/workflows/ # CI
```

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

## Fuentes de datos

| Fuente | Uso | Licencia |
|---|---|---|
| AEMET OpenData | Lluvia observada y previsión horaria | Aviso legal AEMET (reutilización con atribución) |
| PATRICOVA (GVA) | Peligrosidad de inundación, variable y referencia | Por verificar |
| CNIG/IGN | Modelo digital del terreno | CC BY 4.0 |
| Copernicus Land | Impermeabilidad del suelo | Copernicus data policy |
| Copernicus EMS | Inundaciones observadas (etiquetas) | Copernicus data policy |
| Ayuntamiento de Valencia | Barrios y distritos | Por verificar |
