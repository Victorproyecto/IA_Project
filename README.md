# DANA Predictor

Mapa de susceptibilidad a inundación en València, L'Horta Sud y L'Horta Oest, aprendido de la DANA de 2024, con datos
reales de fuentes oficiales.

> Prototipo académico informativo. No sustituye los avisos de las autoridades.

## Qué hace

Para cada celda de 100 m de los 22 municipios del área piloto calcula la probabilidad de que se inunde en un episodio
como la DANA del 29-10-2024 y la clasifica como **Bajo, Medio o Alto**. Usa 9 variables del terreno: altitud,
pendiente, % de suelo sellado, distancia al cauce principal, altura sobre el cauce (HAND) y 4 tipos de geología
(IGME). Aprende de las zonas que se
inundaron en la DANA según Copernicus EMS (satélite) y la huella oficial de la GVA. Los mapas oficiales de
peligrosidad (PATRICOVA y SNCZI) no son variables: son la referencia con la que se compara.

Es un mapa fijo de susceptibilidad aprendido de un solo episodio, no una predicción en tiempo real: no usa la lluvia y
no generaliza a otra cuenca (ver [Modelo](#modelo)).

## Requisitos

- Python 3.12 (o Docker)
- Unos 3 GB libres (los datos originales ocupan 1,7 GB)
- Conexión a internet en el navegador, porque el mapa de fondo (OpenStreetMap), la librería del mapa (MapLibre) y
  las sugerencias de direcciones (CartoCiudad) se cargan desde la red
- Los datos originales, que no van en git. Se descarga el zip de
  [Google Drive](https://drive.google.com/file/d/176-uFyjY6HrutmeO81buOrQoFvy12swU/view) y se descomprime dentro de
  `data/`, de modo que quede `data/raw/` (detalle en [docs/DATOS.md](docs/DATOS.md))

No hace falta ninguna variable de entorno. `.env.example` solo documenta `DATA_DIR`, opcional.

## Ejecución local

```bash
git clone https://github.com/Victorproyecto/IA_Project.git
cd IA_Project
python3 -m venv .venv && source .venv/bin/activate
pip install -r environment/requirements.txt

# antes: descomprimir el zip de Google Drive dentro de data/ (queda data/raw/)
python -m src.pipeline.ingest     # 1. data/raw → data/preparados: recorta y unifica las 11 fuentes y calcula la altura sobre el cauce, ~3 min
python -m src.pipeline.run        # 2. data/preparados → data/procesados (celdas, variables, etiquetas), ~20 min
python -m src.pipeline.train      # 3. entrena, valida y predice → data/modelos y data/procesados, ~1 min
streamlit run src/frontend/app.py # 4. aplicación en http://localhost:8501

python -m pytest                  # tests (~35 s, sobre un recorte pequeño del área)
```

### Con Docker

```bash
# con data/raw ya descomprimido, como arriba
docker compose -f environment/docker-compose.yml build pipeline
docker compose -f environment/docker-compose.yml run --rm ingest
docker compose -f environment/docker-compose.yml run --rm pipeline
docker compose -f environment/docker-compose.yml run --rm train
docker compose -f environment/docker-compose.yml up app          # http://localhost:8501
```

## Arquitectura

```
data/raw ─ingest─► data/preparados ─run─► data/procesados ─train─► data/modelos + predicciones
(Google Drive; nada de data/ va en git)                                   │
                                   frontend (Streamlit) ◄── backend (services.py)
```

- **Pipeline** (por lotes): prepara los datos, construye las celdas y entrena. Escribe Parquet / GeoParquet.
- **Backend**: única puerta de acceso a los resultados del pipeline.
- **Frontend**: Streamlit, con dos componentes JavaScript propios: el mapa de celdas (MapLibre GL JS) y el
  buscador con sugerencias (CartoCiudad, del IGN). Solo lee los resultados a través del backend.

## Estructura

```
src/
├── config.py              # rutas, ficheros, áreas (en metros), parámetros del pipeline y criterios de éxito
├── pipeline/
│   ├── ingest.py          # data/raw → data/preparados: recorta y unifica las 11 fuentes; altura sobre el cauce
│   ├── grid.py            # cuadrícula de celdas de 100 m y pendiente
│   ├── sources.py         # lleva cada fuente de data/preparados a la celda
│   ├── features.py        # variables derivadas (PATRICOVA invertido, unidad territorial)
│   ├── blocks.py          # bloques espaciales de 1 km y grupos de validación
│   ├── checks.py          # comprobaciones de datos antes de escribir
│   ├── run.py             # orquesta la preparación: zonas, etiquetas, tabla de entrenamiento y prueba externa
│   ├── evaluation.py      # métricas, umbrales Bajo/Medio/Alto y comprobación con calados de campo
│   ├── train.py           # validación por bloques, selección, referencias oficiales, prueba externa y ficha
│   └── predict.py         # predicción por celda
├── backend/
│   └── services.py        # lectura de resultados para el frontend (también la cuadrícula del mapa)
└── frontend/
    ├── app.py             # aplicación Streamlit
    └── components/        # componentes JavaScript: mapa de celdas y buscador con sugerencias
tests/                     # pytest: unitarios, datos, modelo, backend y frontend
environment/               # requirements.txt, Dockerfile, docker-compose.yml, entrypoint.sh
docs/
├── DATOS.md               # estructura de data/, fuentes, licencias y qué hace la ingesta
├── ENTREGA_NF3.md         # documento de la entrega NF3
├── MEJORAS_MODELO_NF3.md  # cambios del modelo respecto a la versión de prueba de NF3
├── modelo_datos.png       # modelo de datos
├── capturas/              # capturas de la aplicación y del terminal
└── entregas/              # PDF entregados (Fase 0 y Fase 1)
data/                      # no se versiona
├── raw/                   # originales (zip en Google Drive)
├── preparados/            # se genera con ingest; lo que lee el pipeline
├── procesados/            # se genera con run y train
└── modelos/               # se genera con train
```

## Datos

| Fichero en `data/preparados/` | Fuente | Licencia |
|---|---|---|
| `mdt_altitud.tif` | IGN / CNIG, PNOA MDT05 | CC BY 4.0 |
| `impermeabilidad_suelo.tif` | Copernicus Land, Imperviousness Density 2024 | Copernicus data policy |
| `zonas_inundadas_dana2024.gpkg` | Copernicus EMS (EMSR773) + huella oficial de la DANA (GVA / ICV) | Copernicus data policy; GVA por verificar |
| `calados_campo_dana2024.gpkg` | GVA / ICV, calado medido en campo (no se usa para entrenar) | Por verificar |
| `cauces_principales.gpkg` | IGN, Hidrografía de España (INSPIRE HY) | CC BY 4.0 |
| `altura_sobre_cauce.tif` | Calculada en la ingesta con el MDT y los cauces (HAND) | La de sus fuentes (CC BY 4.0) |
| `geologia_igme.gpkg` | IGME, GEODE 50 | Por verificar |
| `suelo_no_urbano_clc2018.gpkg` | Copernicus Land, Corine Land Cover 2018 | Copernicus data policy |
| `municipios.gpkg` | IGN / CNIG, límites municipales | CC BY 4.0 |
| `barrios_valencia.gpkg` | Ayuntamiento de València, datos abiertos | CC BY 4.0 |
| `patricova_peligrosidad.gpkg` | GVA, PATRICOVA (solo referencia) | Por verificar |
| `snczi_zona_inundable_t500.gpkg` | MITECO, SNCZI T=500 años (solo referencia) | Por verificar |

Detalle de cada fuente, cómo descargarla y qué hace la ingesta: [docs/DATOS.md](docs/DATOS.md).

## Modelo

- Una fila por celda: 41.401 celdas de 100 m, 9.765 inundadas en la DANA (24 %). Se excluyen el mar, los arrozales
  y el agua permanente (Corine Land Cover).
- Regresión logística y Random Forest, calibrados y con pesos de clase. Se elige el de mayor F1 entre los que alcanzan
  Recall ≥ 0,80.
- Validación cruzada en 5 grupos de bloques espaciales de 1 km: cada celda se predice con un modelo que no ha visto su
  zona.
- Resultado: Random Forest `0.4.0-rf`, Recall 0,80, **F1 0,72** (objetivo de la Fase 1: 0,70; marcarlo todo daría
  0,38), ROC-AUC 0,92. Los mapas oficiales tal cual: PATRICOVA F1 0,46 y SNCZI T=500 F1 0,65. El objetivo de F1 se
  alcanza con poco margen al añadir la altura sobre el cauce (sin ella, 0,687) y no se mantiene con bloques de 3 km
  (0,63 con y sin ella): detalle en [docs/MEJORAS_MODELO_NF3.md](docs/MEJORAS_MODELO_NF3.md), apartado 2.5.
- Calados medidos en campo (≥ 0,3 m): el 83 % cae en celdas Medio o Alto, frente al 67 % dentro del SNCZI y el 34 %
  dentro de PATRICOVA. No es una validación independiente: la huella de la GVA, parte de la etiqueta, cubre casi
  todos esos puntos.
- **Prueba externa en la Ribera** (otra cuenca, que el modelo nunca ve): ROC-AUC 0,59. El modelo no generaliza fuera
  del área en la que aprende; allí los mapas oficiales funcionan mucho mejor.
- Ficha completa en `data/modelos/model_card.json` y en la pestaña *Modelo y datos* de la aplicación. Cambios
  respecto a la versión de prueba de NF3: [docs/MEJORAS_MODELO_NF3.md](docs/MEJORAS_MODELO_NF3.md).

## Contribuir

Cada integrante sube su parte en su propia rama, otro miembro del equipo la revisa y se integra en `develop`; de ahí
pasa a `main`. La versión entregada en cada fase se marca con un tag (NF3: `nf3-entrega`).

## Equipo

Grupo 10, Proyecto Final de Big Data e IA

| Nombre | Rol |
|---|---|
| Dennis Bonilla | Data |
| Jaume Palau | ML |
| Víctor Martínez | Platform |
| Rafael Chambilla | BI |
