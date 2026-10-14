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

Las instrucciones de ejecución se añadirán con la primera versión del MVP (v0.1).

## Estructura

```
├── docs/          # Documentación y entregas
├── src/           # pipeline, models, data_access, app
├── tests/
├── data/          # No se versiona
└── environment/
```

## Contribuir

Todo cambio entra por pull request a `develop`, con revisión de otro miembro del equipo. Los commits siguen [Conventional Commits](https://www.conventionalcommits.org/).

## Fuentes de datos

AEMET · PATRICOVA · CNIG/IGN · Copernicus Land · Copernicus EMS · Ayuntamiento de Valencia
