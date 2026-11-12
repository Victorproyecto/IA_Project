# Mejoras del modelo y del producto (NF3)

Este documento recoge los cambios hechos sobre la versión de prueba de NF3 (rama `nf3-prueba`), por qué se hicieron,
qué resultados dan y cómo hay que leerlos.

## 1. Resumen

| | NF3 (antes) | Ahora |
|---|---|---|
| Área piloto | Ciudad de València | **València, L'Horta Sud y L'Horta Oest** (22 municipios, la cuenca del Poyo y lo que inundó) |
| Tamaño de celda | 250 m (1.397 celdas) | **100 m (41.401 celdas)** |
| Etiqueta (qué se inundó) | Copernicus EMS (satélite) | **Copernicus EMS + huella oficial de la GVA** |
| Celdas inundadas | 19 % | 24 % |
| Variables | 4 de terreno | **8: altitud, pendiente, suelo sellado, distancia al cauce principal y 4 de geología** |
| Modelo elegido | Regresión logística `0.2.0-lr` | **Random Forest `0.3.0-rf`** |
| Recall (validación) | 0,80 | 0,80 |
| F1 (validación) | 0,39 | **0,69** (marcarlo todo daría 0,38) |
| ROC-AUC | 0,74 | **0,90** |
| Frente a los mapas oficiales en el área | - | F1 0,69 frente a 0,65 (SNCZI T=500) y 0,46 (PATRICOVA) |
| En otra cuenca que no ha visto (la Ribera) | No se probó | **No generaliza** (ROC-AUC 0,50; PATRICOVA 0,80 y SNCZI 0,78) |
| Celdas de otros municipios con barrio de València | 344 (error) | 0 |
| Tests automáticos | 20 (no se ejecutaban en Windows) | **32, todos pasan también en Windows y tardan ~35 s** |

En resumen, el producto es un mapa de susceptibilidad aprendido de la DANA de 2024. Cubre la zona más afectada y
supera a los mapas oficiales dentro del área en que se entrena, pero no es un predictor que funcione en cualquier zona
o con cualquier tormenta, como muestra la prueba en la Ribera. El F1 (0,687) queda justo por debajo del objetivo de
la Fase 1 (0,70).

## 2. Cambios y motivo

### 2.1 Área y datos

| Cambio | Motivo | Ficheros |
|---|---|---|
| Área piloto ampliada a 22 municipios completos (València, L'Horta Sud y L'Horta Oest) y fijada en metros UTM, con coordenadas múltiplos de 100 y 250 m | Es la cuenca del Poyo y donde la DANA causó más daño; la ciudad de València apenas se inundó. Fijarla en metros evita perder franjas del borde al cambiar el tamaño de celda | `config.py`, `grid.py`, `run.py` |
| Celdas de 100 m | Mismo rendimiento que 250 m con 6 veces más detalle; a 50 m el modelo empeora | `config.py` |
| Límites municipales (CNIG): municipio de cada celda por mayor superficie; barrio y distrito solo dentro de València | En NF3 cada celda recibía el barrio de València más cercano, también las de Mislata o Alboraya (344 celdas mal asignadas). Las celdas de playa, fuera de los recintos (que siguen la costa), toman el municipio más cercano | `ingest.py`, `sources.py`, `features.py`, `checks.py` |
| Distancia al cauce principal (IGN): nuevo cauce del Turia y cursos naturales con nombre, en superficie | Las inundaciones de la zona son desbordamientos de barrancos. Se excluyen acequias, tramos subterráneos (el IGN guarda así 10,5 km del antiguo cauce bajo el Jardí), cursos sin nombre (desagües menores, como el de Natzaret) y el antiguo cauce del Turia | `ingest.py`, `sources.py` |
| Geología del IGME (GEODE 50): fracción de la celda en 4 grupos (llanura de inundación o fondo de valle, limos de albufera o marjal, abanico aluvial, glacis o coluvión) | Describe el terreno que el agua encontró: las zonas de abanico y marjal son las que más pesan después de la altitud y los cauces | `ingest.py`, `sources.py`, `config.py` |
| Etiqueta Copernicus EMS + huella oficial de la GVA | El satélite no ve el agua entre edificios: en L'Horta Sud, el 39 % de los puntos donde se midió agua (1,3-1,6 m de media) caía en celdas que Copernicus daba por no inundadas. La huella de la GVA cubre esos cascos urbanos | `ingest.py`, `sources.py`, `checks.py` |
| Gravedad por calados de campo en las celdas que solo marca la GVA | La huella no trae calado; si en la celda se midió agua, manda el calado medido | `sources.py` |
| Reparación de la huella de la GVA | Llega con geometría inválida (superficie negativa). Se repara y se comprueba que mide lo que declara la GVA (42.550 ha) | `ingest.py`, `tests/test_data.py` |
| Exclusión del agua permanente y los arrozales (Corine Land Cover 2018) | No son suelo urbano, el satélite no distingue esa agua de la inundación y son aciertos fáciles que inflan las métricas (lago de la Albufera, cauces, arrozales) | `ingest.py`, `sources.py`, `run.py` |
| PATRICOVA y SNCZI solo como referencia, no como variables | El modelo aprende del terreno y se compara con los mapas oficiales; si los usara como variable, repetiría parte de su información (con PATRICOVA como variable el F1 sube de 0,687 a 0,704) | `config.py`, `train.py` |
| Prueba externa en la Ribera (Magro y Xúquer: Algemesí, l'Alcúdia, Carlet, Guadassuar…) | Otra cuenca que también se inundó en la DANA y que el modelo nunca ve: mide si generaliza | `run.py`, `train.py` |

### 2.2 Entrenamiento y validación

| Cambio | Para qué | Ficheros |
|---|---|---|
| Regla de selección explícita: entre los modelos con Recall ≥ 0,80, el de mayor F1 (desempate por PR-AUC) | Criterio trazable. Candidatos: regresión logística y Random Forest, calibrados y con pesos de clase | `train.py` |
| Niveles por recall: *Medio* es el umbral más alto que mantiene Recall ≥ 0,80 y *Alto*, el más alto con Recall ≥ 0,50 | Los niveles no dependen de cuántas celdas se inundaron: Medio + Alto encuentran el 80 % de lo inundado y Alto concentra la mitad, la de mayor probabilidad | `evaluation.py`, `config.py` |
| Referencia trivial (F1 de marcarlo todo como inundado) | El F1 hay que leerlo frente a esa cifra (0,38) | `train.py` |
| Comparación con PATRICOVA y el SNCZI (T=500) tal cual, y prueba con PATRICOVA como variable | Ver cuánto añade el modelo a los mapas oficiales y cuánto añadiría usarlos | `train.py` |
| Importancia por permutación (caída de PR-AUC al desordenar cada variable) | Mismo criterio para los dos modelos | `train.py` |
| Prueba externa en la Ribera, a más de 1 km del área piloto | Medir si el modelo funciona en una cuenca que no ha visto | `train.py` |
| Comprobación con calados de campo (GVA: IVE, SGRT, UPV) | Ver cuánto de lo medido en campo cae en zona de riesgo del modelo y de los mapas oficiales. No es una validación independiente (la huella de la GVA forma parte de la etiqueta) | `evaluation.py`, `train.py` |
| Ficha del modelo en UTF-8 | En Windows se escribía en cp1252 y las tildes salían rotas en Linux/Docker | `train.py`, `services.py` |

### 2.3 Producto

Los tiempos son los de abrir la aplicación en local con Edge (mediana de 7 cargas, antes y después, en el mismo
ordenador).

| Cambio | Resultado | Ficheros |
|---|---|---|
| Mapa de celdas con JavaScript (MapLibre GL JS, componente propio de Streamlit). Python envía un byte por celda (~130 KB) y el navegador dibuja un cuadrado exacto por celda y resuelve el clic | Carga hasta ver el mapa: 1,39 s → 0,70 s. Clic hasta ver la ficha: 0,56 s → 0,09 s. Datos al cargar: 7,3 → 3,4 MB. Los bordes son nítidos a cualquier zoom (antes, una imagen con escalones de hasta 5 m) | `services.py`, `app.py`, `components/` |
| Buscador con sugerencias: municipios, distritos y barrios del área al instante y, a partir de 3 letras, calles y direcciones de CartoCiudad (IGN), limitadas a los códigos postales del área | Ya no hace falta escribir la dirección exacta; sustituye a Nominatim, cuyas normas no permiten el autocompletado | `components/place_search.js`, `config.py` |
| Ficha de celda explicada: posición relativa ("más que el X % de las celdas"), qué significa cada nivel, cauce más cercano y su distancia, fuente de lo observado (Copernicus EMS y/o GVA) | | `services.py`, `app.py` |
| Tabla con municipio, percentil y distancia al cauce; resumen por distrito o municipio | La tabla pesa 3,2 MB en lugar de 5,9 (los textos repetidos van como categorías) | `app.py` |
| Pestaña del modelo: comparativa con PATRICOVA y el SNCZI, referencia trivial, prueba externa en la Ribera, comprobación con calados, importancia y limitaciones | | `app.py` |

### 2.4 Correcciones y repositorio

| Cambio | Motivo | Ficheros |
|---|---|---|
| Suelo sellado regenerado desde la descarga de Copernicus | El ráster usado en las pruebas anteriores estaba desplazado unos 1,4 km respecto al original (correlación 0,33 con el original; 0,97 al compensar el desplazamiento). Corregido, el área tiene 41.401 celdas en lugar de 39.770 (entran las de la costa: El Saler, Pinedo…) y las métricas apenas cambian (F1 0,690 → 0,687) | `ingest.py` |
| Datos fuera del repositorio y estructura `raw / preparados / procesados / modelos` | Los originales ocupan 1,7 GB y GitHub no admite ficheros de más de 100 MB: están en Google Drive y el resto se genera con `ingest`, `run` y `train`. El MDT preparado va en centímetros enteros (146 MB → 47 MB) sin pérdida por encima de 1 cm | `.gitignore`, `config.py`, `docs/DATOS.md` |
| Tests en 35 s en lugar de más de 5 min | Usan datos reales sobre un recorte de 6 x 6 km, entrenan un bosque de 10 árboles y comparten una única preparación. En la CI, sin datos, se saltan los que los necesitan | `tests/conftest.py`, `train.py` |
| Tests en Windows: si no se puede crear el enlace simbólico a `data/preparados`, se crea una unión de directorio | Los tests fallaban en Windows antes de ejecutarse | `tests/conftest.py` |
| Constantes con nombre en lugar de números sueltos | Cada parámetro tiene nombre y explicación en un solo sitio | `config.py` y `src/` |
| DATOS.md con las fuentes nuevas y cómo descargarlas | El portal anterior del Ayuntamiento ya no existe: el proyecto no se podía reproducir | `docs/DATOS.md` |

## 3. Resultados

Todas las cifras salen de `data/modelos/model_card.json` (ejecución del 11/11/2026).

### 3.1 Modelos y referencias (validación cruzada por bloques de 1 km)

| Modelo | Recall | F1 | Precisión | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Random Forest (elegido)** | 0,80 | **0,687** | **0,602** | **0,902** | **0,685** |
| Regresión logística | 0,80 | 0,613 | 0,497 | 0,818 | 0,523 |
| Random Forest con PATRICOVA como variable | 0,80 | 0,704 | 0,629 | 0,918 | 0,762 |
| SNCZI T=500 tal cual (cualquier celda en la zona = riesgo) | 0,575 | 0,646 | 0,738 | 0,756 | 0,525 |
| PATRICOVA tal cual (cualquier nivel = riesgo) | 0,375 | 0,458 | 0,588 | 0,651 | 0,409 |
| Marcarlo todo como inundado | 1,00 | 0,382 | 0,236 | 0,500 | - |

- El modelo cumple Recall ≥ 0,80 y **se queda justo por debajo de F1 ≥ 0,70** (criterios de la Fase 1). Supera a la
  regresión logística, al SNCZI y a PATRICOVA.
- **El F1 hay que leerlo frente a 0,38**, que es lo que daría marcarlo todo: la mejora real es de +0,31. El ROC-AUC
  (0,90) no depende de cuántas celdas se inundaron y va de 0,88 a 0,93 según el grupo de bloques.
- **Con PATRICOVA como variable** el modelo sube a F1 0,704 (+0,017): lo que PATRICOVA añade a lo que ya se sabe del
  terreno es poco.
- **La comparación favorece algo al modelo**, que aprende de esta misma DANA, mientras que los mapas oficiales son
  anteriores y describen otra cosa (periodos de retorno).
- Los mapas oficiales dejan fuera una parte grande de lo inundado: el 62 % de las celdas inundadas (6.102 de 9.765)
  estaba fuera de PATRICOVA y el 42 % (4.150) fuera de la zona de T=500.

### 3.2 Prueba externa en la Ribera

El modelo final, entrenado con todo el área piloto, predice la Ribera (otra cuenca, a más de 1 km del área): 20.419
celdas, el 63 % inundadas.

| | F1 | ROC-AUC | Recall |
|---|---|---|---|
| Marcarlo todo como inundado | 0,775 | 0,500 | 1,00 |
| **Modelo** | **0,060** | **0,505** | 0,03 |
| PATRICOVA | 0,814 | 0,800 | 0,78 |
| SNCZI T=500 | 0,842 | 0,776 | 0,86 |

**En una cuenca que no ha visto, el modelo no sirve:** su ROC-AUC de 0,50 es el de tirar una moneda y los mapas
oficiales funcionan mucho mejor. Con un solo episodio, el modelo aprende cómo se inundó cada zona en 2024, no reglas
que se trasladen a otra. En L'Horta casi no hay llanura de inundación de la que aprender lo que pasa en la Ribera.

Consecuencias:

- La validación por bloques de 1 km mide si el modelo **rellena bien los huecos dentro de una zona que conoce**, que es
  lo que hace el mapa. Es válida para el área cubierta, pero algo optimista y no dice nada de otras zonas.
- Para que el modelo generalice hace falta **entrenar con varios episodios** (trabajo futuro).

### 3.3 Comprobación con calados de campo

| | Modelo (celdas Medio o Alto, en validación) | SNCZI T=500 | PATRICOVA |
|---|---|---|---|
| Puntos con ≥ 0,3 m de agua medida (3.192 puntos en 1.409 celdas) | **86 %** | 67 % | 34 % |

No es una validación independiente: la huella de la GVA, que forma parte de la etiqueta, cubre casi todos estos
puntos. Lo que sí muestra es que **dos de cada tres puntos donde se midió agua quedaban fuera de PATRICOVA**.

### 3.4 Importancia de las variables (permutación)

| Variable | Caída de PR-AUC al desordenarla |
|---|---|
| Distancia al cauce principal | 0,317 |
| Altitud media | 0,288 |
| Pendiente | 0,191 |
| Geología: abanico aluvial | 0,174 |
| Geología: limos de albufera o marjal | 0,143 |
| Suelo sellado | 0,114 |
| Geología: glacis o coluvión | 0,029 |
| Geología: llanura de inundación | 0,012 |

### 3.5 Niveles y falsos negativos

Umbrales fijados en validación: **Medio ≥ 0,356** (Medio + Alto encuentran el 80 % de lo inundado) y **Alto ≥ 0,532**
(concentra el 50 %). El mapa tiene 28.324 celdas en Bajo, 5.601 en Medio y 7.476 en Alto.

Las zonas con más celdas inundadas no detectadas en validación son Picassent (55 %, 183 de 333), Alfafar (53 %, 173
de 328), Manises (34 %) y Aldaia (31 %): cascos urbanos que la huella de la GVA marca como inundados y que el terreno
por sí solo no distingue. La lista completa está en la pestaña *Modelo y datos*.

## 4. Pruebas descartadas

| Idea | Resultado | Por qué se descarta |
|---|---|---|
| Celdas de 50 m | F1 0,35 y ROC-AUC 0,69 (piloto de solo València) | Más ruido que detalle con las variables actuales |
| Densidad de imbornales (Ayuntamiento) | Subía el F1 de forma engañosa | Solo hay datos dentro de València: fuera vale 0 y el modelo aprendía "fuera de València = riesgo". Además, el número de imbornales no mide la capacidad de drenaje |
| Lluvia (local o aguas arriba) | - | Con un solo episodio, la lluvia aguas arriba equivale a decirle al modelo qué barranco se desbordó: mejora las cifras sin aprender nada trasladable. La lluvia local es casi igual en todo el piloto |
| Gradient Boosting como tercer candidato | Empata con Random Forest | No aporta y añade complejidad |
| Solo Copernicus como etiqueta | Cascos urbanos de L'Horta Sud en nivel Bajo | Ver 2.1: el satélite no ve el agua entre edificios |
| PATRICOVA como variable del modelo | F1 0,704 (+0,017) | Se deja como referencia con la que comparar, no como variable |
| SNCZI para T=100 años | - | Se deja solo T=500, el más comparable a un episodio extremo como la DANA |
| Generalización entre València y L'Horta Sud | El modelo no superaba a PATRICOVA | Con el área ampliada a las dos comarcas, ya no hay dos zonas separables; la Ribera es una prueba externa real |
| Mapa que cambia de barrios a celdas según el zoom | Las celdas de nivel Bajo desaparecían al acercarse | Se deja solo el mapa de celdas |

## 5. Pendiente

- **Licencias de los datos:** por verificar las del IGME (GEODE 50), el SNCZI, PATRICOVA y los datos de la GVA (la
  etiqueta depende de la huella).
- **F1 por debajo del objetivo:** 0,687 frente a 0,70. El Recall se cumple, el F1 hay que leerlo frente a la referencia
  trivial (0,38) y el ROC-AUC (0,90) no depende de la proporción de inundadas.

## 6. Trabajo futuro (NF4)

- **Entrenar con varios episodios** (otras activaciones de Copernicus EMS en zonas mediterráneas) y validar dejando
  fuera un episodio entero. Es lo único que permitiría afirmar que el modelo funciona en una inundación que no ha visto.
- Con varios episodios, la **lluvia en la cuenca** de cada uno sí sería una variable útil.

## 7. Cómo reproducirlo

```bash
python -m src.pipeline.ingest   # data/raw → data/preparados (incluye la huella de la GVA, reparada)
python -m src.pipeline.run      # celdas de 100 m, variables y etiqueta (~5 min)
python -m src.pipeline.train    # validación, prueba externa, ficha del modelo y predicciones (~2 min)
streamlit run src/frontend/app.py
python -m pytest                # 32 tests (~35 s)
```
