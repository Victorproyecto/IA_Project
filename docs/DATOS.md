# Datos: estructura, fuentes y cómo rehacerlos

## Estructura

```
data/
├── raw/          originales de cada fuente, tal como se descargan (1,7 GB; zip en Google Drive)
├── preparados/   un fichero por fuente, recortado al área de estudio (lo genera ingest, ~95 MB)
├── procesados/   celdas, etiquetas, tabla de entrenamiento y predicciones (lo genera run y train)
└── modelos/      modelo entrenado, ficha del modelo y registro de experimentos (lo genera train)
```

```
raw ──ingest──► preparados ──run──► procesados ──train──► modelos + predicciones
```

**Nada de `data/` se versiona** (`.gitignore` ignora la carpeta entera). Para ejecutar el proyecto:

1. Descarga el zip de [Google Drive](https://drive.google.com/file/d/176-uFyjY6HrutmeO81buOrQoFvy12swU/view) y
   descomprímelo dentro de `data/`, de modo que quede `data/raw/` con la estructura de abajo.
2. `python -m src.pipeline.ingest` → `data/preparados` (unos minutos).
3. `python -m src.pipeline.run` y `python -m src.pipeline.train` → `data/procesados` y `data/modelos`.

### Por qué no se versiona nada de `data/`

- Los originales ocupan 1,7 GB (solo la zona inundable del SNCZI pesa 1,1 GB, de toda España) y GitHub no admite
  ficheros de más de 100 MB.
- Todo lo demás se genera a partir de `data/raw` con la ingesta, `run` y `train`: versionarlo duplicaría binarios
  (cada cambio guarda una copia entera en el historial) y podría quedar desfasado respecto al código.
- Sin datos, la CI ejecuta los tests que no los necesitan y salta el resto (los que usan `data/preparados` y el de la
  huella de la GVA). En local, con `data/raw` e `ingest` hecho, se ejecutan todos.

### Ficheros preparados

- Están recortados al área de estudio (València, L'Horta Sud, L'Horta Oest y la Ribera) y ocupan ~95 MB en total.
- El MDT se guarda en **centímetros enteros** (con el factor de escala 0,01 dentro del fichero, que el pipeline aplica
  al leer): en metros con decimales ocuparía 146 MB. No se pierde nada por encima de 1 cm, muy por debajo de la
  precisión del propio MDT (decímetros). El mar (altitud 0 exacta) se conserva tal cual.

## Ficheros preparados (`data/preparados/`)

| Fichero | Contenido | Fuente | Licencia |
|---|---|---|---|
| `mdt_altitud.tif` | Altitud (cm, píxel de 5 m); el mar es 0 | IGN / CNIG, PNOA MDT05 | CC BY 4.0 |
| `impermeabilidad_suelo.tif` | % de suelo sellado (10 m, 2024) | Copernicus Land, Imperviousness Density | Copernicus data policy |
| `zonas_inundadas_dana2024.gpkg` | **Etiqueta:** zonas inundadas en la DANA por gravedad y fuente | Copernicus EMS (EMSR773) + huella oficial de la GVA | Copernicus data policy; GVA por verificar |
| `calados_campo_dana2024.gpkg` | Calado medido en campo tras la DANA (no se usa para entrenar) | GVA / ICV | Por verificar |
| `cauces_principales.gpkg` | Cauces con nombre y en superficie | IGN, Hidrografía de España (INSPIRE HY) | CC BY 4.0 |
| `geologia_igme.gpkg` | Unidades cuaternarias agrupadas (aluvial, marjal, abanico, glacis) | IGME, GEODE 50 | Por verificar |
| `suelo_no_urbano_clc2018.gpkg` | Arrozales y agua permanente (esas celdas se excluyen) | Copernicus Land, Corine Land Cover 2018 | Copernicus data policy |
| `municipios.gpkg` | Términos municipales | IGN / CNIG | CC BY 4.0 |
| `barrios_valencia.gpkg` | Barrios de València con su distrito | Ayuntamiento de València | CC BY 4.0 |
| `patricova_peligrosidad.gpkg` | Peligrosidad de inundación, niveles 1-6 (**solo referencia**) | GVA, PATRICOVA | Por verificar |
| `snczi_zona_inundable_t500.gpkg` | Zona inundable T=500 años (**solo referencia**) | MITECO, SNCZI | Por verificar |

## Originales (`data/raw/`), la entrada de la ingesta

Están en el zip de [Google Drive](https://drive.google.com/file/d/176-uFyjY6HrutmeO81buOrQoFvy12swU/view). Si hay
que volver a descargarlos de su origen, la tabla indica de dónde sale cada uno.

```
data/raw/
├── cnig_mdt05/                PNOA_MDT05_ETRS89_HU30_{0695,0696,0721,0722,0746,0747}_LID.tif
├── copernicus_land_imd/       CLMS_NVLCC_IMD_S2024_R10m_E34N18_03035_V01_R01.tif
├── copernicus_ems_EMSR773/    EMSR773_AOI{01,03,04}_DEL_PRODUCT_{observedEventA,floodDepthA}_v1.shp (+ .dbf, .shx, .prj)
├── gva_dana2024/              huella_inundacion.geojson
│                              altura_agua_ive.geojson, altura_agua_sgrt.geojson, altura_agua_upv.geojson
├── ign_hidrografia/           watercourse.geojson
├── igme_geode50/              geode50.geojson
├── copernicus_clc/            clc2018_no_urbano.geojson
├── cnig_limites_municipales/  recintos_municipales_inspire_peninbal_etrs89.shp (+ .dbf, .shx, .prj, .cpg)
├── ayuntamiento_valencia/     barrios.geojson, distritos.geojson
├── gva_patricova/             orde_patricova_peligrosidad_inun0.shp (+ .dbf, .shx, .prj, .cpg)
└── snczi/                     laminasPB-q500.zip (tal cual, sin descomprimir)
```

| Carpeta | Qué es | Dónde se descarga | Licencia |
|---|---|---|---|
| `cnig_mdt05` | Modelo digital del terreno, píxel de 5 m | Centro de Descargas del CNIG, producto "MDT05", hojas MTN50 0695, 0696, 0721, 0722, 0746 y 0747: https://centrodedescargas.cnig.es | CC BY 4.0 (IGN) |
| `copernicus_land_imd` | % de suelo sellado, 10 m, 2024 | Copernicus Land Monitoring Service, "Imperviousness Density 2024 (raster 10 m and 100 m), Europe, 3-yearly": https://land.copernicus.eu (requiere cuenta EU Login). Colección **Imperviousness Density 10 m**, proyección **EPSG:3035**, GeoTIFF. El fichero debe empezar por `CLMS_NVLCC_IMD_` | Copernicus data policy. DOI 10.2909/f0bcb6f0-d775-4218-bec1-13adea01ff8f |
| `copernicus_ems_EMSR773` | Zonas inundadas y calado de la DANA (satélite) | Copernicus EMS Rapid Mapping, activación EMSR773, productos DEL de AOI01, AOI03 y AOI04 (descarga directa): `https://rapidmapping.emergency.copernicus.eu/backend/EMSR773/AOI01/DEL_PRODUCT/EMSR773_AOI01_DEL_PRODUCT_v1.zip` y lo mismo con `AOI03` y `AOI04` | Copernicus data policy |
| `gva_dana2024` | **Huella oficial de la zona inundada** (parte de la etiqueta) y **altura de agua medida en campo** | WFS del ICV `https://terramapas.icv.gva.es/00_DANA2024`, en GeoJSON y EPSG:25830 (`...?service=WFS&version=2.0.0&request=GetFeature&typenames=ms:<capa>&srsName=urn:ogc:def:crs:EPSG::25830&outputFormat=application/json; subtype=geojson`). Capas: `DANA2024.ZonasInundadas.HuellaInundacion` → `huella_inundacion.geojson`; `DANA2024.ZonasInundadas.AlturaAgua.IVE`, `.SGRT` y `.UPV` → `altura_agua_ive.geojson`, `altura_agua_sgrt.geojson`, `altura_agua_upv.geojson` | **Por verificar** (crítico: la etiqueta depende de la huella) |
| `ign_hidrografia` | Cauces (Hidrografía de España, INSPIRE HY, origen BTN25) | API OGC del IGN: `https://api-features.idee.es/collections/watercourse/items?bbox=-0.70000,39.10000,-0.14000,39.65000&f=json&limit=1000` (paginado), guardado como `watercourse.geojson` | CC BY 4.0 (IGN) |
| `igme_geode50` | Mapa geológico continuo 1:50.000 (GEODE 50) | Servicio ArcGIS REST del IGME `https://mapas.igme.es/gis/rest/services/Cartografia_Geologica/IGME_Geode_50/MapServer`, polígonos del área de estudio en GeoJSON (WGS84), guardado como `geode50.geojson` | Por verificar |
| `copernicus_clc` | Corine Land Cover 2018: arrozales (213) y agua permanente (511, 512, 521, 522, 523) | Servicio ArcGIS REST de la EEA `https://image.discomap.eea.europa.eu/arcgis/rest/services/Corine/CLC2018_WM/MapServer/0`, polígonos del área de estudio con esos códigos en GeoJSON, guardado como `clc2018_no_urbano.geojson` | Copernicus data policy |
| `cnig_limites_municipales` | Límites municipales (recintos) | Centro de Descargas del CNIG → Información geográfica de referencia → Delimitaciones territoriales → **Límites municipales, provinciales y autonómicos** → "Límites y Unidades Administrativas Actuales" (SHAPEFILE). Del zip solo hace falta la carpeta `recintos_municipales_inspire_peninbal_etrs89` | CC BY 4.0 (IGN) |
| `ayuntamiento_valencia` | Barrios y distritos | Geoportal del Ayuntamiento de València: barrios `https://geoportal.valencia.es/apps/OpenData/UrbanismoEInfraestructuras/UrbanismoEInfraestructuras__224.zip`, distritos `..._225.zip`. Vienen en shapefile: convertir a GeoJSON con el nombre indicado | CC BY 4.0 |
| `gva_patricova` | Peligrosidad de inundación (niveles 1-7) | Institut Cartogràfic Valencià, capa "orde_patricova_peligrosidad_inun" en SHP: http://descargas.icv.gva.es/server_api/gdb/descarga/index.html?capas=%22infraestructuras.gdb/orde_patricova_peligrosidad_inun%22&formato=Shapefile%20-%20SHP%20-%20.shp&geocat=DATO_DIRECTA_SHP | Por verificar |
| `snczi` | Zonas inundables del SNCZI, periodo de retorno de 500 años (toda España) | MITECO, descarga de láminas de zonas inundables: https://www.miteco.gob.es/es/cartografia-y-sig/ide/descargas/agua/zi-lamina.html, fichero `laminasPB-q500.zip` | Por verificar |

## Qué hace la ingesta (`src/pipeline/ingest.py`)

Todo se recorta al rectángulo que contiene el área piloto y la Ribera (prueba externa), definidos en metros en
`src/config.py`. Las capas vectoriales llevan un margen de 2 km (5 km los cauces, para no perder el cauce más cercano
a las celdas del borde); los rásteres no lo necesitan, porque solo se leen dentro de las celdas.

| Fichero preparado | Transformación |
|---|---|
| `mdt_altitud.tif` | Une las 6 hojas, recorta y guarda la altitud en centímetros enteros. La tierra que redondearía a 0 cm se queda a ±1 cm, para que 0 siga siendo solo el mar |
| `impermeabilidad_suelo.tif` | Recorta la tesela y declara 255 (mar o sin dato) como nodata |
| `zonas_inundadas_dana2024.gpkg` | **Etiqueta.** Copernicus EMS: zona inundada = agua o rastro de agua (`observedEvent`); gravedad por calado (`floodDepth`): < 0,5 m Leve, 0,5-1 m Moderada, > 1 m Grave; lo que no tiene calado cuenta como Leve. **Huella de la GVA:** se añade lo que no marca Copernicus (sobre todo cascos urbanos, donde el satélite no ve el agua entre edificios), como Leve. Cada polígono lleva su fuente (`source`) |
| `calados_campo_dana2024.gpkg` | Puntos con calado medido (`altura_agua_m` → `depth_m`). No se usan para entrenar: dan gravedad a las celdas que solo marca la GVA y sirven para comprobar el modelo |
| `cauces_principales.gpkg` | El nuevo cauce del Turia (*Canal del Riu Túria*) y los cursos naturales **con nombre y en superficie**. Se excluyen acequias, canales de riego, tramos subterráneos, cursos sin nombre (desagües menores) y el antiguo cauce del Turia (no lleva agua desde que el río se desvió tras la riada de 1957) |
| `geologia_igme.gpkg` | Agrupa las unidades por palabras de su descripción (`config.GEOLOGY_GROUPS`); el resto no se guarda |
| `suelo_no_urbano_clc2018.gpkg` | Se queda con los códigos de arrozal y agua permanente (`config.NON_URBAN_CLC_CODES`) |
| `municipios.gpkg` | Recintos municipales (`NAMEUNIT` → `municipio`) |
| `barrios_valencia.gpkg` | Añade a cada barrio el nombre de su distrito |
| `patricova_peligrosidad.gpkg` | Renombra `n_pelig` a `nivel` y se queda con los niveles 1-6 (el 7, geomorfológico, no tiene frecuencia ni calado) |
| `snczi_zona_inundable_t500.gpkg` | Lee solo el área de estudio del fichero de toda España y quita la coordenada Z |

## Nota sobre la huella de la GVA

El servicio del ICV entrega la huella como **un único polígono con geometría inválida**: las partes separadas de la
zona inundada vienen guardadas como "huecos" fuera del borde exterior, de modo que su superficie sale negativa
(−49.515 ha). Es un fallo de formato al exportar, no de los datos. La ingesta la repara con `make_valid` y comprueba
que la superficie resultante coincide con la que declara la propia GVA en el campo `area_ha` (42.550 ha); si no
coincide, se detiene. Hay un test para ello (`tests/test_data.py`), que necesita la descarga original.

## Privacidad

Ninguna fuente contiene datos personales. Las sugerencias del buscador las da CartoCiudad (IGN): el navegador le
envía lo que se escribe a partir de 3 letras, limitado a los códigos postales del área piloto. La aplicación no guarda
las direcciones que se buscan.
