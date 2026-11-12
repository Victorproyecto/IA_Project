// Mapa de celdas (componente de Streamlit, st.components.v2) dibujado con MapLibre GL JS.
//
// Python envía la cuadrícula como un byte por celda (nivel de riesgo y gravedad observada en la DANA) y el
// navegador construye los polígonos: los cuadrados son exactos a cualquier zoom y el clic se resuelve aquí mismo,
// sin volver a enviar el mapa. Datos que recibe (data):
//   grid: {origin: [x0, y0], cell_m, rows, cols, utm_zone, codes (base64)}
//         codes[row * cols + col] = nivel (0 = sin celda, 1 Bajo, 2 Medio, 3 Alto) + 4 * gravedad (0 no inundada,
//         1 Leve, 2 Moderada, 3 Grave). row 0 es la fila sur. zone_id = row * cols + col + 1 (como en grid.py)
//   colors, opacity: color y opacidad de cada nivel; observed_colors: color de cada gravedad
//   selected: zone_id de la celda seleccionada (o null)
//   view: {id, center: [lat, lon], zoom}: el mapa solo se mueve cuando cambia view.id (búsqueda), no al pulsar
//   height: alto del mapa en píxeles
// Devuelve a Python el evento "cell" con el zone_id pulsado (0 si se pulsa fuera de las celdas).

const MAPLIBRE_URL = "https://cdn.jsdelivr.net/npm/maplibre-gl@4.7.1/dist/maplibre-gl";
const LEVELS = ["Bajo", "Medio", "Alto"];
const SEVERITIES = ["Leve", "Moderada", "Grave"];
const OSM_STYLE = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      maxzoom: 19,
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    },
  },
  layers: [{ id: "osm", type: "raster", source: "osm" }],
};
const LAYERS = [
  { id: "cells", label: "Susceptibilidad (celdas)", visible: true },
  { id: "observed", label: "Inundado en la DANA (observado)", visible: false },
  { id: "selected", label: "Celda seleccionada", visible: true },
];

// --- Carga de MapLibre (una sola vez por página) ---------------------------------------------------------------

let maplibrePromise;

function loadMapLibre() {
  if (window.maplibregl) return Promise.resolve(window.maplibregl);
  maplibrePromise ??= new Promise((resolve, reject) => {
    const css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = `${MAPLIBRE_URL}.css`;
    document.head.appendChild(css);
    const script = document.createElement("script");
    script.src = `${MAPLIBRE_URL}.js`;
    script.onload = () => resolve(window.maplibregl);
    script.onerror = () => reject(new Error("No se ha podido cargar MapLibre (¿sin conexión?)"));
    document.head.appendChild(script);
  });
  return maplibrePromise;
}

// --- UTM (ETRS89, elipsoide GRS80) <-> longitud y latitud --------------------------------------------------------
// Fórmulas de Snyder (USGS, Map Projections: A Working Manual, 1987): error de milímetros dentro del huso.

const A = 6378137;
const F = 1 / 298.257222101;
const E2 = F * (2 - F);
const EP2 = E2 / (1 - E2);
const K0 = 0.9996;
const FALSE_EASTING = 500000;
const RAD = Math.PI / 180;

function centralMeridian(zone) {
  return (zone * 6 - 183) * RAD;
}

function utmToLonLat(x, y, zone) {
  const e1 = (1 - Math.sqrt(1 - E2)) / (1 + Math.sqrt(1 - E2));
  const mu = y / K0 / (A * (1 - E2 / 4 - (3 * E2 ** 2) / 64 - (5 * E2 ** 3) / 256));
  const phi1 = mu + (1.5 * e1 - (27 / 32) * e1 ** 3) * Math.sin(2 * mu)
    + ((21 / 16) * e1 ** 2 - (55 / 32) * e1 ** 4) * Math.sin(4 * mu)
    + ((151 / 96) * e1 ** 3) * Math.sin(6 * mu) + ((1097 / 512) * e1 ** 4) * Math.sin(8 * mu);
  const sin = Math.sin(phi1);
  const cos = Math.cos(phi1);
  const tan = Math.tan(phi1);
  const n1 = A / Math.sqrt(1 - E2 * sin ** 2);
  const t1 = tan ** 2;
  const c1 = EP2 * cos ** 2;
  const r1 = (A * (1 - E2)) / (1 - E2 * sin ** 2) ** 1.5;
  const d = (x - FALSE_EASTING) / (n1 * K0);
  const lat = phi1 - ((n1 * tan) / r1) * (d ** 2 / 2 - ((5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * EP2) * d ** 4) / 24
    + ((61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * EP2 - 3 * c1 ** 2) * d ** 6) / 720);
  const lon = centralMeridian(zone) + (d - ((1 + 2 * t1 + c1) * d ** 3) / 6
    + ((5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * EP2 + 24 * t1 ** 2) * d ** 5) / 120) / cos;
  return [lon / RAD, lat / RAD];
}

function lonLatToUtm(lonDeg, latDeg, zone) {
  const phi = latDeg * RAD;
  const sin = Math.sin(phi);
  const cos = Math.cos(phi);
  const tan = Math.tan(phi);
  const n = A / Math.sqrt(1 - E2 * sin ** 2);
  const t = tan ** 2;
  const c = EP2 * cos ** 2;
  const a = cos * (lonDeg * RAD - centralMeridian(zone));
  const m = A * ((1 - E2 / 4 - (3 * E2 ** 2) / 64 - (5 * E2 ** 3) / 256) * phi
    - ((3 * E2) / 8 + (3 * E2 ** 2) / 32 + (45 * E2 ** 3) / 1024) * Math.sin(2 * phi)
    + ((15 * E2 ** 2) / 256 + (45 * E2 ** 3) / 1024) * Math.sin(4 * phi) - ((35 * E2 ** 3) / 3072) * Math.sin(6 * phi));
  const x = FALSE_EASTING + K0 * n * (a + ((1 - t + c) * a ** 3) / 6 + ((5 - 18 * t + t ** 2 + 72 * c - 58 * EP2) * a ** 5) / 120);
  const y = K0 * (m + n * tan * (a ** 2 / 2 + ((5 - t + 9 * c + 4 * c ** 2) * a ** 4) / 24
    + ((61 - 58 * t + t ** 2 + 600 * c - 330 * EP2) * a ** 6) / 720));
  return [x, y];
}

// --- Geometría de la cuadrícula ----------------------------------------------------------------------------------

function decodeGrid(grid) {
  const binary = atob(grid.codes);
  const codes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) codes[i] = binary.charCodeAt(i);
  // Esquinas de todas las celdas en longitud y latitud: (rows + 1) x (cols + 1) puntos, compartidos entre vecinas
  const corners = new Float64Array((grid.rows + 1) * (grid.cols + 1) * 2);
  const [x0, y0] = grid.origin;
  for (let r = 0; r <= grid.rows; r++) {
    for (let c = 0; c <= grid.cols; c++) {
      const [lon, lat] = utmToLonLat(x0 + c * grid.cell_m, y0 + r * grid.cell_m, grid.utm_zone);
      const i = (r * (grid.cols + 1) + c) * 2;
      corners[i] = lon;
      corners[i + 1] = lat;
    }
  }
  const corner = (r, c) => {
    const i = (r * (grid.cols + 1) + c) * 2;
    return [corners[i], corners[i + 1]];
  };
  return { ...grid, codes, corner };
}

// Celdas por nivel, un cuadrado por celda. Las vecinas comparten exactamente sus esquinas para que la GPU no deje
// huecos de un píxel entre ellas
function cellFeatures(g) {
  const polygons = LEVELS.map(() => []);
  for (let r = 0; r < g.rows; r++) {
    for (let c = 0; c < g.cols; c++) {
      const level = g.codes[r * g.cols + c] & 3;
      if (level > 0) {
        polygons[level - 1].push([[g.corner(r, c), g.corner(r, c + 1), g.corner(r + 1, c + 1), g.corner(r + 1, c),
          g.corner(r, c)]]);
      }
    }
  }
  return {
    type: "FeatureCollection",
    features: polygons.map((coordinates, i) => ({
      type: "Feature", properties: { level: LEVELS[i] }, geometry: { type: "MultiPolygon", coordinates },
    })),
  };
}

// Contorno de lo inundado por gravedad: los bordes entre una celda de esa gravedad y una vecina que no lo es
function observedFeatures(g) {
  const severity = (r, c) => (r < 0 || c < 0 || r >= g.rows || c >= g.cols ? 0 : g.codes[r * g.cols + c] >> 2);
  const features = [];
  for (let s = 1; s <= SEVERITIES.length; s++) {
    const lines = [];
    // Bordes horizontales (entre la fila r - 1 y la r), unidos en tramos
    for (let r = 0; r <= g.rows; r++) {
      let start = -1;
      for (let c = 0; c <= g.cols; c++) {
        const edge = c < g.cols && (severity(r - 1, c) === s) !== (severity(r, c) === s);
        if (edge && start < 0) start = c;
        if (!edge && start >= 0) {
          lines.push([g.corner(r, start), g.corner(r, c)]);
          start = -1;
        }
      }
    }
    // Bordes verticales (entre la columna c - 1 y la c)
    for (let c = 0; c <= g.cols; c++) {
      let start = -1;
      for (let r = 0; r <= g.rows; r++) {
        const edge = r < g.rows && (severity(r, c - 1) === s) !== (severity(r, c) === s);
        if (edge && start < 0) start = r;
        if (!edge && start >= 0) {
          lines.push([g.corner(start, c), g.corner(r, c)]);
          start = -1;
        }
      }
    }
    features.push({
      type: "Feature", properties: { severity: SEVERITIES[s - 1] }, geometry: { type: "MultiLineString", coordinates: lines },
    });
  }
  return { type: "FeatureCollection", features };
}

function selectedFeature(g, zoneId) {
  if (!zoneId) return { type: "FeatureCollection", features: [] };
  const r = Math.floor((zoneId - 1) / g.cols);
  const c = (zoneId - 1) % g.cols;
  const ring = [g.corner(r, c), g.corner(r, c + 1), g.corner(r + 1, c + 1), g.corner(r + 1, c), g.corner(r, c)];
  return { type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [ring] } }] };
}

function zoneAt(g, lngLat) {
  const [x, y] = lonLatToUtm(lngLat.lng, lngLat.lat, g.utm_zone);
  const c = Math.floor((x - g.origin[0]) / g.cell_m);
  const r = Math.floor((y - g.origin[1]) / g.cell_m);
  if (r < 0 || c < 0 || r >= g.rows || c >= g.cols || (g.codes[r * g.cols + c] & 3) === 0) return 0;
  return r * g.cols + c + 1;
}

// --- Mapa ------------------------------------------------------------------------------------------------------

function matchExpression(property, values, fallback) {
  return ["match", ["get", property], ...Object.entries(values).flat(), fallback];
}

function addLayers(map, g, data) {
  map.addSource("cells", { type: "geojson", data: cellFeatures(g) });
  map.addSource("observed", { type: "geojson", data: observedFeatures(g) });
  map.addSource("selected", { type: "geojson", data: selectedFeature(g, data.selected) });
  map.addLayer({
    id: "cells", type: "fill", source: "cells",
    paint: { "fill-color": matchExpression("level", data.colors, "#000000"),
      "fill-opacity": matchExpression("level", data.opacity, 0), "fill-antialias": false },
  });
  map.addLayer({
    id: "observed", type: "line", source: "observed", layout: { visibility: "none" },
    paint: { "line-color": matchExpression("severity", data.observed_colors, "#000000"), "line-width": 1.5,
      "line-dasharray": [3, 2] },
  });
  map.addLayer({ id: "selected", type: "line", source: "selected", paint: { "line-color": "#000000", "line-width": 3 } });
}

function layerControl(map) {
  const box = document.createElement("div");
  box.className = "maplibregl-ctrl maplibregl-ctrl-group dana-layers";
  for (const layer of LAYERS) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = layer.visible;
    input.onchange = () => map.setLayoutProperty(layer.id, "visibility", input.checked ? "visible" : "none");
    label.append(input, ` ${layer.label}`);
    box.appendChild(label);
  }
  return { onAdd: () => box, onRemove: () => box.remove() };
}

export default async function ({ data, parentElement, setTriggerValue }) {
  let root = parentElement.querySelector(".dana-map");
  if (!root) {
    root = document.createElement("div");
    root.className = "dana-map";
    parentElement.appendChild(root);
  }
  root.style.height = `${data.height}px`;
  const state = (root.__dana ??= {});
  // Streamlit solo guarda la última función de limpieza devuelta: siempre la misma, que quita el mapa actual
  state.cleanup ??= () => {
    state.map?.remove();
    state.map = undefined;
  };

  // Misma cuadrícula: solo se actualizan la celda seleccionada y, si hay una búsqueda nueva, la vista
  if (state.map && state.gridCodes === data.grid.codes) {
    state.data = data;
    state.map.getSource("selected")?.setData(selectedFeature(state.grid, data.selected));
    if (data.view.id !== state.viewId) {
      state.viewId = data.view.id;
      state.map.flyTo({ center: [data.view.center[1], data.view.center[0]], zoom: data.view.zoom });
    }
    return state.cleanup;
  }

  const maplibregl = await loadMapLibre();
  state.cleanup();
  root.removeAttribute("data-map-ready");
  const grid = decodeGrid(data.grid);
  Object.assign(state, { grid, data, gridCodes: data.grid.codes, viewId: data.view.id });
  const map = new maplibregl.Map({
    container: root,
    style: OSM_STYLE,
    center: [data.view.center[1], data.view.center[0]],
    zoom: data.view.zoom,
    attributionControl: { compact: false },
  });
  state.map = map;
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-left");
  map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
  map.addControl(layerControl(map), "top-right");
  // Las celdas se añaden en cuanto está el estilo, sin esperar a las teselas del mapa de fondo (vienen de
  // OpenStreetMap). data-map-ready marca que ya se ven las celdas
  map.once("style.load", () => {
    addLayers(map, grid, state.data);
    const onData = (event) => {
      if (event.sourceId !== "cells" || !map.isSourceLoaded("cells")) return;
      map.off("sourcedata", onData);
      map.once("render", () => root.setAttribute("data-map-ready", "1"));
    };
    map.on("sourcedata", onData);
  });
  map.on("click", (event) => {
    const zone = zoneAt(grid, event.lngLat);
    map.getSource("selected")?.setData(selectedFeature(grid, zone));
    setTriggerValue("cell", zone);
  });
  map.getCanvas().style.cursor = "pointer";
  return state.cleanup;
}
