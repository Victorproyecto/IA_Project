// Buscador con sugerencias (componente de Streamlit, st.components.v2).
//
// Mientras se escribe sugiere, en este orden:
//   1. Municipios, distritos y barrios del área piloto (lista propia, al instante y sin conexión).
//   2. Calles, direcciones y lugares de CartoCiudad (IGN), limitados a los códigos postales del área piloto.
//      Las consultas van del navegador a CartoCiudad: la aplicación no guarda lo que se busca.
// Datos que recibe (data): places [{name, kind, lat, lon}], postal_codes, label, placeholder.
// Devuelve a Python el evento "place" con {lat, lon, label, detail, kind}: detail es el tipo que se muestra
// (Municipio, Barrio, Calle...) y kind, "place" (lista propia) o "address" (CartoCiudad).

const CARTOCIUDAD = "https://www.cartociudad.es/geocoder/api/geocoder";
const MIN_CHARS = 3; // CartoCiudad solo se consulta a partir de 3 letras
const DEBOUNCE_MS = 250; // espera tras la última tecla antes de consultar
const MAX_PLACES = 5;
const MAX_ADDRESSES = 8;
// Tipo de resultado de CartoCiudad -> etiqueta que se muestra
const TYPE_LABELS = { callejero: "Calle", portal: "Dirección", carretera: "Carretera", poblacion: "Población",
  Municipio: "Municipio", toponimo: "Lugar", ngbe: "Lugar" };

const normalize = (text) => text.normalize("NFD").replace(/\p{Diacritic}/gu, "").toLowerCase().trim();

function matchPlaces(places, query) {
  const q = normalize(query);
  if (!q) return [];
  return places
    .map((p) => ({ ...p, rank: normalize(p.name).startsWith(q) ? 0 : normalize(p.name).includes(q) ? 1 : 2 }))
    .filter((p) => p.rank < 2)
    .sort((a, b) => a.rank - b.rank || a.name.localeCompare(b.name))
    .slice(0, MAX_PLACES)
    .map((p) => ({ label: p.name, detail: p.kind, lat: p.lat, lon: p.lon, kind: "place" }));
}

async function fetchAddresses(query, postalCodes, signal) {
  const params = new URLSearchParams({ q: query, limit: String(MAX_ADDRESSES), cod_postal_filter: postalCodes.join(",") });
  const response = await fetch(`${CARTOCIUDAD}/candidates?${params}`, { signal });
  if (!response.ok) throw new Error(`CartoCiudad ${response.status}`);
  return (await response.json())
    .filter((c) => c.type !== "Municipio") // los municipios ya salen de la lista propia
    .map((c) => ({ label: c.address, detail: TYPE_LABELS[c.type] ?? "Lugar", lat: c.lat, lon: c.lng, kind: "address",
      candidate: c }));
}

// Las calles y poblaciones llegan sin coordenadas: se piden a CartoCiudad al elegirlas
async function locate(item) {
  if (item.lat && item.lon) return item;
  const c = item.candidate;
  const params = new URLSearchParams({ q: c.address, type: c.type, id: c.id });
  if (c.tip_via) params.set("tip_via", c.tip_via);
  if (c.portalNumber) params.set("portal", String(c.portalNumber));
  const response = await fetch(`${CARTOCIUDAD}/find?${params}`);
  const found = response.ok ? await response.json() : null;
  if (!found?.lat) throw new Error("Sin coordenadas");
  return { ...item, lat: found.lat, lon: found.lng };
}

export default function ({ data, parentElement, setTriggerValue }) {
  if (parentElement.querySelector(".search")) return; // ya montado: conserva lo escrito

  const box = document.createElement("div");
  box.className = "search";
  box.innerHTML = `
    <label class="label"></label>
    <input class="input" type="search" autocomplete="off" spellcheck="false" role="combobox" aria-expanded="false"
           aria-autocomplete="list" aria-controls="dana-suggestions">
    <ul class="list" id="dana-suggestions" role="listbox" hidden></ul>`;
  parentElement.appendChild(box);
  const input = box.querySelector(".input");
  const list = box.querySelector(".list");
  box.querySelector(".label").textContent = data.label;
  input.placeholder = data.placeholder;
  input.setAttribute("aria-label", data.label);

  let items = [];
  let active = -1;
  let timer;
  let controller;

  function render(message) {
    list.replaceChildren();
    items.forEach((item, i) => {
      const li = document.createElement("li");
      li.className = `item${i === active ? " active" : ""}`;
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", String(i === active));
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = item.label;
      const detail = document.createElement("span");
      detail.className = "detail";
      detail.textContent = item.detail;
      li.append(name, detail);
      li.onmousedown = (event) => {
        event.preventDefault(); // que el campo no pierda el foco antes de elegir
        choose(item);
      };
      list.appendChild(li);
    });
    if (message) {
      const li = document.createElement("li");
      li.className = "message";
      li.textContent = message;
      list.appendChild(li);
    }
    const open = items.length > 0 || Boolean(message);
    list.hidden = !open;
    input.setAttribute("aria-expanded", String(open));
  }

  async function choose(item) {
    clearTimeout(timer);
    controller?.abort();
    input.value = item.label;
    items = [];
    render("Localizando…");
    try {
      const located = await locate(item);
      render();
      setTriggerValue("place", { lat: located.lat, lon: located.lon, label: located.label, detail: located.detail,
        kind: located.kind });
    } catch {
      render("No se ha podido localizar esa ubicación");
    }
  }

  function update() {
    const query = input.value;
    items = matchPlaces(data.places, query);
    active = items.length ? 0 : -1;
    clearTimeout(timer);
    controller?.abort();
    if (query.trim().length < MIN_CHARS) {
      render(query.trim() && !items.length ? "Escribe al menos 3 letras para buscar calles" : undefined);
      return;
    }
    render("Buscando direcciones…");
    timer = setTimeout(async () => {
      controller = new AbortController();
      try {
        const addresses = await fetchAddresses(query, data.postal_codes, controller.signal);
        items = [...matchPlaces(data.places, query), ...addresses];
        active = items.length ? Math.max(active, 0) : -1;
        render(items.length ? undefined : "Sin resultados en el área piloto");
      } catch (error) {
        if (error.name !== "AbortError") render("No se ha podido consultar el buscador de direcciones");
      }
    }, DEBOUNCE_MS);
  }

  input.addEventListener("input", update);
  input.addEventListener("focus", () => input.value && update());
  input.addEventListener("blur", () => {
    list.hidden = true;
    input.setAttribute("aria-expanded", "false");
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (!items.length) return;
      active = (active + (event.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      render();
    } else if (event.key === "Enter") {
      event.preventDefault();
      if (items[Math.max(active, 0)]) choose(items[Math.max(active, 0)]);
    } else if (event.key === "Escape") {
      list.hidden = true;
      input.setAttribute("aria-expanded", "false");
    }
  });
}
