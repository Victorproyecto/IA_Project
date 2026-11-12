// Botón de tema claro / oscuro (componente de Streamlit, st.components.v2).
//
// Streamlit no permite cambiar el tema desde Python: lo elige el usuario en el menú de ajustes, que lo guarda en el
// navegador (localStorage, clave "stActiveTheme-<ruta>-v2", valor "Light", "Dark" o "System"). Este botón escribe esa
// misma clave y recarga la página. Si una versión futura de Streamlit cambia la clave, el botón deja de tener efecto y
// el tema se sigue pudiendo cambiar desde el menú de ajustes.

const THEME_KEY = `stActiveTheme-${window.location.pathname}-v2`;

// Tema que se ve ahora: se deduce del fondo de la aplicación (claro u oscuro), como hace Streamlit
function isDark() {
  const app = document.querySelector(".stApp") ?? document.body;
  const [r, g, b] = getComputedStyle(app).backgroundColor.match(/\d+/g).map(Number);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b < 128;
}

export default function ({ parentElement }) {
  let button = parentElement.querySelector(".dana-theme");
  if (!button) {
    button = document.createElement("button");
    button.type = "button";
    button.className = "dana-theme";
    button.onclick = () => {
      window.localStorage.setItem(THEME_KEY, JSON.stringify(isDark() ? "Light" : "Dark"));
      window.location.reload();
    };
    parentElement.appendChild(button);
  }
  const dark = isDark();
  button.textContent = dark ? "☀️ Tema claro" : "🌙 Tema oscuro";
  button.title = dark ? "Cambiar a tema claro" : "Cambiar a tema oscuro";
  button.setAttribute("aria-label", button.title);
}
