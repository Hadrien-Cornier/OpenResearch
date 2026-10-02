// The desktop app paints its native titlebar with the color the page shows
// beneath it (src/commands/app.rs), so the two read as one surface.

import type { ThemePreference } from "./theme";

let preference: ThemePreference = "system";
let sent = "";
let pending = false;
const swatch = document
  .createElement("canvas")
  .getContext("2d", { willReadFrequently: true });

/** Any CSS color as rrggbb hex, or null when it is not fully opaque. */
function opaqueHex(color: string): string | null {
  if (!swatch) return null;
  swatch.clearRect(0, 0, 1, 1);
  // An unparseable color leaves fillStyle as it was.
  swatch.fillStyle = "transparent";
  swatch.fillStyle = color;
  swatch.fillRect(0, 0, 1, 1);
  const [r, g, b, a] = swatch.getImageData(0, 0, 1, 1).data;
  if (a !== 255) return null;
  return [r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("");
}

function topEdgeColor(): string | null {
  const hits = document.elementsFromPoint(window.innerWidth / 2, 0);
  // Content scrolled under the titlebar shouldn't recolor it; match the surface it scrolls over.
  const scroller = hits.findIndex(
    (el) => el.scrollHeight > el.clientHeight && /auto|scroll/.test(getComputedStyle(el).overflowY),
  );
  for (const element of hits.slice(Math.max(scroller, 0))) {
    const hex = opaqueHex(getComputedStyle(element).backgroundColor);
    if (hex) return hex;
  }
  return opaqueHex(getComputedStyle(document.documentElement).backgroundColor);
}

/** Re-sends the titlebar color; pass the theme preference when it changes. */
export function syncDesktopTitlebar(nextPreference?: ThemePreference): void {
  if (nextPreference) preference = nextPreference;
  if (!window.ipc || pending) return;
  pending = true;
  setTimeout(() => {
    pending = false;
    const color = topEdgeColor();
    const message = `titlebar:${preference}:${color}`;
    if (!color || message === sent) return;
    sent = message;
    window.ipc?.postMessage(message);
  }, 100);
}

if (window.ipc) {
  new MutationObserver(() => syncDesktopTitlebar()).observe(
    document.documentElement,
    {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ["class", "style", "data-theme"],
    },
  );
  window.addEventListener("resize", () => syncDesktopTitlebar());
}

const INTERACTIVE =
  "a, button, input, select, textarea, label, summary, [role='button'], [role='tab'], [role='menuitem'], [role='option'], [role='switch'], [role='checkbox'], [contenteditable]:not([contenteditable='false']), [tabindex]:not([tabindex='-1'])";

// The macOS app runs the page under its titlebar, so the page's top strip
// drags and zooms the window in its place.
if ("__ORX_MAC_TITLEBAR__" in window) {
  document.documentElement.classList.add("mac-titlebar");
  window.addEventListener("mousedown", (event) => {
    if (event.button !== 0 || event.defaultPrevented || event.clientY >= 28) return;
    if (event.target instanceof Element && event.target.closest(INTERACTIVE)) return;
    window.ipc?.postMessage(event.detail === 2 ? "titlebar:zoom" : "titlebar:drag");
  });
}
