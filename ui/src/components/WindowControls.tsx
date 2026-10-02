import { useSyncExternalStore, type MouseEvent } from "react";
import { m } from "../paraglide/messages.js";

// Segoe Fluent Icons (Windows 11) and Segoe MDL2 Assets (Windows 10) share these glyphs.
const GLYPH = { minimize: "", maximize: "", restore: "", close: "" };

const BUTTON_CLASS_NAME =
  "inline-flex h-8 w-11.5 items-center justify-center text-xs text-text font-['Segoe_Fluent_Icons','Segoe_MDL2_Assets']";

function subscribeMaximized(onChange: () => void) {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-maximized"] });
  return () => observer.disconnect();
}

function send(message: string) {
  window.ipc?.postMessage(`titlebar:${message}`);
}

function resizeFrom(edge: string) {
  return (event: MouseEvent) => {
    // Keeps desktopTitlebar.ts from also starting a window drag.
    event.preventDefault();
    send(`resize:${edge}`);
  };
}

/** The Windows app's caption buttons and top resize edge, which its frameless window lacks. */
export function WindowControls() {
  const maximized = useSyncExternalStore(subscribeMaximized, () =>
    document.documentElement.hasAttribute("data-maximized"),
  );
  return (
    <>
      {!maximized && (
        <>
          <div className="fixed inset-x-0 top-0 z-110 h-1 cursor-ns-resize" onMouseDown={resizeFrom("n")} />
          <div className="fixed start-0 top-0 z-110 size-2 cursor-nwse-resize" onMouseDown={resizeFrom("nw")} />
          <div className="fixed end-0 top-0 z-110 size-2 cursor-nesw-resize" onMouseDown={resizeFrom("ne")} />
        </>
      )}
      <div className="fixed end-0 top-0 z-100 flex">
        <button className={`${BUTTON_CLASS_NAME} hover:bg-panel`} title={m.window_minimize()} aria-label={m.window_minimize()} onClick={() => send("minimize")}>
          {GLYPH.minimize}
        </button>
        <button
          className={`${BUTTON_CLASS_NAME} hover:bg-panel`}
          title={maximized ? m.window_restore() : m.window_maximize()}
          aria-label={maximized ? m.window_restore() : m.window_maximize()}
          onClick={() => send("zoom")}
        >
          {maximized ? GLYPH.restore : GLYPH.maximize}
        </button>
        <button className={`${BUTTON_CLASS_NAME} hover:bg-accent-red hover:text-white`} title={m.window_close()} aria-label={m.window_close()} onClick={() => send("close")}>
          {GLYPH.close}
        </button>
      </div>
    </>
  );
}
