import { useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Ellipsis } from "lucide-react";
import { m } from "../paraglide/messages.js";
import { usePopover } from "./ModelPicker";
import { MenuItem } from "./ui";

export function ArchiveMenu({ id, name, hasParent, onArchive, compact = false }: {
  id: string;
  name: string;
  hasParent: boolean;
  onArchive: (id: string, direction: "up" | "down", archived: boolean) => void;
  compact?: boolean;
}) {
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menu = usePopover(triggerRef);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const toggle = () => {
    if (!menu.open && triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      const below = window.innerHeight - rect.bottom >= 160;
      setPosition({
        top: below ? rect.bottom + 4 : Math.max(4, rect.top - 160),
        left: Math.max(4, Math.min(rect.right - 176, window.innerWidth - 180)),
      });
    }
    menu.setOpen((open) => !open);
  };
  const choose = (direction: "up" | "down", archived: boolean) => {
    menu.setOpen(false);
    onArchive(id, direction, archived);
  };

  return (
    <div className="nodrag">
      {compact ? (
        <button
          type="button"
          ref={triggerRef}
          className="node-action"
          aria-label={m.tree_archive_manage_experiment({ name })}
          aria-expanded={menu.open}
          onClick={toggle}
        >
          <Ellipsis size={15} aria-hidden="true" />
        </button>
      ) : (
        <button
          type="button"
          className="btn inline-flex h-7 items-center justify-center rounded-sm border border-border bg-background px-2.5 text-sm text-text hover:bg-surface"
          ref={triggerRef}
          aria-label={m.tree_archive_manage_experiment({ name })}
          aria-expanded={menu.open}
          onClick={toggle}
        >
          {m.tree_archive_manage()}
        </button>
      )}
      {menu.open && createPortal(
        <div ref={menu.ref} className="option-menu fixed z-50 min-w-44 rounded-lg border border-border bg-background p-1.5 shadow-menu" style={position}>
          {hasParent && <MenuItem onClick={() => choose("up", true)}>{m.tree_archive_above()}</MenuItem>}
          <MenuItem onClick={() => choose("down", true)}>{m.tree_archive_down()}</MenuItem>
          {hasParent && <MenuItem onClick={() => choose("up", false)}>{m.tree_restore_above()}</MenuItem>}
          <MenuItem onClick={() => choose("down", false)}>{m.tree_restore_down()}</MenuItem>
        </div>, document.body,
      )}
    </div>
  );
}
