import { useRef } from "react";
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
  const choose = (direction: "up" | "down", archived: boolean) => {
    menu.setOpen(false);
    onArchive(id, direction, archived);
  };

  return (
    <div className="relative nodrag" ref={menu.ref}>
      {compact ? (
        <button
          type="button"
          ref={triggerRef}
          className="node-action"
          aria-label={m.tree_archive_manage_experiment({ name })}
          aria-expanded={menu.open}
          onClick={() => menu.setOpen((open) => !open)}
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
          onClick={() => menu.setOpen((open) => !open)}
        >
          {m.tree_archive_manage()}
        </button>
      )}
      {menu.open && (
        <div className={`option-menu absolute end-0 z-50 min-w-44 rounded-lg border border-border bg-background p-1.5 shadow-menu ${compact ? "bottom-[calc(100%_+_4px)]" : "top-[calc(100%_+_4px)]"}`}>
          {hasParent && <MenuItem onClick={() => choose("up", true)}>{m.tree_archive_above()}</MenuItem>}
          <MenuItem onClick={() => choose("down", true)}>{m.tree_archive_down()}</MenuItem>
          {hasParent && <MenuItem onClick={() => choose("up", false)}>{m.tree_restore_above()}</MenuItem>}
          <MenuItem onClick={() => choose("down", false)}>{m.tree_restore_down()}</MenuItem>
        </div>
      )}
    </div>
  );
}
