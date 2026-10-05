# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib==3.11.2", "numpy"]
# ///
"""Plot the measured primary comparison. Run with uv run --no-project."""
import json
from pathlib import Path
from orx_figstyle import BASELINE, PALETTE, TEXT, figure, save, use_style

HERE = Path(__file__).resolve().parent

def main():
    paths = [HERE.parents[1] / "scholarcatalyst-pilot/data/replication.json",
             HERE.parent / "data/held-out.json"]
    reports = [json.loads(path.read_text()) for path in paths]
    if not all(report["complete"] for report in reports):
        raise ValueError("An incomplete experiment cannot support this comparison plot.")
    values = [report["paired_comparisons"]["orx_skill_minus_plain_tools"]
              ["balanced_pilot"]["recall@5"] for report in reports]
    use_style()
    fig, ax = figure(width=TEXT, ratio=0.32)
    ax.axvline(0, color=BASELINE, linewidth=0.8)
    labels = ["Pilot (50 projects)", "Held-out (157 projects)"]
    for y, value in enumerate(values):
        mean = value["mean_delta"] * 100
        lo, hi = [x * 100 for x in value["interval_95"]]
        color = PALETTE["blue"] if y == 1 else "#666666"
        ax.plot([lo, hi], [y, y], color=color, linewidth=1.3)
        ax.plot(mean, y, "o", color=color, markersize=4)
    ax.set_yticks([0, 1], labels)
    ax.set_ylim(-0.6, 1.6)
    ax.invert_yaxis()
    ax.set_xlabel("Skill minus ordinary tools: Recall@5 (percentage points)")
    ax.set_ylabel("Cohort")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)
    fig.tight_layout()
    save(fig, str(HERE / "paired-recall"))
    svg = HERE / "paired-recall.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig("/tmp/scholarcatalyst-expanded-preview.png", dpi=180)

if __name__ == "__main__":
    main()
