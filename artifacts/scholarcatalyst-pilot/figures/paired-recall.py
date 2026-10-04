# /// script
# dependencies = ["matplotlib>=3.9"]
# ///
"""Plot measured paired recall differences. Run with uv run --no-project."""

import json
import math
from pathlib import Path

from orx_figstyle import BASELINE, PALETTE, WIDE, figure_grid, panel_labels, save, use_style

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/replication.json"
COMPARISONS = [
    ("plain_tools_minus_closed_book", "Ordinary tools - model only"),
    ("orx_skill_minus_closed_book", "Skill - model only"),
    ("orx_skill_minus_plain_tools", "Skill - ordinary tools"),
]


def main():
    report = json.loads(DATA.read_text())
    if not report["complete"]:
        raise ValueError("plot requires a complete recorded run")
    use_style()
    fig, axes = figure_grid(1, 2, width=WIDE, ratio=0.44, sharex=True, sharey=True)
    estimates = []
    for axis, kind in zip(axes, ("core_query", "subfield_query")):
        axis.axvline(0, color=BASELINE, linewidth=0.8, zorder=1)
        for index, (key, _) in enumerate(COMPARISONS):
            value = report["paired_comparisons"][key][kind]["recall@5"]
            point = 100 * value["mean_delta"]
            low, high = [100 * endpoint for endpoint in value["interval_95"]]
            y = 2 - index
            axis.plot([low, high], [y, y], color=PALETTE["blue"], linewidth=1.2, zorder=2)
            axis.plot(point, y, "o", color=PALETTE["blue"], zorder=3)
            axis.annotate(f"{point:+.1f}", (high, y), xytext=(5, 0), textcoords="offset points",
                          va="center", fontsize=7)
            estimates.extend((low, high))
        axis.set_yticks([2, 1, 0], [label for _, label in COMPARISONS])
        axis.set_ylim(-0.6, 2.6)
        axis.set_xlabel("Recall@5 change (percentage points)")
        axis.grid(axis="y", visible=False)
        axis.grid(axis="x")
        axis.tick_params(axis="y", length=0)
    low, high = min(0, *estimates), max(0, *estimates)
    span = max(10, high - low)
    axes[0].set_xlim(math.floor((low - span * 0.08) / 5) * 5,
                     math.ceil((high + span * 0.25) / 5) * 5)
    axes[0].set_ylabel("Paired comparison")
    panel_labels(axes, labels=["(a) Core questions", "(b) Subfield questions"])
    stem = str(Path(__file__).with_suffix(""))
    save(fig, stem, close=False)
    svg_path = Path(stem + ".svg")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    fig.savefig("/tmp/scholarcatalyst-paired-recall-preview.png", dpi=180)
    print("One cohort seed. Intervals resample source projects, not model seeds.")


if __name__ == "__main__":
    main()
