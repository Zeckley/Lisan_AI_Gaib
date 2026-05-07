"""
simulation/snapshot.py
======================
Snapshot and plotting utilities for Colony and Faction.

Usage
-----
    from snapshot import take_snapshot, plot_history

    history = []
    for t in range(50):
        faction.tick()
        history.append(take_snapshot(home_colony))   # or take_snapshot(faction)

    plot_history(history)

take_snapshot(target)
    Works on either a Colony or a Faction instance.
    Returns a plain dict — safe to pickle, JSON-serialise, or stack into a list.
    Faction snapshots aggregate across all colonies plus treasury totals.

plot_history(snapshots, title="")
    Takes the list of snapshot dicts and produces a 4-panel figure:
      1. Resources (stockpile + rate)        — per-resource subplots, dual axis
      2. Workers    (level × assigned)        — stacked bar per tick
      3. Buildings  (type × level × active)  — per-type subplot, stacked area
      4. Flags & Directive                   — event-style timeline

Snapshot dict schema
--------------------
{
  "tick"       : int,
  "label"      : str,          # colony/faction name

  # ── Resources ──────────────────────────────────────────────────────────
  "stockpile"  : {resource_name: float, ...},   # local stockpile (or treasury for Faction)
  "net_rate"   : {resource_name: float, ...},   # production − consumption per tick

  # ── Workers ────────────────────────────────────────────────────────────
  "workers"    : {
      level_int: {"assigned": int, "unassigned": int},
      ...
  },
  "population" : float,

  # ── Buildings ──────────────────────────────────────────────────────────
  # building_name → level → state_bucket → count
  "buildings"  : {
      building_name: {
          level_int: {"producing": int, "constructing": int, "idle": int}
      }
  },

  # ── Flags & Directive ──────────────────────────────────────────────────
  "critical_flags"  : [str, ...],
  "strategic_flags" : [str, ...],
  "directive"       : str,         # DirectiveType.name or "MULTI" for Faction
  "tax_rate"        : float,
  "urgency"         : float,
}
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

matplotlib.rcParams.update({
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "axes.titlecolor":   "#e6edf3",
    "grid.color":        "#21262d",
    "grid.linewidth":    0.6,
    "axes.grid":         True,
})

# ── Palette ────────────────────────────────────────────────────────────────
_RES_COLORS = {
    "MINERALS":  "#58a6ff",
    "ENERGY":    "#f0c133",
    "ORGANICS":  "#56d364",
    "RARE_MATS": "#bc8cff",
}

_LEVEL_COLORS = [
    "#1f3a5c", "#2d5986", "#3b78b0", "#4997da", "#57b6ff",
]   # darker → lighter as level increases 1 → 5

_FLAG_COLORS = {
    # critical
    "FOOD_SHORTAGE":       "#f85149",
    "POWER_DEFICIT":       "#ff7b72",
    "POPULATION_COLLAPSE": "#ffa198",
    # strategic
    "DEFENSE_NEEDED":      "#d29922",
    "WORKER_SHORTAGE":     "#e3b341",
    "RESOURCE_LOW":        "#f0c133",
    "EXPORT_STRAINED":     "#a5d6ff",
    "CONSTRUCTION_BLOCKED":"#79c0ff",
}

_DIRECTIVE_COLORS = {
    "HARVEST": "#56d364",
    "DEFEND":  "#f85149",
    "EXPAND":  "#79c0ff",
    "EXPORT":  "#bc8cff",
    "IDLE":    "#8b949e",
    "MULTI":   "#e3b341",
}

_BUILD_STATE_COLORS = {
    "producing":    "#58a6ff",
    "constructing": "#e3b341",
    "idle":         "#30363d",
}


# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _colony_snapshot(colony) -> Dict[str, Any]:
    """Extract a snapshot dict from a Colony instance."""
    from buildings import BuildingState, ResourceType

    tick  = colony._tick
    label = colony.name

    # ── Stockpile & net rates ──────────────────────────────────────────────
    stockpile = {
        rt.name: colony.stockpile.get(int(rt), 0.0)
        for rt in ResourceType
    }

    net = colony._net_rates()
    net_rate = {rt.name: net.get(int(rt), 0.0) for rt in ResourceType}
    power_net_rate = net.get(4, 0.0)   # synthetic key 4 = POWER

    # ── Workers ────────────────────────────────────────────────────────────
    workers: Dict[int, Dict[str, int]] = {}
    for w in colony._workers:
        lvl = int(w.level)
        bucket = workers.setdefault(lvl, {"assigned": 0, "unassigned": 0})
        if w.is_assigned:
            bucket["assigned"]   += 1
        else:
            bucket["unassigned"] += 1

    # ── Buildings ──────────────────────────────────────────────────────────
    buildings: Dict[str, Dict[int, Dict[str, int]]] = {}
    for b in colony._buildings:
        bname = b.building_type.name
        by_type = buildings.setdefault(bname, {})
        by_level = by_type.setdefault(b.level, {"producing": 0, "constructing": 0, "idle": 0})
        if b.state == BuildingState.CONSTRUCTING:
            by_level["constructing"] += 1
        elif b.is_producing:
            by_level["producing"]    += 1
        else:
            by_level["idle"]         += 1

    # ── Flags & Directive ──────────────────────────────────────────────────
    critical_flags  = [f.name for f in colony.critical_flags]
    strategic_flags = [f.name for f in colony.strategic_flags]
    directive  = colony.directive.directive_type.name
    tax_rate   = colony.directive.tax_rate
    urgency    = colony.directive.urgency

    return {
        "tick":            tick,
        "label":           label,
        "stockpile":       stockpile,
        "net_rate":        net_rate,
        "power_net_rate":  power_net_rate,
        "workers":         workers,
        "population":      colony.population,
        "buildings":       buildings,
        "critical_flags":  critical_flags,
        "strategic_flags": strategic_flags,
        "directive":       directive,
        "tax_rate":        tax_rate,
        "urgency":         urgency,
    }


def _faction_snapshot(faction) -> Dict[str, Any]:
    """
    Aggregate snapshot across all faction colonies + treasury.
    Stockpile = faction treasury (not colony reserves).
    Buildings/workers = sums across all colonies.
    Flags = union of all colony flags.
    Directive = "MULTI" if colonies differ, else the common directive name.
    """
    from buildings import ResourceType

    tick  = faction._tick
    label = faction.name

    # Treasury as the "stockpile"
    stockpile = {rt.name: faction.treasury.get(int(rt), 0.0) for rt in ResourceType}

    # Net rates: sum across colonies
    net_rate: Dict[str, float] = {rt.name: 0.0 for rt in ResourceType}
    power_net_rate: float = 0.0
    for c in faction._colonies:
        c_net = c._net_rates()
        for rt in ResourceType:
            net_rate[rt.name] += c_net.get(int(rt), 0.0)
        power_net_rate += c_net.get(4, 0.0)

    # Workers: merge all colonies
    workers: Dict[int, Dict[str, int]] = {}
    population = 0.0
    for c in faction._colonies:
        population += c.population
        for lvl, counts in _colony_snapshot(c)["workers"].items():
            bkt = workers.setdefault(lvl, {"assigned": 0, "unassigned": 0})
            bkt["assigned"]   += counts["assigned"]
            bkt["unassigned"] += counts["unassigned"]

    # Buildings: merge all colonies
    buildings: Dict[str, Dict[int, Dict[str, int]]] = {}
    for c in faction._colonies:
        for bname, by_level in _colony_snapshot(c)["buildings"].items():
            fac_bt = buildings.setdefault(bname, {})
            for lvl, counts in by_level.items():
                fac_bl = fac_bt.setdefault(lvl, {"producing": 0, "constructing": 0, "idle": 0})
                for k in ("producing", "constructing", "idle"):
                    fac_bl[k] += counts[k]

    # Flags: union
    critical_flags  = sorted({f for c in faction._colonies for f in c.critical_flags},  key=lambda x: x.value)
    strategic_flags = sorted({f for c in faction._colonies for f in c.strategic_flags}, key=lambda x: x.value)
    critical_flags  = [f.name for f in critical_flags]
    strategic_flags = [f.name for f in strategic_flags]

    # Directive: unanimous or MULTI
    directives = {c.directive.directive_type.name for c in faction._colonies}
    directive  = directives.pop() if len(directives) == 1 else "MULTI"
    tax_rates  = [c.directive.tax_rate for c in faction._colonies]
    urgencies  = [c.directive.urgency  for c in faction._colonies]
    tax_rate   = float(np.mean(tax_rates)) if tax_rates else 0.0
    urgency    = float(np.mean(urgencies)) if urgencies else 0.0

    return {
        "tick":            tick,
        "label":           label,
        "stockpile":       stockpile,
        "net_rate":        net_rate,
        "power_net_rate":  power_net_rate,
        "workers":         workers,
        "population":      population,
        "buildings":       buildings,
        "critical_flags":  critical_flags,
        "strategic_flags": strategic_flags,
        "directive":       directive,
        "tax_rate":        tax_rate,
        "urgency":         urgency,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def take_snapshot(target) -> Dict[str, Any]:
    """
    Capture the current state of a Colony or Faction as a plain dict.

    Parameters
    ----------
    target : Colony or Faction instance

    Returns
    -------
    dict — see module docstring for full schema
    """
    # Duck-type: Faction has _colonies; Colony has colony_id
    if hasattr(target, "_colonies"):
        return _faction_snapshot(target)
    return _colony_snapshot(target)


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def plot_history(
    snapshots:   List[Dict[str, Any]],
    title:       str  = "",
    save_path:   str  = "",
    show:        bool = True,
) -> plt.Figure:
    """
    Render a 4-panel dashboard from a list of snapshots.

    Parameters
    ----------
    snapshots  : list of dicts produced by take_snapshot()
    title      : optional super-title
    save_path  : if non-empty, save figure to this path
    show       : call plt.show() at the end (set False in scripts)

    Returns
    -------
    matplotlib Figure
    """
    if not snapshots:
        raise ValueError("snapshots list is empty")

    ticks = [s["tick"] for s in snapshots]
    label = snapshots[-1]["label"]
    all_resources = list(_RES_COLORS.keys())

    # ── Figure layout ─────────────────────────────────────────────────────
    # Row 0 → resources (1 subplot per resource, 2 cols each for 4 resources)
    # Row 1 → workers  (full width)
    # Row 2 → buildings (one subplot per building type, up to 8)
    # Row 3 → flags / directive  (full width)

    n_res   = len(all_resources)
    btypes  = _all_building_types(snapshots)
    n_bld   = len(btypes)

    # Use a gridspec with 4 rows of varying heights
    fig = plt.figure(figsize=(20, 22), facecolor="#0d1117")
    fig.subplots_adjust(hspace=0.55, wspace=0.35)

    gs_top  = matplotlib.gridspec.GridSpec(
        4, max(n_res, n_bld, 2),
        figure=fig,
        height_ratios=[3, 2, 3, 2],
        hspace=0.55,
        wspace=0.35,
    )

    # ── Section headers ────────────────────────────────────────────────────
    _section_label(fig, 0.985, "[1] RESOURCES")
    _section_label(fig, 0.72,  "[2] WORKERS")
    _section_label(fig, 0.575, "[3] BUILDINGS")
    _section_label(fig, 0.155, "[4] FLAGS & DIRECTIVE")

    # ══════════════════════════════════════════════════════════════════════
    # Panel 1 — Resources
    # ══════════════════════════════════════════════════════════════════════
    for i, rname in enumerate(all_resources):
        col = i % max(n_res, 1)
        ax  = fig.add_subplot(gs_top[0, col])
        _plot_resource(ax, snapshots, ticks, rname)

    # ══════════════════════════════════════════════════════════════════════
    # Panel 2 — Workers
    # ══════════════════════════════════════════════════════════════════════
    ax_workers = fig.add_subplot(gs_top[1, :])
    _plot_workers(ax_workers, snapshots, ticks)

    # ══════════════════════════════════════════════════════════════════════
    # Panel 3 — Buildings
    # ══════════════════════════════════════════════════════════════════════
    n_cols = max(n_bld, 1)
    for i, btype in enumerate(btypes):
        ax = fig.add_subplot(gs_top[2, i % n_cols])
        _plot_building(ax, snapshots, ticks, btype)

    # ══════════════════════════════════════════════════════════════════════
    # Panel 4 — Flags & Directive
    # ══════════════════════════════════════════════════════════════════════
    ax_flags = fig.add_subplot(gs_top[3, :])
    _plot_flags(ax_flags, snapshots, ticks)

    # ── Super-title ────────────────────────────────────────────────────────
    sup = title or f"{label} — simulation history"
    fig.suptitle(
        sup,
        fontsize=15,
        color="#e6edf3",
        fontweight="bold",
        y=1.01,
        fontfamily="monospace",
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# SUB-PANEL RENDERERS
# ──────────────────────────────────────────────────────────────────────────────

def _section_label(fig: plt.Figure, y: float, text: str) -> None:
    fig.text(
        0.01, y, text,
        fontsize=9, color="#484f58",
        fontfamily="monospace",
        transform=fig.transFigure,
        va="top",
    )


def _plot_resource(ax, snapshots, ticks, rname: str) -> None:
    """Stockpile (filled area) on left axis, net rate (dashed line) on right axis.
    For the ENERGY subplot, also overlays POWER net rate (synthetic key 4) in orange."""
    stock  = [s["stockpile"].get(rname, 0.0) for s in snapshots]
    rate   = [s["net_rate"].get(rname, 0.0)  for s in snapshots]
    color  = _RES_COLORS.get(rname, "#c9d1d9")

    ax.fill_between(ticks, stock, alpha=0.25, color=color)
    ax.plot(ticks, stock, color=color, linewidth=1.8, label="stockpile")
    ax.set_title(rname, fontsize=9, color=color, pad=4)
    ax.set_xlabel("tick", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    ax2 = ax.twinx()
    ax2.plot(ticks, rate, color=color, linewidth=1.2, linestyle="--", alpha=0.7, label="ENERGY Δ/tick")
    ax2.axhline(0, color="#30363d", linewidth=0.8)

    # Overlay POWER net rate on the ENERGY subplot
    if rname == "ENERGY":
        power_color = "#ff9f43"   # warm orange — distinct from the yellow ENERGY tones
        power_rate  = [s.get("power_net_rate", 0.0) for s in snapshots]
        ax2.plot(ticks, power_rate, color=power_color, linewidth=1.4,
                 linestyle="-.", alpha=0.9, label="POWER Δ/tick")
        ax2.axhline(0, color="#30363d", linewidth=0.8)
        ax.set_title("ENERGY  &  POWER", fontsize=9, color=color, pad=4)

    ax2.tick_params(labelsize=7, colors="#8b949e")
    ax2.set_ylabel("Δ/tick", fontsize=7, color="#8b949e")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#30363d")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}"))

    # Combined legend
    lines  = ax.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=6, loc="upper left",
              framealpha=0.3, facecolor="#161b22", edgecolor="#30363d")


def _plot_workers(ax, snapshots, ticks) -> None:
    """
    Stacked bar chart: each level split into assigned (solid) / unassigned (hatched).
    Population shown as a secondary line.
    """
    all_levels = sorted({lvl for s in snapshots for lvl in s["workers"]})
    bar_width  = max(0.6, 0.8 * (ticks[-1] - ticks[0] + 1) / max(len(ticks), 1))
    bar_width  = min(bar_width, 1.0)

    bottom_a = np.zeros(len(ticks))
    bottom_u = np.zeros(len(ticks))
    bottom   = np.zeros(len(ticks))

    for lvl in all_levels:
        col = _LEVEL_COLORS[min(lvl - 1, 4)]
        assigned   = np.array([s["workers"].get(lvl, {}).get("assigned",   0) for s in snapshots], float)
        unassigned = np.array([s["workers"].get(lvl, {}).get("unassigned", 0) for s in snapshots], float)
        total = assigned + unassigned

        ax.bar(ticks, total, bottom=bottom, color=col, width=bar_width * 0.9,
               alpha=0.9, label=f"L{lvl}")
        # Hatch the unassigned portion
        ax.bar(ticks, unassigned, bottom=bottom, color="none", width=bar_width * 0.9,
               edgecolor="#8b949e", hatch="///", linewidth=0.4, alpha=0.5)
        bottom = bottom + total

    # Population secondary axis
    ax2 = ax.twinx()
    pop = [s["population"] for s in snapshots]
    ax2.plot(ticks, pop, color="#f0c133", linewidth=1.4, linestyle=":", label="population")
    ax2.set_ylabel("population", fontsize=7, color="#f0c133")
    ax2.tick_params(labelsize=7, colors="#8b949e")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#30363d")

    ax.set_title("Workers  (solid=assigned, hatched=unassigned)", fontsize=9, pad=4)
    ax.set_xlabel("tick", fontsize=7)
    ax.set_ylabel("worker count", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left",
              framealpha=0.3, facecolor="#161b22", edgecolor="#30363d", ncol=5)


def _all_building_types(snapshots) -> List[str]:
    seen = []
    for s in snapshots:
        for bname in s["buildings"]:
            if bname not in seen:
                seen.append(bname)
    return seen


def _plot_building(ax, snapshots, ticks, btype: str) -> None:
    """
    Stacked area chart: each level's *producing* buildings stacked.
    Constructing and idle counts shown as separate thin dashed/dotted lines.
    """
    all_levels = sorted({
        lvl
        for s in snapshots
        for lvl, _ in s["buildings"].get(btype, {}).items()
    })

    producing_by_level = {
        lvl: np.array([
            s["buildings"].get(btype, {}).get(lvl, {}).get("producing", 0)
            for s in snapshots
        ], float)
        for lvl in all_levels
    }
    constructing = np.array([
        sum(s["buildings"].get(btype, {}).get(lvl, {}).get("constructing", 0)
            for lvl in all_levels)
        for s in snapshots
    ], float)
    idle = np.array([
        sum(s["buildings"].get(btype, {}).get(lvl, {}).get("idle", 0)
            for lvl in all_levels)
        for s in snapshots
    ], float)
    repairing = np.array([
        sum(s["buildings"].get(btype, {}).get(lvl, {}).get("repairing", 0)
            for lvl in all_levels)
        for s in snapshots
    ], float)

    # Stacked area for producing by level
    y_stack = np.zeros(len(ticks))
    for lvl in all_levels:
        col = _LEVEL_COLORS[min(lvl - 1, 4)]
        y   = producing_by_level[lvl]
        ax.fill_between(ticks, y_stack, y_stack + y, alpha=0.85,
                        color=col, label=f"lv{lvl} active")
        ax.plot(ticks, y_stack + y, color=col, linewidth=0.7, alpha=0.6)
        y_stack = y_stack + y

    # Constructing + idle as overlay lines
    if constructing.any():
        ax.plot(ticks, constructing, color="#e3b341", linewidth=1.1,
                linestyle="--", label="building", alpha=0.8)
    if repairing.any():
        ax.plot(ticks, repairing, color="#ff6c32", linewidth=1.1,
                linestyle="-.", label="repairing", alpha=0.8)
    if idle.any():
        ax.plot(ticks, idle, color="#B9B9B9", linewidth=1.1,
                linestyle="--", label="idle", alpha=0.8)

    ax.set_title(btype, fontsize=9, pad=4)
    ax.set_xlabel("tick", fontsize=7)
    ax.set_ylabel("count", fontsize=7)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper left",
              framealpha=0.3, facecolor="#161b22", edgecolor="#30363d")


def _plot_flags(ax, snapshots, ticks) -> None:
    """
    Horizontal timeline per flag (active = filled rectangle).
    Directive shown as a colored background band.
    Tax rate and urgency plotted as lines on a secondary axis.
    """
    all_flags = []
    for s in snapshots:
        for f in s["critical_flags"] + s["strategic_flags"]:
            if f not in all_flags:
                all_flags.append(f)

    # ── Directive background ───────────────────────────────────────────────
    prev_dir   = None
    span_start = ticks[0]
    for i, s in enumerate(snapshots):
        d = s["directive"]
        if d != prev_dir:
            if prev_dir is not None:
                c = _DIRECTIVE_COLORS.get(prev_dir, "#8b949e")
                ax.axvspan(span_start, ticks[i], color=c, alpha=0.08)
            span_start = ticks[i]
            prev_dir   = d
    if prev_dir:
        c = _DIRECTIVE_COLORS.get(prev_dir, "#8b949e")
        ax.axvspan(span_start, ticks[-1], color=c, alpha=0.08)

    # ── Flag rows ─────────────────────────────────────────────────────────
    y_positions = {flag: idx for idx, flag in enumerate(all_flags)}
    for s in snapshots:
        active = set(s["critical_flags"] + s["strategic_flags"])
        for flag in all_flags:
            if flag in active:
                y   = y_positions[flag]
                col = _FLAG_COLORS.get(flag, "#c9d1d9")
                ax.barh(y, 1, left=s["tick"] - 0.5, height=0.7,
                        color=col, alpha=0.75)

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()), fontsize=7)
    ax.set_xlabel("tick", fontsize=7)
    ax.set_title("Flags  (background = directive)", fontsize=9, pad=4)
    ax.tick_params(labelsize=7)

    # ── Tax & urgency secondary axis ──────────────────────────────────────
    ax2 = ax.twinx()
    tax     = [s["tax_rate"] for s in snapshots]
    urgency = [s["urgency"]  for s in snapshots]
    ax2.plot(ticks, tax,     color="#bc8cff", linewidth=1.2, linestyle="--", label="tax rate",  alpha=0.8)
    ax2.plot(ticks, urgency, color="#58a6ff", linewidth=1.2, linestyle=":",  label="urgency",   alpha=0.8)
    ax2.set_ylabel("rate / urgency [0-1]", fontsize=7, color="#8b949e")
    ax2.set_ylim(0, 1.2)
    ax2.tick_params(labelsize=7, colors="#8b949e")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#30363d")
    ax2.legend(fontsize=7, loc="upper right",
               framealpha=0.3, facecolor="#161b22", edgecolor="#30363d")

    # ── Directive annotation at changes ───────────────────────────────────
    prev = None
    for s in snapshots:
        if s["directive"] != prev:
            col = _DIRECTIVE_COLORS.get(s["directive"], "#8b949e")
            ax.text(
                s["tick"], len(all_flags) + 0.1,
                s["directive"],
                fontsize=6, color=col, va="bottom", ha="center",
                fontfamily="monospace",
            )
            prev = s["directive"]

    ax.set_ylim(-0.5, len(all_flags) + 0.8)
    ax.invert_yaxis()


# ──────────────────────────────────────────────────────────────────────────────
# QUICK-START DEMO
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from buildings import BuildingType, ResourceType as R
    from colony import (
        Colony, Faction, Directive, DirectiveType,
        CriticalFlag, StrategicFlag,
    )

    faction = Faction(faction_id=0, name="House Atreides")

    home = Colony(
        colony_id=0, name="Arrakeen", system_id=0, population=800.0,
        stockpile={int(R.MINERALS): 2000., int(R.ENERGY): 500.,
                   int(R.ORGANICS): 8000., int(R.RARE_MATS): 20.},
    )
    outpost = Colony(
        colony_id=1, name="Sietch Tabr", system_id=1, population=200.0,
        stockpile={int(R.MINERALS): 500., int(R.ENERGY): 100.,
                   int(R.ORGANICS): 800., int(R.RARE_MATS): 5.},
    )

    faction.add_colony(home)
    faction.add_colony(outpost)

    home.construct_building(BuildingType.FARM,        planet_index=0)
    home.construct_building(BuildingType.MINE,        planet_index=0)
    home.construct_building(BuildingType.POWER_PLANT, planet_index=0)
    outpost.construct_building(BuildingType.FARM,     planet_index=0)

    faction.issue_directive(0, DirectiveType.HARVEST, tax_rate=0.15, urgency=0.7)
    faction.issue_directive(1, DirectiveType.EXPAND,  tax_rate=0.05, urgency=0.8)

    home_history    = []
    outpost_history = []
    faction_history = []

    for _ in range(40):
        faction.tick()
        home_history.append(take_snapshot(home))
        outpost_history.append(take_snapshot(outpost))
        faction_history.append(take_snapshot(faction))

        # Switch directives mid-run for a more interesting plot
        if _ == 19:
            faction.issue_directive(0, DirectiveType.EXPAND,  tax_rate=0.10, urgency=0.9)
            faction.issue_directive(1, DirectiveType.DEFEND,  tax_rate=0.20, urgency=1.0)

    fig = plot_history(home_history,    title="Arrakeen Colony — 40 tick history",   show=False, save_path="/mnt/user-data/outputs/arrakeen.png")
    fig = plot_history(outpost_history, title="Sietch Tabr Colony — 40 tick history", show=False, save_path="/mnt/user-data/outputs/sietch_tabr.png")
    fig = plot_history(faction_history, title="House Atreides Faction — 40 tick history", show=False, save_path="/mnt/user-data/outputs/house_atreides_faction.png")
    print("Plots saved.")
