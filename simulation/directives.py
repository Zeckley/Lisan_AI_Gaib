"""
simulation/directive.py
=======================
Directive-driven colony management.

Design
------
- Faction-level AI issues directives to colonies via DirectiveIssuer.
  Colonies are obliged to respond; failure erodes colony happiness.
- Each Directive targets one ResourceType and carries an urgency (0.0–10.0).
  Urgency is set by the faction AI based on its analysis of the colony's
  situation and strategic goals (e.g. pre-evacuation urgency=10.0).
- Urgency 0 = balanced mode: grow all resources; no single directive dominates.
- Urgency > 0 = aggressive mode: directives sorted by urgency, highest first.
  Only ONE action executes per tick — first successful directive wins.
  Remaining directives are skipped for that tick.
- reserve_fraction holds reserves from ALL resources to protect against
  e.g. a minerals-focus draining organics and causing population decay.
- Transport directives: faction AI issues is_transport=True when a colony
  is asked to export resources.  priority_list prioritizes transport
  infrastructure (railyard, shipyard) first.  If total transport throughput
  (railyard + shipyard output) meets or exceeds export_demand, priority
  shifts to production buildings for the export resource to maintain surplus.
- export_demand is the recipient colony's required import per tick.  If the
  exporting colony cannot supply it without going into deficit, it must build
  more production.  The faction AI should not issue demands the colony cannot
  reasonably meet.
- Revolt: Colony.faction_happiness drops each tick a directive fails.
  Revolt trigger at 0.2.  The colony breaks away and becomes a new
  "rebellion forces of {faction.name}" faction.  Nearby colonies and colonies
  receiving resources from the revolting colony lose happiness.  This can
  cascade — a second colony dropping below 0.2 joins the rebel faction.
- revolted bool is True once revolt occurs.  If the colony is conquered,
  revolted resets to False and faction_happiness returns to 0.7.
- Distance is calculated between solar systems (colony.solar_system.position).

Priority lists are hardcoded IntEnums per ResourceType (AI will override later).
Factory defaults provided; AI can swap them out or set from config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

from buildings import (
    BuildingType,
    BuildingState,
    BUILDING_STATS,
    MAX_BUILDING_LEVEL,
    ResourceType,
)

R = ResourceType

REVOLT_TRIGGER = 0.2
CONQUEST_HAPPINESS_RESET = 0.7
MIN_HAPPINESS_FROM_REVOLT = 5.0


# ---------------------------------------------------------------------------
# Priority List IntEnums
# ---------------------------------------------------------------------------

class MineralsPriority(IntEnum):
    MINE         = 0
    FACTORY      = 1
    POWER_PLANT  = 2
    RAILYARD     = 3
    SHIPYARD     = 4
    FARM         = 5
    LAB          = 6
    FORT         = 7


class EnergyPriority(IntEnum):
    POWER_PLANT  = 0
    RAILYARD     = 1
    MINE         = 2
    FACTORY     = 3
    FARM         = 4
    LAB          = 5
    SHIPYARD     = 6
    FORT         = 7


class OrganicsPriority(IntEnum):
    FARM         = 0
    POWER_PLANT  = 1
    LAB          = 2
    RAILYARD     = 3
    FACTORY      = 4
    SHIPYARD     = 5
    MINE         = 6
    FORT         = 7


class RareMatsPriority(IntEnum):
    MINE         = 0
    FACTORY      = 1
    LAB          = 2
    POWER_PLANT  = 3
    SHIPYARD     = 4
    FARM         = 5
    RAILYARD     = 6
    FORT         = 7


DEFAULT_PRIORITY_LISTS: Dict[ResourceType, Tuple[BuildingType, ...]] = {
    R.MINERALS:   tuple(BuildingType[b.name] for b in MineralsPriority),
    R.POWER:      tuple(BuildingType[b.name] for b in EnergyPriority),
    R.ORGANICS:   tuple(BuildingType[b.name] for b in OrganicsPriority),
    R.RARE_MATS:  tuple(BuildingType[b.name] for b in RareMatsPriority),
}


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class DirectiveAction(IntEnum):
    NONE    = 0
    BUILD   = 1
    UPGRADE = 2


@dataclass(frozen=True)
class DirectiveActionResult:
    action:        DirectiveAction
    building_type: Optional[BuildingType] = None
    target_id:     Optional[int]         = None


# ---------------------------------------------------------------------------
# Directive
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Directive:
    target_resource: ResourceType
    urgency:         float
    priority_list:   Tuple[BuildingType, ...] = field(compare=False)
    active:          bool      = True
    is_transport:    bool      = False
    export_demand:   float     = 0.0
    export_destinations: Tuple[str, ...] = field(
        default_factory=tuple,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not (0.0 <= self.urgency <= 10.0):
            raise ValueError(f"urgency must be in [0.0, 10.0], got {self.urgency}")

    @classmethod
    def make(
        cls,
        target_resource:   ResourceType,
        urgency:            float          = 0.0,
        priority_list:    Optional[Tuple[BuildingType, ...]] = None,
        active:             bool           = True,
        is_transport:       bool           = False,
        export_demand:      float          = 0.0,
        export_destinations: Tuple[str, ...] = field(default_factory=tuple),
    ) -> Directive:
        if priority_list is None:
            priority_list = DEFAULT_PRIORITY_LISTS.get(target_resource, ())
        return cls(
            target_resource     = target_resource,
            urgency             = urgency,
            priority_list       = priority_list,
            active              = active,
            is_transport        = is_transport,
            export_demand       = export_demand,
            export_destinations = export_destinations,
        )


# ---------------------------------------------------------------------------
# DirectiveManager
# ---------------------------------------------------------------------------

@dataclass
class DirectiveManager:
    """
    Lives on Colony.  Holds the colony's active directives and decides
    which action to take each tick.

    Only ONE action executes per tick — highest-urgency directive that
    can afford something wins.  Remaining directives are skipped.
    """

    directives:        List[Directive] = field(default_factory=list)
    reserve_fraction:  float           = 0.0

    last_result: Optional[DirectiveActionResult] = field(default=None, init=False)

    def add(self, directive: Directive) -> None:
        self.directives.append(directive)

    def remove(self, target_resource: ResourceType) -> bool:
        for i, d in enumerate(self.directives):
            if d.target_resource == target_resource:
                self.directives.pop(i)
                return True
        return False

    def get(self, target_resource: ResourceType) -> Optional[Directive]:
        for d in self.directives:
            if d.target_resource == target_resource:
                return d
        return None

    def get_transport(self) -> List[Directive]:
        return [d for d in self.directives if d.active and d.is_transport]

    def update_urgency(self, target_resource: ResourceType, urgency: float) -> bool:
        d = self.get(target_resource)
        if d is None:
            return False
        idx = self.directives.index(d)
        self.directives[idx] = Directive(
            target_resource     = d.target_resource,
            urgency             = urgency,
            priority_list       = d.priority_list,
            active              = d.active,
            is_transport        = d.is_transport,
            export_demand       = d.export_demand,
            export_destinations = d.export_destinations,
        )
        return True

    @staticmethod
    def is_balanced(directives: List[Directive]) -> bool:
        return all(d.urgency == 0.0 for d in directives if d.active)

    @staticmethod
    def _tolerance(urgency: float) -> float:
        if urgency <= 0.0:
            return 0.0
        return (10.0 ** (urgency / 5.0) - 1.0) / 9.0 * 200.0

    @staticmethod
    def _can_afford(
        cost:        Dict[int, float],
        stockpile:   Dict[int, float],
        is_balanced: bool,
        tolerance:   float,
    ) -> bool:
        for res, amount in cost.items():
            available = stockpile.get(res, 0.0)
            deficit   = amount - available
            if is_balanced:
                if deficit > 0.0:
                    return False
            else:
                if deficit > tolerance:
                    return False
        return True

    @staticmethod
    def _transport_throughput(active_buildings) -> float:
        total = 0.0
        for b in active_buildings:
            if b.state == BuildingState.ACTIVE:
                if b.building_type in (BuildingType.RAILYARD, BuildingType.SHIPYARD):
                    total += b.stats.production_rate.get(R.TRANSFER, 0.0)
        return total

    def _resolve(
        self,
        directive:        Directive,
        stockpile:        Dict[int, float],
        active_buildings, # List[Building]
    ) -> Optional[DirectiveActionResult]:
        
        is_balanced = directive.urgency == 0.0
        tolerance   = self._tolerance(directive.urgency)

        if directive.is_transport and directive.export_demand > 0.0:
            throughput = self._transport_throughput(active_buildings)
            if throughput >= directive.export_demand:
                priority_list = directive.priority_list
            else:
                priority_list = (BuildingType.RAILYARD, BuildingType.SHIPYARD)
        else:
            priority_list = directive.priority_list

        by_type: Dict[BuildingType, List] = {}
        for b in active_buildings:
            if b.state == BuildingState.ACTIVE:
                by_type.setdefault(b.building_type, []).append(b)

        for btype in priority_list:
            candidates = [
                b for b in by_type.get(btype, [])
                if b.level < MAX_BUILDING_LEVEL
            ]

            if candidates:
                target     = max(candidates, key=lambda b: b.level)
                next_level = target.level + 1
                cost       = BUILDING_STATS[btype][next_level].build_cost
                if self._can_afford(cost, stockpile, is_balanced, tolerance):
                    return DirectiveActionResult(
                        action        = DirectiveAction.UPGRADE,
                        building_type = btype,
                        target_id     = target.id,
                    )

            cost = BUILDING_STATS[btype][1].build_cost
            if self._can_afford(cost, stockpile, is_balanced, tolerance):
                return DirectiveActionResult(
                    action        = DirectiveAction.BUILD,
                    building_type = btype,
                    target_id     = None,
                )

        return None

    @staticmethod
    def _spend(
        result:           DirectiveActionResult,
        active_buildings, # List[Building]
        stockpile:        Dict[int, float]
    ) -> None:
    
        if result.building_type is None or result.action == DirectiveAction.NONE:
            return

        if result.action == DirectiveAction.BUILD:
            level = 1
        elif result.action == DirectiveAction.UPGRADE:
            level = 1
            if result.target_id is not None:
                for b in active_buildings:
                    if b.id == result.target_id:
                        level = b.level + 1
                        break

        cost = BUILDING_STATS[result.building_type][level].build_cost
        for res, amount in cost.items():
            stockpile[res] = max(0.0, stockpile.get(res, 0.0) - amount)

    def decide(
        self,
        stockpile:        Dict[int, float],
        active_buildings,  # List[Building]
    ) -> Optional[DirectiveActionResult]:
        
        active = [d for d in self.directives if d.active]
        if not active:
            self.last_result = None
            return None

        spendable = {
            res: amount * (1.0 - self.reserve_fraction)
            for res, amount in stockpile.items()
        }

        if self.is_balanced(active):
            sorted_dirs = sorted(active, key=lambda d: d.urgency)
        else:
            sorted_dirs = sorted(active, key=lambda d: d.urgency, reverse=True)

        working_stock = dict(spendable)

        for d in sorted_dirs:
            result = self._resolve(d, working_stock, active_buildings)
            if result is not None and result.action != DirectiveAction.NONE:
                self._spend(result, active_buildings, working_stock)
                self.last_result = result
                return result

        self.last_result = None
        return None

    def summary(self) -> str:
        lines = [f"DirectiveManager ({len(self.directives)} active):"]
        for d in self.directives:
            flags = []
            if d.is_transport:
                flags.append(f"TRANSPORT demand={d.export_demand}")
            if not d.active:
                flags.append("inactive")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            lines.append(
                f"  {d.target_resource.name:<12} "
                f"urgency={d.urgency:.1f}{flag_str}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DirectiveIssuer  — faction-level AI interface
# ---------------------------------------------------------------------------

@dataclass
class DirectiveIssuer:
    """
    Lives on Faction.  Issues directives to colonies.

    Faction AI calls issue() to send a directive to a colony.
    The colony is obliged to respond; failure erodes colony happiness.

    Methods
    -------
    issue(colony, directive)
        Send a directive to a colony.

    revoke(colony, target_resource)
        Remove a directive from a colony.

    set_urgency(colony, target_resource, urgency)
        Adjust urgency on an existing directive.

    record_response(colony, success, directive)
        Call after colony.tick().  Failure deducts colony.faction_happiness.
        If happiness drops below REVOLT_TRIGGER, trigger_revolt() is called.

    trigger_revolt(colony, faction_colonies)
        Colony breaks away and becomes a new "rebellion forces of {faction.name}"
        faction.  Cascades to any other colonies that also drop below
        REVOLT_TRIGGER from proximity or trade effects.

    get_receiving_colonies(colony, faction_colonies)
        Returns colonies that receive resources from the given colony via
        active transport directives.

    on_conquest(colony)
        Called when a revolted colony is conquered.  Resets revolted=False
        and faction_happiness to CONQUEST_HAPPINESS_RESET (0.7).
    """

    issued_log: List[Directive] = field(default_factory=list)

    def issue(
        self,
        colony:    'Colony',
        directive: Directive,
    ) -> None:
        
        colony.directives.add(directive)
        self.issued_log.append(directive)

    def revoke(
        self,
        colony:          'Colony',
        target_resource: ResourceType,
    ) -> None:
        
        colony.directives.remove(target_resource)

    def set_urgency(
        self,
        colony:          'Colony',
        target_resource: ResourceType,
        urgency:         float,
    ) -> None:
        
        colony.directives.update_urgency(target_resource, urgency)

    def record_response(
        self,
        colony:    'Colony',
        success:   bool,
        directive: Directive,
    ) -> None:
        
        if not success:
            colony.faction_happiness = max(0.0, colony.faction_happiness - 1.0)
            if colony.faction_happiness < REVOLT_TRIGGER:
                self.trigger_revolt(colony, [])

    def get_receiving_colonies(
        self,
        source_colony:      'Colony',
        faction_colonies:   List['Colony'],
    ) -> List['Colony']:
        receiving: List['Colony'] = []
        source_id = source_colony.id

        for colony in faction_colonies:
            if colony.id == source_colony.id or colony.revolted:
                continue
            for d in colony.directives.get_transport():
                if source_id in d.export_destinations:
                    receiving.append(colony)
                    break

        return receiving

    def _solar_system_distance(self, a: 'Colony', b: 'Colony') -> float:
        pos_a = a.solar_system.position
        pos_b = b.solar_system.position
        dx = pos_a[0] - pos_b[0]
        dy = pos_a[1] - pos_b[1]
        dz = pos_a[2] - pos_b[2]
        return (dx*dx + dy*dy + dz*dz) ** 0.5

    def apply_proximity_effect(
        self,
        source_colony:      'Colony',
        faction_colonies:   List['Colony'],
        amount:             float,
    ) -> None:
        for colony in faction_colonies:
            if colony.id == source_colony.id or colony.revolted:
                continue
            distance = self._solar_system_distance(source_colony, colony)
            if distance < float('inf'):
                deduction = amount * (1.0 / (distance + 1.0))
                colony.faction_happiness = max(0.0, colony.faction_happiness - deduction)

    def trigger_revolt(
        self,
        revolter:          'Colony',
        faction_colonies:  List['Colony'],
    ) -> None:
        revolt_faction = revolter.faction

        rebel_faction = self._create_rebel_faction(revolt_faction, [revolter])
        revolter.faction = rebel_faction
        revolter.revolted = True
        revolt_faction.colonies = [c for c in revolt_faction.colonies if c.id != revolter.id]
        rebel_faction.colonies.append(revolter)

        affected: Set[str] = set()
        self._cascade_revolt(
            revolter,
            faction_colonies,
            affected,
            rebel_faction,
            revolt_faction,
        )

    def _create_rebel_faction(self, original: 'Faction', colonies: List['Colony']) -> 'Faction':
        from colony import Faction, FactionType
        name = f"rebellion forces of {original.name}"
        return Faction(name=name, faction_type=FactionType.AGGRESSIVE, population=0.0)

    def _cascade_revolt(
        self,
        revolter:           'Colony',
        faction_colonies:   List['Colony'],
        affected:           Set[str],
        rebel_faction:     'Faction',
        original_faction:  'Faction',
    ) -> None:
        queue = [revolter]
        processed: Set[str] = set()

        while queue:
            current = queue.pop(0)
            current_id = current.id

            if current_id in processed:
                continue
            processed.add(current_id)
            affected.add(current_id)

            receiving = self.get_receiving_colonies(current, faction_colonies)
            for colony in receiving:
                if colony.id not in affected:
                    colony.faction_happiness = max(
                        0.0,
                        colony.faction_happiness - MIN_HAPPINESS_FROM_REVOLT,
                    )
                    if colony.faction_happiness < REVOLT_TRIGGER and not colony.revolted:
                        queue.append(colony)

            self.apply_proximity_effect(current, faction_colonies, MIN_HAPPINESS_FROM_REVOLT)

            for colony in faction_colonies:
                if (colony.id not in affected
                        and not colony.revolted
                        and colony.faction == original_faction
                        and colony.faction_happiness < REVOLT_TRIGGER):
                    queue.append(colony)

        for colony in faction_colonies:
            if colony.id in processed and not colony.revolted:
                colony.faction = rebel_faction
                colony.revolted = True
                original_faction.colonies = [c for c in original_faction.colonies if c.id != colony.id]
                rebel_faction.colonies.append(colony)

    def on_conquest(
        self,
        colony: 'Colony',
    ) -> None:
        colony.revolted = False
        colony.faction_happiness = CONQUEST_HAPPINESS_RESET