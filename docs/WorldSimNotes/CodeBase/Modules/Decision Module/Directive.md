---
tags:
  - Class
---
### Overview
[[Faction]] issue directives to each [[Colony]] to tell the colony to focus on specific aspects of development and which resources to deliver to the faction or other colonies. These directives help to make colonies more specialized or adapt to the current situation to help the overall faction become more productive

### Properties

| Property        | Description                                                                                                                                    |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| directive_type  | primary intent. See [[DirectiveType]] for what kinds of directives are available                                                               |
| tax_rate        | fraction [0.0, 1.0+] of each resource produced that flows into the colony's faction sub-stockpile. Values > 1.0 will draw from local reserves. |
| urgency         | scalar [0.0, 1.0] — how aggressively to pursue the directive vs. balanced upkeep. 1.0 = full commitment.                                       |
| target_resource | optional [[ResourceType]] to focus HARVEST directives                                                                                          |
| target_building | optional [[BuildingType]] to focus DEFEND/EXPAND directives                                                                                    |
| export_dest_id  | colony id that should receive faction sub-stockpile transfers                                                                                  |
| override_flags  | set of [[CodeBase/Modules/Decision Module/StrategicFlag]] values the faction explicitly suppresses ([[CriticalFlag]] are never suppressible)                                   |

