---
name: class-documentation-update
description: Create or update structured Obsidian-compatible markdown documentation for a specific Python class, with linked files per method category
metadata:
  scope: project
  output-dir: docs
  style: obsidian
---

## What I do

I read a Python class source file and produce (or update) a set of Obsidian-compatible markdown files in `docs/` that document the class. I split documentation into a main overview file and one file per logical group of methods (e.g. workers, building management, resource collection, decision making, tick loop, reporting, private helpers, flag system).

## When to use me

Use this skill when you need to:
- Generate fresh documentation for a new or undocumented class
- Update existing docs after adding/removing/renaming methods or fields
- Reorganize docs to match the current state of the source code

## Workflow

### Step 1 — Read the source

Read the entire class definition from the source file. Identify:
- All constructor `__init__` / dataclass fields (public and private)
- All `@property` decorated methods
- All public and private methods
- Any module-level constants or enums the class references

### Step 2 — Group methods into categories

Organize methods into logical groups. For a typical simulation agent class the categories are:

| Category | Typical contents |
|---|---|
| **Private Helpers** | Low-level utilities like `_new_id`, `_add_to`, `_deduct`, `_can_afford` |
| **Workers & Population** | Recruitment, assignment, worker queries, lab upskilling |
| **Building Management** | Construction, upgrades, repairs, surge, lifecycle |
| **Resource Collection** | Production collection, tax, upkeep, feeding, rate calculations |
| **Flag System** | Flag enums, thresholds, flag evaluation |
| **Decision Making** | Validation checks, survival loop, directive sub-rules |
| **Decision Validation** | Pipeline analysis, build/upgrade validation |
| **Tick Loop** | The main `tick()` method |
| **Reporting** | `summary()`, `stockpile_summary()`, `flag_summary()` |

Adjust categories to match the actual code structure — don't force methods into ill-fitting groups. If the class has unique groups (e.g. "Combat", "Trade", "Research"), create them.

### Step 3 — Create or update the main overview file

File: `docs/<ClassName>.md`

Contents:
- A brief description of what the class does (from the module/class docstring)
- A table of **constructor fields** (name, type, default, description)
- A table of **runtime state fields** (name, type, description)
- A table of **properties** (name, return type, description)
- A **method index** organized by category, each entry linking to the category file with `[[ClassName - Category#method_name|method_name()]]`
- A **related classes** table referencing other classes with `[[links]]`
- A **related documentation** section linking to all category files

### Step 4 — Create or update one file per method category

File pattern: `docs/<ClassName> - <Category>.md`

Each file should contain:
- A header with the class name and category
- One section per method with:
  - The full method signature (with type annotations)
  - A brief description of what it does
  - Which other methods call it (with `[[links]]`)
  - Which methods it calls (with `[[links]]`)
  - Any side effects or important notes
- A **Related** section at the bottom linking back to the main file and other categories

### Step 5 — Use Obsidian-compatible cross-references

Throughout all files, use `[[ClassName]]` and `[[ClassName#method_name]]` syntax:
- `[[Building]]` → links to Building class doc (if it exists)
- `[[Colony - Resource Collection#collect_resources|collect_resources()]]` → links to specific method in a category file
- `[[ResourceType]]` → links to enum/class doc
- `[[Faction#tick|Faction.tick()]]` → links to a method in another class

### Step 6 — Update existing files (not overwrite blindly)

If `docs/<ClassName>.md` or any category file already exists:
- Read the existing file first
- Update changed method signatures, add new methods, remove deleted ones
- Preserve any user-added notes or cross-references that are still valid
- Maintain the existing file structure and style

## Conventions

- Use GitHub-flavored markdown tables for fields, properties, and method indices
- Use fenced code blocks with `python` language tag for method signatures
- Use bullet lists for calling/called-by relationships
- Keep descriptions concise (1-3 sentences per method)
- Order methods within a category file in the same order they appear in source
- Include line references to the source file when helpful (e.g. `> File: \`simulation/colony.py\``)
- End each category file with a `## Related` section linking to the main file and sibling files

## Example output structure

```
docs/
├── Colony.md
├── Colony - Private Helpers.md
├── Colony - Workers & Population.md
├── Colony - Building Management.md
├── Colony - Resource Collection.md
├── Colony - Flag System.md
├── Colony - Decision Making.md
├── Colony - Tick Loop.md
└── Colony - Reporting.md
```
