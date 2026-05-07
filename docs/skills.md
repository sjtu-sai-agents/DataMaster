# Skills Module

The Skills module provides the skill system for EvoMaster, enabling knowledge and operator capabilities.

## Overview

```
evomaster/skills/
├── base.py           # BaseSkill, SkillRegistry
├── knowledge/        # Knowledge skills
│   └── {skill_name}/
│       ├── SKILL.md     # Skill definition
│       └── references/  # Reference documents
└── rag/              # RAG operator skill
    ├── SKILL.md
    ├── scripts/
    │   ├── database.py
    │   ├── encode.py
    │   └── search.py
    └── references/
```

## Skill Types

EvoMaster supports two types of skills:

### Knowledge Skills

Knowledge skills contain only information, no executable scripts:
- **Level 1 (meta_info)**: ~100 tokens, always in context
- **Level 2 (full_info)**: 500-2000 tokens, loaded on demand

### Operator Skills

Operator skills include executable scripts:
- **Level 1 (meta_info)**: ~100 tokens, always in context
- **Level 2 (full_info)**: 500-2000 tokens, loaded on demand
- **Level 3 (scripts)**: Executable code, run via tool call

## SkillMetaInfo

Metadata parsed from SKILL.md frontmatter.

```python
class SkillMetaInfo(BaseModel):
    """Skill metadata (Level 1)

    Parsed from SKILL.md YAML frontmatter.
    Always in context to help Agent decide whether to use the skill.
    """
    name: str = Field(description="Skill name")
    description: str = Field(description="Skill description with usage scenarios")
    skill_type: str = Field(description="Skill type: knowledge or operator")
    license: str | None = Field(default=None, description="License info")
```

## BaseSkill

Abstract base class for all skills.

```python
class BaseSkill(ABC):
    """Skill base class

    Skills are EvoMaster components containing:
    - Level 1 (meta_info): Skill metadata (~100 tokens), always in context
    - Level 2 (full_info): Complete info (500-2000 tokens), loaded on demand
    - Level 3 (scripts): Executable code (Operator type only)
    """

    skill_type: ClassVar[str] = "base"

    def __init__(self, skill_path: Path):
        """Initialize Skill

        Args:
            skill_path: Skill directory path
        """

    def get_full_info(self) -> str:
        """Get complete info (Level 2)

        Extracted from SKILL.md body, loaded on demand.

        Returns:
            Complete skill info text
        """

    def get_reference(self, reference_name: str) -> str:
        """Get reference document content

        Args:
            reference_name: Reference name (e.g., "forms.md", "reference/api.md")

        Returns:
            Reference document content
        """

    @abstractmethod
    def to_context_string(self) -> str:
        """Convert to context string

        Returns string that should be added to Agent context.
        """
```

## KnowledgeSkill

Knowledge type skill implementation.

```python
class KnowledgeSkill(BaseSkill):
    """Knowledge type Skill

    Contains only knowledge info, no executable scripts.
    - Level 1: meta_info (always in context)
    - Level 2: full_info (loaded on demand)
    """

    skill_type: ClassVar[str] = "knowledge"

    def to_context_string(self) -> str:
        """Returns meta_info description for context"""
        return f"[Knowledge: {self.meta_info.name}] {self.meta_info.description}"
```

## OperatorSkill

Operator type skill implementation.

```python
class OperatorSkill(BaseSkill):
    """Operator type Skill

    Contains executable operation scripts.
    - Level 1: meta_info (always in context)
    - Level 2: full_info (loaded on demand)
    - Level 3: scripts (executable scripts)
    """

    skill_type: ClassVar[str] = "operator"

    def __init__(self, skill_path: Path):
        super().__init__(skill_path)
        self.scripts_dir = self.skill_path / "scripts"
        self.available_scripts = self._scan_scripts()

    def _scan_scripts(self) -> list[Path]:
        """Scan scripts directory for executable scripts

        Returns:
            List of script paths (.py, .sh, .js)
        """

    def get_script_path(self, script_name: str) -> Path | None:
        """Get script path by name

        Args:
            script_name: Script name

        Returns:
            Script path, or None if not exists
        """

    def to_context_string(self) -> str:
        """Returns meta_info with available scripts list"""
        scripts_info = ", ".join([s.name for s in self.available_scripts])
        return f"[Operator: {self.meta_info.name}] {self.meta_info.description} (Scripts: {scripts_info})"
```

## SkillRegistry

Skill registry for managing all available skills.

```python
class SkillRegistry:
    """Skill registry center

    Manages all available Skills, supporting:
    - Auto-discovery and loading
    - On-demand retrieval
    - Providing meta_info for Agent selection
    """

    def __init__(self, skills_root: Path):
        """Initialize SkillRegistry

        Args:
            skills_root: Skills root directory (contains knowledge/ and operator/ subdirs)
        """

    def get_skill(self, name: str) -> BaseSkill | None:
        """Get skill by name

        Args:
            name: Skill name

        Returns:
            Skill object, or None if not exists
        """

    def get_all_skills(self) -> list[BaseSkill]:
        """Get all skills"""

    def get_knowledge_skills(self) -> list[KnowledgeSkill]:
        """Get all Knowledge skills"""

    def get_operator_skills(self) -> list[OperatorSkill]:
        """Get all Operator skills"""

    def get_meta_info_context(self) -> str:
        """Get all skills' meta_info for Agent context

        Returns:
            String containing all skills' meta_info
        """

    def search_skills(self, query: str) -> list[BaseSkill]:
        """Search skills by keyword

        Args:
            query: Search keyword

        Returns:
            List of matching skills
        """
```

## SKILL.md Format

### Frontmatter (YAML)

```yaml
---
name: skill-name
description: Brief description with usage scenarios and trigger conditions
skill_type: knowledge  # or operator
license: MIT
---
```

### Body (Markdown)

The body contains the full_info (Level 2):

```markdown
# Skill Name

## Overview

Detailed description of what this skill does.

## Usage

When to use this skill:
- Scenario 1
- Scenario 2

## Details

Technical details, parameters, examples, etc.

## References

- [Reference 1](./references/ref1.md)
- [Reference 2](./references/ref2.md)
```

## Directory Structure

### Knowledge Skill

```
evomaster/skills/knowledge/
└── my_knowledge_skill/
    ├── SKILL.md           # Skill definition
    └── references/        # Optional reference docs
        ├── guide.md
        └── examples.md
```

### Operator Skill

```
evomaster/skills/
└── my_operator_skill/
    ├── SKILL.md           # Skill definition
    ├── scripts/           # Executable scripts
    │   ├── main.py
    │   └── helper.sh
    └── references/        # Optional reference docs
        └── api.md
```

## Usage Examples

### Loading Skills in Playground

```yaml
# config.yaml
skills:
  enabled: true
  skills_root: "evomaster/skills"
```

```python
from evomaster.skills import SkillRegistry
from pathlib import Path

# Load skills
registry = SkillRegistry(Path("evomaster/skills"))

# Get all skills
all_skills = registry.get_all_skills()

# Get meta_info for Agent context
context = registry.get_meta_info_context()

# Search skills
results = registry.search_skills("rag")
```

### Using Skills via SkillTool

Agent can use skills through the `use_skill` tool:

```python
# Get skill info
{"action": "get_info", "skill_name": "rag"}

# Get reference doc
{"action": "get_reference", "skill_name": "rag", "reference_name": "api.md"}

# Run script (Operator only)
{"action": "run_script", "skill_name": "rag", "script_name": "search.py", "script_args": "--query 'search term'"}
```

### Creating a New Skill

1. Create skill directory:
```bash
mkdir -p evomaster/skills/knowledge/my_skill
```

2. Create SKILL.md:
```markdown
---
name: my-skill
description: A skill that helps with XYZ tasks. Use when you need to do ABC.
skill_type: knowledge
---

# My Skill

## Overview

This skill provides knowledge about XYZ...

## When to Use

- When you need to understand ABC
- When working with DEF concepts

## Details

Detailed information here...
```

3. Add references (optional):
```bash
mkdir -p evomaster/skills/knowledge/my_skill/references
echo "# Reference Doc" > evomaster/skills/knowledge/my_skill/references/guide.md
```

## Related Documentation

- [Architecture Overview](./architecture.md)
- [Tools Module](./tools.md)
- [Agent Module](./agent.md)
