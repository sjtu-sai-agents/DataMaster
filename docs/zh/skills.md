# Skills 模块

Skills 模块为 EvoMaster 提供技能系统，支持知识和操作能力。

## 概述

```
evomaster/skills/
├── base.py           # BaseSkill, SkillRegistry
├── knowledge/        # Knowledge 技能
│   └── {skill_name}/
│       ├── SKILL.md     # 技能定义
│       └── references/  # 参考文档
└── rag/              # RAG 操作技能
    ├── SKILL.md
    ├── scripts/
    │   ├── database.py
    │   ├── encode.py
    │   └── search.py
    └── references/
```

## 技能类型

EvoMaster 支持两种类型的技能：

### Knowledge Skills

Knowledge 技能只包含信息，没有可执行脚本：
- **Level 1 (meta_info)**：~100 tokens，始终在上下文中
- **Level 2 (full_info)**：500-2000 tokens，按需加载

### Operator Skills

Operator 技能包含可执行脚本：
- **Level 1 (meta_info)**：~100 tokens，始终在上下文中
- **Level 2 (full_info)**：500-2000 tokens，按需加载
- **Level 3 (scripts)**：可执行代码，通过工具调用运行

## SkillMetaInfo

从 SKILL.md frontmatter 解析的元数据。

```python
class SkillMetaInfo(BaseModel):
    """技能元信息（Level 1）

    从 SKILL.md 的 YAML frontmatter 解析得到。
    始终在上下文中，帮助 Agent 决定是否使用该技能。
    """
    name: str = Field(description="技能名称")
    description: str = Field(description="技能描述，包含使用场景")
    skill_type: str = Field(description="技能类型：knowledge 或 operator")
    license: str | None = Field(default=None, description="许可证信息")
```

## BaseSkill

所有技能的抽象基类。

```python
class BaseSkill(ABC):
    """技能基类

    Skills 是 EvoMaster 的技能组件，包含：
    - Level 1 (meta_info)：技能元信息（~100 tokens），始终在上下文
    - Level 2 (full_info)：完整信息（500-2000 tokens），按需加载
    - Level 3 (scripts)：可执行代码（仅 Operator 类型）
    """

    skill_type: ClassVar[str] = "base"

    def __init__(self, skill_path: Path):
        """初始化 Skill

        Args:
            skill_path: 技能目录路径
        """

    def get_full_info(self) -> str:
        """获取完整信息（Level 2）

        从 SKILL.md body 部分提取，按需加载。

        Returns:
            完整的技能信息文本
        """

    def get_reference(self, reference_name: str) -> str:
        """获取参考文档内容

        Args:
            reference_name: 参考文档名称（如 "forms.md", "reference/api.md"）

        Returns:
            参考文档内容
        """

    @abstractmethod
    def to_context_string(self) -> str:
        """转换为上下文字符串

        返回应该添加到 Agent 上下文中的字符串。
        """
```

## KnowledgeSkill

Knowledge 类型技能实现。

```python
class KnowledgeSkill(BaseSkill):
    """Knowledge 类型 Skill

    只包含知识信息，没有可执行脚本。
    - Level 1：meta_info（始终在上下文）
    - Level 2：full_info（按需加载）
    """

    skill_type: ClassVar[str] = "knowledge"

    def to_context_string(self) -> str:
        """返回上下文的 meta_info 描述"""
        return f"[Knowledge: {self.meta_info.name}] {self.meta_info.description}"
```

## OperatorSkill

Operator 类型技能实现。

```python
class OperatorSkill(BaseSkill):
    """Operator 类型 Skill

    包含可执行的操作脚本。
    - Level 1：meta_info（始终在上下文）
    - Level 2：full_info（按需加载）
    - Level 3：scripts（可执行脚本）
    """

    skill_type: ClassVar[str] = "operator"

    def __init__(self, skill_path: Path):
        super().__init__(skill_path)
        self.scripts_dir = self.skill_path / "scripts"
        self.available_scripts = self._scan_scripts()

    def _scan_scripts(self) -> list[Path]:
        """扫描 scripts 目录获取可执行脚本

        Returns:
            脚本路径列表（.py, .sh, .js）
        """

    def get_script_path(self, script_name: str) -> Path | None:
        """按名称获取脚本路径

        Args:
            script_name: 脚本名称

        Returns:
            脚本路径，如果不存在则返回 None
        """

    def to_context_string(self) -> str:
        """返回 meta_info 和可用脚本列表"""
        scripts_info = ", ".join([s.name for s in self.available_scripts])
        return f"[Operator: {self.meta_info.name}] {self.meta_info.description} (Scripts: {scripts_info})"
```

## SkillRegistry

管理所有可用技能的注册表。

```python
class SkillRegistry:
    """技能注册中心

    管理所有可用的 Skills，支持：
    - 自动发现和加载
    - 按需检索
    - 提供 meta_info 供 Agent 选择
    """

    def __init__(self, skills_root: Path):
        """初始化 SkillRegistry

        Args:
            skills_root: skills 根目录（包含 knowledge/ 和 operator/ 子目录）
        """

    def get_skill(self, name: str) -> BaseSkill | None:
        """按名称获取技能

        Args:
            name: 技能名称

        Returns:
            Skill 对象，如果不存在则返回 None
        """

    def get_all_skills(self) -> list[BaseSkill]:
        """获取所有技能"""

    def get_knowledge_skills(self) -> list[KnowledgeSkill]:
        """获取所有 Knowledge 技能"""

    def get_operator_skills(self) -> list[OperatorSkill]:
        """获取所有 Operator 技能"""

    def get_meta_info_context(self) -> str:
        """获取所有技能的 meta_info，用于添加到 Agent 上下文

        Returns:
            包含所有技能 meta_info 的字符串
        """

    def search_skills(self, query: str) -> list[BaseSkill]:
        """按关键词搜索技能

        Args:
            query: 搜索关键词

        Returns:
            匹配的技能列表
        """
```

## SKILL.md 格式

### Frontmatter（YAML）

```yaml
---
name: skill-name
description: 简要描述，包含使用场景和触发条件
skill_type: knowledge  # 或 operator
license: MIT
---
```

### Body（Markdown）

body 部分包含 full_info（Level 2）：

```markdown
# 技能名称

## 概述

详细描述此技能的功能。

## 使用场景

何时使用此技能：
- 场景 1
- 场景 2

## 详情

技术细节、参数、示例等。

## 参考

- [参考 1](./references/ref1.md)
- [参考 2](./references/ref2.md)
```

## 目录结构

### Knowledge Skill

```
evomaster/skills/knowledge/
└── my_knowledge_skill/
    ├── SKILL.md           # 技能定义
    └── references/        # 可选的参考文档
        ├── guide.md
        └── examples.md
```

### Operator Skill

```
evomaster/skills/
└── my_operator_skill/
    ├── SKILL.md           # 技能定义
    ├── scripts/           # 可执行脚本
    │   ├── main.py
    │   └── helper.sh
    └── references/        # 可选的参考文档
        └── api.md
```

## 使用示例

### 在 Playground 中加载 Skills

```yaml
# config.yaml
skills:
  enabled: true
  skills_root: "evomaster/skills"
```

```python
from evomaster.skills import SkillRegistry
from pathlib import Path

# 加载技能
registry = SkillRegistry(Path("evomaster/skills"))

# 获取所有技能
all_skills = registry.get_all_skills()

# 获取 Agent 上下文的 meta_info
context = registry.get_meta_info_context()

# 搜索技能
results = registry.search_skills("rag")
```

### 通过 SkillTool 使用 Skills

Agent 可以通过 `use_skill` 工具使用技能：

```python
# 获取技能信息
{"action": "get_info", "skill_name": "rag"}

# 获取参考文档
{"action": "get_reference", "skill_name": "rag", "reference_name": "api.md"}

# 运行脚本（仅 Operator）
{"action": "run_script", "skill_name": "rag", "script_name": "search.py", "script_args": "--query 'search term'"}
```

### 创建新技能

1. 创建技能目录：
```bash
mkdir -p evomaster/skills/knowledge/my_skill
```

2. 创建 SKILL.md：
```markdown
---
name: my-skill
description: 一个帮助完成 XYZ 任务的技能。当需要做 ABC 时使用。
skill_type: knowledge
---

# 我的技能

## 概述

此技能提供关于 XYZ 的知识...

## 使用场景

- 当需要理解 ABC 时
- 当处理 DEF 概念时

## 详情

详细信息在这里...
```

3. 添加参考文档（可选）：
```bash
mkdir -p evomaster/skills/knowledge/my_skill/references
echo "# 参考文档" > evomaster/skills/knowledge/my_skill/references/guide.md
```

## 相关文档

- [架构概述](./architecture.md)
- [Tools 模块](./tools.md)
- [Agent 模块](./agent.md)
