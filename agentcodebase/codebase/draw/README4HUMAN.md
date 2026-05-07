# Matplotlib Codebase Settings and Preparations

## Introduction

在论文撰写和可视化的过程中，**将实验得到的结构化的原始数据**转化为**清晰、直观、一目了然**的各式各样的图片和表格是一个关键的步骤，然后 `matplotlib` 支持的复杂配置浩如烟海，花费大量的时间学习会实时更新的绘图接口毫无意义，然而仅凭 `matplotlib` 提供的基础配置很显然无法满足绘图的审美需求。

因此，在 AI 工作流和智能体工具盛行的当下，我希望在 codebase 中设计一种**简单方便、人类可控、可以实时更新**的绘图工作流，它更像是一种说明书，而不是一个封装完善的智能体应用。（形式是什么不重要，能够达到最终的效果就行），而本文档即为该工作流的详细配置文档。

此工作流的优势在于：

- **简单方便**：人类干预者只需要起到**给出需要处理的结构化数据和画出图片的类型指引**以及**检查代码运行得到的 PDF 文件审查**，不需要动手写配置复杂，重复度极高的代码。并且相关内容自动作为上下文写入 Agent（例如 Claude Code）的记忆中，人类无需手动实现。
- **人类可控**：人类可以手动微调 Agent 生成的代码，包括修改图标标题等细节的内容，而不是完全的黑箱封装。
- **实时更新**：CodeBase 是在 [`matplotlib` 官方库示例源代码](https://matplotlib.org/stable/_downloads/46b4cb42d5bb56cc39e2b5b2b520b38d/gallery_python.zip) 的基础上经过手动微调和精调细选实现的，本质只是一份处理过的可直接使用的 References，可以实时更新并且添加个性化的代码示例和提示词。

## Pipeline

### 选择色调

[https://mycolor.space/](https://mycolor.space/) 这个网站可以自适应的选择对应合适的舒服的色调。

<details>
  <summary>个人推荐的色调</summary>

1. 淡色渐变 1
    #05c6b4
    #46d4a8
    #72e099
    #9eea89
    #caf27a
    #f9f871

2. 深色渐变 1
    #05c6b4
    #00b4cd
    #009edd
    #0082db
    #6f60c0
    #99358e

3. 淡色渐变 2
    #fa6306
    #ff7c55
    #ff9b8e
    #ffbec3

4. 深色渐变 2
    #fa6306
    #f03c5d
    #c0438b
    #7b5296
    #3f5380
    #2f4858

5. 浅色渐变 3
    #004712
    #2e6b33
    #559257
    #7cba7c
    #a4e4a3

6. 深色渐变 3
    #a4e4a3
    #55d0a9
    #00b9b7
    #009fc4
    #0083c8
    #0062bb

</details>



### Code Templates

- 推荐下载 [这个链接](https://matplotlib.org/stable/_downloads/46b4cb42d5bb56cc39e2b5b2b520b38d/gallery_python.zip) 然后解压得到文件夹 `gallery_python` 到 `codebase/draw/gallery_python` 中，作为官方文档，实时查阅。

- `codebase/draw/gallery_python_new` 文件夹中的文件是精选过的 **47** 个 Python 画图代码核心示范代码

使用如下命令可以生成对应的图片示例在 `codebase/draw/images` 文件夹中（PDF 格式）：

```bash
python codebase/draw/generate_plots.py
```

### `.mplstyle` Codebase

`mplstyle` files borrowed from [Public Github](https://github.com/hosilva/mplstyle)

> [!IMPORATNT]
> 提前保证对应字体已经安装并且能被 Matplotlib 识别

- `codebase/draw/mplstyle/computer_modern.mplstyle`
- `codebase/draw/mplstyle/times_new_roman.mplstyle`

### Prompts Generation

```bash
# 可以实现提示词的按需动态加载
python codebase/draw/generate_docs.py
```

- [codebase/draw/prompts/GALLERY_ALL.md](./prompts/GALLERY_ALL.md): 总体提示词
- [codebase/draw/prompts/GALLERY_DOCS_RAW.md](./prompts/GALLERY_DOCS_RAW.md)：代码模板指示文件

最终生成的完整提示词在：[`codebase/draw/README4AGENTS.md`](./README4AGENTS.md)