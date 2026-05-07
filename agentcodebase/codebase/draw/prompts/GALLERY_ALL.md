# `matplotlib` 代码绘图指南

## 基本要求（必须遵守）

- 对于结构化的数据，统一使用 `csv` 文件格式或者`npy` 格式，**将计算对应内容的部分和画图的部分耦合**，分在不同的模块中，你只需要在画图的模块中参考这个文件。
- 颜色的选择会主动提及，**不要使用默认颜色**！
- 特定的格式文件在 `codebase/draw/mplstyle`，务必使用他们。（人类使用者会告知使用什么字体，使用什么 style 等等）

```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
# Load custom style
style_file = base_dir / "mplstyle" / "times_new_roman.mplstyle"
plt.style.use(str(style_file))
```

- 画图标的时候**不要使用中文**，全部**使用英文**
- **不要设计具有太多图例的文字**，这样很容易造成视图遮挡影响美观性！图例也应该简洁了当！
- 画图的时候使用 `fig.savefig(output_file, bbox_inches='tight')` 来存储，存储为 **PDF** 格式放在对应的专门文件夹中（建议是 `images` 文件夹），不要使用 `plot.show()`，必须使用: `plt.tight_layout()`
- **画图的时候不要 ax.title**!

## 颜色的选择

```python
COLORS = ['#05c6b4', '#00b4cd', '#009edd', '#0082db', '#6f60c0', '#99358e']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLORS)
```

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


## 其他注意事项

如果需要字体优化，可以参考下面的示例代码：

```python
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
```

- **图例拥挤**: 使用 `bbox_to_anchor=(1.05, ...)` 移到外侧 + `tight_layout()`
- **colorbar数字过大**: 设置 `labelsize=7` 或 `8`
- **标题/标签过大**: 缩小字体到 8-12
- **元素挤压**: 增大 `figsize` 或使用 `tight_layout()`

## 代码模板参考

{template}