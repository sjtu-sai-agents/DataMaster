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


### 目录索引

- ### 1. Basic Plots `codebase/draw/gallery_python_new/basic_plots` (基础图表 - 7个)
- ### 2. Statistical Plots `codebase/draw/gallery_python_new/statistical_plots` (统计图表 - 8个)
- ### 3. Distribution Plots `codebase/draw/gallery_python_new/distribution_plots` (分布图 - 7个)
- ### 4. Heatmaps & Contours `codebase/draw/gallery_python_new/heatmaps_contours` (热图等高线 - 4个)
- ### 5. 3D Plots `codebase/draw/gallery_python_new/3d_plots` (三维图表 - 5个)
- ### 6. Polar Plots `codebase/draw/gallery_python_new/polar_plots` (极坐标图 - 5个)
- ### 7. Time Series `codebase/draw/gallery_python_new/time_series` (时间序列 - 4个)
- ### 8. Composite Plots `codebase/draw/gallery_python_new/composite_plots` (组合图表 - 3个)
- ### 9. Images & Shapes `codebase/draw/gallery_python_new/images_shapes` (图像形状 - 2个)
- ### 10. Misc Plots `codebase/draw/gallery_python_new/misc_plots` (其他图表 - 2个)


### 1. Basic Plots `codebase/draw/gallery_python_new/basic_plots` (基础图表 - 7个)

**simple_plot.py** - 基础折线图

使用 matplotlib 的这套 API（包括 `plt.subplots()`、`ax.plot()`、`ax.set()` 等）可以绘制能够展现数据随时间或其他**连续变量变化趋势的折线图**。这种图表特别适用于可视化序列数据、趋势分析、周期性波动以及多组数据的比较。你可以自定义坐标轴标签、标题、网格线等元素，并通过调整线条样式、颜色、标记点来增强图表的可读性和表现力，从而清晰传达数据背后的信息与规律。

```python
"""
=========
折线图
=========

Create a basic 折线图.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test.png")
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.plot` / `matplotlib.pyplot.plot`
#    - `matplotlib.pyplot.subplots`
#    - `matplotlib.figure.Figure.savefig`
#
# .. tags::
#
#    plot-style: line
#    level: beginner

```

**scatter_demo2.py** - 散点图

使用 `ax.scatter()` 创建带有可变颜色和大小的散点图,可以展示多个变量之间的关系。通过参数 `c`(颜色映射)、`s`(标记大小)和 `alpha`(透明度)可以展示数据的多个维度,特别适合金融数据、相关性分析和多变量数据可视化。

```python
"""
=============
Scatter Demo2
=============

Demo of 散点图 with varying marker colors and sizes.
"""
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.cbook as cbook

# Load a numpy record array from yahoo csv data with fields date, open, high,
# low, close, volume, adj_close from the mpl-data/sample_data directory. The
# record array stores the date as an np.datetime64 with a day unit ('D') in
# the date column.
price_data = cbook.get_sample_data('goog.npz')['price_data']
price_data = price_data[-250:]  # get the most recent 250 trading days

delta1 = np.diff(price_data["adj_close"]) / price_data["adj_close"][:-1]

# Marker size in units of points^2
volume = (15 * price_data["volume"][:-2] / price_data["volume"][0])**2
close = 0.003 * price_data["close"][:-2] / 0.003 * price_data["open"][:-2]

fig, ax = plt.subplots()
ax.scatter(delta1[:-1], delta1[1:], c=close, s=volume, alpha=0.5)

ax.set_xlabel(r'$\Delta_i$', fontsize=15)
ax.set_ylabel(r'$\Delta_{i+1}$', fontsize=15)
ax.set_title('Volume and percent change')

ax.grid(True)
fig.tight_layout()

plt.show()

# %%
# .. tags::
#
#    component: marker
#    component: color
#    plot-style: scatter
#    level: beginner

```

**barchart.py** - 分组柱状图

使用 `ax.bar()` 和偏移量技术创建分组柱状图,适合比较多组分类数据。通过 `bar_label()` 可以在每个柱子上添加数值标签,配合图例、网格线和自定义颜色,能够清晰展示不同类别之间的对比关系。

```python
"""
=============================
Grouped 柱状图 with labels
=============================

This 示例 shows a how to create a grouped 柱状图 and how to annotate
bars with labels.
"""

# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np

species = ("Adelie", "Chinstrap", "Gentoo")
penguin_means = {
    'Bill Depth': (18.35, 18.43, 14.98),
    'Bill Length': (38.79, 48.83, 47.50),
    'Flipper Length': (189.95, 195.82, 217.19),
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(8, 5))

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3, fontsize=9)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Length (mm)')
ax.set_title('Penguin attributes by species')
ax.set_xticks(x + width, species)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 250)

# Adjust layout to prevent overlapping
plt.tight_layout()

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.bar` / `matplotlib.pyplot.bar`
#    - `matplotlib.axes.Axes.bar_label` / `matplotlib.pyplot.bar_label`
#
# .. tags::
#
#    component: label
#    plot-type: bar
#    level: beginner

```

**barh.py** - 水平柱状图

使用 `ax.barh()` 创建水平方向的柱状图,通过 `invert_yaxis()` 让标签从上到下排列,特别适合展示排名、长标签类别或水平方向的数据对比。支持添加误差条(`xerr`)来表示数据的不确定性范围。

```python
"""
====================
Horizontal 柱状图
====================

This 示例 showcases a simple horizontal 柱状图.
"""
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

fig, ax = plt.subplots()

# 示例 data
people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

ax.barh(y_pos, performance, xerr=error, align='center')
ax.set_yticks(y_pos, labels=people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go today?')

plt.show()

# %%
# .. tags::
#
#    plot-type: bar
#    level: beginner

```

**bar_stacked.py** - 堆叠柱状图

使用 `ax.bar()` 的 `bottom` 参数实现柱状图的堆叠效果,能够同时展示各部分数值和累计总量。通过循环更新 `bottom` 变量,可以创建多层堆叠图表,适合展示构成比例、累计数据和部分与整体的关系。

```python
"""
=================
Stacked 柱状图
=================

This is an 示例 of creating a stacked bar plot
using `~matplotlib.pyplot.bar`.
"""

import matplotlib.pyplot as plt
import numpy as np

# data from https://allisonhorst.github.io/palmerpenguins/

species = (
    "Adelie\n $\\mu=$3700.66g",
    "Chinstrap\n $\\mu=$3733.09g",
    "Gentoo\n $\\mu=5076.02g$",
)
weight_counts = {
    "Below": np.array([70, 31, 58]),
    "Above": np.array([82, 37, 66]),
}
width = 0.5

fig, ax = plt.subplots()
bottom = np.zeros(3)

for boolean, weight_count in weight_counts.items():
    p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom)
    bottom += weight_count

ax.set_title("Number of penguins with above average body mass")
ax.legend(loc="upper right")

plt.show()

# %%
# .. tags::
#
#    plot-type: bar
#    level: beginner

```

**stackplot_demo.py** - 堆叠面积图

使用 `ax.stackplot()` 创建堆叠面积图,展示多个序列随时间变化的累积效果。通过设置 `baseline='wiggle'` 参数可以创建流图(streamgraph)效果,适合可视化人口变化、资源分配、资金流向等时间序列的构成分析。

```python
"""
===========================
Stackplots and streamgraphs
===========================
"""

# %%
# Stackplots
# ----------
#
# Stackplots draw multiple datasets as vertically stacked areas. This is
# useful when the individual data values and additionally their cumulative
# value are of interest.


import matplotlib.pyplot as plt
import numpy as np

import matplotlib.ticker as mticker

# data from United Nations World Population Prospects (Revision 2019)
# https://population.un.org/wpp/, license: CC BY 3.0 IGO
year = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2018]
population_by_continent = {
    'Africa': [.228, .284, .365, .477, .631, .814, 1.044, 1.275],
    'the Americas': [.340, .425, .519, .619, .727, .840, .943, 1.006],
    'Asia': [1.394, 1.686, 2.120, 2.625, 3.202, 3.714, 4.169, 4.560],
    'Europe': [.220, .253, .276, .295, .310, .303, .294, .293],
    'Oceania': [.012, .015, .019, .022, .026, .031, .036, .039],
}

fig, ax = plt.subplots()
ax.stackplot(year, population_by_continent.values(),
             labels=population_by_continent.keys(), alpha=0.8)
ax.legend(loc='upper left', reverse=True)
ax.set_title('World population')
ax.set_xlabel('Year')
ax.set_ylabel('Number of people (billions)')
# add tick at every 200 million people
ax.yaxis.set_minor_locator(mticker.MultipleLocator(.2))

plt.show()

# %%
# Streamgraphs
# ------------
#
# Using the *baseline* parameter, you can turn an ordinary stacked area plot
# with baseline 0 into a stream graph.


# Fixing random state for reproducibility
np.random.seed(19680801)


def gaussian_mixture(x, n=5):
    """Return a random mixture of *n* Gaussians, evaluated at positions *x*."""
    def add_random_gaussian(a):
        amplitude = 1 / (.1 + np.random.random())
        dx = x[-1] - x[0]
        x0 = (2 * np.random.random() - .5) * dx
        z = 10 / (.1 + np.random.random()) / dx
        a += amplitude * np.exp(-(z * (x - x0))**2)
    a = np.zeros_like(x)
    for j in range(n):
        add_random_gaussian(a)
    return a


x = np.linspace(0, 100, 101)
ys = [gaussian_mixture(x) for _ in range(3)]

fig, ax = plt.subplots()
ax.stackplot(x, ys, baseline='wiggle')
plt.show()

# %%
# .. tags::
#
#    plot-type: stackplot
#    level: intermediate

```

**horizontal_barchart_distribution.py** - 水平分布柱状图

使用 `ax.barh()` 和 `left` 参数创建水平堆叠柱状图,通过 `data.cumsum()` 计算每个类别的起始位置,结合颜色映射和标签文本颜色自动调整,能够优雅地展示问卷调查结果、评分分布等离散数据分布情况。

```python
"""
=============================================
Discrete distribution as horizontal 柱状图
=============================================

Stacked 柱状图s can be used to visualize discrete distributions.

This 示例 visualizes the result of a survey in which people could rate
their agreement to questions on a five-element scale.

The horizontal stacking is achieved by calling `~.Axes.barh()` for each
category and passing the starting point as the cumulative sum of the
already drawn bars via the parameter ``left``.
"""

import matplotlib.pyplot as plt
import numpy as np

category_names = ['Strongly disagree', 'Disagree',
                  'Neither agree nor disagree', 'Agree', 'Strongly agree']
results = {
    'Question 1': [10, 15, 17, 32, 26],
    'Question 2': [26, 22, 29, 10, 13],
    'Question 3': [35, 37, 7, 2, 19],
    'Question 4': [32, 11, 9, 15, 33],
    'Question 5': [21, 29, 5, 5, 40],
    'Question 6': [8, 19, 5, 30, 38]
}


def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlGn'](
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncols=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax


survey(results, category_names)
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.barh` / `matplotlib.pyplot.barh`
#    - `matplotlib.axes.Axes.bar_label` / `matplotlib.pyplot.bar_label`
#    - `matplotlib.axes.Axes.legend` / `matplotlib.pyplot.legend`
#
# .. tags::
#
#    domain: statistics
#    component: label
#    plot-type: bar
#    level: beginner

```


### 2. Statistical Plots `codebase/draw/gallery_python_new/statistical_plots` (统计图表 - 8个)

**boxplot.py** - 美化箱线图

使用自定义样式配置创建出版级质量的箱线图,通过 `boxprops`、`medianprops`、`meanprops`、`flierprops` 等参数精细控制箱体、中位数线、均值点和异常值的样式。配合自定义配色方案和透明度设置,能够生成适合学术论文和专业报告的高质量统计图表,清晰展示数据的分布特征、中位数、四分位数和异常值。

```python
"""
=================================
Beautiful Boxplot Example
=================================

This example demonstrates how to create publication-quality boxplots
with custom styling and professional appearance.
"""

import matplotlib.pyplot as plt
import numpy as np

# Load Times New Roman style
import os
style_file = os.path.join(os.path.dirname(__file__), '../../mplstyle/times_new_roman.mplstyle')
plt.style.use(style_file)

# Custom color palette
CUSTOM_COLORS = ['#05c6b4', '#00b4cd', '#009edd', '#0082db', '#6f60c0', '#99358e']
COLORS = CUSTOM_COLORS[:4]  # Use first 4 colors for boxes

# Generate sample data
np.random.seed(42)
data = [
    np.random.normal(0, 1, 100),
    np.random.normal(2, 1.2, 100),
    np.random.normal(1, 0.8, 100),
    np.random.normal(3, 1.5, 100)
]

labels = ['Group A', 'Group B', 'Group C', 'Group D']

# Create figure with professional styling
fig, ax = plt.subplots(figsize=(10, 6))

# Custom boxplot styling with thin black borders
boxprops = {
    'linewidth': 1.0,
    'facecolor': 'white',
    'edgecolor': 'black'
}

whiskerprops = {
    'linewidth': 1.0,
    'color': 'black'
}

capprops = {
    'linewidth': 1.0,
    'color': 'black'
}

medianprops = {
    'linewidth': 2.0,
    'color': '#99358e',
    'solid_capstyle': 'round'
}

meanprops = {
    'marker': 'D',
    'markerfacecolor': CUSTOM_COLORS[4],
    'markeredgecolor': 'white',
    'markersize': 7,
    'markeredgewidth': 1.0
}

flierprops = {
    'marker': 'o',
    'markerfacecolor': 'white',
    'markeredgecolor': CUSTOM_COLORS[5],
    'markersize': 5,
    'markeredgewidth': 1.0,
    'alpha': 0.7
}

# Create boxplot
bp = ax.boxplot(
    data,
    labels=labels,
    patch_artist=True,
    showmeans=True,
    boxprops=boxprops,
    whiskerprops=whiskerprops,
    capprops=capprops,
    medianprops=medianprops,
    meanprops=meanprops,
    flierprops=flierprops,
    widths=0.6
)

# Color the boxes
for patch, color in zip(bp['boxes'], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add grid
ax.grid(True, axis='y', linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

# Set labels and title
ax.set_xlabel('Experimental Groups', fontsize=12, fontweight='bold')
ax.set_ylabel('Measured Values', fontsize=12, fontweight='bold')

# Customize spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('/Users/anonymous-user/Desktop/Dev/AgentCodeBase/codebase/draw/images/boxplot.pdf',
            bbox_inches='tight', dpi=150)

print("Boxplot saved successfully!")

```

**boxplot_demo.py** - 箱线图基础

展示箱线图的各种样式选项,包括 `notch`(缺口)、`sym`(异常值符号)、`orientation`(方向)等参数的使用方法。通过多个子图对比不同配置效果,演示如何自定义箱线图的外观、隐藏异常值、改变须长度等,是掌握箱线图基础配置的完整参考。

```python
"""
========
Boxplots
========

Visualizing boxplots with matplotlib.

The following 示例s show off how to visualize boxplots with
Matplotlib. There are many options to control their appearance and
the statistics that they use to summarize the data.

.. redirect-from:: /gallery/pyplots/boxplot_demo_pyplot
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Polygon

# Fixing random state for reproducibility
np.random.seed(19680801)

# fake up some data
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low))

fig, axs = plt.subplots(2, 3)

# basic plot
axs[0, 0].boxplot(data)
axs[0, 0].set_title('basic plot')

# notched plot
axs[0, 1].boxplot(data, notch=True)
axs[0, 1].set_title('notched plot')

# change outlier point symbols
axs[0, 2].boxplot(data, sym='gD')
axs[0, 2].set_title('change outlier\npoint symbols')

# don't show outlier points
axs[1, 0].boxplot(data, sym='')
axs[1, 0].set_title("don't show\noutlier points")

# horizontal boxes
axs[1, 1].boxplot(data, sym='rs', orientation='horizontal')
axs[1, 1].set_title('horizontal boxes')

# change whisker length
axs[1, 2].boxplot(data, sym='rs', orientation='horizontal', whis=0.75)
axs[1, 2].set_title('change whisker length')

fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.4, wspace=0.3)

# fake up some more data
spread = np.random.rand(50) * 100
center = np.ones(25) * 40
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
d2 = np.concatenate((spread, center, flier_high, flier_low))
# Making a 2-D array only works if all the columns are the
# same length.  If they are not, then use a list instead.
# This is actually more efficient because boxplot converts
# a 2-D array into a list of vectors internally anyway.
data = [data, d2, d2[::2]]

# Multiple 箱线图s on one Axes
fig, ax = plt.subplots()
ax.boxplot(data)

plt.show()


# %%
# Below we'll generate data from five different probability distributions,
# each with different characteristics. We want to play with how an IID
# bootstrap resample of the data preserves the distributional
# properties of the original sample, and a boxplot is one visual tool
# to make this assessment

random_dists = ['Normal(1, 1)', 'Lognormal(1, 1)', 'Exp(1)', 'Gumbel(6, 4)',
                'Triangular(2, 9, 11)']
N = 500

norm = np.random.normal(1, 1, N)
logn = np.random.lognormal(1, 1, N)
expo = np.random.exponential(1, N)
gumb = np.random.gumbel(6, 4, N)
tria = np.random.triangular(2, 9, 11, N)

# Generate some random indices that we'll use to resample the original data
# arrays. For code brevity, just use the same random indices for each array
bootstrap_indices = np.random.randint(0, N, N)
data = [
    norm, norm[bootstrap_indices],
    logn, logn[bootstrap_indices],
    expo, expo[bootstrap_indices],
    gumb, gumb[bootstrap_indices],
    tria, tria[bootstrap_indices],
]

fig, ax1 = plt.subplots(figsize=(10, 6))
fig.canvas.manager.set_window_title('A Boxplot Example')
fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

bp = ax1.boxplot(data, notch=False, sym='+', orientation='vertical', whis=1.5)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

ax1.set(
    axisbelow=True,  # Hide the grid behind plot objects
    title='Comparison of IID Bootstrap Resampling Across Five Distributions',
    xlabel='Distribution',
    ylabel='Value',
)

# Now fill the boxes with desired colors
box_colors = ['darkkhaki', 'royalblue']
num_boxes = len(data)
medians = np.empty(num_boxes)
for i in range(num_boxes):
    box = bp['boxes'][i]
    box_x = []
    box_y = []
    for j in range(5):
        box_x.append(box.get_xdata()[j])
        box_y.append(box.get_ydata()[j])
    box_coords = np.column_stack([box_x, box_y])
    # Alternate between Dark Khaki and Royal Blue
    ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
    # Now draw the median lines back over what we just filled in
    med = bp['medians'][i]
    median_x = []
    median_y = []
    for j in range(2):
        median_x.append(med.get_xdata()[j])
        median_y.append(med.get_ydata()[j])
        ax1.plot(median_x, median_y, 'k')
    medians[i] = median_y[0]
    # Finally, overplot the sample averages, with horizontal alignment
    # in the center of each box
    ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
             color='w', marker='*', markeredgecolor='k')

# Set the axes ranges and axes labels
ax1.set_xlim(0.5, num_boxes + 0.5)
top = 40
bottom = -5
ax1.set_ylim(bottom, top)
ax1.set_xticklabels(np.repeat(random_dists, 2),
                    rotation=45, fontsize=8)

# Due to the Y-axis scale being different across samples, it can be
# hard to compare differences in medians across the samples. Add upper
# X-axis tick labels with the sample medians to aid in comparison
# (just use two decimal places of precision)
pos = np.arange(num_boxes) + 1
upper_labels = [str(round(s, 2)) for s in medians]
weights = ['bold', 'semibold']
for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
    k = tick % 2
    ax1.text(pos[tick], .95, upper_labels[tick],
             transform=ax1.get_xaxis_transform(),
             horizontalalignment='center', size='x-small',
             weight=weights[k], color=box_colors[k])

# Finally, add a basic legend
fig.text(0.80, 0.08, f'{N} Random Numbers',
         backgroundcolor=box_colors[0], color='black', weight='roman',
         size='x-small')
fig.text(0.80, 0.045, 'IID Bootstrap Resample',
         backgroundcolor=box_colors[1],
         color='white', weight='roman', size='x-small')
fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
         weight='roman', size='medium')
fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
         size='x-small')

plt.show()

# %%
# Here we write a custom function to bootstrap confidence intervals.
# We can then use the boxplot along with this function to show these intervals.


def fake_bootstrapper(n):
    """
    This is just a placeholder for the user's method of
    bootstrapping the median and its confidence intervals.

    Returns an arbitrary median and confidence interval packed into a tuple.
    """
    if n == 1:
        med = 0.1
        ci = (-0.25, 0.25)
    else:
        med = 0.2
        ci = (-0.35, 0.50)
    return med, ci

inc = 0.1
e1 = np.random.normal(0, 1, size=500)
e2 = np.random.normal(0, 1, size=500)
e3 = np.random.normal(0, 1 + inc, size=500)
e4 = np.random.normal(0, 1 + 2*inc, size=500)

treatments = [e1, e2, e3, e4]
med1, ci1 = fake_bootstrapper(1)
med2, ci2 = fake_bootstrapper(2)
medians = [None, None, med1, med2]
conf_intervals = [None, None, ci1, ci2]

fig, ax = plt.subplots()
pos = np.arange(len(treatments)) + 1
bp = ax.boxplot(treatments, sym='k+', positions=pos,
                notch=True, bootstrap=5000,
                usermedians=medians,
                conf_intervals=conf_intervals)

ax.set_xlabel('treatment')
ax.set_ylabel('response')
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], markersize=3.0)
plt.show()


# %%
# Here we customize the widths of the caps .

x = np.linspace(-7, 7, 140)
x = np.hstack([-25, x, 25])
fig, ax = plt.subplots()

ax.boxplot([x, x], notch=True, capwidths=[0.01, 0.2])

plt.show()

# %%
#
# .. tags:: domain: statistics, plot-type: boxplot
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.boxplot` / `matplotlib.pyplot.boxplot`
#    - `matplotlib.artist.Artist.set` / `matplotlib.pyplot.setp`

```

**violinplot.py** - 小提琴图

使用 `ax.violinplot()` 创建小提琴图,通过核密度估计(KDE)展示数据的概率分布形状。通过 `points`(KDE评估点数)、`bw_method`(带宽方法)、`showmeans`(显示均值)、`showmedians`(显示中位数)、`quantiles`(分位数)等参数可以精细控制小提琴图的显示效果,适合比较多个数据组的分布特征。

```python
"""
==================
小提琴图 basics
==================

小提琴图s are similar to 直方图s and 箱线图s in that they show
an abstract representation of the probability distribution of the
sample. Rather than showing counts of data points that fall into bins
or order statistics, 小提琴图s use kernel density estimation (KDE) to
compute an empirical distribution of the sample. That computation
is controlled by several parameters. This 示例 演示s how to
modify the number of points at which the KDE is evaluated (``points``)
and how to modify the bandwidth of the KDE (``bw_method``).

For more information on 小提琴图s and KDE, the scikit-learn docs
have a great section: https://scikit-learn.org/stable/modules/density.html
"""

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


# fake data
fs = 10  # fontsize
pos = [1, 2, 4, 5, 7, 8]
data = [np.random.normal(0, std, size=100) for std in pos]

fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(10, 4))

axs[0, 0].violinplot(data, pos, points=20, widths=0.3,
                     showmeans=True, showextrema=True, showmedians=True)
axs[0, 0].set_title('Custom violin 1', fontsize=fs)

axs[0, 1].violinplot(data, pos, points=40, widths=0.5,
                     showmeans=True, showextrema=True, showmedians=True,
                     bw_method='silverman')
axs[0, 1].set_title('Custom violin 2', fontsize=fs)

axs[0, 2].violinplot(data, pos, points=60, widths=0.7, showmeans=True,
                     showextrema=True, showmedians=True, bw_method=0.5)
axs[0, 2].set_title('Custom violin 3', fontsize=fs)

axs[0, 3].violinplot(data, pos, points=60, widths=0.7, showmeans=True,
                     showextrema=True, showmedians=True, bw_method=0.5,
                     quantiles=[[0.1], [], [], [0.175, 0.954], [0.75], [0.25]])
axs[0, 3].set_title('Custom violin 4', fontsize=fs)

axs[0, 4].violinplot(data[-1:], pos[-1:], points=60, widths=0.7,
                     showmeans=True, showextrema=True, showmedians=True,
                     quantiles=[0.05, 0.1, 0.8, 0.9], bw_method=0.5)
axs[0, 4].set_title('Custom violin 5', fontsize=fs)

axs[0, 5].violinplot(data[-1:], pos[-1:], points=60, widths=0.7,
                     showmeans=True, showextrema=True, showmedians=True,
                     quantiles=[0.05, 0.1, 0.8, 0.9], bw_method=0.5, side='low')

axs[0, 5].violinplot(data[-1:], pos[-1:], points=60, widths=0.7,
                     showmeans=True, showextrema=True, showmedians=True,
                     quantiles=[0.05, 0.1, 0.8, 0.9], bw_method=0.5, side='high')
axs[0, 5].set_title('Custom violin 6', fontsize=fs)

axs[1, 0].violinplot(data, pos, points=80, orientation='horizontal', widths=0.7,
                     showmeans=True, showextrema=True, showmedians=True)
axs[1, 0].set_title('Custom violin 7', fontsize=fs)

axs[1, 1].violinplot(data, pos, points=100, orientation='horizontal', widths=0.9,
                     showmeans=True, showextrema=True, showmedians=True,
                     bw_method='silverman')
axs[1, 1].set_title('Custom violin 8', fontsize=fs)

axs[1, 2].violinplot(data, pos, points=200, orientation='horizontal', widths=1.1,
                     showmeans=True, showextrema=True, showmedians=True,
                     bw_method=0.5)
axs[1, 2].set_title('Custom violin 9', fontsize=fs)

axs[1, 3].violinplot(data, pos, points=200, orientation='horizontal', widths=1.1,
                     showmeans=True, showextrema=True, showmedians=True,
                     quantiles=[[0.1], [], [], [0.175, 0.954], [0.75], [0.25]],
                     bw_method=0.5)
axs[1, 3].set_title('Custom violin 10', fontsize=fs)

axs[1, 4].violinplot(data[-1:], pos[-1:], points=200, orientation='horizontal',
                     widths=1.1, showmeans=True, showextrema=True, showmedians=True,
                     quantiles=[0.05, 0.1, 0.8, 0.9], bw_method=0.5)
axs[1, 4].set_title('Custom violin 11', fontsize=fs)

axs[1, 5].violinplot(data[-1:], pos[-1:], points=200, orientation='horizontal',
                     widths=1.1, showmeans=True, showextrema=True, showmedians=True,
                     quantiles=[0.05, 0.1, 0.8, 0.9], bw_method=0.5, side='low')

axs[1, 5].violinplot(data[-1:], pos[-1:], points=200, orientation='horizontal',
                     widths=1.1, showmeans=True, showextrema=True, showmedians=True,
                     quantiles=[0.05, 0.1, 0.8, 0.9], bw_method=0.5, side='high')
axs[1, 5].set_title('Custom violin 12', fontsize=fs)


for ax in axs.flat:
    ax.set_yticklabels([])

fig.suptitle("Violin Plotting Examples")
fig.subplots_adjust(hspace=0.4)
plt.show()

# %%
#
# .. tags:: plot-type: violin, domain: statistics
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.violinplot` / `matplotlib.pyplot.violinplot`

```

**customized_violin.py** - 自定义小提琴图

完全自定义小提琴图的样式,通过修改返回的 `ViolinPlot` 对象的 `bodies` 属性来设置填充颜色和透明度,使用 `vlines()` 和 `scatter()` 添加自定义的统计标记。这种方法可以实现完全个性化的视觉效果,适合需要特定品牌风格或出版要求的图表制作。

```python
"""
=========================
小提琴图 customization
=========================

This 示例 演示s how to fully customize 小提琴图s. The first plot
shows the default style by providing only the data. The second plot first
limits what Matplotlib draws with additional keyword arguments. Then a
simplified representation of a 箱线图 is drawn on top. Lastly, the styles of
the artists of the violins are modified.

For more information on 小提琴图s, the scikit-learn docs have a great
section: https://scikit-learn.org/stable/modules/density.html
"""

import matplotlib.pyplot as plt
import numpy as np


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


# create test data
np.random.seed(19680801)
data = [sorted(np.random.normal(0, std, 100)) for std in range(1, 5)]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)

ax1.set_title('Default violin plot')
ax1.set_ylabel('Observed values')
ax1.violinplot(data)

ax2.set_title('Customized violin plot')
parts = ax2.violinplot(
        data, showmeans=False, showmedians=False,
        showextrema=False)

for pc in parts['bodies']:
    pc.set_facecolor('#D43F3A')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

# set style for the axes
labels = ['A', 'B', 'C', 'D']
for ax in [ax1, ax2]:
    set_axis_style(ax, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()

# %%
#
# .. tags:: plot-type: violin, domain: statistics
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.violinplot` / `matplotlib.pyplot.violinplot`
#    - `matplotlib.axes.Axes.vlines` / `matplotlib.pyplot.vlines`

```

**errorbar.py** - 误差条图

使用 `ax.errorbar()` 创建带误差条的折线图,通过 `xerr` 和 `yerr` 参数指定X和Y方向的误差,配合 `fmt`(数据点格式)、`capsize`(误差帽大小)、`capthick`(误差帽粗细)、`elinewidth`(误差线粗细)等参数,能够清晰展示实验数据的不确定性范围,是科学数据可视化的标准方法。

```python
"""
=================
Errorbar function
=================

This exhibits the most basic use of the error bar method.
In this case, constant values are provided for the error
in both the x- and y-directions.
"""

import matplotlib.pyplot as plt
import numpy as np

# 示例 data
x = np.arange(0.1, 4, 0.5)
y = np.exp(-x)

fig, ax = plt.subplots()
ax.errorbar(x, y, xerr=0.2, yerr=0.4)
plt.show()

# %%
#
#
# .. tags:: plot-type: errorbar, domain: statistics,
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.errorbar` / `matplotlib.pyplot.errorbar`

```

**errorbar_features.py** - 误差条特性

展示误差条的高级功能,包括不对称误差(`xerr=[lower, upper]`)、对数坐标轴上的误差条、自定义误差线样式等。通过多种配置演示如何处理复杂的误差表示需求,适合需要在特殊坐标系统或展示非对称不确定性的科学数据分析场景。

```python
"""
=======================================
Different ways of specifying error bars
=======================================

Errors can be specified as a constant value (as shown in
:doc:`/gallery/statistics/errorbar`). However, this 示例 演示s
how they vary by specifying arrays of error values.

If the raw ``x`` and ``y`` data have length N, there are two options:

Array of shape (N,):
    Error varies for each point, but the error values are
    symmetric (i.e. the lower and upper values are equal).

Array of shape (2, N):
    Error varies for each point, and the lower and upper limits
    (in that order) are different (asymmetric case)

In addition, this 示例 演示s how to use log
scale with error bars.
"""

import matplotlib.pyplot as plt
import numpy as np

# 示例 data
x = np.arange(0.1, 4, 0.5)
y = np.exp(-x)

# 示例 error bar values that vary with x-position
error = 0.1 + 0.2 * x

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
ax0.errorbar(x, y, yerr=error, fmt='-o')
ax0.set_title('variable, symmetric error')

# error bar values w/ different -/+ errors that
# also vary with the x-position
lower_error = 0.4 * error
upper_error = error
asymmetric_error = [lower_error, upper_error]

ax1.errorbar(x, y, xerr=asymmetric_error, fmt='o')
ax1.set_title('variable, asymmetric error')
ax1.set_yscale('log')
plt.show()

# %%
#
# .. tags:: plot-type: errorbar, domain: statistics
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.errorbar` / `matplotlib.pyplot.errorbar`

```

**hexbin_demo.py** - 六边形分箱图

使用 `ax.hexbin()` 创建六边形分箱图,通过将数据点聚合到六边形网格中来展示二维数据的密度分布。相比散点图,这种方法更适合大数据集的可视化,通过 `gridsize`(网格大小)、`bins`(分箱方式,如 `'log'`)和 `cmap`(颜色映射)参数可以优化视觉效果,避免过度绘制问题。

```python
"""
=====================
Hexagonal binned plot
=====================

`~.Axes.hexbin` is a 2D 直方图 plot, in which the bins are hexagons and
the color represents the number of data points within each bin.
"""

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

n = 100_000
x = np.random.standard_normal(n)
y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)
xlim = x.min(), x.max()
ylim = y.min(), y.max()

fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True, figsize=(9, 4))

hb = ax0.hexbin(x, y, gridsize=50, cmap='inferno')
ax0.set(xlim=xlim, ylim=ylim)
ax0.set_title("Hexagon binning")
cb = fig.colorbar(hb, ax=ax0, label='counts')

hb = ax1.hexbin(x, y, gridsize=50, bins='log', cmap='inferno')
ax1.set(xlim=xlim, ylim=ylim)
ax1.set_title("With a log color scale")
cb = fig.colorbar(hb, ax=ax1, label='counts')

plt.show()

# %%
#
# .. tags:: plot-type: 直方图, plot-type: hexbin, domain: statistics
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.hexbin` / `matplotlib.pyplot.hexbin`

```

**confidence_ellipse.py** - 置信椭圆

绘制二维数据集的置信椭圆,通过计算协方差矩阵和特征值,使用 `Ellipse` patch 和 `Affine2D` 变换创建表示数据置信区间的椭圆。可以通过 `n_std` 参数控制置信水平(如2倍标准差对应95%置信区间),适合展示相关性分析、统计推断和数据分布的置信范围。

```python
"""
======================================================
Plot a confidence ellipse of a two-dimensional dataset
======================================================

This 示例 shows how to plot a confidence ellipse of a
two-dimensional dataset, using its pearson correlation coefficient.

The approach that is used to obtain the correct geometry is
explained and proved here:

https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html

The method avoids the use of an iterative eigen decomposition algorithm
and makes use of the fact that a normalized covariance matrix (composed of
pearson correlation coefficients and ones) is particularly easy to handle.
"""


import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# %%
#
# The plotting function itself
# """"""""""""""""""""""""""""
#
# This function plots the confidence ellipse of the covariance of the given
# array-like variables x and y. The ellipse is plotted into the given
# Axes object *ax*.
#
# The radiuses of the ellipse can be controlled by n_std which is the number
# of standard deviations. The default value is 3 which makes the ellipse
# enclose 98.9% of the points if the data is normally distributed
# like in these 示例s (3 standard deviations in 1-D contain 99.7%
# of the data, which is 98.9% of the data in 2-D).


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# %%
#
# A helper function to create a correlated dataset
# """"""""""""""""""""""""""""""""""""""""""""""""
#
# Creates a random two-dimensional dataset with the specified
# two-dimensional mean (mu) and dimensions (scale).
# The correlation can be controlled by the param 'dependency',
# a 2x2 matrix.

def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]


# %%
#
# Positive, negative and weak correlation
# """""""""""""""""""""""""""""""""""""""
#
# Note that the shape for the weak correlation (right) is an ellipse,
# not a circle because x and y are differently scaled.
# However, the fact that x and y are uncorrelated is shown by
# the axes of the ellipse being aligned with the x- and y-axis
# of the coordinate system.

np.random.seed(0)

PARAMETERS = {
    'Positive correlation': [[0.85, 0.35],
                             [0.15, -0.65]],
    'Negative correlation': [[0.9, -0.4],
                             [0.1, -0.6]],
    'Weak correlation': [[1, 0],
                         [0, 1]],
}

mu = 2, 4
scale = 3, 5

fig, axs = plt.subplots(1, 3, figsize=(9, 3))
for ax, (title, dependency) in zip(axs, PARAMETERS.items()):
    x, y = get_correlated_dataset(800, dependency, mu, scale)
    ax.scatter(x, y, s=0.5)

    ax.axvline(c='grey', lw=1)
    ax.axhline(c='grey', lw=1)

    confidence_ellipse(x, y, ax, edgecolor='red')

    ax.scatter(mu[0], mu[1], c='red', s=3)
    ax.set_title(title)

plt.show()


# %%
#
# Different number of standard deviations
# """""""""""""""""""""""""""""""""""""""
#
# A plot with n_std = 3 (blue), 2 (purple) and 1 (red)

fig, ax_nstd = plt.subplots(figsize=(6, 6))

dependency_nstd = [[0.8, 0.75],
                   [-0.2, 0.35]]
mu = 0, 0
scale = 8, 5

ax_nstd.axvline(c='grey', lw=1)
ax_nstd.axhline(c='grey', lw=1)

x, y = get_correlated_dataset(500, dependency_nstd, mu, scale)
ax_nstd.scatter(x, y, s=0.5)

confidence_ellipse(x, y, ax_nstd, n_std=1,
                   label=r'$1\sigma$', edgecolor='firebrick')
confidence_ellipse(x, y, ax_nstd, n_std=2,
                   label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
confidence_ellipse(x, y, ax_nstd, n_std=3,
                   label=r'$3\sigma$', edgecolor='blue', linestyle=':')

ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
ax_nstd.set_title('Different standard deviations')
ax_nstd.legend()
plt.show()


# %%
#
# Using the keyword arguments
# """""""""""""""""""""""""""
#
# Use the keyword arguments specified for `matplotlib.patches.Patch` in order
# to have the ellipse rendered in different ways.

fig, ax_kwargs = plt.subplots(figsize=(6, 6))
dependency_kwargs = [[-0.8, 0.5],
                     [-0.2, 0.5]]
mu = 2, -3
scale = 6, 5

ax_kwargs.axvline(c='grey', lw=1)
ax_kwargs.axhline(c='grey', lw=1)

x, y = get_correlated_dataset(500, dependency_kwargs, mu, scale)
# Plot the ellipse with zorder=0 in order to 演示
# its transparency (caused by the use of alpha).
confidence_ellipse(x, y, ax_kwargs,
                   alpha=0.5, facecolor='pink', edgecolor='purple', zorder=0)

ax_kwargs.scatter(x, y, s=0.5)
ax_kwargs.scatter(mu[0], mu[1], c='red', s=3)
ax_kwargs.set_title('Using keyword arguments')

fig.subplots_adjust(hspace=0.25)
plt.show()

# %%
#
# .. tags::
#
#    plot-type: speciality
#    plot-type: scatter
#    component: ellipse
#    component: patch
#    domain: statistics
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.transforms.Affine2D`
#    - `matplotlib.patches.Ellipse`

```


### 3. Distribution Plots `codebase/draw/gallery_python_new/distribution_plots` (分布图 - 7个)

**hist.py** - 基础直方图

使用 `ax.hist()` 创建直方图来展示数据的分布特征,通过 `bins`(分箱数量)、`density`(归一化为概率密度)、`alpha`(透明度)、`color`(颜色)和 `edgecolor`(边框颜色)等参数自定义样式。还展示二维直方图 `hist2d` 和百分比格式化,是数据分析中最常用的分布可视化方法。

```python
"""
==========
直方图s
==========

How to plot 直方图s with Matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors
from matplotlib.ticker import PercentFormatter

# Create a random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(19680801)

# %%
# Generate data and plot a simple 直方图
# -----------------------------------------
#
# To generate a 1D 直方图 we only need a single vector of numbers. For a 2D
# 直方图 we'll need a second vector. We'll generate both below, and show
# the 直方图 for each vector.

N_points = 100000
n_bins = 20

# Generate two normal distributions
dist1 = rng.standard_normal(N_points)
dist2 = 0.4 * rng.standard_normal(N_points) + 5

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(dist1, bins=n_bins)
axs[1].hist(dist2, bins=n_bins)

plt.show()


# %%
# Updating 直方图 colors
# -------------------------
#
# The 直方图 method returns (among other things) a ``patches`` object. This
# gives us access to the properties of the objects drawn. Using this, we can
# edit the 直方图 to our liking. Let's change the color of each bar
# based on its y value.

fig, axs = plt.subplots(1, 2, tight_layout=True)

# N is the count in each bin, bins is the lower-limit of the bin
N, bins, patches = axs[0].hist(dist1, bins=n_bins)

# We'll color code by height, but you could use any scalar
fracs = N / N.max()

# we need to normalize the data to 0..1 for the full range of the colormap
norm = colors.Normalize(fracs.min(), fracs.max())

# Now, we'll loop through our objects and set the color of each accordingly
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

# We can also normalize our inputs by the total number of counts
axs[1].hist(dist1, bins=n_bins, density=True)

# Now we format the y-axis to display percentage
axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))


# %%
# Plot a 2D 直方图
# -------------------
#
# To plot a 2D 直方图, one only needs two vectors of the same length,
# corresponding to each axis of the 直方图.

fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(dist1, dist2)


# %%
# Customizing your 直方图
# --------------------------
#
# Customizing a 2D 直方图 is similar to the 1D case, you can control
# visual components such as the bin size or color normalization.

fig, axs = plt.subplots(3, 1, figsize=(5, 15), sharex=True, sharey=True,
                        tight_layout=True)

# We can increase the number of bins on each axis
axs[0].hist2d(dist1, dist2, bins=40)

# As well as define normalization of the colors
axs[1].hist2d(dist1, dist2, bins=40, norm=colors.LogNorm())

# We can also define custom numbers of bins for each axis
axs[2].hist2d(dist1, dist2, bins=(80, 10), norm=colors.LogNorm())

# %%
#
# .. tags::
#
#    plot-type: 直方图,
#    plot-type: 直方图2d
#    domain: statistics
#    styling: color,
#    component: normalization
#    component: patch
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.hist` / `matplotlib.pyplot.hist`
#    - `matplotlib.pyplot.hist2d`
#    - `matplotlib.ticker.PercentFormatter`

```

**histogram_histtypes.py** - 直方图类型

演示 `histtype` 参数的不同选项,包括 `'bar'`(传统柱状)、`'barstacked'`(堆叠柱状)、`'step'`(阶梯线)、`'stepfilled'`(填充阶梯)等样式。通过对比不同类型的效果,帮助选择最适合数据和出版需求的直方图样式。

```python
"""
================================================================
Demo of the 直方图 function's different ``histtype`` settings
================================================================

* 直方图 with step curve that has a color fill.
* 直方图 with step curve with no fill.
* 直方图 with custom and unequal bin widths.
* Two 直方图s with stacked bars.

Selecting different bin counts and sizes can significantly affect the
shape of a 直方图. The Astropy docs have a great section on how to
select these parameters:
http://docs.astropy.org/en/stable/visualization/直方图.html
"""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)

mu_x = 200
sigma_x = 25
x = np.random.normal(mu_x, sigma_x, size=100)

mu_w = 200
sigma_w = 10
w = np.random.normal(mu_w, sigma_w, size=100)

fig, axs = plt.subplots(nrows=2, ncols=2)

axs[0, 0].hist(x, 20, density=True, histtype='stepfilled', facecolor='g',
               alpha=0.75)
axs[0, 0].set_title('stepfilled')

axs[0, 1].hist(x, 20, density=True, histtype='step', facecolor='g',
               alpha=0.75)
axs[0, 1].set_title('step')

axs[1, 0].hist(x, density=True, histtype='barstacked', rwidth=0.8)
axs[1, 0].hist(w, density=True, histtype='barstacked', rwidth=0.8)
axs[1, 0].set_title('barstacked')

# Create a 直方图 by providing the bin edges (unequally spaced).
bins = [100, 150, 180, 195, 205, 220, 250, 300]
axs[1, 1].hist(x, bins, density=True, histtype='bar', rwidth=0.8)
axs[1, 1].set_title('bar, unequal bins')

fig.tight_layout()
plt.show()

# %%
#
# .. tags:: plot-type: 直方图, domain: statistics, purpose: reference
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.hist` / `matplotlib.pyplot.hist`

```

**histogram_cumulative.py** - 累积直方图

使用 `cumulative=True` 参数创建累积分布函数(CDF),展示数据随变量的累积概率。配合 `density=True` 和 `histtype='step'` 可以生成理论CDF的对比图,适合概率分布分析、统计建模和数据质量检查。

```python
"""
========================
Cumulative distributions
========================

This 示例 shows how to plot the empirical cumulative distribution function
(ECDF) of a sample. We also show the theoretical CDF.

In engineering, ECDFs are sometimes called "non-exceedance" curves: the y-value
for a given x-value gives probability that an observation from the sample is
below that x-value. For 示例, the value of 220 on the x-axis corresponds to
about 0.80 on the y-axis, so there is an 80% chance that an observation in the
sample does not exceed 220. Conversely, the empirical *complementary*
cumulative distribution function (the ECCDF, or "exceedance" curve) shows the
probability y that an observation from the sample is above a value x.

A direct method to plot ECDFs is `.Axes.ecdf`.  Passing ``complementary=True``
results in an ECCDF instead.

Alternatively, one can use ``ax.hist(data, density=True, cumulative=True)`` to
first bin the data, as if plotting a 直方图, and then compute and plot the
cumulative sums of the frequencies of entries in each bin.  Here, to plot the
ECCDF, pass ``cumulative=-1``.  Note that this approach results in an
approximation of the E(C)CDF, whereas `.Axes.ecdf` is exact.
"""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)

mu = 200
sigma = 25
n_bins = 25
data = np.random.normal(mu, sigma, size=100)

fig = plt.figure(figsize=(9, 4), layout="constrained")
axs = fig.subplots(1, 2, sharex=True, sharey=True)

# Cumulative distributions.
axs[0].ecdf(data, label="CDF")
n, bins, patches = axs[0].hist(data, n_bins, density=True, histtype="step",
                               cumulative=True, label="Cumulative histogram")
x = np.linspace(data.min(), data.max())
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (x - mu))**2))
y = y.cumsum()
y /= y[-1]
axs[0].plot(x, y, "k--", linewidth=1.5, label="Theory")

# Complementary cumulative distributions.
axs[1].ecdf(data, complementary=True, label="CCDF")
axs[1].hist(data, bins=bins, density=True, histtype="step", cumulative=-1,
            label="Reversed cumulative histogram")
axs[1].plot(x, 1 - y, "k--", linewidth=1.5, label="Theory")

# Label the figure.
fig.suptitle("Cumulative distributions")
for ax in axs:
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("Annual rainfall (mm)")
    ax.set_ylabel("Probability of occurrence")
    ax.label_outer()

plt.show()

# %%
#
# .. tags:: plot-type: ecdf, plot-type: 直方图, domain: statistics
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.hist` / `matplotlib.pyplot.hist`
#    - `matplotlib.axes.Axes.ecdf` / `matplotlib.pyplot.ecdf`

```

**histogram_multihist.py** - 多组直方图

在同一图表中并排展示多个数据组的直方图,通过循环和偏移技术或 `histtype='barstacked'` 实现多组数据的对比。适合比较多组实验数据的分布差异,或展示不同条件下的数据分布特征。

```python
"""
=====================================================
The 直方图 (hist) function with multiple data sets
=====================================================

Plot 直方图 with multiple sample sets and 演示:

* Use of legend with multiple sample sets
* Stacked bars
* Step curve with no fill
* Data sets of different sample sizes

Selecting different bin counts and sizes can significantly affect the
shape of a 直方图. The Astropy docs have a great section on how to
select these parameters:
http://docs.astropy.org/en/stable/visualization/直方图.html

.. redirect-from:: /gallery/lines_bars_and_markers/filled_step

"""
# %%
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)

n_bins = 10
x = np.random.randn(1000, 3)

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)

colors = ['red', 'tan', 'lime']
ax0.hist(x, n_bins, density=True, histtype='bar', color=colors, label=colors)
ax0.legend(prop={'size': 10})
ax0.set_title('bars with legend')

ax1.hist(x, n_bins, density=True, histtype='bar', stacked=True)
ax1.set_title('stacked bar')

ax2.hist(x, n_bins, histtype='step', stacked=True, fill=False)
ax2.set_title('stack step (unfilled)')

# Make a multiple-直方图 of data-sets with different length.
x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
ax3.hist(x_multi, n_bins, histtype='bar')
ax3.set_title('different sample sizes')

fig.tight_layout()
plt.show()

# %%
# -----------------------------------
# Setting properties for each dataset
# -----------------------------------
#
# You can style the 直方图s individually by passing a list of values to the
# following parameters:
#
# * edgecolor
# * facecolor
# * hatch
# * linewidth
# * linestyle
#
#
# edgecolor
# .........

fig, ax = plt.subplots()

edgecolors = ['green', 'red', 'blue']

ax.hist(x, n_bins, fill=False, histtype="step", stacked=True,
        edgecolor=edgecolors, label=edgecolors)
ax.legend()
ax.set_title('Stacked Steps with Edgecolors')

plt.show()

# %%
# facecolor
# .........

fig, ax = plt.subplots()

facecolors = ['green', 'red', 'blue']

ax.hist(x, n_bins, histtype="barstacked", facecolor=facecolors, label=facecolors)
ax.legend()
ax.set_title("Bars with different Facecolors")

plt.show()

# %%
# hatch
# .....

fig, ax = plt.subplots()

hatches = [".", "o", "x"]

ax.hist(x, n_bins, histtype="barstacked", hatch=hatches, label=hatches)
ax.legend()
ax.set_title("Hatches on Stacked Bars")

plt.show()

# %%
# linewidth
# .........

fig, ax = plt.subplots()

linewidths = [1, 2, 3]
edgecolors = ["green", "red", "blue"]

ax.hist(x, n_bins, fill=False, histtype="bar", linewidth=linewidths,
        edgecolor=edgecolors, label=linewidths)
ax.legend()
ax.set_title("Bars with Linewidths")

plt.show()

# %%
# linestyle
# .........

fig, ax = plt.subplots()

linestyles = ['-', ':', '--']

ax.hist(x, n_bins, fill=False, histtype='bar', linestyle=linestyles,
        edgecolor=edgecolors, label=linestyles)
ax.legend()
ax.set_title('Bars with Linestyles')

plt.show()

# %%
#
# .. tags:: plot-type: 直方图, domain: statistics, purpose: reference
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.hist` / `matplotlib.pyplot.hist`

```

**histogram_normalization.py** - 标准化直方图

演示直方图的多种归一化方法,包括 `density=True`(概率密度)、`weights` 参数(自定义权重)等,可以将计数转换为概率密度或其他标准化形式。适合需要比较不同样本量数据或展示概率密度分布的场景。

```python
"""
.. redirect-from:: /gallery/statistics/直方图_features

===================================
直方图 bins, density, and weight
===================================

The `.Axes.hist` method can flexibly create 直方图s in a few different ways,
which is flexible and helpful, but can also lead to confusion.  In particular,
you can:

- bin the data as you want, either with an automatically chosen number of
  bins, or with fixed bin edges,
- normalize the 直方图 so that its integral is one,
- and assign weights to the data points, so that each data point affects the
  count in its bin differently.

The Matplotlib ``hist`` method calls `numpy.直方图` and plots the results,
therefore users should consult the numpy documentation for a definitive guide.

直方图s are created by defining bin edges, and taking a dataset of values
and sorting them into the bins, and counting or summing how much data is in
each bin.  In this simple 示例, 9 numbers between 1 and 4 are sorted into 3
bins:
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(19680801)

xdata = np.array([1.2, 2.3, 3.3, 3.1, 1.7, 3.4, 2.1, 1.25, 1.3])
xbins = np.array([1, 2, 3, 4])

# changing the style of the 直方图 bars just to make it
# very clear where the boundaries of the bins are:
style = {'facecolor': 'none', 'edgecolor': 'C0', 'linewidth': 3}

fig, ax = plt.subplots()
ax.hist(xdata, bins=xbins, **style)

# plot the xdata locations on the x axis:
ax.plot(xdata, 0*xdata, 'd')
ax.set_ylabel('Number per bin')
ax.set_xlabel('x bins (dx=1.0)')

# %%
# Modifying bins
# ==============
#
# Changing the bin size changes the shape of this sparse 直方图, so its a
# good idea to choose bins with some care with respect to your data.  Here we
# make the bins half as wide.

xbins = np.arange(1, 4.5, 0.5)

fig, ax = plt.subplots()
ax.hist(xdata, bins=xbins, **style)
ax.plot(xdata, 0*xdata, 'd')
ax.set_ylabel('Number per bin')
ax.set_xlabel('x bins (dx=0.5)')

# %%
# We can also let numpy (via Matplotlib) choose the bins automatically, or
# specify a number of bins to choose automatically:

fig, ax = plt.subplot_mosaic([['auto', 'n4']],
                             sharex=True, sharey=True, layout='constrained')

ax['auto'].hist(xdata, **style)
ax['auto'].plot(xdata, 0*xdata, 'd')
ax['auto'].set_ylabel('Number per bin')
ax['auto'].set_xlabel('x bins (auto)')

ax['n4'].hist(xdata, bins=4, **style)
ax['n4'].plot(xdata, 0*xdata, 'd')
ax['n4'].set_xlabel('x bins ("bins=4")')

# %%
# Normalizing 直方图s: density and weight
# ==========================================
#
# Counts-per-bin is the default length of each bar in the 直方图.  However,
# we can also normalize the bar lengths as a probability density function using
# the ``density`` parameter:

fig, ax = plt.subplots()
ax.hist(xdata, bins=xbins, density=True, **style)
ax.set_ylabel('Probability density [$V^{-1}$])')
ax.set_xlabel('x bins (dx=0.5 $V$)')

# %%
# This normalization can be a little hard to interpret when just exploring the
# data. The value attached to each bar is divided by the total number of data
# points *and* the width of the bin, and thus the values _integrate_ to one
# when integrating across the full range of data.
# e.g. ::
#
#     density = counts / (sum(counts) * np.diff(bins))
#     np.sum(density * np.diff(bins)) == 1
#
# This normalization is how `probability density functions
# <https://en.wikipedia.org/wiki/Probability_density_function>`_ are defined in
# statistics.  If :math:`X` is a random variable on :math:`x`, then :math:`f_X`
# is is the probability density function if :math:`P[a<X<b] = \int_a^b f_X dx`.
# If the units of x are Volts, then the units of :math:`f_X` are :math:`V^{-1}`
# or probability per change in voltage.
#
# The usefulness of this normalization is a little more clear when we draw from
# a known distribution and try to compare with theory.  So, choose 1000 points
# from a `normal distribution
# <https://en.wikipedia.org/wiki/Normal_distribution>`_, and also calculate the
# known probability density function:

xdata = rng.normal(size=1000)
xpdf = np.arange(-4, 4, 0.1)
pdf = 1 / (np.sqrt(2 * np.pi)) * np.exp(-xpdf**2 / 2)

# %%
# If we don't use ``density=True``, we need to scale the expected probability
# distribution function by both the length of the data and the width of the
# bins:

fig, ax = plt.subplot_mosaic([['False', 'True']], layout='constrained')
dx = 0.1
xbins = np.arange(-4, 4, dx)
ax['False'].hist(xdata, bins=xbins, density=False, histtype='step', label='Counts')

# scale and plot the expected pdf:
ax['False'].plot(xpdf, pdf * len(xdata) * dx, label=r'$N\,f_X(x)\,\delta x$')
ax['False'].set_ylabel('Count per bin')
ax['False'].set_xlabel('x bins [V]')
ax['False'].legend()

ax['True'].hist(xdata, bins=xbins, density=True, histtype='step', label='density')
ax['True'].plot(xpdf, pdf, label='$f_X(x)$')
ax['True'].set_ylabel('Probability density [$V^{-1}$]')
ax['True'].set_xlabel('x bins [$V$]')
ax['True'].legend()

# %%
# One advantage of using the density is therefore that the shape and amplitude
# of the 直方图 does not depend on the size of the bins.  Consider an
# extreme case where the bins do not have the same width.  In this 示例, the
# bins below ``x=-1.25`` are six times wider than the rest of the bins.   By
# normalizing by density, we preserve the shape of the distribution, whereas if
# we do not, then the wider bins have much higher counts than the thinner bins:

fig, ax = plt.subplot_mosaic([['False', 'True']], layout='constrained')
dx = 0.1
xbins = np.hstack([np.arange(-4, -1.25, 6*dx), np.arange(-1.25, 4, dx)])
ax['False'].hist(xdata, bins=xbins, density=False, histtype='step', label='Counts')
ax['False'].plot(xpdf, pdf * len(xdata) * dx, label=r'$N\,f_X(x)\,\delta x_0$')
ax['False'].set_ylabel('Count per bin')
ax['False'].set_xlabel('x bins [V]')
ax['False'].legend()

ax['True'].hist(xdata, bins=xbins, density=True, histtype='step', label='density')
ax['True'].plot(xpdf, pdf, label='$f_X(x)$')
ax['True'].set_ylabel('Probability density [$V^{-1}$]')
ax['True'].set_xlabel('x bins [$V$]')
ax['True'].legend()

# %%
# Similarly, if we want to compare 直方图s with different bin widths, we may
# want to use ``density=True``:

fig, ax = plt.subplot_mosaic([['False', 'True']], layout='constrained')

# expected PDF
ax['True'].plot(xpdf, pdf, '--', label='$f_X(x)$', color='k')

for nn, dx in enumerate([0.1, 0.4, 1.2]):
    xbins = np.arange(-4, 4, dx)
    # expected 直方图:
    ax['False'].plot(xpdf, pdf*1000*dx, '--', color=f'C{nn}')
    ax['False'].hist(xdata, bins=xbins, density=False, histtype='step')

    ax['True'].hist(xdata, bins=xbins, density=True, histtype='step', label=dx)

# Labels:
ax['False'].set_xlabel('x bins [$V$]')
ax['False'].set_ylabel('Count per bin')
ax['True'].set_ylabel('Probability density [$V^{-1}$]')
ax['True'].set_xlabel('x bins [$V$]')
ax['True'].legend(fontsize='small', title='bin width:')

# %%
# Sometimes people want to normalize so that the sum of counts is one.  This is
# analogous to a `probability mass function
# <https://en.wikipedia.org/wiki/Probability_mass_function>`_ for a discrete
# variable where the sum of probabilities for all the values equals one.  Using
# ``hist``, we can get this normalization if we set the *weights* to 1/N.
# Note that the amplitude of this normalized 直方图 still depends on
# width and/or number of the bins:

fig, ax = plt.subplots(layout='constrained', figsize=(3.5, 3))

for nn, dx in enumerate([0.1, 0.4, 1.2]):
    xbins = np.arange(-4, 4, dx)
    ax.hist(xdata, bins=xbins, weights=1/len(xdata) * np.ones(len(xdata)),
                   histtype='step', label=f'{dx}')
ax.set_xlabel('x bins [$V$]')
ax.set_ylabel('Bin count / N')
ax.legend(fontsize='small', title='bin width:')

# %%
# The value of normalizing 直方图s is comparing two distributions that have
# different sized populations.  Here we compare the distribution of ``xdata``
# with a population of 1000, and ``xdata2`` with 100 members.

xdata2 = rng.normal(size=100)

fig, ax = plt.subplot_mosaic([['no_norm', 'density', 'weight']],
                             layout='constrained', figsize=(8, 4))

xbins = np.arange(-4, 4, 0.25)

ax['no_norm'].hist(xdata, bins=xbins, histtype='step')
ax['no_norm'].hist(xdata2, bins=xbins, histtype='step')
ax['no_norm'].set_ylabel('Counts')
ax['no_norm'].set_xlabel('x bins [$V$]')
ax['no_norm'].set_title('No normalization')

ax['density'].hist(xdata, bins=xbins, histtype='step', density=True)
ax['density'].hist(xdata2, bins=xbins, histtype='step', density=True)
ax['density'].set_ylabel('Probability density [$V^{-1}$]')
ax['density'].set_title('Density=True')
ax['density'].set_xlabel('x bins [$V$]')

ax['weight'].hist(xdata, bins=xbins, histtype='step',
                  weights=1 / len(xdata) * np.ones(len(xdata)),
                  label='N=1000')
ax['weight'].hist(xdata2, bins=xbins, histtype='step',
                  weights=1 / len(xdata2) * np.ones(len(xdata2)),
                  label='N=100')
ax['weight'].set_xlabel('x bins [$V$]')
ax['weight'].set_ylabel('Counts / N')
ax['weight'].legend(fontsize='small')
ax['weight'].set_title('Weight = 1/N')

plt.show()

# %%
#
# .. tags:: plot-type: 直方图, domain: statistics
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.hist` / `matplotlib.pyplot.hist`
#    - `matplotlib.axes.Axes.set_title`
#    - `matplotlib.axes.Axes.set_xlabel`
#    - `matplotlib.axes.Axes.set_ylabel`
#    - `matplotlib.axes.Axes.legend`

```

**histogram_bihistogram.py** - 双向直方图

通过 `weights=-np.ones_like()` 创建向下的直方图,与向上直方图对比展示两个数据集的分布差异。使用 `ax.axhline(0)` 添加零线作为基准,适合可视化前后对比、实验对照组差异分析等场景。

```python
"""
===========
Bi直方图
===========

How to plot a bi直方图 with Matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np

# Create a random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(19680801)

# %%
# Generate data and plot a bi直方图
# ------------------------------------
#
# To generate a bi直方图 we need two datasets (each being a vector of numbers).
# We will plot both 直方图s using plt.hist() and set the weights of the second
# one to be negative. We'll generate data below and plot the bi直方图.

N_points = 10_000

# Generate two normal distributions
dataset1 = np.random.normal(0, 1, size=N_points)
dataset2 = np.random.normal(1, 2, size=N_points)

# Use a constant bin width to make the two 直方图s easier to compare visually
bin_width = 0.25
bins = np.arange(np.min([dataset1, dataset2]),
                    np.max([dataset1, dataset2]) + bin_width, bin_width)

fig, ax = plt.subplots()

# Plot the first 直方图
ax.hist(dataset1, bins=bins, label="Dataset 1")

# Plot the second 直方图
# (notice the negative weights, which flip the 直方图 upside down)
ax.hist(dataset2, weights=-np.ones_like(dataset2), bins=bins, label="Dataset 2")
ax.axhline(0, color="k")
ax.legend()

plt.show()

# %%
#
# .. tags:: plot-type: 直方图, domain: statistics, purpose: showcase

```

**scatter_hist.py** - 散点+直方图组合

使用 `subplot_mosaic` 或 `GridSpec` 创建复合布局,主区域显示散点图展示变量关系,顶部和右侧添加边际直方图展示各变量的分布。这种布局在多变量数据分析和相关性研究中非常有用,能够同时提供关系和分布两个维度的信息。

```python
"""
============================
散点图 with 直方图s
============================

Add 直方图s to the x-axes and y-axes margins of a 散点图.

This layout features a central 散点图 illustrating the relationship
between x and y, a 直方图 at the top displaying the distribution of x, and a
直方图 on the right showing the distribution of y.

For a nice alignment of the main Axes with the marginals, two options are shown
below:

.. contents::
   :local:

While `.Axes.inset_axes` may be a bit more complex, it allows correct handling
of main Axes with a fixed aspect ratio.

Let us first define a function that takes x and y data as input, as well as
three Axes, the main Axes for the scatter, and two marginal Axes. It will then
create the scatter and 直方图s inside the provided Axes.
"""

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

# some random data
x = np.random.randn(1000)
y = np.random.randn(1000)


def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the 散点图:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')


# %%
# Defining the Axes positions using subplot_mosaic
# ------------------------------------------------
#
# We use the `~.pyplot.subplot_mosaic` function to define the positions and
# names of the three axes; the empty axes is specified by ``'.'``.  We manually
# specify the size of the figure, and can make the different axes have
# different sizes by specifying the *width_ratios* and *height_ratios*
# arguments. The *layout* argument is set to ``'constrained'`` to optimize the
# spacing between the axes.

fig, axs = plt.subplot_mosaic([['histx', '.'],
                               ['scatter', 'histy']],
                              figsize=(6, 6),
                              width_ratios=(4, 1), height_ratios=(1, 4),
                              layout='constrained')
scatter_hist(x, y, axs['scatter'], axs['histx'], axs['histy'])


# %%
#
# Defining the Axes positions using inset_axes
# --------------------------------------------
#
# `~.Axes.inset_axes` can be used to position marginals *outside* the main
# Axes.  The advantage of doing so is that the aspect ratio of the main Axes
# can be fixed, and the marginals will always be drawn relative to the position
# of the Axes.

# Create a Figure, which doesn't have to be square.
fig = plt.figure(layout='constrained')
# Create the main Axes.
ax = fig.add_subplot()
# The main Axes' aspect can be fixed.
ax.set_aspect('equal')
# Create marginal Axes, which have 25% of the size of the main Axes.  Note that
# the inset Axes are positioned *outside* (on the right and the top) of the
# main Axes, by specifying axes coordinates greater than 1.  Axes coordinates
# less than 0 would likewise specify positions on the left and the bottom of
# the main Axes.
ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
# Draw the 散点图 and marginals.
scatter_hist(x, y, ax, ax_histx, ax_histy)

plt.show()


# %%
#
# While we recommend using one of the two methods described above, there are
# number of other ways to achieve a similar layout:
#
# - The Axes can be positioned manually in relative coordinates using
#   `~matplotlib.figure.Figure.add_axes`.
# - A gridspec can be used to create the layout
#   (`~matplotlib.figure.Figure.add_gridspec`) and adding only the three desired
#   axes (`~matplotlib.figure.Figure.add_subplot`).
# - Four subplots can be created  using `~.pyplot.subplots`,  and the unused
#   axes in the upper right can be removed manually.
# - The ``axes_grid1`` toolkit can be used, as shown in
#   :doc:`/gallery/axes_grid1/scatter_hist_locatable_axes`.
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.figure.Figure.subplot_mosaic`
#    - `matplotlib.pyplot.subplot_mosaic`
#    - `matplotlib.figure.Figure.add_subplot`
#    - `matplotlib.axes.Axes.inset_axes`
#    - `matplotlib.axes.Axes.scatter`
#    - `matplotlib.axes.Axes.hist`
#
# .. tags::
#
#    component: axes
#    plot-type: scatter
#    plot-type: 直方图
#    level: intermediate

```


### 4. Heatmaps & Contours `codebase/draw/gallery_python_new/heatmaps_contours` (热图等高线 - 4个)

**contour_label_demo.py** - 等高线标注

使用 `ax.contour()` 和 `ax.clabel()` 创建带标签的等高线图,通过 `levels`(等高线数量或数值)和 `cmap`(颜色映射)参数控制样式。`clabel()` 函数可以在等高线上添加数值标签,通过 `inline=True`(标签在线上)和 `fontsize`(字体大小)等参数优化标签显示,适合地形图、气象图等科学可视化场景。

```python
"""
==================
Contour Label Demo
==================

Illustrate some of the more advanced things that one can do with
contour labels.

See also the :doc:`contour demo 示例
</gallery/images_contours_and_fields/contour_demo>`.
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.ticker as ticker

# %%
# Define our surface

delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

# %%
# Make contour labels with custom level formatters


# This custom formatter removes trailing zeros, e.g. "1.0" becomes "1", and
# then adds a percent sign.
def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"


# Basic contour plot
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)

ax.clabel(CS, CS.levels, fmt=fmt, fontsize=10)

# %%
# Label contours with arbitrary strings using a dictionary

fig1, ax1 = plt.subplots()

# Basic contour plot
CS1 = ax1.contour(X, Y, Z)

fmt = {}
strs = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh']
for l, s in zip(CS1.levels, strs):
    fmt[l] = s

# Label every other level using strings
ax1.clabel(CS1, CS1.levels[::2], fmt=fmt, fontsize=10)

# %%
# Use a Formatter

fig2, ax2 = plt.subplots()

CS2 = ax2.contour(X, Y, 100**Z, locator=plt.LogLocator())
fmt = ticker.LogFormatterMathtext()
fmt.create_dummy_axis()
ax2.clabel(CS2, CS2.levels, fmt=fmt)
ax2.set_title("$100^Z$")

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.contour` / `matplotlib.pyplot.contour`
#    - `matplotlib.axes.Axes.clabel` / `matplotlib.pyplot.clabel`
#    - `matplotlib.ticker.LogFormatterMathtext`
#    - `matplotlib.ticker.TickHelper.create_dummy_axis`

```

**contourf_demo.py** - 填充等高线

使用 `ax.contourf()` 创建填充颜色的等高线图,通过 `levels` 参数控制等高线密度,使用 `cmap` 设置连续的颜色映射。结合 `colorbar()` 可以添加颜色条来标识数值范围,适合可视化温度场、压力场、海拔分布等二维标量场数据。

```python
"""
=============
Contourf demo
=============

How to use the `.axes.Axes.contourf` method to create filled contour plots.
"""
import matplotlib.pyplot as plt
import numpy as np

delta = 0.025

x = y = np.arange(-3.0, 3.01, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

nr, nc = Z.shape

# put NaNs in one corner:
Z[-nr // 6:, -nc // 6:] = np.nan
# contourf will convert these to masked


Z = np.ma.array(Z)
# mask another corner:
Z[:nr // 6, :nc // 6] = np.ma.masked

# mask a circle in the middle:
interior = np.sqrt(X**2 + Y**2) < 0.5
Z[interior] = np.ma.masked

# %%
# Automatic contour levels
# ------------------------
# We are using automatic selection of contour levels; this is usually not such
# a good idea, because they don't occur on nice boundaries, but we do it here
# for purposes of illustration.

fig1, ax2 = plt.subplots(layout='constrained')
CS = ax2.contourf(X, Y, Z, 10, cmap=plt.cm.bone)

# Note that in the following, we explicitly pass in a subset of the contour
# levels used for the filled contours.  Alternatively, we could pass in
# additional levels to provide extra resolution, or leave out the *levels*
# keyword argument to use all of the original levels.

CS2 = ax2.contour(CS, levels=CS.levels[::2], colors='r')

ax2.set_title('Nonsense (3 masked regions)')
ax2.set_xlabel('word length anomaly')
ax2.set_ylabel('sentence length anomaly')

# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig1.colorbar(CS)
cbar.ax.set_ylabel('verbosity coefficient')
# Add the contour line levels to the colorbar
cbar.add_lines(CS2)

# %%
# Explicit contour levels
# -----------------------
# Now make a contour plot with the levels specified, and with the colormap
# generated automatically from a list of colors.

fig2, ax2 = plt.subplots(layout='constrained')
levels = [-1.5, -1, -0.5, 0, 0.5, 1]
CS3 = ax2.contourf(X, Y, Z, levels, colors=('r', 'g', 'b'), extend='both')
# Our data range extends outside the range of levels; make
# data below the lowest contour level yellow, and above the
# highest level cyan:
CS3.cmap.set_under('yellow')
CS3.cmap.set_over('cyan')

CS4 = ax2.contour(X, Y, Z, levels, colors=('k',), linewidths=(3,))
ax2.set_title('Listed colors (3 masked regions)')
ax2.clabel(CS4, fmt='%2.1f', colors='w', fontsize=14)

# Notice that the colorbar gets all the information it
# needs from the ContourSet object, CS3.
fig2.colorbar(CS3)

# %%
# Extension settings
# ------------------
# Illustrate all 4 possible "extend" settings:
extends = ["neither", "both", "min", "max"]
cmap = plt.colormaps["winter"].with_extremes(under="magenta", over="yellow")
# Note: contouring simply excludes masked or nan regions, so
# instead of using the "bad" colormap value for them, it draws
# nothing at all in them.  Therefore, the following would have
# no effect:
# cmap.set_bad("red")

fig, axs = plt.subplots(2, 2, layout="constrained")

for ax, extend in zip(axs.flat, extends):
    cs = ax.contourf(X, Y, Z, levels, cmap=cmap, extend=extend)
    fig.colorbar(cs, ax=ax, shrink=0.9)
    ax.set_title("extend = %s" % extend)
    ax.locator_params(nbins=4)

plt.show()

# %%
# Orient contour plots using the origin keyword
# ---------------------------------------------
# This code 演示s orienting contour plot data using the "origin" keyword

x = np.arange(1, 10)
y = x.reshape(-1, 1)
h = x * y

fig, (ax1, ax2) = plt.subplots(ncols=2)

ax1.set_title("origin='upper'")
ax2.set_title("origin='lower'")
ax1.contourf(h, levels=np.arange(5, 70, 5), extend='both', origin="upper")
ax2.contourf(h, levels=np.arange(5, 70, 5), extend='both', origin="lower")

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.contour` / `matplotlib.pyplot.contour`
#    - `matplotlib.axes.Axes.contourf` / `matplotlib.pyplot.contourf`
#    - `matplotlib.axes.Axes.clabel` / `matplotlib.pyplot.clabel`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
#    - `matplotlib.colors.Colormap`
#    - `matplotlib.colors.Colormap.set_bad`
#    - `matplotlib.colors.Colormap.set_under`
#    - `matplotlib.colors.Colormap.set_over`

```

**pcolormesh_levels.py** - 伪彩色网格

使用 `ax.pcolormesh()` 绘制非均匀网格或规则网格的伪彩色图,通过 `shading` 参数(`'flat'`、`'nearest'`、`'auto'`、`'gouraud'`)控制渲染方式。结合 `BoundaryNorm` 和 `MaxNLocator` 可以精确控制颜色级别,适合大规模网格数据、地理信息系统(GIS)和科学计算结果的可视化。

```python
"""
==========
pcolormesh
==========

`.axes.Axes.pcolormesh` allows you to generate 2D image-style plots.
Note that it is faster than the similar `~.axes.Axes.pcolor`.

"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

# %%
# Basic pcolormesh
# ----------------
#
# We usually specify a pcolormesh by defining the edge of quadrilaterals and
# the value of the quadrilateral.  Note that here *x* and *y* each have one
# extra element than Z in the respective dimension.

np.random.seed(19680801)
Z = np.random.rand(6, 10)
x = np.arange(-0.5, 10, 1)  # len = 11
y = np.arange(4.5, 11, 1)  # len = 7

fig, ax = plt.subplots()
ax.pcolormesh(x, y, Z)

# %%
# Non-rectilinear pcolormesh
# --------------------------
#
# Note that we can also specify matrices for *X* and *Y* and have
# non-rectilinear quadrilaterals.

x = np.arange(-0.5, 10, 1)  # len = 11
y = np.arange(4.5, 11, 1)  # len = 7
X, Y = np.meshgrid(x, y)
X = X + 0.2 * Y  # tilt the coordinates.
Y = Y + 0.3 * X

fig, ax = plt.subplots()
ax.pcolormesh(X, Y, Z)

# %%
# Centered Coordinates
# ---------------------
#
# Often a user wants to pass *X* and *Y* with the same sizes as *Z* to
# `.axes.Axes.pcolormesh`. This is also allowed if ``shading='auto'`` is
# passed (default set by :rc:`pcolor.shading`). Pre Matplotlib 3.3,
# ``shading='flat'`` would drop the last column and row of *Z*, but now gives
# an error. If this is really what you want, then simply drop the last row and
# column of Z manually:

x = np.arange(10)  # len = 10
y = np.arange(6)  # len = 6
X, Y = np.meshgrid(x, y)

fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
axs[0].pcolormesh(X, Y, Z, vmin=np.min(Z), vmax=np.max(Z), shading='auto')
axs[0].set_title("shading='auto' = 'nearest'")
axs[1].pcolormesh(X, Y, Z[:-1, :-1], vmin=np.min(Z), vmax=np.max(Z),
                  shading='flat')
axs[1].set_title("shading='flat'")

# %%
# Making levels using Norms
# -------------------------
#
# Shows how to combine Normalization and Colormap instances to draw
# "levels" in `.axes.Axes.pcolor`, `.axes.Axes.pcolormesh`
# and `.axes.Axes.imshow` type plots in a similar
# way to the levels keyword argument to contour/contourf.

# make these smaller to increase the resolution
dx, dy = 0.05, 0.05

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[slice(1, 5 + dy, dy),
                slice(1, 5 + dx, dx)]

z = np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)

# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())


# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = plt.colormaps['PiYG']
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

fig, (ax0, ax1) = plt.subplots(nrows=2)

im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
cbar0 = fig.colorbar(im, ax=ax0)
cbar0.ax.tick_params(labelsize=8)  # Smaller font for colorbar labels
ax0.set_title('pcolormesh with levels')


# contours are *point* based plots, so convert our bound into point
# centers
cf = ax1.contourf(x[:-1, :-1] + dx/2.,
                  y[:-1, :-1] + dy/2., z, levels=levels,
                  cmap=cmap)
cbar1 = fig.colorbar(cf, ax=ax1)
cbar1.ax.tick_params(labelsize=8)  # Smaller font for colorbar labels
ax1.set_title('contourf with levels')

# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap
fig.tight_layout()

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.pcolormesh` / `matplotlib.pyplot.pcolormesh`
#    - `matplotlib.axes.Axes.contourf` / `matplotlib.pyplot.contourf`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
#    - `matplotlib.colors.BoundaryNorm`
#    - `matplotlib.ticker.MaxNLocator`

```

**image_annotated_heatmap.py** - 标注热图

使用 `ax.imshow()` 创建热图,并通过嵌套循环和 `ax.text()` 在每个格子中添加数值标注。通过调整文本颜色(根据背景亮度自动选择黑/白)来确保可读性,适合展示相关性矩阵、混淆矩阵、距离矩阵等需要同时显示颜色编码和数值的场景。

```python
"""
=================
Annotated 热图
=================

It is often desirable to show data which depends on two independent
variables as a color coded image plot. This is often referred to as a
热图. If the data is categorical, this would be called a categorical
热图.

Matplotlib's `~matplotlib.axes.Axes.imshow` function makes
production of such plots particularly easy.

The following 示例s show how to create a 热图 with annotations.
We will start with an easy 示例 and expand it to be usable as a
universal function.
"""


# %%
#
# A simple categorical 热图
# ----------------------------
#
# We may start by defining some data. What we need is a 2D list or array
# which defines the data to color code. We then also need two lists or arrays
# of categories; of course the number of elements in those lists
# need to match the data along the respective axes.
# The 热图 itself is an `~matplotlib.axes.Axes.imshow` plot
# with the labels set to the categories we have.
# Note that it is important to set both, the tick locations
# (`~matplotlib.axes.Axes.set_xticks`) as well as the
# tick labels (`~matplotlib.axes.Axes.set_xticklabels`),
# otherwise they would become out of sync. The locations are just
# the ascending integer numbers, while the ticklabels are the labels to show.
# Finally, we can label the data itself by creating a `~matplotlib.text.Text`
# within each cell showing the value of that cell.


import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib as mpl

# sphinx_gallery_thumbnail_number = 2

vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])


fig, ax = plt.subplots()
im = ax.imshow(harvest)

# Show all ticks and label them with the respective list entries
ax.set_xticks(range(len(farmers)), labels=farmers,
              rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(vegetables)), labels=vegetables)

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
plt.show()


# %%
# Using the helper function code style
# ------------------------------------
#
# As discussed in the :ref:`Coding styles <coding_styles>`
# one might want to reuse such code to create some kind of 热图
# for different input data and/or on different axes.
# We create a function that takes the data and the row and column labels as
# input, and allows arguments that are used to customize the plot
#
# Here, in addition to the above we also want to create a colorbar and
# position the labels above of the 热图 instead of below it.
# The annotations shall get different colors depending on a threshold
# for better contrast against the pixel color.
# Finally, we turn the surrounding axes spines off and create
# a grid of white lines to separate the cells.


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a 热图 from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the 热图 is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the 热图
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a 热图.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the 热图.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


# %%
# The above now allows us to keep the actual plot creation pretty compact.
#

fig, ax = plt.subplots()

im, cbar = heatmap(harvest, vegetables, farmers, ax=ax,
                   cmap="YlGn", cbarlabel="harvest [t/year]")
texts = annotate_heatmap(im, valfmt="{x:.1f} t")

fig.tight_layout()
plt.show()


# %%
# Some more complex 热图 示例s
# ----------------------------------
#
# In the following we show the versatility of the previously created
# functions by applying it in different cases and using different arguments.
#

np.random.seed(19680801)

fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

# Replicate the above 示例 with a different font size and colormap.

im, _ = heatmap(harvest, vegetables, farmers, ax=ax,
                cmap="Wistia", cbarlabel="harvest [t/year]")
annotate_heatmap(im, valfmt="{x:.1f}", size=7)

# Create some new data, give further arguments to imshow (vmin),
# use an integer format on the annotations and provide some colors.

data = np.random.randint(2, 100, size=(7, 7))
y = [f"Book {i}" for i in range(1, 8)]
x = [f"Store {i}" for i in list("ABCDEFG")]
im, _ = heatmap(data, y, x, ax=ax2, vmin=0,
                cmap="magma_r", cbarlabel="weekly sold copies")
annotate_heatmap(im, valfmt="{x:d}", size=7, threshold=20,
                 textcolors=("red", "white"))

# Sometimes even the data itself is categorical. Here we use a
# `matplotlib.colors.BoundaryNorm` to get the data into classes
# and use this to colorize the plot, but also to obtain the class
# labels from an array of classes.

data = np.random.randn(6, 6)
y = [f"Prod. {i}" for i in range(10, 70, 10)]
x = [f"Cycle {i}" for i in range(1, 7)]

qrates = list("ABCDEFG")
norm = matplotlib.colors.BoundaryNorm(np.linspace(-3.5, 3.5, 8), 7)
fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: qrates[::-1][norm(x)])

im, _ = heatmap(data, y, x, ax=ax3,
                cmap=mpl.colormaps["PiYG"].resampled(7), norm=norm,
                cbar_kw=dict(ticks=np.arange(-3, 4), format=fmt),
                cbarlabel="Quality Rating")

annotate_heatmap(im, valfmt=fmt, size=9, fontweight="bold", threshold=-1,
                 textcolors=("red", "black"))

# We can nicely plot a correlation matrix. Since this is bound by -1 and 1,
# we use those as vmin and vmax. We may also remove leading zeros and hide
# the diagonal elements (which are all 1) by using a
# `matplotlib.ticker.FuncFormatter`.

corr_matrix = np.corrcoef(harvest)
im, _ = heatmap(corr_matrix, vegetables, vegetables, ax=ax4,
                cmap="PuOr", vmin=-1, vmax=1,
                cbarlabel="correlation coeff.")


def func(x, pos):
    return f"{x:.2f}".replace("0.", ".").replace("1.00", "")

annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)


plt.tight_layout()
plt.show()


# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.imshow` / `matplotlib.pyplot.imshow`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`

```


### 5. 3D Plots `codebase/draw/gallery_python_new/3d_plots` (三维图表 - 5个)

**scatter3d.py** - 3D散点图

使用 `projection='3d'` 创建三维坐标轴,通过 `ax.scatter(x, y, z)` 绘制三维散点图。可以通过 `c`(颜色映射)、`s`(标记大小)、`cmap`(颜色映射)和 `alpha`(透明度)等参数展示数据的第四维度,适合三维数据分布、空间聚类分析等场景。

```python
"""
==============
3D scatterplot
==============

Demonstration of a basic scatterplot in 3D.
"""

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


def randrange(n, vmin, vmax):
    """
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    """
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# Using only circles ('o') for cleaner appearance
for m, zlow, zhigh in [('o', -50, -25), ('o', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
# Removed Z label for cleaner look

plt.show()

# %%
# .. tags::
#    plot-type: 3D, plot-type: scatter,
#    level: beginner

```

**surface3d.py** - 3D曲面图

使用 `ax.plot_surface(X, Y, Z)` 创建三维表面图,需要通过 `np.meshgrid()` 生成网格数据。通过 `cmap`(颜色映射)、`linewidth`(网格线宽)、`antialiased`(抗锯齿)、`alpha`(透明度)等参数控制表面效果,配合 `colorbar()` 显示数值-颜色对应关系,适合数学函数可视化、科学计算结果展示。

```python
"""
=====================
3D surface (colormap)
=====================

演示s plotting a 3D surface colored with the coolwarm colormap.
The surface is made opaque by using ``antialiased=False``.

Also 演示s using the `.LinearLocator` and custom formatting for the
z axis tick labels.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
cbar = fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.15)
cbar.ax.tick_params(labelsize=7)

plt.show()


# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.pyplot.subplots`
#    - `matplotlib.axis.Axis.set_major_formatter`
#    - `matplotlib.axis.Axis.set_major_locator`
#    - `matplotlib.ticker.LinearLocator`
#    - `matplotlib.ticker.StrMethodFormatter`
#
# .. tags::
#    plot-type: 3D,
#    styling: colormap,
#    level: advanced

```

**wire3d.py** - 3D线框图

使用 `ax.plot_wireframe(X, Y, Z)` 创建三维线框图,通过 `linewidth`(线宽)和 `color`(颜色)参数控制网格线样式。线框图相比表面图更能看到背后的结构,适合几何形状展示、数学函数拓扑结构可视化等场景。

```python
"""
=================
3D wireframe plot
=================

A very basic demonstration of a wireframe plot.
"""

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Grab some test data.
X, Y, Z = axes3d.get_test_data(0.05)

# Plot a basic wireframe.
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    level: beginner

```

**lines3d.py** - 3D折线图

使用 `ax.plot(x, y, z)` 在三维空间中绘制折线,可以展示轨迹、路径或三维时间序列。通过 `linewidth`(线宽)、`linestyle`(线型)、`marker`(标记点)和 `color`(颜色)等参数自定义样式,适合粒子运动轨迹、飞行路径、三维曲线展示等场景。

```python
"""
================
Parametric curve
================

This 示例 演示s plotting a parametric curve in 3D.
"""

import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection='3d')

# Prepare arrays x, y, z
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    level: beginner

```

**contour3d.py** - 3D等高线

使用 `ax.contour(X, Y, Z, zdir='z')` 在三维空间的指定平面上绘制等高线,通过 `offset` 参数控制等高线平面的位置。可以在XY、XZ、YZ平面上同时显示等高线,结合表面图或线框图使用,能够更清晰地理解三维数据场的结构。

```python
"""
=================================
Plot contour (level) curves in 3D
=================================

This is like a contour plot in 2D except that the ``f(x, y)=c`` curve is
plotted on the plane ``z=c``.
"""

import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

ax = plt.figure().add_subplot(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)

ax.contour(X, Y, Z, cmap=cm.coolwarm)  # Plot contour curves

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    level: beginner

```


### 6. Polar Plots `codebase/draw/gallery_python_new/polar_plots` (极坐标图 - 5个)

**polar_demo.py** - 极坐标基础

使用 `subplot_kw={'projection': 'polar'}` 创建极坐标轴,通过 `ax.plot(theta, r)` 绘制极坐标折线图。可以通过 `set_theta_zero_location`(设置0度位置)、`set_theta_direction`(角度方向)、`set_rmax`(最大半径)和 `fill`(填充区域)等参数自定义极坐标图,适合方向性数据、周期性数据和雷达图等场景。

```python
"""
==========
Polar plot
==========

Demo of a 折线图 on a polar axis.

The second plot shows the same data, but with the radial axis starting at r=1
and the angular axis starting at 0 degrees and ending at 225 degrees. Setting
the origin of the radial axis to 0 allows the radial ticks to be placed at the
same location as the first plot.
"""
import matplotlib.pyplot as plt
import numpy as np

r = np.arange(0, 2, 0.01)
theta = 2 * np.pi * r

fig, axs = plt.subplots(2, 1, figsize=(5, 8), subplot_kw={'projection': 'polar'},
                        layout='constrained')
ax = axs[0]
ax.plot(theta, r)
ax.set_rmax(2)
ax.set_rticks([0.5, 1, 1.5, 2])  # Fewer radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)

ax.set_title("A line plot on a polar axis", va='bottom')

ax = axs[1]
ax.plot(theta, r)
ax.set_rmax(2)
ax.set_rmin(1)  # Change the radial axis to only go from 1 to 2
ax.set_rorigin(0)  # Set the origin of the radial axis to 0
ax.set_thetamin(0)
ax.set_thetamax(225)
ax.set_rticks([1, 1.5, 2])  # Fewer radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line

ax.grid(True)
ax.set_title("Same plot, but with reduced axis limits", va='bottom')
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.plot` / `matplotlib.pyplot.plot`
#    - `matplotlib.projections.polar`
#    - `matplotlib.projections.polar.PolarAxes`
#    - `matplotlib.projections.polar.PolarAxes.set_rticks`
#    - `matplotlib.projections.polar.PolarAxes.set_rmin`
#    - `matplotlib.projections.polar.PolarAxes.set_rorigin`
#    - `matplotlib.projections.polar.PolarAxes.set_rmax`
#    - `matplotlib.projections.polar.PolarAxes.set_rlabel_position`
#
# .. tags::
#
#    plot-type: polar
#    level: beginner

```

**polar_scatter.py** - 极坐标散点图

在极坐标系中使用 `ax.scatter(theta, r)` 绘制散点图,通过 `c`(颜色)、`s`(标记大小)、`cmap`(颜色映射)和 `alpha`(透明度)等参数展示多维度数据。适合风向数据分布、极坐标聚类、周期性模式识别等场景。

```python
"""
==========================
散点图 on polar axis
==========================

Size increases radially in this 示例 and color increases with angle
(just to verify the symbols are being scattered correctly).
"""
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

# Compute areas and colors
N = 150
r = 2 * np.random.rand(N)
theta = 2 * np.pi * np.random.rand(N)
area = 200 * r**2
colors = theta

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)

# %%
# 散点图 on polar axis, with offset origin
# ----------------------------------------------
#
# The main difference with the previous plot is the configuration of the origin
# radius, producing an annulus. Additionally, the theta zero location is set to
# rotate the plot.

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)

ax.set_rorigin(-2.5)
ax.set_theta_zero_location('W', offset=10)

# %%
# 散点图 on polar axis confined to a sector
# -----------------------------------------------
#
# The main difference with the previous plots is the configuration of the
# theta start and end limits, producing a sector instead of a full circle.

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)

ax.set_thetamin(45)
ax.set_thetamax(135)

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.scatter` / `matplotlib.pyplot.scatter`
#    - `matplotlib.projections.polar`
#    - `matplotlib.projections.polar.PolarAxes.set_rorigin`
#    - `matplotlib.projections.polar.PolarAxes.set_theta_zero_location`
#    - `matplotlib.projections.polar.PolarAxes.set_thetamin`
#    - `matplotlib.projections.polar.PolarAxes.set_thetamax`
#
# .. tags::
#
#    plot-style: polar
#    plot-style: scatter
#    level: beginner

```

**polar_bar.py** - 极坐标柱状图

在极坐标系中使用 `ax.bar(theta, values)` 绘制柱状图,通过 `width`(柱宽)和 `bottom`(起始位置)参数控制柱子样式。可以创建类似南丁格尔玫瑰图的视觉效果,适合展示方向性强度、周期性统计量等数据。

```python
"""
=======================
柱状图 on polar axis
=======================

Demo of bar plot on a polar axis.
"""
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

# Compute pie slices
N = 20
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)
colors = plt.cm.viridis(radii / 10.)

ax = plt.subplot(projection='polar')
ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.5)

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.bar` / `matplotlib.pyplot.bar`
#    - `matplotlib.projections.polar`
#
# .. tags::
#
#    plot-type: pie
#    plot-type: bar
#    level: beginner
#    purpose: showcase

```

**pie_features.py** - 饼图特性

使用 `ax.pie(sizes, labels=labels)` 创建饼图,通过 `explode`(突出显示)、`autopct`(自动百分比)、`startangle`(起始角度)、`shadow`(阴影)、`colors`(颜色列表)等参数自定义样式。还可以通过 `pctdistance`(百分比标签距离)和 `labeldistance`(标签距离)精细调整布局,适合占比展示、构成分析等场景。

```python
"""
.. redirect-from:: gallery/pie_and_polar_charts/pie_demo2

==========
Pie charts
==========

Demo of plotting a pie chart.

This 示例 illustrates various parameters of `~matplotlib.axes.Axes.pie`.
"""

# %%
# Label slices
# ------------
#
# Plot a pie chart of animals and label the slices. To add
# labels, pass a list of labels to the *labels* parameter

import matplotlib.pyplot as plt

labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels)

# %%
# Each slice of the pie chart is a `.patches.Wedge` object; therefore in
# addition to the customizations shown here, each wedge can be customized using
# the *wedgeprops* argument, as 演示d in
# :doc:`/gallery/pie_and_polar_charts/nested_pie`.
#
# Auto-label slices
# -----------------
#
# Pass a function or format string to *autopct* to label slices.

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%')

# %%
# By default, the label values are obtained from the percent size of the slice.
#
# Color slices
# ------------
#
# Pass a list of colors to *colors* to set the color of each slice.

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels,
       colors=['olivedrab', 'rosybrown', 'gray', 'saddlebrown'])

# %%
# Hatch slices
# ------------
#
# Pass a list of hatch patterns to *hatch* to set the pattern of each slice.

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, hatch=['**O', 'oO', 'O.O', '.||.'])

# %%
# Swap label and autopct text positions
# -------------------------------------
# Use the *labeldistance* and *pctdistance* parameters to position the *labels*
# and *autopct* text respectively.

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%',
       pctdistance=1.25, labeldistance=.6)

# %%
# *labeldistance* and *pctdistance* are ratios of the radius; therefore they
# vary between ``0`` for the center of the pie and ``1`` for the edge of the
# pie, and can be set to greater than ``1`` to place text outside the pie.
#
# Explode, shade, and rotate slices
# ---------------------------------
#
# In addition to the basic pie chart, this demo shows a few optional features:
#
# * offsetting a slice using *explode*
# * add a drop-shadow using *shadow*
# * custom start angle using *startangle*
#
# This 示例 orders the slices, separates (explodes) them, and rotates them.

explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig, ax = plt.subplots()
ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
       shadow=True, startangle=90)
plt.show()

# %%
# The default *startangle* is 0, which would start the first slice ("Frogs") on
# the positive x-axis. This 示例 sets ``startangle = 90`` such that all the
# slices are rotated counter-clockwise by 90 degrees, and the frog slice starts
# on the positive y-axis.
#
# Controlling the size
# --------------------
#
# By changing the *radius* parameter, and often the text size for better visual
# appearance, the pie chart can be scaled.

fig, ax = plt.subplots()

ax.pie(sizes, labels=labels, autopct='%.0f%%',
       textprops={'size': 'smaller'}, radius=0.5)
plt.show()

# %%
# Modifying the shadow
# --------------------
#
# The *shadow* parameter may optionally take a dictionary with arguments to
# the `.Shadow` patch. This can be used to modify the default shadow.

fig, ax = plt.subplots()
ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
       shadow={'ox': -0.04, 'edgecolor': 'none', 'shade': 0.9}, startangle=90)
plt.show()

# %%
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.pie` / `matplotlib.pyplot.pie`
#
# .. tags::
#
#    plot-type: pie
#    level: beginner

```

**nested_pie.py** - 嵌套饼图

通过多次调用 `ax.pie()` 并使用不同的 `radius`(半径)参数创建嵌套的饼图或环形图,可以展示分层数据或多级分类结构。通过调整每层的半径和宽度,可以创建优雅的嵌套效果,适合展示层级占比、多级分类数据等场景。

```python
"""
=================
Nested pie charts
=================

The following 示例s show two ways to build a nested pie chart
in Matplotlib. Such charts are often referred to as donut charts.

See also the :doc:`/gallery/specialty_plots/leftventricle_bullseye` 示例.
"""

import matplotlib.pyplot as plt
import numpy as np

# %%
# The most straightforward way to build a pie chart is to use the
# `~matplotlib.axes.Axes.pie` method.
#
# In this case, pie takes values corresponding to counts in a group.
# We'll first generate some fake data, corresponding to three groups.
# In the inner circle, we'll treat each number as belonging to its
# own group. In the outer circle, we'll plot them as members of their
# original 3 groups.
#
# The effect of the donut shape is achieved by setting a ``width`` to
# the pie's wedges through the *wedgeprops* argument.


fig, ax = plt.subplots()

size = 0.3
vals = np.array([[60., 32.], [37., 40.], [29., 10.]])

tab20c = plt.color_sequences["tab20c"]
outer_colors = [tab20c[i] for i in [0, 4, 8]]
inner_colors = [tab20c[i] for i in [1, 2, 5, 6, 9, 10]]

ax.pie(vals.sum(axis=1), radius=1, colors=outer_colors,
       wedgeprops=dict(width=size, edgecolor='w'))

ax.pie(vals.flatten(), radius=1-size, colors=inner_colors,
       wedgeprops=dict(width=size, edgecolor='w'))

ax.set(aspect="equal", title='Pie plot with `ax.pie`')
plt.show()

# %%
# However, you can accomplish the same output by using a bar plot on
# Axes with a polar coordinate system. This may give more flexibility on
# the exact design of the plot.
#
# In this case, we need to map x-values of the 柱状图 onto radians of
# a circle. The cumulative sum of the values are used as the edges
# of the bars.

fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))

size = 0.3
vals = np.array([[60., 32.], [37., 40.], [29., 10.]])
# Normalize vals to 2 pi
valsnorm = vals/np.sum(vals)*2*np.pi
# Obtain the ordinates of the bar edges
valsleft = np.cumsum(np.append(0, valsnorm.flatten()[:-1])).reshape(vals.shape)

cmap = plt.colormaps["tab20c"]
outer_colors = cmap(np.arange(3)*4)
inner_colors = cmap([1, 2, 5, 6, 9, 10])

ax.bar(x=valsleft[:, 0],
       width=valsnorm.sum(axis=1), bottom=1-size, height=size,
       color=outer_colors, edgecolor='w', linewidth=1, align="edge")

ax.bar(x=valsleft.flatten(),
       width=valsnorm.flatten(), bottom=1-2*size, height=size,
       color=inner_colors, edgecolor='w', linewidth=1, align="edge")

ax.set(title="Pie plot with `ax.bar` and polar coordinates")
ax.set_axis_off()
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.pie` / `matplotlib.pyplot.pie`
#    - `matplotlib.axes.Axes.bar` / `matplotlib.pyplot.bar`
#    - `matplotlib.projections.polar`
#    - ``Axes.set`` (`matplotlib.artist.Artist.set`)
#    - `matplotlib.axes.Axes.set_axis_off`
#
# .. tags::
#
#    plot-type: pie
#    level: beginner
#    purpose: showcase

```


### 7. Time Series `codebase/draw/gallery_python_new/time_series` (时间序列 - 4个)

**timeline.py** - 时间线图

使用 `matplotlib.dates` 模块处理日期数据,通过 `mdates.YearLocator()`、`MonthLocator()` 设置刻度定位器,使用 `DateFormatter()` 格式化日期显示。结合 `ax.vlines()` 可以在时间线上标记重要事件,适合项目进度展示、历史事件时间线、版本发布历史等场景。

```python
"""
====================================
Timeline with lines, dates, and text
====================================

How to create a simple timeline using Matplotlib release dates.

Timelines can be created with a collection of dates and text. In this 示例,
we show how to create a simple timeline using the dates for recent releases
of Matplotlib. First, we'll pull the data from GitHub.
"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.dates as mdates

try:
    # Try to fetch a list of Matplotlib releases and their dates
    # from https://api.github.com/repos/matplotlib/matplotlib/releases
    import json
    import urllib.request

    url = 'https://api.github.com/repos/matplotlib/matplotlib/releases'
    url += '?per_page=100'
    data = json.loads(urllib.request.urlopen(url, timeout=1).read().decode())

    dates = []
    releases = []
    for item in data:
        if 'rc' not in item['tag_name'] and 'b' not in item['tag_name']:
            dates.append(item['published_at'].split("T")[0])
            releases.append(item['tag_name'].lstrip("v"))

except Exception:
    # In case the above fails, e.g. because of missing internet connection
    # use the following lists as fallback.
    releases = ['2.2.4', '3.0.3', '3.0.2', '3.0.1', '3.0.0', '2.2.3',
                '2.2.2', '2.2.1', '2.2.0', '2.1.2', '2.1.1', '2.1.0',
                '2.0.2', '2.0.1', '2.0.0', '1.5.3', '1.5.2', '1.5.1',
                '1.5.0', '1.4.3', '1.4.2', '1.4.1', '1.4.0']
    dates = ['2019-02-26', '2019-02-26', '2018-11-10', '2018-11-10',
             '2018-09-18', '2018-08-10', '2018-03-17', '2018-03-16',
             '2018-03-06', '2018-01-18', '2017-12-10', '2017-10-07',
             '2017-05-10', '2017-05-02', '2017-01-17', '2016-09-09',
             '2016-07-03', '2016-01-10', '2015-10-29', '2015-02-16',
             '2014-10-26', '2014-10-18', '2014-08-26']

dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]  # Convert strs to dates.
releases = [tuple(release.split('.')) for release in releases]  # Split by component.
dates, releases = zip(*sorted(zip(dates, releases)))  # Sort by increasing date.

# %%
# Next, we'll create a stem plot with some variation in levels as to
# distinguish even close-by events. We add markers on the baseline for visual
# emphasis on the one-dimensional nature of the timeline.
#
# For each event, we add a text label via `~.Axes.annotate`, which is offset
# in units of points from the tip of the event line.
#
# Note that Matplotlib will automatically plot datetime inputs.

# Choose some nice levels: alternate meso releases between top and bottom, and
# progressively shorten the stems for micro releases.
levels = []
macro_meso_releases = sorted({release[:2] for release in releases})
for release in releases:
    macro_meso = release[:2]
    micro = int(release[2])
    h = 1 + 0.8 * (5 - micro)
    level = h if macro_meso_releases.index(macro_meso) % 2 == 0 else -h
    levels.append(level)


def is_feature(release):
    """Return whether a version (split into components) is a feature release."""
    return release[-1] == '0'


# The figure and the axes.
fig, ax = plt.subplots(figsize=(8.8, 4), layout="constrained")
ax.set(title="Matplotlib release dates")

# The vertical stems.
ax.vlines(dates, 0, levels,
          color=[("tab:red", 1 if is_feature(release) else .5) for release in releases])
# The baseline.
ax.axhline(0, c="black")
# The markers on the baseline.
meso_dates = [date for date, release in zip(dates, releases) if is_feature(release)]
micro_dates = [date for date, release in zip(dates, releases)
               if not is_feature(release)]
ax.plot(micro_dates, np.zeros_like(micro_dates), "ko", mfc="white")
ax.plot(meso_dates, np.zeros_like(meso_dates), "ko", mfc="tab:red")

# Annotate the lines.
for date, level, release in zip(dates, levels, releases):
    version_str = '.'.join(release)
    ax.annotate(version_str, xy=(date, level),
                xytext=(-3, np.sign(level)*3), textcoords="offset points",
                verticalalignment="bottom" if level > 0 else "top",
                weight="bold" if is_feature(release) else "normal",
                bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))

ax.xaxis.set(major_locator=mdates.YearLocator(),
             major_formatter=mdates.DateFormatter("%Y"))

# Remove the y-axis and some spines.
ax.yaxis.set_visible(False)
ax.spines[["left", "top", "right"]].set_visible(False)

ax.margins(y=0.1)
plt.show()


# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.annotate`
#    - `matplotlib.axes.Axes.vlines`
#    - `matplotlib.axis.Axis.set_major_locator`
#    - `matplotlib.axis.Axis.set_major_formatter`
#    - `matplotlib.dates.MonthLocator`
#    - `matplotlib.dates.DateFormatter`
#
# .. tags::
#
#    component: annotate
#    plot-type: line
#    level: intermediate

```

**time_series_histogram.py** - 时间序列直方图

创建基于时间序列的直方图,展示事件在时间维度上的分布特征。可以使用 `mdates` 模块处理日期格式的坐标轴,或使用 `np.histogram2d()` 创建二维时间-数值直方图,适合分析时间模式、周期性事件、活动高峰时段等场景。

```python
"""
=====================
Time Series 直方图
=====================

This 示例 演示s how to efficiently visualize large numbers of time
series in a way that could potentially reveal hidden substructure and patterns
that are not immediately obvious, and display them in a visually appealing way.

In this 示例, we generate multiple sinusoidal "signal" series that are
buried under a larger number of random walk "noise/background" series. For an
unbiased Gaussian random walk with standard deviation of σ, the RMS deviation
from the origin after n steps is σ*sqrt(n). So in order to keep the sinusoids
visible on the same scale as the random walks, we scale the amplitude by the
random walk RMS. In addition, we also introduce a small random offset ``phi``
to shift the sines left/right, and some additive random noise to shift
individual data points up/down to make the signal a bit more "realistic" (you
wouldn't expect a perfect sine wave to appear in your data).

The first plot shows the typical way of visualizing multiple time series by
overlaying them on top of each other with ``plt.plot`` and a small value of
``alpha``. The second and third plots show how to reinterpret the data as a 2d
直方图, with optional interpolation between data points, by using
``np.直方图2d`` and ``plt.pcolormesh``.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(nrows=3, figsize=(6, 8), layout='constrained')

# Fix random state for reproducibility
np.random.seed(19680801)
# Make some data; a 1D random walk + small fraction of sine waves
num_series = 1000
num_points = 100
SNR = 0.10  # Signal to Noise Ratio
x = np.linspace(0, 4 * np.pi, num_points)
# Generate unbiased Gaussian random walks
Y = np.cumsum(np.random.randn(num_series, num_points), axis=-1)
# Generate sinusoidal signals
num_signal = round(SNR * num_series)
phi = (np.pi / 8) * np.random.randn(num_signal, 1)  # small random offset
Y[-num_signal:] = (
    np.sqrt(np.arange(num_points))  # random walk RMS scaling factor
    * (np.sin(x - phi)
       + 0.05 * np.random.randn(num_signal, num_points))  # small random noise
)


# Plot series using `plot` and a small value of `alpha`. With this view it is
# very difficult to observe the sinusoidal behavior because of how many
# overlapping series there are. It also takes a bit of time to run because so
# many individual artists need to be generated.
tic = time.time()
axes[0].plot(x, Y.T, color="C0", alpha=0.1)
toc = time.time()
axes[0].set_title("Line plot with alpha")
print(f"{toc-tic:.3f} sec. elapsed")


# Now we will convert the multiple time series into a 直方图. Not only will
# the hidden signal be more visible, but it is also a much quicker procedure.
tic = time.time()
# Linearly interpolate between the points in each time series
num_fine = 800
x_fine = np.linspace(x.min(), x.max(), num_fine)
y_fine = np.concatenate([np.interp(x_fine, x, y_row) for y_row in Y])
x_fine = np.broadcast_to(x_fine, (num_series, num_fine)).ravel()


# Plot (x, y) points in 2d 直方图 with log colorscale
# It is pretty evident that there is some kind of structure under the noise
# You can tune vmax to make signal more visible
cmap = plt.colormaps["plasma"]
cmap = cmap.with_extremes(bad=cmap(0))
h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[400, 100])
pcm = axes[1].pcolormesh(xedges, yedges, h.T, cmap=cmap,
                         norm="log", vmax=1.5e2, rasterized=True)
fig.colorbar(pcm, ax=axes[1], label="# points", pad=0)
axes[1].set_title("2d histogram and log color scale")

# Same data but on linear color scale
pcm = axes[2].pcolormesh(xedges, yedges, h.T, cmap=cmap,
                         vmax=1.5e2, rasterized=True)
fig.colorbar(pcm, ax=axes[2], label="# points", pad=0)
axes[2].set_title("2d histogram and linear color scale")

toc = time.time()
print(f"{toc-tic:.3f} sec. elapsed")
plt.show()

# %%
#
# .. tags::
#
#    plot-type: 直方图2d
#    plot-type: pcolormesh
#    purpose: storytelling
#    styling: color
#    component: colormap
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.pcolormesh` / `matplotlib.pyplot.pcolormesh`
#    - `matplotlib.figure.Figure.colorbar`

```

**fill_between_demo.py** - 填充区域图

使用 `ax.fill_between(x, y1, y2)` 在两条曲线之间填充颜色,通过 `alpha`(透明度)和 `color`(颜色)参数控制填充效果。可以用于表示置信区间、误差范围、目标区域等,配合 `where` 参数可以实现条件填充,适合不确定性可视化、趋势带、阈值范围展示等场景。

```python
"""
===============================
Fill the area between two lines
===============================

This 示例 shows how to use `~.axes.Axes.fill_between` to color the area
between two lines.
"""

import matplotlib.pyplot as plt
import numpy as np

# %%
#
# Basic usage
# -----------
# The parameters *y1* and *y2* can be scalars, indicating a horizontal
# boundary at the given y-values. If only *y1* is given, *y2* defaults to 0.

x = np.arange(0.0, 2, 0.01)
y1 = np.sin(2 * np.pi * x)
y2 = 0.8 * np.sin(4 * np.pi * x)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 6))

ax1.fill_between(x, y1)
ax1.set_title('fill between y1 and 0')

ax2.fill_between(x, y1, 1)
ax2.set_title('fill between y1 and 1')

ax3.fill_between(x, y1, y2)
ax3.set_title('fill between y1 and y2')
ax3.set_xlabel('x')
fig.tight_layout()

# %%
#
# 示例: Confidence bands
# -------------------------
# A common application for `~.axes.Axes.fill_between` is the indication of
# confidence bands.
#
# `~.axes.Axes.fill_between` uses the colors of the color cycle as the fill
# color. These may be a bit strong when applied to fill areas. It is
# therefore often a good practice to lighten the color by making the area
# semi-transparent using *alpha*.

# sphinx_gallery_thumbnail_number = 2

N = 21
x = np.linspace(0, 10, 11)
y = [3.9, 4.4, 10.8, 10.3, 11.2, 13.1, 14.1,  9.9, 13.9, 15.1, 12.5]

# fit a linear curve and estimate its y-values and their error.
a, b = np.polyfit(x, y, deg=1)
y_est = a * x + b
y_err = x.std() * np.sqrt(1/len(x) +
                          (x - x.mean())**2 / np.sum((x - x.mean())**2))

fig, ax = plt.subplots()
ax.plot(x, y_est, '-')
ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
ax.plot(x, y, 'o', color='tab:brown')

# %%
#
# Selectively filling horizontal regions
# --------------------------------------
# The parameter *where* allows to specify the x-ranges to fill. It's a boolean
# array with the same size as *x*.
#
# Only x-ranges of contiguous *True* sequences are filled. As a result the
# range between neighboring *True* and *False* values is never filled. This
# often undesired when the data points should represent a contiguous quantity.
# It is therefore recommended to set ``interpolate=True`` unless the
# x-distance of the data points is fine enough so that the above effect is not
# noticeable. Interpolation approximates the actual x position at which the
# *where* condition will change and extends the filling up to there.

x = np.array([0, 1, 2, 3])
y1 = np.array([0.8, 0.8, 0.2, 0.2])
y2 = np.array([0, 0, 1, 1])

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.set_title('interpolation=False')
ax1.plot(x, y1, 'o--')
ax1.plot(x, y2, 'o--')
ax1.fill_between(x, y1, y2, where=(y1 > y2), color='C0', alpha=0.3)
ax1.fill_between(x, y1, y2, where=(y1 < y2), color='C1', alpha=0.3)

ax2.set_title('interpolation=True')
ax2.plot(x, y1, 'o--')
ax2.plot(x, y2, 'o--')
ax2.fill_between(x, y1, y2, where=(y1 > y2), color='C0', alpha=0.3,
                 interpolate=True)
ax2.fill_between(x, y1, y2, where=(y1 <= y2), color='C1', alpha=0.3,
                 interpolate=True)
fig.tight_layout()

# %%
#
# .. note::
#
#    Similar gaps will occur if *y1* or *y2* are masked arrays. Since missing
#    values cannot be approximated, *interpolate* has no effect in this case.
#    The gaps around masked values can only be reduced by adding more data
#    points close to the masked values.

# %%
#
# Selectively marking horizontal regions across the whole Axes
# ------------------------------------------------------------
# The same selection mechanism can be applied to fill the full vertical height
# of the Axes. To be independent of y-limits, we add a transform that
# interprets the x-values in data coordinates and the y-values in Axes
# coordinates.
#
# The following 示例 marks the regions in which the y-data are above a
# given threshold.

fig, ax = plt.subplots()
x = np.arange(0, 4 * np.pi, 0.01)
y = np.sin(x)
ax.plot(x, y, color='black')

threshold = 0.75
ax.axhline(threshold, color='green', lw=2, alpha=0.7)
ax.fill_between(x, 0, 1, where=y > threshold,
                color='green', alpha=0.5, transform=ax.get_xaxis_transform())

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.fill_between` / `matplotlib.pyplot.fill_between`
#    - `matplotlib.axes.Axes.get_xaxis_transform`
#
# .. tags::
#
#    styling: conditional
#    plot-type: fill_between
#    level: beginner
#    purpose: showcase

```

**fill_betweenx_demo.py** - 横向填充区域图

使用 `ax.fill_betweenx(y, x1, x2)` 在水平方向上填充区域,与 `fill_between` 类似但是沿Y轴方向。适合需要在水平方向展示区间的场景,比如垂直剖面的数值范围、Y轴条件下的X值范围等。

```python
"""
========================================
Fill the area between two vertical lines
========================================

Using `~.Axes.fill_betweenx` to color along the horizontal direction between
two curves.
"""
import matplotlib.pyplot as plt
import numpy as np

y = np.arange(0.0, 2, 0.01)
x1 = np.sin(2 * np.pi * y)
x2 = 1.2 * np.sin(4 * np.pi * y)

fig, [ax1, ax2, ax3] = plt.subplots(1, 3, sharey=True, figsize=(6, 6))

ax1.fill_betweenx(y, 0, x1)
ax1.set_title('between (x1, 0)')

ax2.fill_betweenx(y, x1, 1)
ax2.set_title('between (x1, 1)')
ax2.set_xlabel('x')

ax3.fill_betweenx(y, x1, x2)
ax3.set_title('between (x1, x2)')

# %%
# Now fill between x1 and x2 where a logical condition is met. Note this is
# different than calling::
#
#   fill_between(y[where], x1[where], x2[where])
#
# because of edge effects over multiple contiguous regions.

fig, [ax, ax1] = plt.subplots(1, 2, sharey=True, figsize=(6, 6))
ax.plot(x1, y, x2, y, color='black')
ax.fill_betweenx(y, x1, x2, where=x2 >= x1, facecolor='green')
ax.fill_betweenx(y, x1, x2, where=x2 <= x1, facecolor='red')
ax.set_title('fill_betweenx where')

# Test support for masked arrays.
x2 = np.ma.masked_greater(x2, 1.0)
ax1.plot(x1, y, x2, y, color='black')
ax1.fill_betweenx(y, x1, x2, where=x2 >= x1, facecolor='green')
ax1.fill_betweenx(y, x1, x2, where=x2 <= x1, facecolor='red')
ax1.set_title('regions with x2 > 1 are masked')

# %%
# This 示例 illustrates a problem; because of the data gridding, there are
# undesired unfilled triangles at the crossover points. A brute-force solution
# would be to interpolate all arrays to a very fine grid before plotting.

plt.show()

# %%
# .. tags::
#
#    plot-type: fill_between
#    level: beginner

```


### 8. Composite Plots `codebase/draw/gallery_python_new/composite_plots` (组合图表 - 3个)

**subplots_demo.py** - 多子图

演示创建多子图的各种方法,包括 `plt.subplots(nrows, ncols)`(规则网格)、`subplot_mosaic`(复杂布局)和 `GridSpec`(完全自定义布局)。通过 `sharex`、`sharey` 参数可以共享坐标轴,使用 `supxlabel`、`supylabel` 可以添加全局标签,适合多面板对比分析、多维度数据展示等场景。

```python
"""
===============================================
Create multiple subplots using ``plt.subplots``
===============================================

`.pyplot.subplots` creates a figure and a grid of subplots with a single call,
while providing reasonable control over how the individual plots are created.
For more advanced use cases you can use `.GridSpec` for a more general subplot
layout or `.Figure.add_subplot` for adding subplots at arbitrary locations
within the figure.
"""

# sphinx_gallery_thumbnail_number = 11

import matplotlib.pyplot as plt
import numpy as np

# Set smaller font sizes for better display
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['figure.titlesize'] = 10

# Some 示例 data to display
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

# %%
# A figure with just one subplot
# """"""""""""""""""""""""""""""
#
# ``subplots()`` without arguments returns a `.Figure` and a single
# `~.axes.Axes`.
#
# This is actually the simplest and recommended way of creating a single
# Figure and Axes.

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('A single plot')

# %%
# Stacking subplots in one direction
# """"""""""""""""""""""""""""""""""
#
# The first two optional arguments of `.pyplot.subplots` define the number of
# rows and columns of the subplot grid.
#
# When stacking in one direction only, the returned ``axs`` is a 1D numpy array
# containing the list of created Axes.

fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(x, y)
axs[1].plot(x, -y)

# %%
# If you are creating just a few Axes, it's handy to unpack them immediately to
# dedicated variables for each Axes. That way, we can use ``ax1`` instead of
# the more verbose ``axs[0]``.

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
ax1.plot(x, y)
ax2.plot(x, -y)

# %%
# To obtain side-by-side subplots, pass parameters ``1, 2`` for one row and two
# columns.

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Horizontally stacked subplots')
ax1.plot(x, y)
ax2.plot(x, -y)

# %%
# Stacking subplots in two directions
# """""""""""""""""""""""""""""""""""
#
# When stacking in two directions, the returned ``axs`` is a 2D NumPy array.
#
# If you have to set parameters for each subplot it's handy to iterate over
# all subplots in a 2D grid using ``for ax in axs.flat:``.

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x, y)
axs[0, 0].set_title('Axis [0, 0]')
axs[0, 1].plot(x, y, 'tab:orange')
axs[0, 1].set_title('Axis [0, 1]')
axs[1, 0].plot(x, -y, 'tab:green')
axs[1, 0].set_title('Axis [1, 0]')
axs[1, 1].plot(x, -y, 'tab:red')
axs[1, 1].set_title('Axis [1, 1]')

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

# %%
# You can use tuple-unpacking also in 2D to assign all subplots to dedicated
# variables:

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('Sharing x per column, y per row')
ax1.plot(x, y)
ax2.plot(x, y**2, 'tab:orange')
ax3.plot(x, -y, 'tab:green')
ax4.plot(x, -y**2, 'tab:red')

for ax in fig.get_axes():
    ax.label_outer()

# %%
# Sharing axes
# """"""""""""
#
# By default, each Axes is scaled individually. Thus, if the ranges are
# different the tick values of the subplots do not align.

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Axes values are scaled individually by default')
ax1.plot(x, y)
ax2.plot(x + 1, -y)

# %%
# You can use *sharex* or *sharey* to align the horizontal or vertical axis.

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Aligning x-axis using sharex')
ax1.plot(x, y)
ax2.plot(x + 1, -y)

# %%
# Setting *sharex* or *sharey* to ``True`` enables global sharing across the
# whole grid, i.e. also the y-axes of vertically stacked subplots have the
# same scale when using ``sharey=True``.

fig, axs = plt.subplots(3, sharex=True, sharey=True)
fig.suptitle('Sharing both axes')
axs[0].plot(x, y ** 2)
axs[1].plot(x, 0.3 * y, 'o')
axs[2].plot(x, y, '+')

# %%
# For subplots that are sharing axes one set of tick labels is enough. Tick
# labels of inner Axes are automatically removed by *sharex* and *sharey*.
# Still there remains an unused empty space between the subplots.
#
# To precisely control the positioning of the subplots, one can explicitly
# create a `.GridSpec` with `.Figure.add_gridspec`, and then call its
# `~.GridSpecBase.subplots` method.  For 示例, we can reduce the height
# between vertical subplots using ``add_gridspec(hspace=0)``.
#
# `.label_outer` is a handy method to remove labels and ticks from subplots
# that are not at the edge of the grid.

fig = plt.figure()
gs = fig.add_gridspec(3, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
fig.suptitle('Sharing both axes')
axs[0].plot(x, y ** 2)
axs[1].plot(x, 0.3 * y, 'o')
axs[2].plot(x, y, '+')

# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()

# %%
# Apart from ``True`` and ``False``, both *sharex* and *sharey* accept the
# values 'row' and 'col' to share the values only per row or column.

fig = plt.figure()
gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
(ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
fig.suptitle('Sharing x per column, y per row')
ax1.plot(x, y)
ax2.plot(x, y**2, 'tab:orange')
ax3.plot(x + 1, -y, 'tab:green')
ax4.plot(x + 2, -y**2, 'tab:red')

for ax in fig.get_axes():
    ax.label_outer()

# %%
# If you want a more complex sharing structure, you can first create the
# grid of Axes with no sharing, and then call `.axes.Axes.sharex` or
# `.axes.Axes.sharey` to add sharing info a posteriori.

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x, y)
axs[0, 0].set_title("main")
axs[1, 0].plot(x, y**2)
axs[1, 0].set_title("shares x with main")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(x + 1, y + 1)
axs[0, 1].set_title("unrelated")
axs[1, 1].plot(x + 2, y + 2)
axs[1, 1].set_title("also unrelated")
fig.tight_layout()

# %%
# Polar Axes
# """"""""""
#
# The parameter *subplot_kw* of `.pyplot.subplots` controls the subplot
# properties (see also `.Figure.add_subplot`). In particular, this can be used
# to create a grid of polar Axes.

fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
ax1.plot(x, y)
ax2.plot(x, y ** 2)

plt.show()

# %%
# .. tags::
#
#    component: subplot,
#    component: axes,
#    component: axis
#    plot-type: line,
#    plot-type: polar
#    level: beginner
#    purpose: showcase

```

**two_scales.py** - 双Y轴图

使用 `ax.twinx()` 创建共享X轴的双Y轴图表,左右两个Y轴可以有不同的刻度和标签。通过设置不同的颜色和 `tick_params` 可以区分两个数据系列,适合同时展示不同量级或单位的两个相关变量,如温度与降雨量、价格与交易量等。

```python
"""
===========================
Plots with different scales
===========================

Two plots on the same Axes with different left and right scales.

The trick is to use *two different Axes* that share the same *x* axis.
You can use separate `matplotlib.ticker` formatters and locators as
desired since the two Axes are independent.

Such Axes are generated by calling the `.Axes.twinx` method. Likewise,
`.Axes.twiny` is available to generate Axes that share a *y* axis but
have different top and bottom scales.
"""
import matplotlib.pyplot as plt
import numpy as np

# Create some mock data
t = np.arange(0.01, 10.0, 0.01)
data1 = np.exp(t)
data2 = np.sin(2 * np.pi * t)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('exp', color=color)
ax1.plot(t, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.twinx` / `matplotlib.pyplot.twinx`
#    - `matplotlib.axes.Axes.twiny` / `matplotlib.pyplot.twiny`
#    - `matplotlib.axes.Axes.tick_params` / `matplotlib.pyplot.tick_params`
#
# .. tags::
#
#    component: axes
#    plot-type: line
#    level: beginner

```

**scatter_with_legend.py** - 散点图+图例

创建带图例的散点图,通过自定义图例的位置(`loc`)、样式(`frameon`、`shadow`)和布局(`ncol`)来优化图表可读性。可以手动添加图例项,使用自定义的颜色和标记,适合多类别数据可视化、分组数据展示等场景。

```python
"""
==========================
散点图 with a legend
==========================

To create a 散点图 with a legend one may use a loop and create one
`~.Axes.scatter` plot per item to appear in the legend and set the ``label``
accordingly.

The following also 演示s how transparency of the markers
can be adjusted by giving ``alpha`` a value between 0 and 1.
"""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)


fig, ax = plt.subplots()
for color in ['tab:blue', 'tab:orange', 'tab:green']:
    n = 750
    x, y = np.random.rand(2, n)
    scale = 200.0 * np.random.rand(n)
    ax.scatter(x, y, c=color, s=scale, label=color,
               alpha=0.3, edgecolors='none')

ax.legend(fontsize=9, markerscale=0.7, bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)
plt.tight_layout()

plt.show()


# %%
# .. _automatedlegendcreation:
#
# Automated legend creation
# -------------------------
#
# Another option for creating a legend for a scatter is to use the
# `.PathCollection.legend_elements` method.  It will automatically try to
# determine a useful number of legend entries to be shown and return a tuple of
# handles and labels. Those can be passed to the call to `~.axes.Axes.legend`.


N = 45
x, y = np.random.rand(2, N)
c = np.random.randint(1, 5, size=N)
s = np.random.randint(10, 220, size=N)

fig, ax = plt.subplots()

scatter = ax.scatter(x, y, c=c, s=s)

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="center left", bbox_to_anchor=(1.05, 0.5),
                    title="Classes", fontsize=8)
ax.add_artist(legend1)

# produce a legend with a cross-section of sizes from the scatter
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
legend2 = ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.05, 0.3),
                    title="Sizes", fontsize=8, markerscale=0.7)

plt.tight_layout()
plt.show()


# %%
# Further arguments to the `.PathCollection.legend_elements` method
# can be used to steer how many legend entries are to be created and how they
# should be labeled. The following shows how to use some of them.

volume = np.random.rayleigh(27, size=40)
amount = np.random.poisson(10, size=40)
ranking = np.random.normal(size=40)
price = np.random.uniform(1, 10, size=40)

fig, ax = plt.subplots()

# Because the price is much too small when being provided as size for ``s``,
# we normalize it to some useful point sizes, s=0.3*(price*3)**2
scatter = ax.scatter(volume, amount, c=ranking, s=0.3*(price*3)**2,
                     vmin=-3, vmax=3, cmap="Spectral")

# Produce a legend for the ranking (colors). Even though there are 40 different
# rankings, we only want to show 5 of them in the legend.
legend1 = ax.legend(*scatter.legend_elements(num=5),
                    loc="center left", bbox_to_anchor=(1.05, 0.6),
                    title="Ranking", fontsize=8)
ax.add_artist(legend1)

# Produce a legend for the price (sizes). Because we want to show the prices
# in dollars, we use the *func* argument to supply the inverse of the function
# used to calculate the sizes from above. The *fmt* ensures to show the price
# in dollars. Note how we target at 5 elements here, but obtain only 4 in the
# created legend due to the automatic round prices that are chosen for us.
kw = dict(prop="sizes", num=5, color=scatter.cmap(0.7), fmt="$ {x:.2f}",
          func=lambda s: np.sqrt(s/.3)/3)
legend2 = ax.legend(*scatter.legend_elements(**kw),
                    loc="center left", bbox_to_anchor=(1.05, 0.3),
                    title="Price", fontsize=8, markerscale=0.7)

plt.tight_layout()
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.scatter` / `matplotlib.pyplot.scatter`
#    - `matplotlib.axes.Axes.legend` / `matplotlib.pyplot.legend`
#    - `matplotlib.collections.PathCollection.legend_elements`
#
# .. tags::
#
#    component: legend
#    plot-type: scatter
#    level: intermediate

```


### 9. Images & Shapes `codebase/draw/gallery_python_new/images_shapes` (图像形状 - 2个)

**layer_images.py** - 图像叠加

使用 `ax.imshow()` 的 `alpha`(透明度)参数叠加多层图像,通过 `extent` 参数控制图像的位置和范围。可以通过调整每层的透明度来控制叠加效果,适合图像融合、前后对比、多通道数据叠加等场景。

```python
"""
================================
Layer images with alpha blending
================================

Layer images above one another using alpha blending
"""
import matplotlib.pyplot as plt
import numpy as np


def func3(x, y):
    return (1 - x / 2 + x**5 + y**3) * np.exp(-(x**2 + y**2))


# make these smaller to increase the resolution
dx, dy = 0.05, 0.05

x = np.arange(-3.0, 3.0, dx)
y = np.arange(-3.0, 3.0, dy)
X, Y = np.meshgrid(x, y)

# when layering multiple images, the images need to have the same
# extent.  This does not mean they need to have the same shape, but
# they both need to render to the same coordinate system determined by
# xmin, xmax, ymin, ymax.  Note if you use different interpolations
# for the images their apparent extent could be different due to
# interpolation edge effects

extent = np.min(x), np.max(x), np.min(y), np.max(y)
fig = plt.figure(frameon=False)

Z1 = np.add.outer(range(8), range(8)) % 2  # chessboard
im1 = plt.imshow(Z1, cmap=plt.cm.gray, interpolation='nearest',
                 extent=extent)

Z2 = func3(X, Y)

im2 = plt.imshow(Z2, cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear',
                 extent=extent)

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.imshow` / `matplotlib.pyplot.imshow`

```

**triinterp_demo.py** - 三角插值

使用 `matplotlib.tri` 模块进行三角网格插值,通过 `Triangulation` 创建非规则网格,使用 `tripcolor` 或 `tricontour` 在三角网格上绘制数据。适合地理信息系统、有限元分析、非规则采样数据的可视化等场景。

```python
"""
==============
Triinterp Demo
==============

Interpolation from triangular grid to quad grid.
"""
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.tri as mtri

# Create triangulation.
x = np.asarray([0, 1, 2, 3, 0.5, 1.5, 2.5, 1, 2, 1.5])
y = np.asarray([0, 0, 0, 0, 1.0, 1.0, 1.0, 2, 2, 3.0])
triangles = [[0, 1, 4], [1, 2, 5], [2, 3, 6], [1, 5, 4], [2, 6, 5], [4, 5, 7],
             [5, 6, 8], [5, 8, 7], [7, 8, 9]]
triang = mtri.Triangulation(x, y, triangles)

# Interpolate to regularly-spaced quad grid.
z = np.cos(1.5 * x) * np.cos(1.5 * y)
xi, yi = np.meshgrid(np.linspace(0, 3, 20), np.linspace(0, 3, 20))

interp_lin = mtri.LinearTriInterpolator(triang, z)
zi_lin = interp_lin(xi, yi)

interp_cubic_geom = mtri.CubicTriInterpolator(triang, z, kind='geom')
zi_cubic_geom = interp_cubic_geom(xi, yi)

interp_cubic_min_E = mtri.CubicTriInterpolator(triang, z, kind='min_E')
zi_cubic_min_E = interp_cubic_min_E(xi, yi)

# Set up the figure
fig, axs = plt.subplots(nrows=2, ncols=2)
axs = axs.flatten()

# Plot the triangulation.
axs[0].tricontourf(triang, z)
axs[0].triplot(triang, 'ko-')
axs[0].set_title('Triangular grid')

# Plot linear interpolation to quad grid.
axs[1].contourf(xi, yi, zi_lin)
axs[1].plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
axs[1].plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
axs[1].set_title("Linear interpolation")

# Plot cubic interpolation to quad grid, kind=geom
axs[2].contourf(xi, yi, zi_cubic_geom)
axs[2].plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
axs[2].plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
axs[2].set_title("Cubic interpolation,\nkind='geom'")

# Plot cubic interpolation to quad grid, kind=min_E
axs[3].contourf(xi, yi, zi_cubic_min_E)
axs[3].plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
axs[3].plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
axs[3].set_title("Cubic interpolation,\nkind='min_E'")

fig.tight_layout()
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.tricontourf` / `matplotlib.pyplot.tricontourf`
#    - `matplotlib.axes.Axes.triplot` / `matplotlib.pyplot.triplot`
#    - `matplotlib.axes.Axes.contourf` / `matplotlib.pyplot.contourf`
#    - `matplotlib.axes.Axes.plot` / `matplotlib.pyplot.plot`
#    - `matplotlib.tri`
#    - `matplotlib.tri.LinearTriInterpolator`
#    - `matplotlib.tri.CubicTriInterpolator`
#    - `matplotlib.tri.Triangulation`

```


### 10. Misc Plots `codebase/draw/gallery_python_new/misc_plots` (其他图表 - 2个)

**scales.py** - 坐标轴刻度

演示不同的坐标轴刻度类型,包括线性刻度(`'linear'`)、对数刻度(`'log'`)、对称对数刻度(`'symlog'`)、逻辑刻度(`'logit'`)等。通过 `set_xscale()` 和 `set_yscale()` 设置刻度类型,适合展示跨越多个数量级的数据、包含正负值的数据等特殊场景。

```python
"""
===============
Scales overview
===============

Illustrate the scale transformations applied to axes, e.g. log, symlog, logit.

See `matplotlib.scale` for a full list of built-in scales, and
:doc:`/gallery/scales/custom_scale` for how to create your own scale.
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(400)
y = np.linspace(0.002, 1, 400)

fig, axs = plt.subplots(3, 2, figsize=(6, 8), layout='constrained')

axs[0, 0].plot(x, y)
axs[0, 0].set_yscale('linear')
axs[0, 0].set_title('linear')
axs[0, 0].grid(True)

axs[0, 1].plot(x, y)
axs[0, 1].set_yscale('log')
axs[0, 1].set_title('log')
axs[0, 1].grid(True)

axs[1, 0].plot(x, y - y.mean())
axs[1, 0].set_yscale('symlog', linthresh=0.02)
axs[1, 0].set_title('symlog')
axs[1, 0].grid(True)

axs[1, 1].plot(x, y)
axs[1, 1].set_yscale('logit')
axs[1, 1].set_title('logit')
axs[1, 1].grid(True)

axs[2, 0].plot(x, y - y.mean())
axs[2, 0].set_yscale('asinh', linear_width=0.01)
axs[2, 0].set_title('asinh')
axs[2, 0].grid(True)


# Function x**(1/2)
def forward(x):
    return x**(1/2)


def inverse(x):
    return x**2


axs[2, 1].plot(x, y)
axs[2, 1].set_yscale('function', functions=(forward, inverse))
axs[2, 1].set_title('function: $x^{1/2}$')
axs[2, 1].grid(True)
axs[2, 1].set_yticks(np.arange(0, 1.2, 0.2))

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this 示例:
#
#    - `matplotlib.axes.Axes.set_xscale`
#    - `matplotlib.axes.Axes.set_yscale`
#    - `matplotlib.scale.LinearScale`
#    - `matplotlib.scale.LogScale`
#    - `matplotlib.scale.SymmetricalLogScale`
#    - `matplotlib.scale.LogitScale`
#    - `matplotlib.scale.FuncScale`

```

**table_demo.py** - 表格演示

使用 `plt.table()` 在图表中嵌入表格,可以设置单元格文本、颜色、边框等样式。表格可以放置在图表的顶部、底部或作为独立元素,适合在图表中展示统计数据、参数列表、汇总信息等场景。

```python
"""
==========
Table Demo
==========

Demo of table function to display a table within a plot.
"""
import matplotlib.pyplot as plt
import numpy as np

data = [[ 66386, 174296,  75131, 577908,  32015],
        [ 58230, 381139,  78045,  99308, 160454],
        [ 89135,  80552, 152558, 497981, 603535],
        [ 78415,  81858, 150656, 193263,  69638],
        [139361, 331509, 343164, 781380,  52269]]

columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

values = np.arange(0, 2500, 500)
value_increment = 1000

# Get some pastel shades for the colors
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
n_rows = len(data)

index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialize the vertical-offset for the stacked 柱状图.
y_offset = np.zeros(len(columns))

# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data[row]
    cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])
# Reverse colors and text labels to display the last value at the top.
colors = colors[::-1]
cell_text.reverse()

# Add a table at the bottom of the Axes
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel(f"Loss in ${value_increment}'s")
plt.yticks(values * value_increment, ['%d' % val for val in values])
plt.xticks([])
plt.title('Loss by Disaster')

plt.show()

```

