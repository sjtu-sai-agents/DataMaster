## 代码模板和范例

在这里，我们给出了常用的 47 中科研绘图可视化的 `matplotlib` 的标准版实现，请你在生成代码的时候**优先参考这里的代码**！这里的代码都是绝对可信并且可行的，同时，**请你务必不要忘记你需要遵守的基本要求**。

### 1. Basic Plots `codebase/draw/gallery_python_new/basic_plots` (基础图表 - 7个)

**simple_plot.py** - 基础折线图
使用 matplotlib 的这套 API（包括 `plt.subplots()`、`ax.plot()`、`ax.set()` 等）可以绘制能够展现数据随时间或其他**连续变量变化趋势的折线图**。这种图表特别适用于可视化序列数据、趋势分析、周期性波动以及多组数据的比较。你可以自定义坐标轴标签、标题、网格线等元素，并通过调整线条样式、颜色、标记点来增强图表的可读性和表现力，从而清晰传达数据背后的信息与规律。
<code_path>codebase/draw/gallery_python_new/basic_plots/simple_plot.py</code_path>


**scatter_demo2.py** - 散点图
使用 `ax.scatter()` 创建带有可变颜色和大小的散点图,可以展示多个变量之间的关系。通过参数 `c`(颜色映射)、`s`(标记大小)和 `alpha`(透明度)可以展示数据的多个维度,特别适合金融数据、相关性分析和多变量数据可视化。<code_path>codebase/draw/gallery_python_new/basic_plots/scatter_demo2.py</code_path>

**barchart.py** - 分组柱状图
使用 `ax.bar()` 和偏移量技术创建分组柱状图,适合比较多组分类数据。通过 `bar_label()` 可以在每个柱子上添加数值标签,配合图例、网格线和自定义颜色,能够清晰展示不同类别之间的对比关系。
<code_path>codebase/draw/gallery_python_new/basic_plots/barchart.py</code_path>

**barh.py** - 水平柱状图
使用 `ax.barh()` 创建水平方向的柱状图,通过 `invert_yaxis()` 让标签从上到下排列,特别适合展示排名、长标签类别或水平方向的数据对比。支持添加误差条(`xerr`)来表示数据的不确定性范围。
<code_path>codebase/draw/gallery_python_new/basic_plots/barh.py</code_path>

**bar_stacked.py** - 堆叠柱状图
使用 `ax.bar()` 的 `bottom` 参数实现柱状图的堆叠效果,能够同时展示各部分数值和累计总量。通过循环更新 `bottom` 变量,可以创建多层堆叠图表,适合展示构成比例、累计数据和部分与整体的关系。
<code_path>codebase/draw/gallery_python_new/basic_plots/bar_stacked.py</code_path>

**stackplot_demo.py** - 堆叠面积图
使用 `ax.stackplot()` 创建堆叠面积图,展示多个序列随时间变化的累积效果。通过设置 `baseline='wiggle'` 参数可以创建流图(streamgraph)效果,适合可视化人口变化、资源分配、资金流向等时间序列的构成分析。
<code_path>codebase/draw/gallery_python_new/basic_plots/stackplot_demo.py</code_path>

**horizontal_barchart_distribution.py** - 水平分布柱状图
使用 `ax.barh()` 和 `left` 参数创建水平堆叠柱状图,通过 `data.cumsum()` 计算每个类别的起始位置,结合颜色映射和标签文本颜色自动调整,能够优雅地展示问卷调查结果、评分分布等离散数据分布情况。
<code_path>codebase/draw/gallery_python_new/basic_plots/horizontal_barchart_distribution.py</code_path>

### 2. Statistical Plots `codebase/draw/gallery_python_new/statistical_plots` (统计图表 - 8个)

**boxplot.py** - 美化箱线图
使用自定义样式配置创建出版级质量的箱线图,通过 `boxprops`、`medianprops`、`meanprops`、`flierprops` 等参数精细控制箱体、中位数线、均值点和异常值的样式。配合自定义配色方案和透明度设置,能够生成适合学术论文和专业报告的高质量统计图表,清晰展示数据的分布特征、中位数、四分位数和异常值。
<code_path>codebase/draw/gallery_python_new/statistical_plots/boxplot.py</code_path>

**boxplot_demo.py** - 箱线图基础
展示箱线图的各种样式选项,包括 `notch`(缺口)、`sym`(异常值符号)、`orientation`(方向)等参数的使用方法。通过多个子图对比不同配置效果,演示如何自定义箱线图的外观、隐藏异常值、改变须长度等,是掌握箱线图基础配置的完整参考。
<code_path>codebase/draw/gallery_python_new/statistical_plots/boxplot_demo.py</code_path>

**violinplot.py** - 小提琴图
使用 `ax.violinplot()` 创建小提琴图,通过核密度估计(KDE)展示数据的概率分布形状。通过 `points`(KDE评估点数)、`bw_method`(带宽方法)、`showmeans`(显示均值)、`showmedians`(显示中位数)、`quantiles`(分位数)等参数可以精细控制小提琴图的显示效果,适合比较多个数据组的分布特征。
<code_path>codebase/draw/gallery_python_new/statistical_plots/violinplot.py</code_path>

**customized_violin.py** - 自定义小提琴图
完全自定义小提琴图的样式,通过修改返回的 `ViolinPlot` 对象的 `bodies` 属性来设置填充颜色和透明度,使用 `vlines()` 和 `scatter()` 添加自定义的统计标记。这种方法可以实现完全个性化的视觉效果,适合需要特定品牌风格或出版要求的图表制作。
<code_path>codebase/draw/gallery_python_new/statistical_plots/customized_violin.py</code_path>

**errorbar.py** - 误差条图
使用 `ax.errorbar()` 创建带误差条的折线图,通过 `xerr` 和 `yerr` 参数指定X和Y方向的误差,配合 `fmt`(数据点格式)、`capsize`(误差帽大小)、`capthick`(误差帽粗细)、`elinewidth`(误差线粗细)等参数,能够清晰展示实验数据的不确定性范围,是科学数据可视化的标准方法。
<code_path>codebase/draw/gallery_python_new/statistical_plots/errorbar.py</code_path>

**errorbar_features.py** - 误差条特性
展示误差条的高级功能,包括不对称误差(`xerr=[lower, upper]`)、对数坐标轴上的误差条、自定义误差线样式等。通过多种配置演示如何处理复杂的误差表示需求,适合需要在特殊坐标系统或展示非对称不确定性的科学数据分析场景。
<code_path>codebase/draw/gallery_python_new/statistical_plots/errorbar_features.py</code_path>

**hexbin_demo.py** - 六边形分箱图
使用 `ax.hexbin()` 创建六边形分箱图,通过将数据点聚合到六边形网格中来展示二维数据的密度分布。相比散点图,这种方法更适合大数据集的可视化,通过 `gridsize`(网格大小)、`bins`(分箱方式,如 `'log'`)和 `cmap`(颜色映射)参数可以优化视觉效果,避免过度绘制问题。
<code_path>codebase/draw/gallery_python_new/statistical_plots/hexbin_demo.py</code_path>

**confidence_ellipse.py** - 置信椭圆
绘制二维数据集的置信椭圆,通过计算协方差矩阵和特征值,使用 `Ellipse` patch 和 `Affine2D` 变换创建表示数据置信区间的椭圆。可以通过 `n_std` 参数控制置信水平(如2倍标准差对应95%置信区间),适合展示相关性分析、统计推断和数据分布的置信范围。
<code_path>codebase/draw/gallery_python_new/statistical_plots/confidence_ellipse.py</code_path>

### 3. Distribution Plots `codebase/draw/gallery_python_new/distribution_plots` (分布图 - 7个)

**hist.py** - 基础直方图
使用 `ax.hist()` 创建直方图来展示数据的分布特征,通过 `bins`(分箱数量)、`density`(归一化为概率密度)、`alpha`(透明度)、`color`(颜色)和 `edgecolor`(边框颜色)等参数自定义样式。还展示二维直方图 `hist2d` 和百分比格式化,是数据分析中最常用的分布可视化方法。
<code_path>codebase/draw/gallery_python_new/distribution_plots/hist.py</code_path>

**histogram_histtypes.py** - 直方图类型
演示 `histtype` 参数的不同选项,包括 `'bar'`(传统柱状)、`'barstacked'`(堆叠柱状)、`'step'`(阶梯线)、`'stepfilled'`(填充阶梯)等样式。通过对比不同类型的效果,帮助选择最适合数据和出版需求的直方图样式。
<code_path>codebase/draw/gallery_python_new/distribution_plots/histogram_histtypes.py</code_path>

**histogram_cumulative.py** - 累积直方图
使用 `cumulative=True` 参数创建累积分布函数(CDF),展示数据随变量的累积概率。配合 `density=True` 和 `histtype='step'` 可以生成理论CDF的对比图,适合概率分布分析、统计建模和数据质量检查。
<code_path>codebase/draw/gallery_python_new/distribution_plots/histogram_cumulative.py</code_path>

**histogram_multihist.py** - 多组直方图
在同一图表中并排展示多个数据组的直方图,通过循环和偏移技术或 `histtype='barstacked'` 实现多组数据的对比。适合比较多组实验数据的分布差异,或展示不同条件下的数据分布特征。
<code_path>codebase/draw/gallery_python_new/distribution_plots/histogram_multihist.py</code_path>

**histogram_normalization.py** - 标准化直方图
演示直方图的多种归一化方法,包括 `density=True`(概率密度)、`weights` 参数(自定义权重)等,可以将计数转换为概率密度或其他标准化形式。适合需要比较不同样本量数据或展示概率密度分布的场景。
<code_path>codebase/draw/gallery_python_new/distribution_plots/histogram_normalization.py</code_path>

**histogram_bihistogram.py** - 双向直方图
通过 `weights=-np.ones_like()` 创建向下的直方图,与向上直方图对比展示两个数据集的分布差异。使用 `ax.axhline(0)` 添加零线作为基准,适合可视化前后对比、实验对照组差异分析等场景。
<code_path>codebase/draw/gallery_python_new/distribution_plots/histogram_bihistogram.py</code_path>

**scatter_hist.py** - 散点+直方图组合
使用 `subplot_mosaic` 或 `GridSpec` 创建复合布局,主区域显示散点图展示变量关系,顶部和右侧添加边际直方图展示各变量的分布。这种布局在多变量数据分析和相关性研究中非常有用,能够同时提供关系和分布两个维度的信息。
<code_path>codebase/draw/gallery_python_new/distribution_plots/scatter_hist.py</code_path>

### 4. Heatmaps & Contours `codebase/draw/gallery_python_new/heatmaps_contours` (热图等高线 - 4个)

**contour_label_demo.py** - 等高线标注
使用 `ax.contour()` 和 `ax.clabel()` 创建带标签的等高线图,通过 `levels`(等高线数量或数值)和 `cmap`(颜色映射)参数控制样式。`clabel()` 函数可以在等高线上添加数值标签,通过 `inline=True`(标签在线上)和 `fontsize`(字体大小)等参数优化标签显示,适合地形图、气象图等科学可视化场景。
<code_path>codebase/draw/gallery_python_new/heatmaps_contours/contour_label_demo.py</code_path>

**contourf_demo.py** - 填充等高线
使用 `ax.contourf()` 创建填充颜色的等高线图,通过 `levels` 参数控制等高线密度,使用 `cmap` 设置连续的颜色映射。结合 `colorbar()` 可以添加颜色条来标识数值范围,适合可视化温度场、压力场、海拔分布等二维标量场数据。
<code_path>codebase/draw/gallery_python_new/heatmaps_contours/contourf_demo.py</code_path>

**pcolormesh_levels.py** - 伪彩色网格
使用 `ax.pcolormesh()` 绘制非均匀网格或规则网格的伪彩色图,通过 `shading` 参数(`'flat'`、`'nearest'`、`'auto'`、`'gouraud'`)控制渲染方式。结合 `BoundaryNorm` 和 `MaxNLocator` 可以精确控制颜色级别,适合大规模网格数据、地理信息系统(GIS)和科学计算结果的可视化。
<code_path>codebase/draw/gallery_python_new/heatmaps_contours/pcolormesh_levels.py</code_path>

**image_annotated_heatmap.py** - 标注热图
使用 `ax.imshow()` 创建热图,并通过嵌套循环和 `ax.text()` 在每个格子中添加数值标注。通过调整文本颜色(根据背景亮度自动选择黑/白)来确保可读性,适合展示相关性矩阵、混淆矩阵、距离矩阵等需要同时显示颜色编码和数值的场景。
<code_path>codebase/draw/gallery_python_new/heatmaps_contours/image_annotated_heatmap.py</code_path>

### 5. 3D Plots `codebase/draw/gallery_python_new/3d_plots` (三维图表 - 5个)

**scatter3d.py** - 3D散点图
使用 `projection='3d'` 创建三维坐标轴,通过 `ax.scatter(x, y, z)` 绘制三维散点图。可以通过 `c`(颜色映射)、`s`(标记大小)、`cmap`(颜色映射)和 `alpha`(透明度)等参数展示数据的第四维度,适合三维数据分布、空间聚类分析等场景。
<code_path>codebase/draw/gallery_python_new/3d_plots/scatter3d.py</code_path>

**surface3d.py** - 3D曲面图
使用 `ax.plot_surface(X, Y, Z)` 创建三维表面图,需要通过 `np.meshgrid()` 生成网格数据。通过 `cmap`(颜色映射)、`linewidth`(网格线宽)、`antialiased`(抗锯齿)、`alpha`(透明度)等参数控制表面效果,配合 `colorbar()` 显示数值-颜色对应关系,适合数学函数可视化、科学计算结果展示。
<code_path>codebase/draw/gallery_python_new/3d_plots/surface3d.py</code_path>

**wire3d.py** - 3D线框图
使用 `ax.plot_wireframe(X, Y, Z)` 创建三维线框图,通过 `linewidth`(线宽)和 `color`(颜色)参数控制网格线样式。线框图相比表面图更能看到背后的结构,适合几何形状展示、数学函数拓扑结构可视化等场景。
<code_path>codebase/draw/gallery_python_new/3d_plots/wire3d.py</code_path>

**lines3d.py** - 3D折线图
使用 `ax.plot(x, y, z)` 在三维空间中绘制折线,可以展示轨迹、路径或三维时间序列。通过 `linewidth`(线宽)、`linestyle`(线型)、`marker`(标记点)和 `color`(颜色)等参数自定义样式,适合粒子运动轨迹、飞行路径、三维曲线展示等场景。
<code_path>codebase/draw/gallery_python_new/3d_plots/lines3d.py</code_path>

**contour3d.py** - 3D等高线
使用 `ax.contour(X, Y, Z, zdir='z')` 在三维空间的指定平面上绘制等高线,通过 `offset` 参数控制等高线平面的位置。可以在XY、XZ、YZ平面上同时显示等高线,结合表面图或线框图使用,能够更清晰地理解三维数据场的结构。
<code_path>codebase/draw/gallery_python_new/3d_plots/contour3d.py</code_path>

### 6. Polar Plots `codebase/draw/gallery_python_new/polar_plots` (极坐标图 - 5个)

**polar_demo.py** - 极坐标基础
使用 `subplot_kw={'projection': 'polar'}` 创建极坐标轴,通过 `ax.plot(theta, r)` 绘制极坐标折线图。可以通过 `set_theta_zero_location`(设置0度位置)、`set_theta_direction`(角度方向)、`set_rmax`(最大半径)和 `fill`(填充区域)等参数自定义极坐标图,适合方向性数据、周期性数据和雷达图等场景。
<code_path>codebase/draw/gallery_python_new/polar_plots/polar_demo.py</code_path>

**polar_scatter.py** - 极坐标散点图
在极坐标系中使用 `ax.scatter(theta, r)` 绘制散点图,通过 `c`(颜色)、`s`(标记大小)、`cmap`(颜色映射)和 `alpha`(透明度)等参数展示多维度数据。适合风向数据分布、极坐标聚类、周期性模式识别等场景。
<code_path>codebase/draw/gallery_python_new/polar_plots/polar_scatter.py</code_path>

**polar_bar.py** - 极坐标柱状图
在极坐标系中使用 `ax.bar(theta, values)` 绘制柱状图,通过 `width`(柱宽)和 `bottom`(起始位置)参数控制柱子样式。可以创建类似南丁格尔玫瑰图的视觉效果,适合展示方向性强度、周期性统计量等数据。
<code_path>codebase/draw/gallery_python_new/polar_plots/polar_bar.py</code_path>

**pie_features.py** - 饼图特性
使用 `ax.pie(sizes, labels=labels)` 创建饼图,通过 `explode`(突出显示)、`autopct`(自动百分比)、`startangle`(起始角度)、`shadow`(阴影)、`colors`(颜色列表)等参数自定义样式。还可以通过 `pctdistance`(百分比标签距离)和 `labeldistance`(标签距离)精细调整布局,适合占比展示、构成分析等场景。
<code_path>codebase/draw/gallery_python_new/polar_plots/pie_features.py</code_path>

**nested_pie.py** - 嵌套饼图
通过多次调用 `ax.pie()` 并使用不同的 `radius`(半径)参数创建嵌套的饼图或环形图,可以展示分层数据或多级分类结构。通过调整每层的半径和宽度,可以创建优雅的嵌套效果,适合展示层级占比、多级分类数据等场景。
<code_path>codebase/draw/gallery_python_new/polar_plots/nested_pie.py</code_path>

### 7. Time Series `codebase/draw/gallery_python_new/time_series` (时间序列 - 4个)

**timeline.py** - 时间线图
使用 `matplotlib.dates` 模块处理日期数据,通过 `mdates.YearLocator()`、`MonthLocator()` 设置刻度定位器,使用 `DateFormatter()` 格式化日期显示。结合 `ax.vlines()` 可以在时间线上标记重要事件,适合项目进度展示、历史事件时间线、版本发布历史等场景。
<code_path>codebase/draw/gallery_python_new/time_series/timeline.py</code_path>

**time_series_histogram.py** - 时间序列直方图
创建基于时间序列的直方图,展示事件在时间维度上的分布特征。可以使用 `mdates` 模块处理日期格式的坐标轴,或使用 `np.histogram2d()` 创建二维时间-数值直方图,适合分析时间模式、周期性事件、活动高峰时段等场景。
<code_path>codebase/draw/gallery_python_new/time_series/time_series_histogram.py</code_path>

**fill_between_demo.py** - 填充区域图
使用 `ax.fill_between(x, y1, y2)` 在两条曲线之间填充颜色,通过 `alpha`(透明度)和 `color`(颜色)参数控制填充效果。可以用于表示置信区间、误差范围、目标区域等,配合 `where` 参数可以实现条件填充,适合不确定性可视化、趋势带、阈值范围展示等场景。
<code_path>codebase/draw/gallery_python_new/time_series/fill_between_demo.py</code_path>

**fill_betweenx_demo.py** - 横向填充区域图
使用 `ax.fill_betweenx(y, x1, x2)` 在水平方向上填充区域,与 `fill_between` 类似但是沿Y轴方向。适合需要在水平方向展示区间的场景,比如垂直剖面的数值范围、Y轴条件下的X值范围等。
<code_path>codebase/draw/gallery_python_new/time_series/fill_betweenx_demo.py</code_path>

### 8. Composite Plots `codebase/draw/gallery_python_new/composite_plots` (组合图表 - 3个)

**subplots_demo.py** - 多子图
演示创建多子图的各种方法,包括 `plt.subplots(nrows, ncols)`(规则网格)、`subplot_mosaic`(复杂布局)和 `GridSpec`(完全自定义布局)。通过 `sharex`、`sharey` 参数可以共享坐标轴,使用 `supxlabel`、`supylabel` 可以添加全局标签,适合多面板对比分析、多维度数据展示等场景。
<code_path>codebase/draw/gallery_python_new/composite_plots/subplots_demo.py</code_path>

**two_scales.py** - 双Y轴图
使用 `ax.twinx()` 创建共享X轴的双Y轴图表,左右两个Y轴可以有不同的刻度和标签。通过设置不同的颜色和 `tick_params` 可以区分两个数据系列,适合同时展示不同量级或单位的两个相关变量,如温度与降雨量、价格与交易量等。
<code_path>codebase/draw/gallery_python_new/composite_plots/two_scales.py</code_path>

**scatter_with_legend.py** - 散点图+图例
创建带图例的散点图,通过自定义图例的位置(`loc`)、样式(`frameon`、`shadow`)和布局(`ncol`)来优化图表可读性。可以手动添加图例项,使用自定义的颜色和标记,适合多类别数据可视化、分组数据展示等场景。
<code_path>codebase/draw/gallery_python_new/composite_plots/scatter_with_legend.py</code_path>

### 9. Images & Shapes `codebase/draw/gallery_python_new/images_shapes` (图像形状 - 2个)

**layer_images.py** - 图像叠加
使用 `ax.imshow()` 的 `alpha`(透明度)参数叠加多层图像,通过 `extent` 参数控制图像的位置和范围。可以通过调整每层的透明度来控制叠加效果,适合图像融合、前后对比、多通道数据叠加等场景。
<code_path>codebase/draw/gallery_python_new/images_shapes/layer_images.py</code_path>

**triinterp_demo.py** - 三角插值
使用 `matplotlib.tri` 模块进行三角网格插值,通过 `Triangulation` 创建非规则网格,使用 `tripcolor` 或 `tricontour` 在三角网格上绘制数据。适合地理信息系统、有限元分析、非规则采样数据的可视化等场景。
<code_path>codebase/draw/gallery_python_new/images_shapes/triinterp_demo.py</code_path>

### 10. Misc Plots `codebase/draw/gallery_python_new/misc_plots` (其他图表 - 2个)

**scales.py** - 坐标轴刻度
演示不同的坐标轴刻度类型,包括线性刻度(`'linear'`)、对数刻度(`'log'`)、对称对数刻度(`'symlog'`)、逻辑刻度(`'logit'`)等。通过 `set_xscale()` 和 `set_yscale()` 设置刻度类型,适合展示跨越多个数量级的数据、包含正负值的数据等特殊场景。
<code_path>codebase/draw/gallery_python_new/misc_plots/scales.py</code_path>

**table_demo.py** - 表格演示
使用 `plt.table()` 在图表中嵌入表格,可以设置单元格文本、颜色、边框等样式。表格可以放置在图表的顶部、底部或作为独立元素,适合在图表中展示统计数据、参数列表、汇总信息等场景。
<code_path>codebase/draw/gallery_python_new/misc_plots/table_demo.py</code_path>
