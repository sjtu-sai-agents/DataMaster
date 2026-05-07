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
