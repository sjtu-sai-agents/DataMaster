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
