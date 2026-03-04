
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

custom_colormap = {
    "Upper Continental Crust": (0.92, 0.85, 0.74),
    "Lower Continental Crust": (0.67, 0.83, 0.90),
    "Sediments": (0.72, 0.53, 0.42),
    "Ocean Crust": (0.34, 0.73, 0.50),
    "Upper Mantle": (0.55, 0.65, 0.80),
    "Lower Mantle": (0.35, 0.45, 0.65),
    "Weak Zone": (0.99, 0.96, 0.91),
    "Lithosphere Mantle": (0.85, 0.85, 0.85)
}


fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('off')

# 创建带颜色的色块（此时先不加边框，或加了也可）
handles = [mpatches.Patch(color=color, label=key) for key, color in custom_colormap.items()]

legend = ax.legend(handles=handles, loc='center', ncol=4,
                   frameon=False, fontsize=12,
                   handlelength=1.0, handleheight=1.0)

# 强制为每个色块添加黑色边框
for patch in legend.get_patches():
    patch.set_edgecolor('black')
    patch.set_linewidth(1)   # 可调整粗细

plt.tight_layout()
plt.savefig('colormap_legend.png', transparent=True, bbox_inches='tight', dpi=300)