# -*- coding: utf-8 -*-
"""
fields.py
=========
ASPECT 输出字段统一配置表

使用方法：
    from fields import ASPECT_FIELDS

    field_key = "shear_stress"
    info = ASPECT_FIELDS[field_key]
    print(info.name, info.unit, info.is_scalar)

特点：
    - 单位支持 LaTeX 格式
    - 自动判断标量/张量
    - 推荐绘图量（plot_quantity）便于后处理
    - 易于扩展、维护
"""

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class FieldInfo:
    """ASPECT 输出字段信息"""
    name: str                           # ASPECT 字段名
    unit: str                           # 单位（LaTeX 可读）
    is_scalar: bool                     # 是否为标量
    components: Optional[int] = None    # 张量分量数（标量为 None）
    plot_quantity: str = ""             # 推荐绘图的标量量
    description: str = ""               # 字段说明（可选）

# ================== ASPECT 常用字段配置表 ==================
ASPECT_FIELDS = {
    # 应力相关
    "stress": FieldInfo(
        name="stress",
        unit="Pa",
        is_scalar=True,
        plot_quantity="II (2nd invariant)",
        description="Total Cauchy stress tensor (2nd invariant)"
    ),
    "shear_stress": FieldInfo(
        name="shear_stress",
        unit="Pa",
        is_scalar=False,
        components=6,
        plot_quantity="|τ_xy|",
        description="Deviatoric stress tensor (xx, xy, xz, yy, yz, zz)"
    ),
    "principal_stress": FieldInfo(
        name="principal_stress",
        unit="Pa",
        is_scalar=False,
        components=3,
        plot_quantity="σ₁ (max principal)",
        description="Principal stresses (σ₁, σ₂, σ₃)"
    ),

    # 应变率
    "strain_rate": FieldInfo(
        name="strain_rate",
        unit="s^{-1}",
        is_scalar=False,
        components=6,
        plot_quantity="II (2nd invariant)",
        description="Strain rate tensor"
    ),

    # 速度与运动
    "velocity": FieldInfo(
        name="velocity",
        unit="m/s",
        is_scalar=False,
        components=3,
        plot_quantity="speed",
        description="Velocity vector (u, v, w)"
    ),

    # 热学与物质属性
    "temperature": FieldInfo(
        name="temperature",
        unit="K",
        is_scalar=True,
        plot_quantity="",
        description="Temperature"
    ),
    "viscosity": FieldInfo(
        name="viscosity",
        unit="Pa·s",
        is_scalar=True,
        plot_quantity="",
        description="Effective viscosity"
    ),
    "density": FieldInfo(
        name="density",
        unit="kg/m³",
        is_scalar=True,
        plot_quantity="",
        description="Density"
    ),

    # 其他常用
    "composition": FieldInfo(
        name="composition",
        unit="1",
        is_scalar=True,
        plot_quantity="",
        description="Compositional field (e.g., C1, C2)"
    ),
    "plastic_strain": FieldInfo(
        name="plastic_strain",
        unit="1",
        is_scalar=True,
        plot_quantity="",
        description="Accumulated plastic strain"
    ),
}

# ================== 辅助函数 ==================
def get_field_info(field_key: str) -> FieldInfo:
    """安全获取字段信息"""
    if field_key not in ASPECT_FIELDS:
        raise KeyError(f"Field '{field_key}' not found. Available: {list(ASPECT_FIELDS.keys())}")
    return ASPECT_FIELDS[field_key]

def list_fields() -> None:
    """打印所有字段信息（调试用）"""
    print("ASPECT 输出字段列表：")
    print("-" * 80)
    for key, info in ASPECT_FIELDS.items():
        comp = f"{info.components} components" if not info.is_scalar else "scalar"
        print(f"{key:18} | {info.unit:10} | {comp:15} | {info.plot_quantity or '—'}")
    print("-" * 80)

__all__ = [
    "ASPECT_FIELDS",
    "FieldInfo",
    "get_field",
    "list_fields"
]

if __name__ == "__main__":
    list_fields()
        # ================== 导出控制 ==================
