# utils.py
import math

# Camera CoC table (典型值，单位 mm). 你可自行扩展。
cameras = {
    "Nikon Z5": {"sensor":"Full Frame", "coc_mm": 0.03},
    "Sony A7 III": {"sensor":"Full Frame", "coc_mm": 0.03},
    "Canon EOS R": {"sensor":"Full Frame", "coc_mm": 0.03},
    "APS-C Example": {"sensor":"APS-C", "coc_mm": 0.019},
    "Micro Four Thirds": {"sensor":"MFT", "coc_mm": 0.015}
}

# Simple lens entries (min/max focal length in mm)
lenses = {
    "Nikkor 14-30mm f/4": {"min_f": 14, "max_f": 30},
    "Standard 50mm": {"min_f": 50, "max_f": 50},
    "Zoom 24-70mm": {"min_f": 24, "max_f": 70}
}

def hyperfocal_mm(f_mm, N, coc_mm):
    """
    超焦距 H(单位 mm)
    H = f^2 / (N*c) + f
    """
    return (f_mm * f_mm) / (N * coc_mm) + f_mm

def calc_dof_mm(f_mm, N, s_mm, coc_mm):
    """
    计算最近/最远清晰点，返回 (D_near_mm, D_far_mm)
    使用经典光学公式，单位：mm
    如果 D_far 为无限，返回 math.inf
    参数:
        f_mm: 焦距(mm)
        N: 光圈(f/值)
        s_mm: 对焦距离(mm)
        coc_mm: CoC(mm)
    """
    H = hyperfocal_mm(f_mm, N, coc_mm)
    # D_near = (H * s) / (H + (s - f))
    # D_far  = (H * s) / (H - (s - f))  if s < H else inf
    denom_near = (H + (s_mm - f_mm))
    if denom_near == 0:
        D_near = float('inf')
    else:
        D_near = (H * s_mm) / denom_near

    if s_mm < H:
        denom_far = (H - (s_mm - f_mm))
        if denom_far == 0:
            D_far = float('inf')
        else:
            D_far = (H * s_mm) / denom_far
    else:
        D_far = float('inf')

    return D_near, D_far

def format_distance_m(mm_val):
    import math
    if math.isinf(mm_val):
        return "∞"
    m = mm_val / 1000.0
    if m < 1.0:
        return f"{m*100:.1f} cm"
    if m < 1000:
        return f"{m:.3f} m"
    else:
        return f"{m/1000.0:.2f} km"
