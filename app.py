# app.py
import streamlit as st
import json
import io
import math
import matplotlib.pyplot as plt
from datetime import datetime
# --- Interactive Plotly Depth of Field visualization ---
import plotly.graph_objects as go
import numpy as np

from utils import calc_dof_mm, hyperfocal_mm, format_distance_m, cameras, lenses
from utils import APERTURE

st.set_page_config(page_title="DOF Calculator — Nikon Z5 & More", layout="centered")

st.title("📷 景深计算器")
st.markdown("""
输入相机/镜头参数来计算最近/最远对焦距离、景深，并显示**景深分布曲线**。
支持相机/镜头 CoC 自动选择；支持导出/导入常用参数(preset JSON)。
""")

# --- Left: Presets: camera + lens
with st.sidebar:
    st.header("📁 Preset / 相机镜头选择")

    cam_names = list(cameras.keys())
    camera_choice = st.selectbox("选择相机(自动设定 CoC)", cam_names, index=cam_names.index("Nikon Z5") if "Nikon Z5" in cam_names else 0)
    cam_info = cameras[camera_choice]
    coc_default = cam_info["coc_mm"]

    lens_names = list(lenses.keys())
    # default lens if present
    default_lens = "Nikkor 14-30mm f/4" if "Nikkor 14-30mm f/4" in lenses else lens_names[0]
    lens_choice = st.selectbox("选择镜头(可选)", lens_names, index=lens_names.index(default_lens))
    lens_info = lenses[lens_choice]

    st.markdown("---")
    st.write("CoC(弥散圆)")
    coc = st.number_input("CoC (mm)", min_value=0.001, value=float(coc_default), format="%.4f")

    st.markdown("---")
    st.write("保存/加载 Preset")
    preset_name = st.text_input("Preset 名称(保存用)", value="my_preset")
    if st.button("导出当前参数为 Preset (下载 JSON)"):
        preset = {
            "name": preset_name or f"preset_{datetime.utcnow().isoformat()}",
            "camera": camera_choice,
            "lens": lens_choice,
            "coc_mm": coc,
            "params": {
                "f_mm": st.session_state.get("f_mm", 24),
                "aperture": st.session_state.get("aperture", 8.0),
                "focus_m": st.session_state.get("focus_m", 2.0),
                "shutter_s": st.session_state.get("shutter_s", 1/125),
                "iso": st.session_state.get("iso", 100)
            }
        }
        b = io.BytesIO(json.dumps(preset, ensure_ascii=False, indent=2).encode("utf-8"))
        st.download_button(label="下载 preset JSON", data=b, file_name=f"{preset['name']}.json", mime="application/json")

    uploaded = st.file_uploader("上传 preset JSON 来加载", type=["json"])
    if uploaded is not None:
        try:
            loaded = json.load(uploaded)
            # apply loaded values if present
            camera_choice = loaded.get("camera", camera_choice)
            lens_choice = loaded.get("lens", lens_choice)
            coc = float(loaded.get("coc_mm", coc))
            params = loaded.get("params", {})
            st.session_state["f_mm"] = float(params.get("f_mm", st.session_state.get("f_mm", 24)))
            st.session_state["aperture"] = float(params.get("aperture", st.session_state.get("aperture", 8.0)))
            st.session_state["focus_m"] = float(params.get("focus_m", 2.0))
            st.session_state["shutter_s"] = float(params.get("shutter_s", 1/125))
            st.session_state["iso"] = int(params.get("iso", 100))
            st.success("已加载 preset(并已应用到界面)")
        except Exception as e:
            st.error(f"加载失败: {e}")

# --- Main UI
st.header("参数输入")

col1, col2, col3 = st.columns(3)

# focus on session_state defaults to keep download preset workable
if "f_mm" not in st.session_state:
    st.session_state["f_mm"] = 24
if "aperture" not in st.session_state:
    st.session_state["aperture"] = 8.0
if "focus_m" not in st.session_state:
    st.session_state["focus_m"] = 2.0
if "shutter_s" not in st.session_state:
    st.session_state["shutter_s"] = 1/125
if "iso" not in st.session_state:
    st.session_state["iso"] = 100

# with col1:
#     f_mm = st.slider("焦距 f (mm)", min_value=int(lens_info.get("min_f",14)), max_value=int(lens_info.get("max_f",30)), value=int(st.session_state["f_mm"]))
#     st.session_state["f_mm"] = f_mm
#     st.write(f"镜头焦距范围：{lens_info.get('min_f')}–{lens_info.get('max_f')} mm")

with col1:
    f_mm = st.number_input(
        "焦距 f (mm)",
        min_value=float(lens_info.get("min_f", 14)),
        max_value=float(lens_info.get("max_f", 30)),
        value=float(st.session_state["f_mm"]),
        step=1,
    )
    st.session_state["f_mm"] = f_mm
    st.write(f"镜头焦距范围：{lens_info.get('min_f')}–{lens_info.get('max_f')} mm")


with col2:
    aperture = st.selectbox("光圈 (f/值)", options=APERTURE, index=APERTURE.index(st.session_state["aperture"]) if st.session_state["aperture"] in APERTURE else 4)
    st.session_state["aperture"] = aperture

with col3:
    focus_m = st.number_input("对焦距离 (m)", min_value=0.01, max_value=1e9, value=float(st.session_state["focus_m"]), step=0.1, format="%.2f")
    st.session_state["focus_m"] = focus_m

st.markdown("---")
st.subheader("曝光 / 曝光值 (EV) 计算")

col4, col5 = st.columns([1,1])
with col4:
    shutter_s = st.text_input("快门速度 (输入秒，例如 1/125 或 0.008)", value=str(st.session_state["shutter_s"]))
    # parse shutter
    try:
        if "/" in shutter_s:
            num, den = shutter_s.split("/")
            shutter_val = float(num) / float(den)
        else:
            shutter_val = float(shutter_s)
    except:
        shutter_val = float(st.session_state["shutter_s"])
    st.session_state["shutter_s"] = shutter_val

with col5:
    iso = st.number_input("ISO", min_value=50, max_value=51200, step=50, value=int(st.session_state["iso"]))
    st.session_state["iso"] = iso

# compute DOF and EV
H_mm = hyperfocal_mm(f_mm, aperture, coc)
H_m = H_mm / 1000.0

Dn_mm, Df_mm = calc_dof_mm(f_mm, aperture, focus_m*1000.0, coc)
# DOF numeric
if math.isinf(Df_mm):
    dof_m = float("inf")
else:
    dof_m = max(0.0, (Df_mm - Dn_mm) / 1000.0)

# EV calculations
# EV at ISO100: EV = log2(N^2 / t)
def ev_at_iso100(N, t_s):
    return math.log2((N * N) / t_s)

ev100 = ev_at_iso100(aperture, shutter_val)
# adjust for ISO
# Effective EV for chosen ISO relative to 100:
ev_iso = ev100 + math.log2(iso / 100.0)

st.markdown("### 结果")
st.write(f"选定相机：**{camera_choice}**(CoC={coc:.4f} mm) ； 选定镜头：**{lens_choice}**")
st.metric("超焦距 H", f"{H_m:.3f} m")
st.metric("最近对焦距离", format_distance_m(Dn_mm))
st.metric("最远对焦距离", "∞" if math.isinf(Df_mm) else format_distance_m(Df_mm))
st.metric("景深 DOF", "无限" if math.isinf(dof_m) else f"{dof_m:.3f} m")

st.markdown("**曝光信息**")
st.write(f"光圈 f/{aperture} · 快门 {shutter_val:.5g} s · ISO {iso}")
st.write(f"EV @ ISO100 = {ev100:.2f} ； 等效 EV (考虑 ISO) = {ev_iso:.2f}")

# Suggest alternative shutter or ISO for a target EV (optional)
st.markdown("建议：若要在 ISO 100 下得到相同曝光，需下列快门速度(给出基于当前光圈)")
# choose recommended shutter speeds near standard series
target_iso = 100
target_ev = ev_iso - math.log2(target_iso/100.0)  # EV at ISO100 for same exposure
recommended_shutters = []
common_shutters = [1/8000,1/4000,1/2000,1/1000,1/500,1/250,1/125,1/60,1/30,1/15,1/8,1/4,1/2,1,2,4,8,15,30]
for t in common_shutters:
    ev_t = ev_at_iso100(aperture, t)
    if abs(ev_t - target_ev) < 0.5:
        recommended_shutters.append(t)
if not recommended_shutters:
    # if none close, suggest the closest
    closest = min(common_shutters, key=lambda t: abs(ev_at_iso100(aperture,t)-target_ev))
    recommended_shutters = [closest]

st.write("建议快门速度(在 ISO100 下、相近曝光)：", ", ".join([f"{s:.5g}s" for s in recommended_shutters]))

st.markdown("---")
st.header("图形化：景深分布曲线")
st.write("横轴为对焦距离(m)，显示最近/最远清晰点(近、远)，以及 DOF 区间。")

# build curves
import numpy as np
x_m = np.logspace(math.log10(0.1), math.log10(100), 300)  # 0.1 m 到 100 m
x_mm = x_m * 1000.0
near_curve = np.zeros_like(x_mm)
far_curve = np.zeros_like(x_mm)
for i, s_mm in enumerate(x_mm):
    dn_mm, df_mm = calc_dof_mm(f_mm, aperture, s_mm, coc)
    near_curve[i] = dn_mm / 1000.0
    far_curve[i] = np.inf if math.isinf(df_mm) else df_mm / 1000.0

# # Plot using matplotlib (single plot)
# fig, ax = plt.subplots(figsize=(7,4))
# ax.set_xscale('log')
# ax.plot(x_m, near_curve, label='Nearest Point', linewidth=1)
# ax.plot(x_m, far_curve, label='Farthest Point', linewidth=1)
# # plot current focus point and its near/far
# ax.scatter([focus_m], [Dn_mm/1000.0], marker='o')
# ax.scatter([focus_m], [float('inf') if math.isinf(Df_mm) else Df_mm/1000.0], marker='o')
# # hyperfocal line
# ax.axhline(H_m, linestyle='--', linewidth=0.8, label=f'H = {H_m:.2f} m')
# ax.set_xlabel("Focus distance (m)")
# ax.set_ylabel("DoF (m)")
# ax.set_title("Depth of Field")
# ax.legend()
# ax.grid(True, which='both', ls=':', linewidth=0.4)
# st.pyplot(fig)

# st.markdown("---")
# st.caption("此工具基于常见光学公式。结果为理论值，实际可见清晰范围受拍摄目标、显示放大倍数及输出尺寸影响。")

# # Build hover info for each focus distance in x_m array
# hover_text = [
#     f"🔴 Focus point: {x:.2f} m<br>"
#     f"🔵 Near limit: {n:.2f} m<br>"
#     f"🟠 Far limit: {'∞' if math.isinf(fa) else f'{fa:.2f} m'}<br>"
#     f"🟦 DoF width: {'∞' if math.isinf(fa) else f'{(fa - n):.2f} m'}"
#     f"🟩 Hyperfocal: {H_m:.2f} m<br>"
#     for x, n, fa in zip(x_m, near_curve, far_curve)
# ]

# Convert values to meters
near = Dn_mm / 1000.0
far = float('inf') if math.isinf(Df_mm) else Df_mm / 1000.0
focus = focus_m
H = H_m

# Define plot range
x_min = max(0.01, near * 0.5)
x_max = (far if not math.isinf(far) else H * 2)
x = np.linspace(x_min, x_max, 500)
y = np.zeros_like(x)

# Create Plotly figure
fig = go.Figure()

# Fill in-focus region between near and far distances
if not math.isinf(far):
    fig.add_trace(go.Scatter(
        x=[near, far, far, near],
        y=[0, 0, 0.05, 0.05],
        fill='toself',
        fillcolor='rgba(135, 206, 250, 0.4)',  # light blue
        line=dict(color='rgba(135, 206, 250, 0)'),
        hoverinfo='skip',
        name='In-focus region'
    ))
else:
    # Extend fill region to hyperfocal distance * 2 if far is infinity
    fig.add_trace(go.Scatter(
        x=[near, focus+10, focus+10, near],
        y=[0, 0, 0.05, 0.05],
        fill='toself',
        fillcolor='rgba(135, 206, 250, 0.4)',
        line=dict(color='rgba(135, 206, 250, 0)'),
        hoverinfo='skip',
        name='In-focus region (extends to ∞)'
    ))

# Mark the focus point
fig.add_trace(go.Scatter(
    x=[focus], y=[0.025],
    mode='markers+text',
    name='Focus Point',
    marker=dict(color='red', size=10),
    text=[f"Focus\n{focus:.2f} m"],
    textposition='bottom center'
))

# Mark the near and far limits
fig.add_trace(go.Scatter(
    x=[near], y=[0.025],
    mode='markers+text',
    name='Near Limit',
    marker=dict(color='blue', size=8),
    text=[f"Near\n{near:.2f} m"],
    textposition='top center'
))

fig.add_trace(go.Scatter(
    x=[far if not math.isinf(far) else (focus+10)], y=[0.025],
    mode='markers+text',
    name='Far Limit',
    marker=dict(color='orange', size=8),
    text=[f"Far\n{'∞' if math.isinf(far) else f'{far:.2f} m'}"],
    textposition='top center'
))

# Add hyperfocal marker (optional)
fig.add_trace(go.Scatter(
    x=[H], y=[0.025],
    mode='markers+text',
    name='Hyperfocal Distance',
    marker=dict(color='green', size=8, symbol='triangle-up'),
    text=[f"H = {H:.2f} m"],
    textposition='bottom center'
))

# Layout configuration
fig.update_layout(
    title="📷 Depth of Field (1D View Along Camera Axis)",
    xaxis=dict(
        title="Distance from camera (m)",
        type="log",
        showgrid=True,
        gridcolor='lightgray',
        zeroline=False
    ),
    yaxis=dict(
        visible=False,
        range=[-0.02, 0.1]
    ),
    showlegend=False,
    margin=dict(l=40, r=40, t=60, b=40),
    template="plotly_white"
)

# # Add hoverable near/far curves for reference (optional)
# fig.add_trace(go.Scatter(
#     x=x_m, y=np.ones_like(x_m),
#     mode='lines',
#     name='Focus position reference',
#     line=dict(color='gray', width=1, dash='dot'),
#     text=hover_text,
#     hoverinfo='text'
# ))

# Display figure in Streamlit
st.plotly_chart(fig, use_container_width=True)
