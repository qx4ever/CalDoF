# DOF Calculator（Streamlit）  
景深计算器，支持：图形化景深分布、相机/镜头 CoC 自动选择、导出/导入 Preset（JSON）、曝光（EV）计算。  

## 功能
- 计算超焦距、最近/最远对焦点、景深范围
- 图形化显示：随对焦距离变化的 D_near / D_far 曲线
- 相机/镜头 CoC 自动选择（可手工覆盖）
- 导出/导入常用参数（preset JSON）
- 曝光值（EV）计算与快门/ISO 建议

## 本地运行
1. 克隆本仓库
2. 创建虚拟环境并安装依赖：
```bash
python -m venv .venv
source .venv/bin/activate   
pip install -r requirements.txt
```
3. 运行
```shell
streamlit run app.py
```

