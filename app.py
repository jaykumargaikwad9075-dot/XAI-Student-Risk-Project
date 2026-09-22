"""Explainable AI for Personalised Student Success and Academic Risk Prediction.
Industry-style Streamlit dashboard.  Run:  streamlit run app.py"""
import sys, json, joblib, numpy as np, pandas as pd, shap, streamlit as st
import plotly.graph_objects as go
import plotly.express as px
sys.path.insert(0, "src")
from xai_core import *

# ============================================================================ PAGE & THEME
st.set_page_config(page_title="XAI Student Risk Dashboard", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

NAVY, INK, MUTE = "#0F2547", "#1F2937", "#64748B"
ACCENT, ACCENT2 = "#3E6AE1", "#7C5CFC"
COL = {"Low Risk": "#1FA97C", "Medium Risk": "#E8A33D", "High Risk": "#D64550"}
COL_SOFT = {"Low Risk": "#E4F7EF", "Medium Risk": "#FDF2E1", "High Risk": "#FCE8E9"}
PLOT_FONT = dict(family="Inter, sans-serif", color=INK, size=12)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
h1,h2,h3,h4,h5 {{ font-family: 'Poppins', sans-serif; }}
.stApp {{ background: linear-gradient(180deg, #F4F7FC 0%, #EEF2FA 100%); }}
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {NAVY} 0%, #081327 100%); }}
section[data-testid="stSidebar"] * {{ color: #EAF0FB !important; }}
section[data-testid="stSidebar"] .stSlider label, section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label, section[data-testid="stSidebar"] .stMultiSelect label {{ color: #C7D4EE !important; font-weight: 500; }}
div[data-baseweb="tab-list"] {{ gap: 4px; }}
button[data-baseweb="tab"] {{ background: white; border-radius: 10px 10px 0 0 !important; padding: 10px 20px !important;
  font-weight: 600 !important; color: {MUTE} !important; border: 1px solid #E3E9F5 !important; border-bottom: none !important;}}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {NAVY} !important; background: white !important; border-bottom: 3px solid {ACCENT} !important; }}

.hero {{ background: linear-gradient(120deg, {NAVY} 0%, #1B3763 55%, {ACCENT} 140%); border-radius: 18px;
  padding: 26px 32px; margin-bottom: 18px; box-shadow: 0 10px 30px rgba(15,37,71,0.25); }}
.hero h1 {{ color: white; font-size: 1.55rem; margin: 0 0 6px 0; font-weight: 700; }}
.hero p {{ color: #CFE0FF; margin: 2px 0; font-size: 0.9rem; }}
.badge {{ display:inline-block; background: rgba(255,255,255,0.14); color:white; padding:3px 12px; border-radius: 20px;
  font-size: 0.74rem; margin-top:10px; margin-right:8px; border: 1px solid rgba(255,255,255,0.25); }}

.card {{ background: white; border-radius: 16px; padding: 20px 22px; box-shadow: 0 4px 18px rgba(15,37,71,0.06); border: 1px solid #EAEFF8; height: 100%; }}
.card h4 {{ margin: 0 0 12px 0; color: {INK}; font-size: 1.0rem; font-weight: 600; }}
.card h5 {{ margin: 0 0 10px 0; color: {MUTE}; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: .5px;}}

.kpi {{ background: white; border-radius: 14px; padding: 16px 18px; box-shadow: 0 3px 12px rgba(15,37,71,0.06); border: 1px solid #EAEFF8;
  border-left: 4px solid var(--kc, {ACCENT}); }}
.kpi .num {{ font-size: 1.6rem; font-weight: 700; color: {NAVY}; font-family:'Poppins',sans-serif; line-height:1.1;}}
.kpi .lab {{ font-size: 0.76rem; color: {MUTE}; margin-top: 4px; }}
.kpi .delta {{ font-size: 0.72rem; margin-top: 4px; font-weight:600; }}

.risk-pill {{ display:inline-block; padding: 7px 20px; border-radius: 999px; font-weight: 700; font-size: 1.0rem; letter-spacing: 0.3px; }}
.metric-box {{ background: #F7FAFF; border-radius: 12px; padding: 12px 14px; text-align:center; border: 1px solid #EAEFF8; }}
.metric-box .num {{ font-size: 1.2rem; font-weight: 700; color: {NAVY}; font-family:'Poppins',sans-serif;}}
.metric-box .lab {{ font-size: 0.7rem; color: {MUTE}; text-transform: uppercase; letter-spacing: 0.5px; margin-top:2px;}}

.rec {{ background: #F7FAFF; border-left: 4px solid {ACCENT}; border-radius: 8px; padding: 10px 14px; margin-bottom: 10px; font-size: 0.9rem; color:{INK};}}
.driver-chip {{ display:inline-block; background:#FCE8E9; color:#B23A45; padding:4px 12px; border-radius:20px; font-size:0.78rem; margin:3px; font-weight:500;}}
.protect-chip {{ display:inline-block; background:#E4F7EF; color:#127A56; padding:4px 12px; border-radius:20px; font-size:0.78rem; margin:3px; font-weight:500;}}
.tag {{ display:inline-block; background:#EEF2FA; color:{NAVY}; padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; margin-right:6px;}}
hr {{ border-color: #E3E9F5; }}
footer {{visibility:hidden;}} #MainMenu {{visibility:hidden;}}
</style>
""", unsafe_allow_html=True)


# ============================================================================ DATA / MODEL
@st.cache_resource
def load():
    bundle = joblib.load("outputs/risk_model.joblib")
    explainer = shap.TreeExplainer(bundle["model"])
    data = pd.read_csv("data/student_dataset_200.csv")
    metrics = json.load(open("outputs/metrics.json"))
    Xall = build_features(data)
    sv_all = shap_risk_score(explainer, Xall)
    proba_all = bundle["model"].predict_proba(Xall)
    preds_all = pd.Series(np.array(CLASSES)[proba_all.argmax(1)], index=data.index)
    return bundle["model"], explainer, data, bundle["name"], metrics, Xall, sv_all, proba_all, preds_all

model, explainer, data, model_name, metrics, Xall, sv_all, proba_all, preds_all = load()
N = len(data)

data_ext = data.copy()
data_ext["Predicted Risk"] = preds_all
data_ext["P(High Risk)"] = proba_all[:, CLASS_ID["High Risk"]].round(3)

def risk_pill(risk, size="1.0rem"):
    icon = "🟢" if risk == "Low Risk" else "🟡" if risk == "Medium Risk" else "🔴"
    return f"<span class='risk-pill' style='background:{COL_SOFT[risk]}; color:{COL[risk]}; font-size:{size};'>{icon} {risk}</span>"

def kpi(col, num, lab, color, delta=None):
    d = f"<div class='delta' style='color:{color}'>{delta}</div>" if delta else ""
    col.markdown(f"<div class='kpi' style='--kc:{color}'><div class='num'>{num}</div><div class='lab'>{lab}</div>{d}</div>", unsafe_allow_html=True)

# ============================================================================ HERO
st.markdown(f"""
<div class="hero">
  <h1>🎓 Explainable AI for Personalised Student Success &amp; Academic Risk Prediction</h1>
  <p>MIT School of Computing · Department of Computer Science &amp; Engineering · M.Sc. (LYMSC)</p>
  <p>Student: <b>Jaykumar Gaikwad</b> &nbsp;|&nbsp; Faculty Guide: <b>Dr. Ranjana Kale</b></p>
  <span class="badge">📊 {N} students monitored</span>
  <span class="badge">🤖 Model: {model_name}</span>
  <span class="badge">✅ {metrics['comparison'][2]['CV Accuracy']:.0%} cross-val accuracy</span>
  <span class="badge">🔍 Explained with SHAP</span>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_explorer, tab_model, tab_about = st.tabs(["🏠  Overview", "🔍  Student Explorer", "📊  Model Performance", "ℹ️  About"])

# ============================================================================ TAB 1 — OVERVIEW
with tab_overview:
    vc = data["Academic Risk"].value_counts().reindex(CLASSES)
    high_pct = vc["High Risk"] / N

    k1, k2, k3, k4, k5 = st.columns(5)
    kpi(k1, N, "Total Students", ACCENT)
    kpi(k2, f"{vc['High Risk']}", "High Risk", COL["High Risk"], f"{vc['High Risk']/N:.0%} of cohort")
    kpi(k3, f"{vc['Medium Risk']}", "Medium Risk", COL["Medium Risk"], f"{vc['Medium Risk']/N:.0%} of cohort")
    kpi(k4, f"{vc['Low Risk']}", "Low Risk", COL["Low Risk"], f"{vc['Low Risk']/N:.0%} of cohort")
    kpi(k5, f"{data['Attendance (%)'].mean():.0f}%", "Avg. Attendance", ACCENT2)

    st.write("")
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.markdown('<div class="card"><h5>Risk Distribution</h5>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(labels=vc.index, values=vc.values, hole=0.6, marker_colors=[COL[c] for c in vc.index],
                                textinfo="percent", textfont=dict(size=13, family="Inter"), sort=False))
        fig.update_layout(height=270, margin=dict(l=10, r=10, t=10, b=10), showlegend=True,
                           legend=dict(orientation="h", y=-0.08, font=dict(size=11)), paper_bgcolor="rgba(0,0,0,0)", font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><h5>What drives risk across the whole cohort</h5>', unsafe_allow_html=True)
        imp = pd.Series(np.abs(sv_all).mean(0), index=FEATURES).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=ACCENT,
                                text=[f"{v:.3f}" for v in imp.values], textposition="outside"))
        fig.update_layout(height=270, margin=dict(l=10, r=40, t=10, b=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(title="Mean |SHAP| (impact on risk score)", showgrid=True, gridcolor="#EEF2FA"),
                           yaxis=dict(showgrid=False), font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="card"><h5>Marks vs. Attendance, coloured by risk</h5>', unsafe_allow_html=True)
        plot_df = data.copy(); plot_df["Average Marks"] = Xall["Average Marks"]
        fig = px.scatter(plot_df, x="Attendance (%)", y="Average Marks", color="Academic Risk",
                          color_discrete_map=COL, category_orders={"Academic Risk": CLASSES}, opacity=0.75,
                          hover_data={"Student ID": True, "Attendance (%)": True, "Average Marks": ":.0f"})
        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           legend=dict(orientation="h", y=-0.22, title=None, font=dict(size=11)),
                           xaxis=dict(gridcolor="#EEF2FA"), yaxis=dict(gridcolor="#EEF2FA"), font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="card"><h5>Behaviour category vs. risk level</h5>', unsafe_allow_html=True)
        ct = pd.crosstab(data["Student Behaviour"], data["Academic Risk"]).reindex(["Poor", "Average", "Good", "Excellent"]).reindex(columns=CLASSES)
        fig = go.Figure()
        for cls in CLASSES:
            fig.add_trace(go.Bar(name=cls, x=ct.index, y=ct[cls], marker_color=COL[cls]))
        fig.update_layout(barmode="stack", height=300, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           legend=dict(orientation="h", y=-0.22, font=dict(size=11)), xaxis=dict(gridcolor="#EEF2FA"),
                           yaxis=dict(title="Students", gridcolor="#EEF2FA"), font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><h5>⚠️ Top 10 students needing the most attention (by P(High Risk))</h5>', unsafe_allow_html=True)
    top = data_ext.sort_values("P(High Risk)", ascending=False).head(10)[
        ["Student ID", "Student Name", "Predicted Risk", "P(High Risk)", "Attendance (%)", "Student Behaviour", "Assignment Record"]]
    st.dataframe(
        top.style.format({"P(High Risk)": "{:.0%}"}).map(lambda v: f"color:{COL.get(v,'inherit')}; font-weight:600" if v in COL else "", subset=["Predicted Risk"]),
        width='stretch', hide_index=True, height=390)
    st.caption("Open the **Student Explorer** tab and select any of these students to see the full explanation and recommended action plan.")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================ TAB 2 — STUDENT EXPLORER
with tab_explorer:
    st.sidebar.markdown("## 🧭 Student Input")
    mode = st.sidebar.radio("Choose input mode", ["📋 Pick from dataset", "✍️ Enter manually"])
    st.sidebar.markdown("---")
    if mode.startswith("📋"):
        risk_filter = st.sidebar.multiselect("Filter by predicted risk", CLASSES, default=CLASSES)
        options = data_ext[data_ext["Predicted Risk"].isin(risk_filter)]
        options = options["Student ID"] + " — " + options["Student Name"]
        if len(options) == 0:
            st.sidebar.warning("No students match this filter."); st.stop()
        sid = st.sidebar.selectbox("Select student", options)
        raw_row = data[data["Student ID"] == sid.split(" — ")[0]].iloc[0]
    else:
        st.sidebar.caption("Subject marks (0–100)")
        d = {}
        for c in MARK_COLS: d[c] = st.sidebar.slider(c.replace(" Marks", ""), 0, 100, 55)
        st.sidebar.caption("Engagement")
        d["Attendance (%)"] = st.sidebar.slider("Attendance (%)", 0, 100, 75)
        d["Study Hours (per day)"] = st.sidebar.slider("Study hours / day", 0, 8, 3)
        d["Student Behaviour"] = st.sidebar.selectbox("Behaviour", list(BEHAVIOUR), index=1)
        d["Assignment Record"] = st.sidebar.selectbox("Assignment record", ["Complete", "Incomplete"])
        d["Other Activities"] = st.sidebar.selectbox("Other activities", ["Present", "Not Present"])
        raw_row = pd.Series(d)
    st.sidebar.markdown("---")
    st.sidebar.caption("Personal identifiers and gender are excluded from the model — predictions use only academic and engagement factors.")

    feats = build_features(raw_row.to_frame().T.astype({c: float for c in MARK_COLS + ["Attendance (%)", "Study Hours (per day)"]})).iloc[0]
    rep = explain_student(model, explainer, raw_row, feats)
    pred, proba = rep["predicted_risk"], rep["probabilities"]
    name_label = raw_row.get("Student Name", "This student")
    sid_label = raw_row.get("Student ID", "")

    st.markdown(f"#### {name_label}  <span class='tag'>{sid_label}</span>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.1, 1.3, 1.1])
    with c1:
        st.markdown('<div class="card"><h4>Predicted Risk</h4>', unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; margin: 6px 0 14px 0;'>{risk_pill(pred)}</div>", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(mode="gauge+number", value=proba["High Risk"] * 100,
            number={"suffix": "%", "font": {"size": 30, "color": NAVY, "family": "Poppins"}},
            gauge={"axis": {"range": [0, 100], "tickcolor": "#94A3B8"}, "bar": {"color": COL[pred], "thickness": 0.28}, "bgcolor": "white", "borderwidth": 0,
                   "steps": [{"range": [0, 33], "color": "#EAF9F1"}, {"range": [33, 66], "color": "#FEF6E7"}, {"range": [66, 100], "color": "#FDECED"}]}))
        fig.update_layout(height=190, margin=dict(l=18, r=18, t=6, b=6), paper_bgcolor="rgba(0,0,0,0)", font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.caption("Probability of High Risk")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h4>Class Probabilities</h4>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=list(proba.values()), y=list(proba.keys()), orientation="h", marker_color=[COL[k] for k in proba],
                                text=[f"{v:.0%}" for v in proba.values()], textposition="outside"))
        fig.update_layout(height=190, margin=dict(l=10, r=30, t=6, b=6), xaxis=dict(range=[0, 1.15], tickformat=".0%", showgrid=False),
                           yaxis=dict(showgrid=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><h4>Student Snapshot</h4>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"<div class='metric-box'><div class='num'>{feats['Average Marks']:.0f}%</div><div class='lab'>Avg Marks</div></div>", unsafe_allow_html=True)
            st.markdown(f"<br><div class='metric-box'><div class='num'>{feats['Study Hours (per day)']:.0f} h</div><div class='lab'>Study/day</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-box'><div class='num'>{feats['Attendance (%)']:.0f}%</div><div class='lab'>Attendance</div></div>", unsafe_allow_html=True)
            st.markdown(f"<br><div class='metric-box'><div class='num'>{BEHAVIOUR_INV[int(feats['Behaviour'])]}</div><div class='lab'>Behaviour</div></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    c4, c5 = st.columns([1.6, 1])
    with c4:
        st.markdown('<div class="card"><h4>🔍 Why this prediction? (SHAP)</h4>', unsafe_allow_html=True)
        s = rep["shap_risk"].sort_values()
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                                marker_color=[COL["High Risk"] if v > 0 else COL["Low Risk"] for v in s.values],
                                text=[f"{v:+.3f}" for v in s.values], textposition="outside"))
        fig.add_vline(x=0, line_width=1, line_color="#94A3B8")
        fig.update_layout(height=260, margin=dict(l=10, r=40, t=10, b=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(title="Contribution to risk score (red = raises risk, green = lowers it)", showgrid=True, gridcolor="#EEF2FA"),
                           yaxis=dict(showgrid=False), font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="card"><h4>Risk Drivers &amp; Protective Factors</h4>', unsafe_allow_html=True)
        if rep["drivers"]:
            st.markdown("**⚠️ Driving risk up:**")
            st.markdown("".join(f"<span class='driver-chip'>{f}</span>" for f in rep["drivers"]), unsafe_allow_html=True)
        else:
            st.markdown("_No significant risk drivers._")
        st.write("")
        if rep["protective"]:
            st.markdown("**🛡️ Protecting this student:**")
            st.markdown("".join(f"<span class='protect-chip'>{f}</span>" for f in rep["protective"]), unsafe_allow_html=True)
        else:
            st.markdown("_No strong protective factors yet._")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    c6, c7 = st.columns([1.5, 1])
    with c6:
        st.markdown(f'<div class="card"><h4>💡 Personalised Recommendations</h4>', unsafe_allow_html=True)
        for r in rep["recommendations"]:
            st.markdown(f"<div class='rec'>{r}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c7:
        st.markdown('<div class="card"><h4>🔮 What-if: follow the plan</h4>', unsafe_allow_html=True)
        if rep["after_plan"]:
            before_p, after_p = proba["High Risk"], rep["after_plan"]["p_high"]
            fig = go.Figure(go.Bar(x=["Now", "After plan"], y=[before_p, after_p], marker_color=[COL["High Risk"], COL["Low Risk"]],
                                    text=[f"{before_p:.0%}", f"{after_p:.0%}"], textposition="outside"))
            fig.update_layout(height=210, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               yaxis=dict(title="P(High Risk)", tickformat=".0%", range=[0, 1.1], showgrid=True, gridcolor="#EEF2FA"), font=PLOT_FONT)
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
            st.success(f"Predicted level moves to **{rep['after_plan']['risk']}** if these targets are met.")
            st.caption("Model-based estimate of association, not a guarantee.")
        else:
            st.info("Already on track — no plan needed.")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================ TAB 3 — MODEL PERFORMANCE
with tab_model:
    comp = pd.DataFrame(metrics["comparison"])
    best = comp[comp.Model == model_name].iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, model_name, "Selected Model", ACCENT)
    kpi(k2, f"{best['CV Accuracy']:.1%}", "5-fold CV Accuracy", COL["Low Risk"], f"± {best['CV Accuracy SD']*100:.1f} pts")
    kpi(k3, f"{best['CV Macro-F1']:.3f}", "CV Macro-F1", ACCENT2)
    kpi(k4, f"{metrics['external_validation']['accuracy']:.1%}", "External Validation Accuracy", COL["Medium Risk"], "on 250-student template set")

    st.write("")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown('<div class="card"><h5>Model comparison (5-fold cross-validation)</h5>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="CV Accuracy", x=comp.Model, y=comp["CV Accuracy"], marker_color=ACCENT,
                              text=[f"{v:.0%}" for v in comp["CV Accuracy"]], textposition="outside"))
        fig.add_trace(go.Bar(name="CV Macro-F1", x=comp.Model, y=comp["CV Macro-F1"], marker_color=ACCENT2,
                              text=[f"{v:.3f}" for v in comp["CV Macro-F1"]], textposition="outside"))
        fig.update_layout(barmode="group", height=320, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           yaxis=dict(range=[0, 1.15], gridcolor="#EEF2FA"), legend=dict(orientation="h", y=-0.18), font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.caption(f"**{model_name}** was selected: best cross-validated macro-F1 among the tree ensembles, which also support exact SHAP attribution.")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h5>Hold-out confusion matrix</h5>', unsafe_allow_html=True)
        cm = np.array(metrics["confusion_matrix"])
        labels = [c.replace(" Risk", "") for c in CLASSES]
        fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale=[[0, "#F4F7FC"], [1, ACCENT]],
                                    text=cm, texttemplate="%{text}", textfont=dict(size=16, color=INK), showscale=False))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(title="Predicted", side="bottom"), yaxis=dict(title="Actual", autorange="reversed"), font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.caption("40-student stratified hold-out set (not used in cross-validation).")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="card"><h5>Global feature importance (mean |SHAP|)</h5>', unsafe_allow_html=True)
        shap_imp = pd.Series(metrics["shap_mean_abs"]).sort_values()
        fig = go.Figure(go.Bar(x=shap_imp.values, y=shap_imp.index, orientation="h", marker_color=ACCENT,
                                text=[f"{v:.3f}" for v in shap_imp.values], textposition="outside"))
        fig.update_layout(height=300, margin=dict(l=10, r=40, t=10, b=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(gridcolor="#EEF2FA"), yaxis=dict(showgrid=False), font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="card"><h5>Explanation fidelity check</h5>', unsafe_allow_html=True)
        st.markdown(f"""
        <p style='font-size:0.9rem; color:{INK}; line-height:1.6;'>
        The dataset's labels come from a known scoring rule, so that rule's factor weights act as ground truth.
        The SHAP importance ranking correlates strongly with those weights:</p>
        <div style='text-align:center; margin: 18px 0;'>
          <div style='font-size:2.2rem; font-weight:700; color:{NAVY}; font-family:Poppins;'>ρ = {metrics['spearman_rho']:.2f}</div>
          <div style='color:{MUTE}; font-size:0.82rem;'>Spearman correlation (p = {metrics['spearman_p']:.3f})</div>
        </div>
        <p style='font-size:0.85rem; color:{MUTE};'>This suggests the model's explanations are consistent with the true drivers of risk, not just accurate predictions.</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><h5>Per-class metrics (hold-out set)</h5>', unsafe_allow_html=True)
    hr = pd.DataFrame(metrics["holdout_report"]).T.loc[CLASSES, ["precision", "recall", "f1-score", "support"]]
    hr.index.name = "Risk Level"
    st.dataframe(hr.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"})
                 .background_gradient(subset=["precision", "recall", "f1-score"], cmap="Blues", vmin=0.7, vmax=1.0),
                 width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================ TAB 4 — ABOUT
with tab_about:
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown('<div class="card"><h4>About this project</h4>', unsafe_allow_html=True)
        st.markdown(f"""
        <p style='color:{INK}; line-height:1.7; font-size:0.94rem;'>
        Many institutions rely on marks, attendance, assignment and engagement data, but identifying students who are
        academically at risk early is difficult, and traditional methods give little insight into <i>why</i> a student
        is at risk. This project builds an <b>Explainable AI (XAI)</b> system that predicts a student's academic risk
        level, explains the prediction with <b>SHAP</b> (Shapley Additive Explanations), and converts that explanation
        into personalised, actionable recommendations.</p>
        <p style='color:{INK}; line-height:1.7; font-size:0.94rem;'>
        Six factors are used: average marks, attendance, study hours, behaviour, assignment completion and
        participation in other activities. Personal identifiers and gender are deliberately excluded from the model.</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="card"><h4>How to read this dashboard</h4>', unsafe_allow_html=True)
        st.markdown(f"""
        <ul style='color:{INK}; line-height:1.9; font-size:0.92rem;'>
          <li><b>Overview</b> — cohort-wide KPIs, risk distribution, global risk drivers, and the students needing the most attention.</li>
          <li><b>Student Explorer</b> — pick any student (or enter data manually) to see their predicted risk, a SHAP-based explanation, personalised recommendations, and a what-if simulation.</li>
          <li><b>Model Performance</b> — how the model was selected, validated and how faithful its explanations are.</li>
        </ul>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h4>Project details</h4>', unsafe_allow_html=True)
        rows = [("Institution", "MIT School of Computing"), ("Department", "Computer Science & Engineering"),
                ("Programme", "M.Sc. (LYMSC) — Part I"), ("Student", "Jaykumar Gaikwad"), ("Faculty Guide", "Dr. Ranjana Kale"),
                ("Dataset", f"{N} students (synthetic)"), ("Model", model_name), ("Explainability", "SHAP (TreeExplainer)")]
        for k, v in rows:
            st.markdown(f"<div style='display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #EEF2FA; font-size:0.88rem;'>"
                         f"<span style='color:{MUTE};'>{k}</span><span style='color:{INK}; font-weight:600;'>{v}</span></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="card"><h4>⚠️ Responsible use</h4>', unsafe_allow_html=True)
        st.markdown(f"""<p style='color:{MUTE}; font-size:0.85rem; line-height:1.6;'>
        Predictions are trained on synthetic data and should support a mentor's judgement, not replace it.
        The what-if simulation is a model-based estimate of association, not a guarantee of outcome.
        Risk labels should not be shown to students without guidance and context.</p>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"<p style='text-align:center; color:#94A3B8; font-size:0.8rem; margin-top:24px;'>"
            f"XAI Student Risk Dashboard · Model: {model_name} · For academic use only — predictions support, not replace, mentor judgement.</p>", unsafe_allow_html=True)
