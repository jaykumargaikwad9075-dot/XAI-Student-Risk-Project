"""Train, evaluate and explain the academic-risk model. Produces charts, metrics and
per-student personalised reports in ./outputs."""
import json, warnings, joblib, numpy as np, pandas as pd, shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from xai_core import *
warnings.filterwarnings("ignore")
SEED = 42
NAVY, RED, AMBER, GREEN = "#1F4E78", "#C0392B", "#E6A100", "#2E8B57"
COL = {"Low Risk": GREEN, "Medium Risk": AMBER, "High Risk": RED}
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

raw = pd.read_csv("data/student_dataset_200.csv")
X, y = build_features(raw), raw["Academic Risk"].map(CLASS_ID)

# ------------------------------------------------------------------ 1. model comparison
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}
Xtr, Xte, ytr, yte, raw_tr, raw_te = train_test_split(X, y, raw, test_size=0.2, stratify=y, random_state=SEED)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows = []
for name, m in models.items():
    s = cross_validate(m, X, y, cv=cv, scoring=["accuracy", "f1_macro"])
    m.fit(Xtr, ytr); p = m.predict(Xte)
    rows.append({"Model": name, "CV Accuracy": s["test_accuracy"].mean(), "CV Accuracy SD": s["test_accuracy"].std(),
                 "CV Macro-F1": s["test_f1_macro"].mean(), "Test Accuracy": accuracy_score(yte, p), "Test Macro-F1": f1_score(yte, p, average="macro")})
comp = pd.DataFrame(rows).round(4)
comp.to_csv("outputs/model_comparison.csv", index=False)
print(comp.to_string(index=False))

# best explainable tree ensemble is used for SHAP (TreeExplainer is exact for trees)
tree_only = comp[comp.Model.isin(["Random Forest", "Gradient Boosting"])]
best_name = tree_only.sort_values(["CV Macro-F1", "Test Macro-F1"], ascending=False).iloc[0]["Model"]
model = models[best_name]
print("Selected model:", best_name)
pred_te = model.predict(Xte)
cm = confusion_matrix(yte, pred_te)
report = classification_report(yte, pred_te, target_names=CLASSES, output_dict=True)
print(classification_report(yte, pred_te, target_names=CLASSES))

# ------------------------------------------------------------------ 2. external validation on the supplied 250-student template
ext_path = "/mnt/user-data/uploads/Explainable_AI_Personalised_Student_Academic_Risk_Prediction_Dataset_Final__1_.xlsx"
ext = pd.read_excel(ext_path, sheet_name="Student Dataset")
Xe, ye = build_features(ext), ext["Academic Risk"].map(CLASS_ID)
pe = model.predict(Xe)
ext_res = {"n": len(ext), "accuracy": accuracy_score(ye, pe), "macro_f1": f1_score(ye, pe, average="macro")}
print("External (250-row template):", ext_res)

# final model = fit on all 200 students for deployment / reporting
model.fit(X, y)
joblib.dump({"model": model, "features": FEATURES, "classes": CLASSES, "name": best_name}, "outputs/risk_model.joblib")

# ------------------------------------------------------------------ 3. SHAP explanations
explainer = shap.TreeExplainer(model)
sv_high = shap_risk_score(explainer, X)
mean_abs = pd.Series(np.abs(sv_high).mean(0), index=FEATURES).sort_values()

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh(mean_abs.index, mean_abs.values, color=NAVY)
ax.set_xlabel("Mean |SHAP value| (impact on risk score, 0 = Low ... 2 = High)"); ax.set_title("Global feature importance", loc="left", fontweight="bold")
for i, v in enumerate(mean_abs.values): ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout(); plt.savefig("outputs/fig_shap_global.png", dpi=200); plt.close()

plt.figure(figsize=(7.5, 4.2))
shap.summary_plot(sv_high, X, feature_names=FEATURES, show=False, plot_size=None)
plt.title("How each factor moves a student's risk score", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig("outputs/fig_shap_beeswarm.png", dpi=200, bbox_inches="tight"); plt.close()

# explanation fidelity: does SHAP rank agree with the known rule weights?
cmp_df = pd.DataFrame({"Mean |SHAP|": mean_abs, "Rule max points": pd.Series(RULE_WEIGHT)}).loc[mean_abs.index]
rho, pval = spearmanr(cmp_df["Mean |SHAP|"], cmp_df["Rule max points"])
cmp_df.to_csv("outputs/shap_vs_rule_weights.csv")
print(cmp_df, "\nSpearman rho =", round(rho, 3), "p =", round(pval, 3))

# ------------------------------------------------------------------ 4. personalised reports for every student
recs_rows, cache = [], {}
for i in range(len(raw)):
    rep = explain_student(model, explainer, raw.iloc[i], X.iloc[i])
    cache[i] = rep
    recs_rows.append({"Student ID": raw.at[i, "Student ID"], "Student Name": raw.at[i, "Student Name"], "Actual Risk (rule-based)": raw.at[i, "Academic Risk"],
                      "Predicted Risk": rep["predicted_risk"], "P(Low)": rep["probabilities"]["Low Risk"], "P(Medium)": rep["probabilities"]["Medium Risk"], "P(High)": rep["probabilities"]["High Risk"],
                      "Top Risk Drivers": "; ".join(rep["drivers"]) or "None", "Protective Factors": "; ".join(rep["protective"]) or "None",
                      "Personalised Recommendations": " | ".join(rep["recommendations"]),
                      "Risk After Following Plan": rep["after_plan"]["risk"] if rep["after_plan"] else rep["predicted_risk"]})
preds = pd.DataFrame(recs_rows)
preds.to_csv("outputs/student_predictions_and_recommendations.csv", index=False)
moved = preds[preds["Predicted Risk"].isin(["High Risk", "Medium Risk"])]
improve = (moved["Risk After Following Plan"].map(CLASS_ID) < moved["Predicted Risk"].map(CLASS_ID)).mean()
print(f"At-risk students whose predicted level drops after following the plan: {improve:.1%}")

# local explanation figures: one example per class
def local_plot(idx, path):
    rep = cache[idx]; s = rep["shap_risk"].sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    vals = s.values
    ax.barh([f"{f} = {X.iloc[idx][f]:.0f}" if f != "Behaviour" else f"Behaviour = {BEHAVIOUR_INV[int(X.iloc[idx][f])]}" for f in s.index], vals,
            color=[RED if v > 0 else GREEN for v in vals])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP contribution to risk score (red = raises risk, green = lowers risk)")
    ax.set_title(f"{raw.at[idx,'Student ID']} - predicted {rep['predicted_risk']} (P(High)={rep['probabilities']['High Risk']:.2f})", loc="left", fontweight="bold", color=COL[rep["predicted_risk"]])
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
examples = {}
for cls in CLASSES:
    cand = preds[(preds["Predicted Risk"] == cls) & (preds["Actual Risk (rule-based)"] == cls)]
    idx = int(cand.index[len(cand)//2]); examples[cls] = idx
    local_plot(idx, f"outputs/fig_local_{cls.split()[0].lower()}.png")

# ------------------------------------------------------------------ 5. remaining figures
fig, ax = plt.subplots(figsize=(5.5, 3.6))
vc = raw["Academic Risk"].value_counts().reindex(CLASSES)
b = ax.bar(vc.index, vc.values, color=[COL[c] for c in vc.index])
for r, v in zip(b, vc.values): ax.text(r.get_x() + r.get_width()/2, v + 1.5, f"{v} ({v/len(raw):.0%})", ha="center")
ax.set_ylabel("Students"); ax.set_title("Class distribution (200 students)", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig("outputs/fig_class_distribution.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(7, 3.8)); w = 0.38; xs = np.arange(len(comp))
ax.bar(xs - w/2, comp["CV Accuracy"], w, label="5-fold CV accuracy", color=NAVY)
ax.bar(xs + w/2, comp["Test Macro-F1"], w, label="Hold-out macro-F1", color="#7FA7CF")
ax.set_xticks(xs); ax.set_xticklabels(comp.Model, rotation=12); ax.set_ylim(0.6, 1.1); ax.legend(loc="upper center", ncol=2, frameon=False)
ax.set_title("Model comparison", loc="left", fontweight="bold"); plt.tight_layout(); plt.savefig("outputs/fig_model_comparison.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(4.6, 4))
ConfusionMatrixDisplay(cm, display_labels=[c.replace(" Risk", "") for c in CLASSES]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"{best_name}: hold-out confusion matrix", fontsize=10, fontweight="bold"); plt.tight_layout(); plt.savefig("outputs/fig_confusion_matrix.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.scatter(cmp_df["Rule max points"], cmp_df["Mean |SHAP|"], s=70, color=NAVY)
for f, r in cmp_df.iterrows(): ax.annotate(f, (r["Rule max points"], r["Mean |SHAP|"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel("Weight in Risk Score Rules (max points)"); ax.set_ylabel("Mean |SHAP|"); ax.set_xlim(0.5, 3.9)
ax.set_title(f"Explanation fidelity (Spearman rho = {rho:.2f})", loc="left", fontweight="bold"); plt.tight_layout(); plt.savefig("outputs/fig_fidelity.png", dpi=200); plt.close()

json.dump({"selected_model": best_name, "comparison": comp.to_dict("records"), "confusion_matrix": cm.tolist(),
           "holdout_report": {k: v for k, v in report.items()}, "external_validation": ext_res,
           "shap_mean_abs": mean_abs.sort_values(ascending=False).round(4).to_dict(), "spearman_rho": rho, "spearman_p": pval,
           "plan_improves_pct": float(improve), "example_students": {k: raw.at[v, "Student ID"] for k, v in examples.items()},
           "class_counts": raw["Academic Risk"].value_counts().to_dict()}, open("outputs/metrics.json", "w"), indent=2, default=float)
print("done")
