"""Shared logic: feature engineering, SHAP explanation helpers and the personalised
recommendation engine. Used by train_and_explain.py and the Streamlit app."""
import numpy as np, pandas as pd

MARK_COLS = ["Mathematics Marks", "Programming Marks", "Database Marks", "Statistics Marks", "AI Marks"]
BEHAVIOUR = {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3}
BEHAVIOUR_INV = {v: k for k, v in BEHAVIOUR.items()}
FEATURES = ["Average Marks", "Attendance (%)", "Study Hours (per day)", "Behaviour", "Assignment Completed", "Other Activities"]
CLASSES = ["Low Risk", "Medium Risk", "High Risk"]          # fixed order -> label encoding 0/1/2
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
# maximum points each factor can add in the template's 'Risk Score Rules' sheet
RULE_WEIGHT = {"Average Marks": 3, "Attendance (%)": 3, "Study Hours (per day)": 2, "Behaviour": 2, "Assignment Completed": 2, "Other Activities": 1}


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw workbook columns -> the 6 model features. Personal data (name, email, phone, gender)
    is deliberately NOT used by the model."""
    return pd.DataFrame({
        "Average Marks": raw[MARK_COLS].mean(axis=1).round(2),
        "Attendance (%)": raw["Attendance (%)"],
        "Study Hours (per day)": raw["Study Hours (per day)"],
        "Behaviour": raw["Student Behaviour"].map(BEHAVIOUR),
        "Assignment Completed": (raw["Assignment Record"] == "Complete").astype(int),
        "Other Activities": (raw["Other Activities"] == "Present").astype(int),
    })[FEATURES]


def shap_for_class(explainer, X, class_id):
    """Return (n_samples, n_features) SHAP values for one class, whatever the SHAP version's layout."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        return np.asarray(sv[class_id])
    sv = np.asarray(sv)
    return sv[:, :, class_id] if sv.ndim == 3 else sv


def shap_risk_score(explainer, X):
    """SHAP values for the expected risk level  R = P(Medium) + 2*P(High)  (0 = surely Low, 2 = surely High).
    SHAP is additive, so phi_R = phi_Medium + 2*phi_High is an exact Shapley attribution of R and lets one
    explanation cover all three classes (a Medium student can have risk drivers even when P(High) is tiny)."""
    return shap_for_class(explainer, X, CLASS_ID["Medium Risk"]) + 2 * shap_for_class(explainer, X, CLASS_ID["High Risk"])


# ---------------------------------------------------------------- recommendations
def _weakest_subjects(raw_row, k=2):
    s = raw_row[MARK_COLS].astype(float).sort_values().head(k)
    return ", ".join(f"{i.replace(' Marks','')} ({int(v)})" for i, v in s.items())


def recommendation_for(feature, raw_row, feats_row):
    """One concrete, student-specific action for a risk-driving factor."""
    if feature == "Average Marks":
        return (f"Academic performance: average is {feats_row[feature]:.0f}%. Prioritise the weakest subjects - "
                f"{_weakest_subjects(raw_row)} - through doubt-clearing sessions, peer study groups and weekly practice tests. Target: average of 55%+ (ideally 65%+).")
    if feature == "Attendance (%)":
        a = feats_row[feature]
        tgt = 75 if a < 75 else 85
        return f"Attendance: currently {a:.0f}%. Attend every scheduled lecture/lab and inform the mentor about absences. Target: {tgt}%+."
    if feature == "Study Hours (per day)":
        h = feats_row[feature]
        return f"Study habits: only {h:.0f} h/day. Follow a fixed daily timetable with 2 focused sessions. Target: at least 3 h/day (4-5 h before exams)."
    if feature == "Behaviour":
        b = BEHAVIOUR_INV[int(feats_row[feature])]
        return f"Classroom behaviour ({b}): schedule a mentoring meeting, set participation goals and seek faculty feedback every two weeks."
    if feature == "Assignment Completed":
        return "Assignments: incomplete submissions. Use a deadline tracker, split work into weekly milestones and submit drafts early for feedback."
    if feature == "Other Activities":
        return "Engagement: not taking part in other activities. Join a technical club, project group or sports/cultural activity to build engagement."
    return ""


def apply_plan(feats_row, drivers):
    """What-if: set every risk-driving factor to its recommended target and return the new feature row."""
    r = feats_row.copy()
    for f in drivers:
        if f == "Average Marks": r[f] = max(r[f], 60)
        elif f == "Attendance (%)": r[f] = max(r[f], 85)
        elif f == "Study Hours (per day)": r[f] = max(r[f], 4)
        elif f == "Behaviour": r[f] = min(3, max(r[f] + 1, 2))
        elif f == "Assignment Completed": r[f] = 1
        elif f == "Other Activities": r[f] = 1
    return r


def explain_student(model, explainer, raw_row, feats_row, max_recs=3):
    """Full personalised report for one student (pandas Series inputs)."""
    X = feats_row.to_frame().T.astype(float)
    proba = model.predict_proba(X)[0]
    pred = CLASSES[int(np.argmax(proba))]
    sv = shap_risk_score(explainer, X)[0]          # contribution to the 0-2 risk score
    contrib = pd.Series(sv, index=FEATURES).sort_values(ascending=False)
    drivers = [f for f, v in contrib.items() if v > 0.05][:max_recs]
    protective = [f for f, v in contrib.sort_values().items() if v < -0.05][:3]
    recs = [recommendation_for(f, raw_row, feats_row) for f in drivers]
    if pred == "Low Risk" and not recs:
        recs = ["Keep up the current routine. Consider peer-mentoring juniors and aim for 65%+ in every subject."]
    after = None
    if drivers:
        p2 = model.predict_proba(apply_plan(feats_row, drivers).to_frame().T.astype(float))[0]
        after = {"risk": CLASSES[int(np.argmax(p2))], "p_high": float(p2[CLASS_ID["High Risk"]])}
    return {"predicted_risk": pred, "probabilities": dict(zip(CLASSES, proba.round(3))),
            "shap_risk": contrib, "drivers": drivers, "protective": protective,
            "recommendations": recs, "after_plan": after}
