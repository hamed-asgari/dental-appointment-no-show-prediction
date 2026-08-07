from __future__ import annotations

import pandas as pd
import streamlit as st

from app.dashboard_data import (
    DashboardIntegrityError,
    gate_rows,
    load_dashboard_data,
    top_feature_rows,
)


st.set_page_config(
    page_title="Dental No-show Model Evaluation",
    page_icon="🦷",
    layout="wide",
)

st.title("Dental Appointment No-show Prediction")
st.subheader("Transparent model-evaluation dashboard")

st.warning(
    "Synthetic-data-only portfolio project. This model is not validated for "
    "clinical or operational use and this dashboard does not provide "
    "individualized patient or appointment risk estimates."
)

try:
    data = load_dashboard_data()
except DashboardIntegrityError as exc:
    st.error(
        "Dashboard integrity validation failed. Frozen R3 evidence will not "
        f"be rendered. Details: {exc}"
    )
    st.stop()

summary = data.summary
final_test = summary["final_test"]
baseline = summary["population_prior_baseline"]

st.markdown(
    """
This application presents the already-opened, frozen Version 2 evaluation
evidence. It does **not** refit the model, access protected targets, score new
appointments, select an operating threshold, or recommend an intervention.
"""
)

st.header("1. Evidence-based application decision")
st.info(
    "`transparent_model_evaluation_dashboard` was selected by the pre-frozen "
    "four-part gate. An appointment-level risk demonstration required every "
    "gate to pass."
)

gate_frame = pd.DataFrame(gate_rows(summary))
gate_frame["Result"] = gate_frame["Pass"].map(
    {True: "PASS", False: "FAIL"}
)
st.dataframe(
    gate_frame[["Gate", "Observed", "Requirement", "Result"]],
    hide_index=True,
    use_container_width=True,
)

st.error(
    "The Brier-score gate failed: the model Brier score "
    f"({final_test['brier_score']:.6f}) was slightly worse than the frozen "
    f"population-prior baseline ({baseline['brier_score']:.6f}). Therefore "
    "appointment-level risk presentation is not authorized."
)

st.header("2. Protected final-test performance")
row1 = st.columns(4)
row1[0].metric("Appointments", f"{final_test['sample_size']:,}")
row1[1].metric("No-shows", f"{final_test['positive_count']:,}")
row1[2].metric("Prevalence", f"{final_test['prevalence']:.3%}")
row1[3].metric(
    "Mean predicted probability",
    f"{final_test['mean_predicted_probability']:.3%}",
)

row2 = st.columns(4)
row2[0].metric(
    "Average Precision",
    f"{final_test['average_precision']:.3f}",
)
row2[1].metric("ROC-AUC", f"{final_test['roc_auc']:.3f}")
row2[2].metric(
    "Brier score",
    f"{final_test['brier_score']:.4f}",
)
row2[3].metric(
    "Log loss",
    f"{final_test['log_loss']:.4f}",
)

row3 = st.columns(2)
row3[0].metric(
    "Calibration intercept",
    f"{final_test['calibration_intercept']:.3f}",
)
row3[1].metric(
    "Calibration slope",
    f"{final_test['calibration_slope']:.3f}",
)

st.caption(
    "Population-prior reference: "
    f"AP {baseline['average_precision']:.3f}, "
    f"ROC-AUC {baseline['roc_auc']:.3f}, "
    f"Brier {baseline['brier_score']:.4f}, "
    f"log loss {baseline['log_loss']:.4f}."
)

st.header("3. Precision-recall evidence")
st.image(
    str(
        data.figure_paths[
            "v2_final_precision_recall_curve.png"
        ]
    ),
    caption=(
        "Frozen protected-test precision-recall evidence. Ranking improvement "
        "does not by itself validate individualized operational probabilities."
    ),
    use_container_width=True,
)

st.header("4. Calibration evidence")
st.image(
    str(
        data.figure_paths[
            "v2_final_calibration_curve.png"
        ]
    ),
    caption="Frozen protected-test calibration evidence.",
    use_container_width=True,
)
st.markdown(
    f"""
The calibration intercept is **{final_test['calibration_intercept']:.3f}** and
the slope is **{final_test['calibration_slope']:.3f}**. The model Brier score
is slightly worse than the population-prior baseline. These results are why
probabilities are presented as evaluation evidence rather than individualized
operational risk estimates.
"""
)

st.header("5. Registered capacity sensitivity")
st.image(
    str(
        data.figure_paths[
            "v2_final_capacity_tradeoff.png"
        ]
    ),
    caption=(
        "Pre-registered 5%, 10%, and 20% capacity sensitivity. Descriptive "
        "analysis only; no operational threshold or clinic policy was selected."
    ),
    use_container_width=True,
)
st.info(
    "Capacity and cost scenarios are sensitivity analyses, not a validated "
    "intervention policy. No single threshold was selected on the final test."
)

st.header("6. Pre-test interpretation")
st.dataframe(
    pd.DataFrame(top_feature_rows(summary)),
    hide_index=True,
    use_container_width=True,
)
st.caption(
    "Permutation importance was computed before protected-test access on the "
    "policy-selection partition. It is descriptive and did not drive "
    "feature selection."
)

st.header("7. Limitations and external-validation plan")
st.markdown(
    """
- Evidence is limited to a fully synthetic longitudinal benchmark.
- No real-patient external validation has been performed.
- No clinical intervention effect or operational workflow benefit is validated.
- No operating threshold, cost ratio, or capacity policy is clinically validated.
- Subgroup diagnostics are descriptive and do not establish fairness.
- Real-world consideration would require governance review, external validation,
  prospective calibration assessment, workflow-impact evaluation, and a
  separately justified operational decision policy.
"""
)

st.caption(
    "R4 presentation boundary: committed opened evaluation only; no protected "
    "target re-access, model refit, recalibration, feature changes, or "
    "post-test tuning."
)
