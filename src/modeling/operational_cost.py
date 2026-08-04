"""Deterministic operational break-even sensitivity."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.modeling.operational_threshold import (
    evaluate_operational_threshold_states,
)
_EFFECTIVENESS_VALUES = (
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
)
_SENSITIVITY_COLUMNS = (
    "effectiveness",
    (
        "ex_ante_break_even_"
        "intervention_to_no_show_cost"
    ),
    (
        "validation_replay_break_even_"
        "intervention_to_no_show_cost"
    ),
)
_DECISION_RULE = (
    "intervene_all preferred when "
    "intervention_cost / no_show_cost "
    "< prevalence * effectiveness"
)
_RESULT_KEYS = (
    "sensitivity_table",
    "selected_probability_model",
    "threshold_rule",
    "distinct_policy_states",
    "decision_rule",
    "ex_ante_prevalence",
    "validation_replay_prevalence",
    "ex_ante_source",
    "validation_role",
    "specific_cost_values_selected",
    "effectiveness_selected",
    "operational_policy_selected",
)
def evaluate_operational_break_even_sensitivity(
    calibration_target: pd.Series,
    validation_target: pd.Series,
) -> dict[
    str,
    pd.DataFrame | str | float | int | bool,
]:
    """Evaluate deterministic policy break-even boundaries.
    The selected Phase 09 probability contract yields only two
    operational threshold states: intervene on every appointment or
    intervene on none.
    The ex-ante boundary uses only calibration prevalence. Validation
    prevalence is reported solely as a replay audit. No specific cost,
    effectiveness, threshold, or operational policy is selected.
    """
    threshold_result = (
        evaluate_operational_threshold_states(
            calibration_target,
            validation_target,
        )
    )
    ex_ante_prevalence = float(
        threshold_result[
            "probability_value"
        ]
    )
    validation_replay_prevalence = float(
        validation_target.mean()
    )
    effectiveness = np.asarray(
        _EFFECTIVENESS_VALUES,
        dtype=np.float64,
    )
    sensitivity_table = pd.DataFrame(
        {
            "effectiveness": effectiveness,
            (
                "ex_ante_break_even_"
                "intervention_to_no_show_cost"
            ): (
                ex_ante_prevalence
                * effectiveness
            ),
            (
                "validation_replay_break_even_"
                "intervention_to_no_show_cost"
            ): (
                validation_replay_prevalence
                * effectiveness
            ),
        },
        columns=_SENSITIVITY_COLUMNS,
    )
    if tuple(
        sensitivity_table.columns
    ) != _SENSITIVITY_COLUMNS:
        raise RuntimeError(
            "break-even sensitivity column "
            "contract is invalid"
        )
    if not np.array_equal(
        sensitivity_table[
            "effectiveness"
        ].to_numpy(
            dtype=np.float64,
            copy=True,
        ),
        effectiveness,
    ):
        raise RuntimeError(
            "effectiveness grid is invalid"
        )
    numeric = sensitivity_table.to_numpy(
        dtype=np.float64,
        copy=True,
    )
    if not np.isfinite(
        numeric
    ).all():
        raise RuntimeError(
            "break-even sensitivity contains "
            "non-finite values"
        )
    ex_ante_boundary = sensitivity_table[
        (
            "ex_ante_break_even_"
            "intervention_to_no_show_cost"
        )
    ].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    replay_boundary = sensitivity_table[
        (
            "validation_replay_break_even_"
            "intervention_to_no_show_cost"
        )
    ].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    if not np.all(
        np.diff(
            ex_ante_boundary
        )
        >= 0.0
    ):
        raise RuntimeError(
            "ex-ante break-even boundary "
            "must be nondecreasing"
        )
    if not np.all(
        np.diff(
            replay_boundary
        )
        >= 0.0
    ):
        raise RuntimeError(
            "validation replay boundary "
            "must be nondecreasing"
        )
    if ex_ante_boundary[0] != 0.0:
        raise RuntimeError(
            "zero effectiveness must have "
            "zero ex-ante boundary"
        )
    if replay_boundary[0] != 0.0:
        raise RuntimeError(
            "zero effectiveness must have "
            "zero replay boundary"
        )
    if (
        ex_ante_boundary[-1]
        != ex_ante_prevalence
    ):
        raise RuntimeError(
            "full-effectiveness ex-ante "
            "boundary must equal prevalence"
        )
    if (
        replay_boundary[-1]
        != validation_replay_prevalence
    ):
        raise RuntimeError(
            "full-effectiveness replay "
            "boundary must equal prevalence"
        )
    result: dict[
        str,
        pd.DataFrame | str | float | int | bool,
    ] = {
        "sensitivity_table": sensitivity_table,
        "selected_probability_model": (
            threshold_result[
                "selected_probability_model"
            ]
        ),
        "threshold_rule": threshold_result[
            "threshold_rule"
        ],
        "distinct_policy_states": int(
            threshold_result[
                "distinct_policy_states"
            ]
        ),
        "decision_rule": _DECISION_RULE,
        "ex_ante_prevalence": (
            ex_ante_prevalence
        ),
        "validation_replay_prevalence": (
            validation_replay_prevalence
        ),
        "ex_ante_source": (
            "calibration_target only"
        ),
        "validation_role": (
            "replay audit only"
        ),
        "specific_cost_values_selected": (
            False
        ),
        "effectiveness_selected": False,
        "operational_policy_selected": False,
    }
    if tuple(result) != _RESULT_KEYS:
        raise RuntimeError(
            "break-even sensitivity result "
            "contract is invalid"
        )
    return result
