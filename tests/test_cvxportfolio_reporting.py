from cvxportfolio_impl.backtest import clamp_for_display, clean_constraint_mapping


def test_clamp_for_display_snaps_near_upper_bound() -> None:
    assert clamp_for_display(0.600828, upper_bound=0.6) == 0.6


def test_clamp_for_display_snaps_near_lower_bound() -> None:
    assert clamp_for_display(-0.000001, lower_bound=0.0) == 0.0


def test_clean_constraint_mapping_respects_declared_bounds() -> None:
    cleaned = clean_constraint_mapping(
        {"equity": 0.600828, "commodity": -0.0, "bond": 0.298828},
        lower_bounds={"commodity": 0.0},
        upper_bounds={"equity": 0.6, "bond": 0.4},
    )
    assert cleaned == {"equity": 0.6, "commodity": 0.0, "bond": 0.298828}
