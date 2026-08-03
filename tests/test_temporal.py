from thermal_occupancy.occupancy.temporal import TemporalOccupancyStabilizer


def test_entry_and_exit_are_debounced() -> None:
    stabilizer = TemporalOccupancyStabilizer(
        fps=10,
        enter_confirm_sec=0.2,
        exit_confirm_sec=0.3,
        state_retention_sec=1.0,
    )
    assert stabilizer.update(0, [7]) == ()
    assert stabilizer.update(1, [7]) == (7,)
    assert stabilizer.update(2, []) == (7,)
    assert stabilizer.update(3, []) == (7,)
    assert stabilizer.update(4, []) == ()


def test_short_dropout_does_not_change_occupancy() -> None:
    stabilizer = TemporalOccupancyStabilizer(
        fps=10,
        enter_confirm_sec=0.1,
        exit_confirm_sec=0.3,
        state_retention_sec=1.0,
    )
    assert stabilizer.update(0, [2]) == (2,)
    assert stabilizer.update(1, []) == (2,)
    assert stabilizer.update(2, [2]) == (2,)

