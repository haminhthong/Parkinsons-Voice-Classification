"""Kiểm thử nested cross-validation ở cấp subject."""

from src.evaluate import make_subject_folds
from src.model_selection import nested_subject_cross_fitted


def test_nested_cv_has_no_subject_overlap(frame):
    outer_folds = make_subject_folds(frame, n_splits=4, random_state=42)

    for outer_fit, outer_valid in outer_folds:
        outer_train = frame.iloc[outer_fit].reset_index(drop=True)
        outer_validation = frame.iloc[outer_valid]
        assert set(outer_train["subject_id"]).isdisjoint(outer_validation["subject_id"])

        inner_folds = make_subject_folds(outer_train, n_splits=3, random_state=43)
        for inner_fit, inner_valid in inner_folds:
            assert set(outer_train.iloc[inner_fit]["subject_id"]).isdisjoint(
                outer_train.iloc[inner_valid]["subject_id"]
            )


def test_nested_cv_returns_one_cross_fitted_prediction_per_subject(frame):
    fold_metrics, predictions, selections = nested_subject_cross_fitted(
        frame,
        C_values=[0.1],
        class_weights=[None],
    )

    assert len(fold_metrics) == 4
    assert len(predictions) == frame["subject_id"].nunique()
    assert predictions["subject_id"].is_unique
    assert predictions["subject_score"].between(0, 1).all()
    assert set(selections["outer_fold"]) == {1, 2, 3, 4}
