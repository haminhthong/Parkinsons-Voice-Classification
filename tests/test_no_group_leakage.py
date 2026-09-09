"""Kiểm thử bất biến không rò rỉ subject trong mọi fold."""

from parkinson_voice.data import SUBJECT_COLUMN
from parkinson_voice.evaluate import make_subject_folds


def test_every_validation_fold_has_both_classes_and_no_overlap(frame):
    for fit_index, valid_index in make_subject_folds(frame, n_splits=4):
        fit = frame.iloc[fit_index]
        valid = frame.iloc[valid_index]
        assert set(fit[SUBJECT_COLUMN]).isdisjoint(valid[SUBJECT_COLUMN])
        assert valid["status"].nunique() == 2
