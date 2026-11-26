import numpy as np
import pandas as pd
import pytest

from src.prediction.predict_dl import build_submission


@pytest.fixture
def test_and_sample():
    test = pd.DataFrame({"id": [10, 11]})
    sample = pd.DataFrame({"id": [10, 11], "is_fraud": [0.0, 0.0]})
    return test, sample


def test_build_submission_preserves_sample_schema_and_id_order(test_and_sample):
    test, sample = test_and_sample

    submission = build_submission(test, sample, np.array([0.2, 0.8]))

    assert list(submission.columns) == ["id", "is_fraud"]
    assert submission["id"].tolist() == [10, 11]
    assert submission["is_fraud"].tolist() == [0.2, 0.8]


def test_build_submission_rejects_ids_in_wrong_order(test_and_sample):
    test, sample = test_and_sample
    test["id"] = [11, 10]

    with pytest.raises(ValueError, match="IDs differ"):
        build_submission(test, sample, np.array([0.2, 0.8]))


@pytest.mark.parametrize("probabilities", [[0.2, np.nan], [-0.1, 0.8], [0.2]])
def test_build_submission_rejects_invalid_probabilities(test_and_sample, probabilities):
    test, sample = test_and_sample

    with pytest.raises(ValueError):
        build_submission(test, sample, np.asarray(probabilities))
