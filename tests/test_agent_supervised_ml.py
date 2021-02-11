import numpy as np

from agents.agent_supervised_ml.classification import linear_svm, k_nearest_neighbours, decision_tree
from agents.agent_supervised_ml.data_prep import clean_scores
from agents.agent_supervised_ml.pytorch_linreg import linear_regression
from agents.agent_supervised_ml.pytorch_logisticreg import logistic_regression


class Tests:

    def test_one(self):
        pass

    def test_linear_regression(self):
        linear_regression()

    def test_logistic_regression(self):
        logistic_regression()

    def test_clean_scores(self):
        a, b = clean_scores()

        assert isinstance(a, np.ndarray)
        assert a.shape[1] == 41
        assert isinstance(b, np.ndarray)

    def test_linear_svm(self):
        linear_svm()

    def test_k_nearest_neighbours(self):
        k_nearest_neighbours()

    def test_decision_tree(self):
        decision_tree()
