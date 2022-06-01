import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeClassifier

from fast_ml_preprocessing.utils.problem_type import ProblemType


class FeatureSelectionOperation:
    def apply(self, features: pd.DataFrame, target: pd.Series, problem_type: ProblemType) -> pd.DataFrame:
        if problem_type == ProblemType.REGRESSION.value:
            selector: SelectFromModel = SelectFromModel(estimator=ElasticNet(), threshold="0.25*median")
            selector.fit(features, target)
            features_after_selection: list[str] = selector.get_feature_names_out()
            features = pd.DataFrame(selector.transform(features), columns=features_after_selection)
        else:
            selector: SelectFromModel = SelectFromModel(estimator=DecisionTreeClassifier(), threshold="0.25*median")
            selector.fit(features, target)
            features_after_selection: list[str] = selector.get_feature_names_out()
            features = pd.DataFrame(selector.transform(features), columns=features_after_selection)
        return features
