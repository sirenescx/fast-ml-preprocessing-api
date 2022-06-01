from enum import Enum


class ProblemType(Enum):
    REGRESSION = 1
    CLASSIFICATION = 2

    @staticmethod
    def get_problem_type_from_string(problem_type: str) -> int:
        if problem_type == "regression":
            return 1
        if problem_type == "classification":
            return 2
