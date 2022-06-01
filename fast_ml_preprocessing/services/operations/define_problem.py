from fast_ml_preprocessing.utils.problem_type import ProblemType


class ProblemTypeDefiningOperation:
    def define(self, problem_type):
        return ProblemType.get_problem_type_from_string(problem_type)
