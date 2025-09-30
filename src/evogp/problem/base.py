class BaseProblem:
    def evaluate(self, forest):
        raise NotImplementedError

    @property
    def problem_dim(self):
        raise NotImplementedError

    @property
    def solution_dim(self):
        raise NotImplementedError
