class GeneralPipeline:

    def __init__(self, algorithm, problem):
        self.algorithm = algorithm
        self.problem = problem

    def step(self, state):
        raise NotImplementedError
