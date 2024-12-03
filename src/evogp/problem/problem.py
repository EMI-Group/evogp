class Problem:

    def __init__(self):
        pass

    def evaluate(self, randkey, prefix_trees):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    def show(self, randkey, prefix_trees):
        """
        show how a genome perform in this problem
        """
        raise NotImplementedError
