from evogp.tree import Forest

class BaseMutation:
    def __call__(self, forest: Forest):
        raise NotImplementedError
