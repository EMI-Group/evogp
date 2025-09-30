class BasePipeline:
    def __init__(self):
        pass
    
    def step(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
