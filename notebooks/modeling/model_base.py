class BaseModel:
    def __init__(self, model_name):
        match model_name:
            case _:
                raise ValueError(f"{model_name} deos not exists")

    def prepare(self, X):
        return X

    def compile(self):
        raise ValueError("BaseModel does not have an implementation for: compile")
