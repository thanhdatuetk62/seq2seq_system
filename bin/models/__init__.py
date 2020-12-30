from .transformer_nmt import TransformerNMT

models = {"transformer_nmt": TransformerNMT}

def find_model(model):
    if model not in models:
        raise ValueError("Model {} did not exist in our system".\
            format(model))
    return models[model]