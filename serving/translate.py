from server import app
from flask import request


def preprocess(text):
    return [sent.strip().split() for sent in text.split("\n")]

def postprocess(sents):
    return "\n".join(sent for sent in sents)

@app.route('/translate', methods=["POST"])
def translate():
    src_raw_text = request.json["src"]
    
    src_sents = preprocess(src_raw_text)
    trg_sents = []
    # Create batch iterator
    batch_iter = app.data.create_infer_iter(src_sents, app.batch_size)
    for src in batch_iter:
        # Append generated result to final results
        trg_sents += [' '.join(app.data.convert_to_str(tokens))
                      for tokens in app.model(src)]

    trg_raw_text = postprocess(trg_sents)
    return {"target": trg_raw_text}
