from .loss_metrics import CrossEntropyLoss, LabelSmoothingLoss
from .eval_metrics import bleu_score

loss_metrics = {"xent": CrossEntropyLoss, \
    "label_smoothing_loss": LabelSmoothingLoss}

eval_metrics = {"bleu": bleu_score}

def find_loss_metric(loss_metric):
    if loss_metric not in loss_metrics:
        raise ValueError("Loss metric {} did not exist in our system".
                            format(loss_metric))
    return loss_metrics[loss_metric]

def find_eval_metric(eval_metric):
    if eval_metric not in eval_metrics:
        raise ValueError("Loss metric {} did not exist in our system".
                            format(eval_metric))
    return eval_metrics[eval_metric]