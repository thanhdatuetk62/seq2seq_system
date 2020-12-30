from torch import nn
import torch

class CrossEntropyLoss(nn.Module):
    def __init__(self, pad_idx, alpha=0.1, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.pad_idx = pad_idx
    
    def forward(self, probs, labels, device="cpu"):
        """
        Arguments:
            probs (Tensor): [batch_size x len x n_classes]
                - values: (float) in range [0.0, 1.0) 
            labels (Tensor): [batch_size x len]
                - values: (int) in range [0, n_classes)
            alpha (float): label smoothing hyperparameter 
        Returns:
            A tensor with dim = 0 contains value of xent function.
        """
        batch_size, seq_len, n_classes = probs.size()
        ignore_mask = (labels == self.pad_idx)

        mask = torch.arange(0, n_classes).view(1, 1, n_classes).expand( \
            batch_size, seq_len, n_classes).to(device)
        mask = (mask == labels.unsqueeze(-1))
        mask = (1.0 - self.alpha) * mask + self.alpha / n_classes
        
        log_softmax = probs - probs.exp().sum(dim=-1).log().unsqueeze(-1)
        scores = (mask * log_softmax).sum(dim=-1).masked_fill(\
            mask=ignore_mask, value=0.0)
        xent = -1.0 * scores.sum() / batch_size
        return xent

class LabelSmoothingLoss(nn.Module):
  """
  With label smoothing,
  KL-divergence between q_{smoothed ground truth prob.}(w)
  and p_{prob. computed by model}(w) is minimized.
  """
  def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
      assert 0.0 < label_smoothing <= 1.0
      self.ignore_index = ignore_index
      super(LabelSmoothingLoss, self).__init__()

      smoothing_value = label_smoothing / (tgt_vocab_size - 2)
      one_hot = torch.full((tgt_vocab_size,), smoothing_value)
      one_hot[ignore_index] = 0
      self.register_buffer('one_hot', one_hot.unsqueeze(0))

      self.confidence = 1.0 - label_smoothing

  def forward(self, output, target):
      """
      output (FloatTensor): batch_size x n_classes
      target (LongTensor): batch_size
      """
      output = output.log_softmax(dim=-1)
      model_prob = self.one_hot.repeat(target.size(0), 1)
      model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
      model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

      return -1.0 * (output * model_prob).sum() / target.size(0)