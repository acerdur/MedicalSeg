import torch 
import torch.nn as nn

class CrossEntropy(nn.Module):
    """
    nn.CrossEntropyLoss modified to work with segmentation_models_pytorch validation epoch

    Arguments:
        use_lstm:  bool     True if the model is LSTM classifier
    """
    __name__ = 'cross_entropy'

    def __init__(self,use_lstm=True):
        super().__init__()
        self.use_lstm = use_lstm
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, inputs, target):
        _, pred = inputs
        target = target.flatten()
        if self.use_lstm:
            pred = pred.view(-1,pred.shape[-1])

        return self.loss(pred, target)

class Accuracy(nn.Module):

    __name__ = 'acc'

    def __init__(self, use_lstm=True):
        super().__init__()
        self.use_lstm = use_lstm
    
    def forward(self, inputs, target):
        _, pred = inputs
        target = target.flatten()
        if self.use_lstm:
            pred = pred.view(-1,pred.shape[-1])

        pred = pred.argmax(dim=1).flatten()
        true_pred = torch.sum(pred == target, dtype=pred.dtype)

        return true_pred / target.shape[0]

class Precision(nn.Module):

    __name__ = 'prec'

    def __init__(self, use_lstm=True, eps=1e-5):
        super().__init__()
        self.use_lstm = use_lstm
        self.eps = eps

    def forward(self, inputs, target):
        _, pred = inputs
        target = target.flatten()
        if self.use_lstm:
            pred = pred.view(-1,pred.shape[-1])

        pred = pred.argmax(dim=1).flatten()
        tp = torch.sum(pred * target)
        fp = torch.sum(pred) - tp 

        return (tp + self.eps) / (tp + fp + self.eps)

class Recall(nn.Module):

    __name__ = 'rec'

    def __init__(self, use_lstm=True, eps=1-5):
        super().__init__()
        self.use_lstm = use_lstm
        self.eps = eps
    
    def forward(self, inputs, target):
        _, pred = inputs
        target = target.flatten()
        if self.use_lstm:
            pred = pred.view(-1,pred.shape[-1])

        pred = pred.argmax(dim=1).flatten()
        tp = torch.sum(pred * target) # num true positive preds
        pos = target.sum() # num all real positives 

        return (tp + self.eps) / (pos + self.eps)


class SoftPredIoU(nn.Module):

    __name__ = ''

    def __init__(self, use_lstm):
        super().__init__()
        self.use_lstm = use_lstm

    def forward(self, inputs, target):
        _, pred = inputs
        if self.use_lstm:
            pred = pred.view(-1,pred.shape[-1])

        pred = pred.argmax(dim=1).flatten()

    