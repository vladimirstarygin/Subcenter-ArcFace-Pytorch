from .labelsmooth import LabelSmoothSoftmaxCEV1
from .Subcenter_arcface import SubcenterArcMarginProduct

def get_loss(params):
    clf_loss = LabelSmoothSoftmaxCEV1()
    metric_loss = SubcenterArcMarginProduct(params['embedding_dim'], params['num_classes'])
    return clf_loss, metric_loss
