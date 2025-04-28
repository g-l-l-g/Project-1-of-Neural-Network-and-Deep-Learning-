import cupy as cp


def accuracy(predicts, labels):
    """
    Metric for MNIST.
    predicts : [batch, D]
    labels : [batch, ]
    """
    assert predicts.shape[0] == labels.shape[0]

    predict_label = cp.argmax(predicts, axis=-1)
    
    return cp.sum(predict_label == labels) / predicts.shape[0]
