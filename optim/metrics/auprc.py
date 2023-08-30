from sklearn.metrics import precision_recall_curve, average_precision_score


class AUPRC():
    def __init__(self):
        super(AUPRC, self).__init__()

    def __call__(self, y_pred, y):
        y_pred = y_pred.flatten()
        y = y.flatten()
        precisions, recalls, thresholds = precision_recall_curve(y.astype(int), y_pred)
        auprc = average_precision_score(y.astype(int), y_pred)
        return auprc, precisions, recalls, thresholds

def compute_auprc(y_pred, y):
    y_pred = y_pred.flatten()
    y = y.flatten()
    precisions, recalls, thresholds = precision_recall_curve(y.astype(int), y_pred)
    auprc = average_precision_score(y.astype(int), y_pred)
    return auprc, precisions, recalls, thresholds