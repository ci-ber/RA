def compute_dice(predictions, gt, th):
    predictions[predictions>th]=1
    predictions[predictions<1]=0
    eps = 1e-6
    # flatten label and prediction tensors
    inputs = predictions.flatten()
    targets = gt.flatten()

    intersection = (inputs * targets).sum()
    dice = (2. * intersection) / (inputs.sum() + targets.sum() + eps)
    return dice