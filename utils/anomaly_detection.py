import numpy as np


def detect_anomaly(
        residual,
        threshold=3):


    mean = np.mean(residual)

    std = np.std(residual)


    limit = mean + threshold * std


    labels = residual > limit


    return labels, limit
