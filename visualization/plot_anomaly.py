import matplotlib.pyplot as plt



def plot_anomaly(
        y_true,
        y_pred,
        anomaly):


    residual = abs(
        y_true-y_pred
    )


    plt.figure(figsize=(12,4))


    plt.plot(
        y_true,
        label="Actual"
    )


    plt.plot(
        y_pred,
        label="Prediction"
    )


    idx = anomaly.nonzero()[0]


    plt.scatter(
        idx,
        y_true[idx],
        color="red",
        label="Anomaly"
    )


    plt.legend()

    plt.title(
        "Prediction Residual Based Anomaly Detection"
    )

    plt.show()
