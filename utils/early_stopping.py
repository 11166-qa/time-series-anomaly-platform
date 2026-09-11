class EarlyStopping:
    """
    根据验证集 Loss 实现 Early Stopping。
    """

    def __init__(
        self,
        patience=8,
        min_delta=1e-5,
    ):
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss):
        """
        返回:
            improved:
                当前模型是否取得新的最佳验证 Loss

            should_stop:
                是否应该提前停止训练
        """

        improved = (
            val_loss
            < self.best_loss - self.min_delta
        )

        if improved:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        should_stop = (
            self.counter >= self.patience
        )

        return improved, should_stop