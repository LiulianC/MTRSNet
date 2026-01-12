class EarlyStopping:
    def __init__(self, patience=15, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            resloss = self.best_loss
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss from {resloss:.5f} improved to {val_loss:.5f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Validation loss did not improve. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True