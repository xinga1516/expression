import torch


class EarlyStopping:
    '''
    Early stopping utility to stop training when validation loss does not improve for a given number of epochs (patience).
    save checkpoint of the best model when validation loss improves and save last model.
    '''
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
    def state_dict(self):
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "early_stop": self.early_stop,
        }
    def load_state_dict(self, state):
        self.counter = state["counter"]
        self.best_score = state["best_score"]
        self.early_stop = state["early_stop"]
    
