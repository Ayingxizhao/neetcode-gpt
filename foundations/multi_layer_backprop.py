import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        # this is the forward pass
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        z1 = x @ W1.T + b1 # x is (n, ) w1 (n, hidden) -> (hidden, )
        a1 = np.maximum(0, z1) # a1 is (hidden, ) 
        # z2 is your output. 
        z2 = a1 @ W2.T + b2 # (hidden, ) w2 (n, hidden) -> (output,)
        def compute_loss(y_hat, y):
            return np.mean((y_hat - y)**2)
        loss = compute_loss(z2, y_true)
        N = len(y_true)
        # dl/dz2 = 2/N * (z2 - y_true)
        # dz2/dW2 = a1
        # dz2/db2 = 1
        delta2 = 2/N * (z2 - y_true) #(n, )
        # chain with delta2 
        dz2_dW2 = np.outer(delta2, a1)
        dz2_db2 = delta2
        # error signal through w2 and relu derivative
        delta1 = W2.T @ delta2 * (z1>0).astype(float) # (n, ) @ (n, hidden) -> (hidden,)
        dw1 = np.outer(delta1, x)
        db1 = delta1
        # now chain rule
        return {
            'loss': np.round(loss, 4),
            'dW1': np.round(dw1, 4).tolist(),
            'db1' : np.round(db1, 4).tolist(),
            'dW2': np.round(dz2_dW2, 4).tolist(),
            'db2': np.round(dz2_db2, 4).tolist()
        }
