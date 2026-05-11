"""
Szkielet dla Osoby 2.
Cel: własna implementacja SSN w NumPy, zgodna z danymi z `src.data_loader`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]


class NeuralNetwork:
    def __init__(
        self,
        layer_sizes: list[int],
        problem_type: str,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        learning_rate: float = 0.01,
        epochs: int = 100,
        batch_size: int = 32,
        weight_init: str = "xavier",
        random_state: int = 42,
    ) -> None:
        self.layer_sizes = layer_sizes
        self.problem_type = problem_type
        self.activation = activation
        
        if output_activation is None:
            if self.problem_type == "classification":
                self.output_activation = "sigmoid"
            else:
                self.output_activation = "linear"
        else:
            self.output_activation = output_activation
            
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_init = weight_init
        self.random_state = random_state

        self.weights = []
        self.biases = []

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        activations = [X]
        zs = []
        
        A = X
        for i in range(len(self.weights)):
            W = self.weights[i]
            b = self.biases[i]
            Z = np.dot(A, W) + b
            zs.append(Z)
            
            if i == len(self.weights) - 1:
                # Output layer
                if self.output_activation == "sigmoid":
                    A = 1.0 / (1.0 + np.exp(-np.clip(Z, -250, 250)))
                elif self.output_activation == "linear":
                    A = Z
                else:
                    raise ValueError(f"Unsupported output_activation: {self.output_activation}")
            else:
                # Hidden layer
                if self.activation == "relu":
                    A = np.maximum(0, Z)
                elif self.activation == "sigmoid":
                    A = 1.0 / (1.0 + np.exp(-np.clip(Z, -250, 250)))
                elif self.activation == "tanh":
                    A = np.tanh(Z)
                else:
                    raise ValueError(f"Unsupported activation: {self.activation}")
                    
            activations.append(A)
            
        return activations, zs
        
    def _backward(self, y_true: np.ndarray, activations: list[np.ndarray], zs: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        m = y_true.shape[0]
        y_pred = activations[-1]
        
        if self.problem_type == "classification":
            # Derivative for BCE loss + Sigmoid activation
            dZ = (y_pred - y_true) / m
        elif self.problem_type == "regression":
            # Derivative for MSE loss + Linear activation
            dZ = 2.0 * (y_pred - y_true) / m
        else:
            raise ValueError(f"Unsupported problem_type: {self.problem_type}")
            
        dW_list = []
        db_list = []
        
        for i in reversed(range(len(self.weights))):
            W = self.weights[i]
            A_prev = activations[i]
            
            dW = np.dot(A_prev.T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)
            
            dW_list.append(dW)
            db_list.append(db)
            
            if i > 0:
                dA_prev = np.dot(dZ, W.T)
                Z_prev = zs[i-1]
                
                if self.activation == "relu":
                    dZ = dA_prev * (Z_prev > 0).astype(float)
                elif self.activation == "sigmoid":
                    s = 1.0 / (1.0 + np.exp(-np.clip(Z_prev, -250, 250)))
                    dZ = dA_prev * (s * (1.0 - s))
                elif self.activation == "tanh":
                    t = np.tanh(Z_prev)
                    dZ = dA_prev * (1.0 - t**2)
                    
        dW_list.reverse()
        db_list.reverse()
        return dW_list, db_list

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TrainingHistory:
        """
        Oczekiwane minimum:
        - forward pass
        - backward pass
        - update wag
        - zapis historii loss
        """
        np.random.seed(self.random_state)
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i+1]
            
            if self.weight_init == "xavier":
                limit = np.sqrt(6.0 / (n_in + n_out))
                W = np.random.uniform(-limit, limit, (n_in, n_out))
            elif self.weight_init == "he":
                W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            elif self.weight_init == "random":
                W = np.random.randn(n_in, n_out) * 0.01
            else:
                raise ValueError(f"Unknown weight initialization: {self.weight_init}")
                
            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.biases.append(b)

        history = TrainingHistory(train_loss=[], val_loss=[])
        
        y_train = y_train.reshape(-1, 1)
        if y_val is not None:
            y_val = y_val.reshape(-1, 1)
            
        m = X_train.shape[0]
        
        for epoch in range(self.epochs):
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            for i in range(0, m, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                activations, zs = self._forward(X_batch)
                dW_list, db_list = self._backward(y_batch, activations, zs)
                
                for j in range(len(self.weights)):
                    self.weights[j] -= self.learning_rate * dW_list[j]
                    self.biases[j] -= self.learning_rate * db_list[j]

            # Comput loss at the end of epoch
            train_preds, _ = self._forward(X_train)
            if self.problem_type == "regression":
                train_l = np.mean((y_train - train_preds[-1])**2)
            else:
                eps = 1e-15
                p = np.clip(train_preds[-1], eps, 1 - eps)
                train_l = -np.mean(y_train * np.log(p) + (1 - y_train) * np.log(1 - p))
                
            history.train_loss.append(float(train_l))
            
            if X_val is not None and y_val is not None:
                val_preds, _ = self._forward(X_val)
                if self.problem_type == "regression":
                    val_l = np.mean((y_val - val_preds[-1])**2)
                else:
                    eps = 1e-15
                    p = np.clip(val_preds[-1], eps, 1 - eps)
                    val_l = -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))
                history.val_loss.append(float(val_l))
                
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        - regresja: zwróć wartości ciągłe
        - klasyfikacja binarna: zwróć 0/1 albo probability + osobna metoda predict_proba
        """
        activations, _ = self._forward(X)
        out = activations[-1]
        
        if self.problem_type == "regression":
            return out.flatten()
        else:
            return (out > 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Opcjonalne, ale przydatne dla klasyfikacji.
        """
        if self.problem_type != "classification":
            raise ValueError("predict_proba is only available for classification.")
        activations, _ = self._forward(X)
        return activations[-1].flatten()
