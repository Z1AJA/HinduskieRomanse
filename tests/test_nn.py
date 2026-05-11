import numpy as np
import sys
import os

# Add src to path just in case
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.neural_network import NeuralNetwork

def test_regression():
    print("Testing regression...")
    X = np.random.randn(100, 3)
    # y = sum(X) + noise
    y = np.sum(X, axis=1) + np.random.randn(100) * 0.1
    
    nn = NeuralNetwork(
        layer_sizes=[3, 10, 1],
        problem_type="regression",
        activation="relu",
        learning_rate=0.01,
        epochs=100,
        batch_size=16
    )
    
    history = nn.fit(X, y)
    print(f"Regression train loss start: {history.train_loss[0]:.4f}, end: {history.train_loss[-1]:.4f}")
    assert history.train_loss[-1] < history.train_loss[0]
    
    preds = nn.predict(X)
    assert preds.shape == (100,)
    print("Regression OK!")

def test_classification():
    print("Testing classification...")
    X = np.random.randn(100, 3)
    # Target: 1 if sum > 0 else 0
    y = (np.sum(X, axis=1) > 0).astype(int)
    
    nn = NeuralNetwork(
        layer_sizes=[3, 10, 1],
        problem_type="classification",
        activation="tanh",
        learning_rate=0.1,
        epochs=100,
        batch_size=16
    )
    
    history = nn.fit(X, y)
    print(f"Classif train loss start: {history.train_loss[0]:.4f}, end: {history.train_loss[-1]:.4f}")
    assert history.train_loss[-1] < history.train_loss[0]
    
    preds = nn.predict(X)
    assert preds.shape == (100,)
    assert set(np.unique(preds)).issubset({0, 1})
    
    probs = nn.predict_proba(X)
    assert probs.shape == (100,)
    assert np.all((probs >= 0) & (probs <= 1))
    print("Classification OK!")

if __name__ == "__main__":
    test_regression()
    test_classification()
