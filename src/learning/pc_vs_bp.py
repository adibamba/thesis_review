"""Predictive coding (PC) and backpropagation (BP) on the SAME small network, side by side.

Network: input x (256) -> hidden h = tanh(W1 x) (64) -> output o = W2 h (80 classes)
Task:    predict the one-hot label y.   Loss: L = 0.5 * || y - o ||^2

BP  : compute the output error, send it backwards with the chain rule, get exact gradients.
PC  : keep the hidden layer "free". Let it settle for T small steps so that it trades off
      (a) staying close to what the layer below predicts and (b) helping the layer above
      predict y. Then update each weight matrix from its OWN local error only:
      weight change = (local error) x (activity of the layer below).
      No error travels backwards through the weights after the settling phase.
Known result (Whittington & Bogacz 2017): with a small output error PC's update approaches BP's gradient.
The self-test at the bottom measures exactly that (cosine similarity close to 1).

Run the self-test:   python src/learning/pc_vs_bp.py
"""
import numpy as np

f = np.tanh                                   # activation
def fprime(a): return 1.0 - np.tanh(a) ** 2   # its derivative


def forward(W1, W2, x):
    """Plain forward pass. Returns hidden pre-activation a1 and output o."""
    a1 = W1 @ x
    return a1, W2 @ f(a1)


def loss(W1, W2, x, y):
    return 0.5 * float(np.sum((y - forward(W1, W2, x)[1]) ** 2))


# ----------------------------------------------------------------------------- BP
def bp_update(W1, W2, x, y):
    """Return (dW1, dW2) = minus the gradient of L, computed with the chain rule."""
    a1, o = forward(W1, W2, x)
    e2 = y - o                                # output error
    e1 = (W2.T @ e2) * fprime(a1)             # error sent BACKWARDS through W2  <- the BP step
    return np.outer(e1, x), np.outer(e2, f(a1))


# ----------------------------------------------------------------------------- PC
def pc_update(W1, W2, x, y, steps=60, eta=0.1):
    """Return (dW1, dW2) from predictive coding.

    mu1 = what layer 0 predicts for the hidden layer (W1 x).  x1 = the hidden state, free to move.
    e1  = x1 - mu1       (hidden layer is surprised by its own value)
    e2  = y - W2 f(x1)   (output is surprised by the label)
    x1 moves downhill on the total error  F = 0.5*|e1|^2 + 0.5*|e2|^2.
    """
    mu1 = W1 @ x
    x1 = mu1.copy()                           # start where the bottom-up prediction says
    for _ in range(steps):                    # settling (inference) phase
        e1 = x1 - mu1
        e2 = y - W2 @ f(x1)
        x1 = x1 + eta * (-e1 + (W2.T @ e2) * fprime(x1))
    e1 = x1 - mu1
    e2 = y - W2 @ f(x1)
    return np.outer(e1, x), np.outer(e2, f(x1))   # local errors x local activities


# ----------------------------------------------------------------------------- training
def train(method, W1, W2, X, Y, lr=0.02, epochs=50, seed=0):
    """Train copies of W1, W2 with 'bp' or 'pc'. Returns weights and the mean loss per epoch."""
    update = {"bp": bp_update, "pc": pc_update}[method]
    W1, W2 = W1.copy(), W2.copy()
    rng = np.random.default_rng(seed)
    history = [np.mean([loss(W1, W2, x, y) for x, y in zip(X, Y)])]
    for _ in range(epochs):
        for i in rng.permutation(len(X)):
            d1, d2 = update(W1, W2, X[i], Y[i])
            W1 += lr * d1
            W2 += lr * d2
        history.append(np.mean([loss(W1, W2, x, y) for x, y in zip(X, Y)]))
    return W1, W2, history


def cosine(a, b):
    a, b = np.concatenate([m.ravel() for m in a]), np.concatenate([m.ravel() for m in b])
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def init_weights(d_in=256, d_hidden=64, d_out=80, seed=0):
    r = np.random.default_rng(seed)
    return r.standard_normal((d_hidden, d_in)) / np.sqrt(d_in), r.standard_normal((d_out, d_hidden)) / np.sqrt(d_hidden)


# ----------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    # 1) check the BP formula against a numerical gradient (is my BP right?)
    r = np.random.default_rng(1)
    W1, W2 = init_weights(seed=1)
    x, y = r.standard_normal(256), np.eye(80)[7]
    g1, g2 = bp_update(W1, W2, x, y)
    eps, W1p = 1e-6, W1.copy()
    W1p[3, 5] += eps
    numeric = -(loss(W1p, W2, x, y) - loss(W1, W2, x, y)) / eps
    print(f"BP check   analytic dW1[3,5] = {g1[3, 5]:+.6f}   numeric = {numeric:+.6f}")
    assert abs(g1[3, 5] - numeric) < 1e-4

    # 2) PC update vs BP gradient on the same weights and the same example
    cos = cosine(bp_update(W1, W2, x, y), pc_update(W1, W2, x, y))
    print(f"PC vs BP   cosine similarity of one update (random weights) = {cos:.3f}")

    # 3) both methods learn a toy task: 40 random inputs with 5 classes
    X = r.standard_normal((40, 256))
    Y = np.eye(80)[r.integers(0, 5, 40)]
    for m in ("bp", "pc"):
        _, _, h = train(m, W1, W2, X, Y, epochs=60)
        print(f"{m.upper()} training loss  {h[0]:.3f} -> {h[-1]:.4f}")
        assert h[-1] < 0.2 * h[0], f"{m} did not learn"
    Wb1, Wb2, _ = train("bp", W1, W2, X, Y, epochs=60)
    Wp1, Wp2, _ = train("pc", W1, W2, X, Y, epochs=60)
    cos_t = np.mean([cosine(bp_update(Wp1, Wp2, X[i], Y[i]), pc_update(Wp1, Wp2, X[i], Y[i])) for i in range(10)])
    print(f"PC vs BP   cosine on the trained PC network = {cos_t:.3f}")
    print("all checks passed")