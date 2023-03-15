import numpy as np

class Linear_Regression():
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    self.w = 0.5
    self.b = 0.5
  def get_weights(self):
    return self.w, self.b
  def predict(self, X=[]):
    if not X: X = self.X
    w = self.w
    b = self.b
    Y_pred = np.array([])
    for x in X:
      Y_pred = np.append(Y_pred, w * x + b)
    return Y_pred
  def return_loss(self):
    Y_pred = self.predict()
    m = self.X.shape[0]
    return (1 / 2*m) * np.sum((self.Y - Y_pred)**2)
  def update_parameters(self, lr):
    Y_pred = self.predict()
    m = len(self.Y)
    self.w = self.w - (lr * ((1/m) * np.sum((Y_pred - self.Y) * self.X)))
    self.b = self.b - (lr * (1/m) * np.sum(Y_pred - self.Y))