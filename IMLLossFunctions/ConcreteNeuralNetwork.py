import torch
import sklearn
import torch.nn as nn
import pandas as pd
import sklearn.metrics
import numpy as np

# create MLP from literature
class ConcreteNN(nn.Module):

  def __init__(self):
    super().__init__()
    # layers
    self.input_layer = nn.Linear(9, 9)     
    self.hidden_layer = nn.Linear(9, 9)
    self.output_layer = nn.Linear(9, 1) 
    self.activation = nn.Sigmoid()
    # data
    self.data = pd.read_csv("slump_test.csv")

  def forward(self, x):
    x = self.input_layer(x)
    x = self.activation(x)
    x = self.hidden_layer(x)
    x = self.activation(x)
    x = self.output_layer(x)
    return x

# additional stuff for training and evaluation

def test_loss(model, X_test, y_test):
    model.eval()
    output = model(X_test)
    loss = sklearn.metrics.mean_squared_error(output.detach().numpy(), y_test.unsqueeze(1).detach().numpy())
    return loss.item()

def eval_model(model, X, y):
    model.eval()
    y_preds = model(X).detach().numpy()
    return sklearn.metrics.r2_score(y, y_preds), np.sqrt(sklearn.metrics.mean_squared_error(y, y_preds))

def train(model, X_train, y_train, X_test, y_test):
  criterion = nn.MSELoss()
  optimizer= torch.optim.SGD(model.parameters(), lr=0.001, momentum= 0.5)
  epochs = 2000
  loss_over_time = []
  test_loss_over_time = []
  for i in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    target = y_train.unsqueeze(1)
    loss = criterion(output, target)
    loss_over_time.append(loss.item())

    test_loss_over_time.append(test_loss(model=model, X_test=X_test, y_test=y_test))

    loss.backward()
    optimizer.step()
  return loss_over_time, test_loss_over_time

# The Linear Relationship between Water, Cement and Concrete Strength
def linear_relationship(X_train, y_train):
    # Linear relationship between (Water/Cement) and Concrete Strength.
    # Input: Training Data
    # Output: W: [1 (Water/Cement)] Matrix
    #         LR.coef_: Linear Regression Coefficients

    W = (X_train[:, 3]/X_train[:, 0]).reshape(-1, 1) # Water/Cement
    ones = torch.ones(W.shape)
    W = torch.hstack((ones, W)) # add ones for bias
    LR = sklearn.linear_model.LinearRegression(fit_intercept=False)
    LR.fit(W, y_train)
    return W, LR.coef_

# custom losses v1... check influence of normalization and standardization of data on MSE....
def informed_loss1(inputs, outputs, targets, lamb):
    W, b = linear_relationship(inputs, targets)
    mse = nn.MSELoss()
    relu = nn.ReLU()
    info = torch.sum(outputs - (b[0][0] + b[0][1]*W))
    return mse(outputs, targets) + lamb*relu(info)

def informed_loss2(inputs, outputs, targets, lamb):
    W, b = linear_relationship(inputs, targets)
    mse = nn.MSELoss()
    relu = nn.ReLU()
    info = torch.abs(torch.sum(outputs - (b[0][0] + b[0][1]*W)))
    return mse(outputs, targets) + lamb*relu(info)
    
def informed_loss3(inputs, outputs, targets, lamb):
    W, b = linear_relationship(inputs, targets)
    mse = nn.MSELoss()
    relu = nn.ReLU()
    info = torch.sum(outputs - (b[0][0] + b[0][1]*W))
    return mse(outputs, targets) + lamb*torch.abs(info)

def train_with_closs(model, closs, lamb, X_train, y_train, X_test, y_test):
  optimizer= torch.optim.SGD(model.parameters(), lr=0.001, momentum= 0.5)
  epochs = 2000
  loss_over_time = []
  test_loss_over_time = []
  for i in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    target = y_train.unsqueeze(1)
    loss = closs(X_train, output, target, lamb)
    loss_over_time.append(loss.item())
    test_loss_over_time.append(test_loss(model=model, X_test=X_test, y_test=y_test))
    loss.backward()
    optimizer.step()
  return loss_over_time, test_loss_over_time
