import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def test(device_str: str, eps_num: int, piece: str, leraning_rate = 0.03):
  x_file = f"X_{piece}.pt"
  y_file = f"Y_{piece}.pt"
  model_file = f"model_{piece}.pth"
  X = torch.load(x_file)
  y = torch.load(y_file)
  assert len(X) == len(y)

  ratio = 0.8
  idx = int(X.size(0)*ratio)
  device = torch.device(device_str if torch.cuda.is_available() else "cpu")
  if device_str == "cuda":
    X_train, X_test = X[:idx].to(device), X[idx:].to(device)
    Y_train, Y_test = y[:idx].to(device), y[idx:].to(device)
  else:
    X_train, X_test = X[:idx], X[idx:]
    Y_train, Y_test = y[:idx], y[idx:]
  print(torch.cuda.is_available())
  print(device)

  model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(8*8*7, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 8*8),
  ).to(device)

  X_train.to(device)
  X_test.to(device)
  Y_train.to(device)
  Y_test.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=leraning_rate)
  num_epochs = eps_num
  eps = []
  train_acc = []
  test_acc = []

  for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(X_train.view(-1, 8*8*7))
    loss = criterion(output, Y_train.view(-1, 8*8))
    loss.backward()
    optimizer.step()

    if (epoch+1) % 1000 == 0:
      model.eval()
      correct = 0
      total = 0
      with torch.no_grad():
        val_inputs = X_test.view(-1, 8*8*7)
        val_labels = Y_test.view(-1, 8*8)
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs, val_labels)
        max_idxs, predicted = torch.max(val_outputs, 1)
        total = val_labels.size(0)
        correct = torch.sum((val_labels[torch.arange(len(predicted)), predicted] == 1).float())
        accuracy_test = correct / total

        val_inputs = X_train.view(-1, 8*8*7)
        val_labels = Y_train.view(-1, 8*8)
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs, val_labels)
        _, predicted = torch.max(val_outputs, 1)
        total = val_labels.size(0)
        correct = torch.sum((val_labels[torch.arange(len(predicted)), predicted] == 1).float())
        accuracy_train = correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], train_acc: {accuracy_train:4f}, test_acc: {accuracy_test:4f}')
        eps.append(epoch)
        train_acc.append(accuracy_train.cpu().numpy())  # Konwersja tensora na numpy array
        test_acc.append(accuracy_test.cpu().numpy()) 

  torch.save(model.state_dict(), model_file)
  plt.plot(eps, train_acc, label="train_acc")
  plt.plot(eps, test_acc, label="test_acc")
  plt.xlabel("eps")
  plt.ylabel("accuracy")
  plt.legend()
  plt.title(f"Accuracy, lr: {leraning_rate}")
  plt.show()


if __name__ == "__main__":
  # current_time = datetime.datetime.now()
  eps = 100000
  test("cuda", eps, "bishop", 0.01)