from matplotlib import pyplot as plt
# Epoch [14000/150000], train_acc: 0.089935, test_acc: 0.087635
import torch

def generate(filename: str):
    eps = []
    val_accs = []
    train_accs = []
    with open(filename, "r") as file:
        for line in file:
            words = line.split(',')
            train_acc_val = float(words[1].split(':')[1])
            test_acc_val = float(words[2].split(':')[1])
            eps_line = words[0].split(' ')
            eps_num = eps_line[1][1:].split('/')[0]
            eps.append(int(eps_num))
            val_accs.append(test_acc_val)
            train_accs.append(train_acc_val)
    plt.plot(eps, train_accs, label="train accs")
    plt.plot(eps, val_accs, label="val accs")
    plt.legend()
    plt.xlabel("eps")
    plt.ylabel("accuracy")
    plt.title(f"Accuracy, leraning rate: {filename.split('.')[0].split('-')[0]}")
    plt.show()

generate("nwm.txt")
# print(torch.cuda.is_available())