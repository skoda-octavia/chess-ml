import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, dropout):
        self.flat = nn.Flatten(),
        super(Classifier, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.input_batch_norm = nn.BatchNorm1d(hidden_sizes[0])

        self.act_input =  nn.Tanh()
        self.hidden_layers = [ nn.Linear(hidden_sizes[idx], hidden_sizes[idx + 1]) for idx in range(len(hidden_sizes) - 1) ]
        self.hidden_batch_norms = [ nn.BatchNorm1d(size) for size in hidden_sizes[1:]]
        self.act_hid = nn.Tanh()

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.output_act = nn.ReLU()

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flat(x)
        x = self.input_layer(x)
        x = self.input_batch_norm(x)
        x = self.act_input(x)
        x = self.dropout(x)
        for hidden_layer, batch_norm in zip(self.hidden_layers, self.hidden_batch_norms):
            x = hidden_layer(x)
            x = batch_norm(x)
            x = self.act_hid(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x

# create like
# model_class = Classifier(input_size_class, output_size_class, [50, 100, 100, 50], 0.05).to(device)