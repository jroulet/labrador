import pandas as pd
import numpy as np

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# options to change
sim_dir = Path('/home/lucy.thomas/projects/cog/cogwheel-machine/data/10k_set')
output_dir = Path('/home/lucy.thomas/projects/cog/cogwheel-machine/classifier/10k_test')
layer_size = 100 # size of hidden layers
learning_rate = 0.001
n_epochs=1000
batch_size=100

# x data
simulation_data = np.load(sim_dir/'simulation_data.npy')
folded_sampled_params = np.load(sim_dir/'folded_sampled_params.npy')
input_data = np.hstack((folded_sampled_params,simulation_data))
X = torch.tensor(input_data, dtype=torch.float32)

# y data (targets)
unfolding_labels = np.load(sim_dir/'unfolding_labels.npy')
unfolding_labels_expanded = np.zeros((np.shape(unfolding_labels)[0], 16), dtype=int)
for i, label in enumerate(unfolding_labels):
    unfolding_labels_expanded[i, label] = 1
y = torch.tensor(unfolding_labels_expanded, dtype=torch.float32)

# initialise the model
input_dim = np.shape(simulation_data)[1] + np.shape(folded_sampled_params)[1]
output_dim = np.shape(unfolding_labels_expanded)[1]
class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()
    self.hidden1 = nn.Linear(input_dim, layer_size)
    self.activation1 = nn.ReLU()
    self.hidden2 = nn.Linear(layer_size, layer_size)
    self.activation2 = nn.ReLU()
    self.output = nn.Linear(layer_size, output_dim)
    self.activation_output = nn.LogSoftmax(dim=1)
 
  def forward(self, x):
      x = self.activation1(self.hidden1(x))
      x = self.activation2(self.hidden2(x))
      x = self.activation_output(self.output(x))
      return x
    
 
model = Network()
print(model)

# loss function and optimiser
kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train the network
epoch_losses=[]
for epoch in range(n_epochs):
    losses = []
    print(f'EPOCH NUMBER {epoch}')
    for batch in range(0, len(X), batch_size):
        Xbatch = X[batch:batch+batch_size]
        y_pred = model(Xbatch)
        #print('PREDICTED:')
        #print(y_pred)
        ybatch = y[batch:batch+batch_size]
        #loss = loss_fn(y_pred,ybatch)
        loss = kl_loss(y_pred,ybatch)
        loss_np = loss.detach().numpy()
        #print('LOSS:')
        #print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(f'Finished batch {batch}, latest loss {loss}')
        losses = np.append(losses,loss_np)
    epoch_losses = np.append(epoch_losses, np.average(losses))
    print(f'Finished epoch {epoch}, latest loss {loss}')

plt.plot(range(n_epochs),epoch_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(output_dir/'training_loss.png')

# save the trained model
torch.save(model.state_dict(), output_dir/'trained_model')