import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random as rd
import torch.optim as optim
import utils

device = utils.device

INIT_P = utils.INIT_P
LAMBDA = utils.LAMBDA
NUM_OF_LINKS = utils.NUM_OF_LINKS
MAX_DEADLINE = utils.MAX_DEADLINE
d_max = utils.d_max

Buffer = utils.Buffer
Deficit = utils.Deficit
totalArrival = utils.totalArrival
totalDelivered = utils.totalDelivered
p_next = utils.p_next
e = utils.e
Arrival = utils.Arrival
sumArrival = utils.sumArrival
Action = utils.Action
sumAction = utils.sumAction
totalBuff = utils.totalBuff

N_ACTIONS = utils.N_ACTIONS
N_STATES = utils.N_STATES
HIDDEN_SIZE = utils.HIDDEN_SIZE    # the dim of hidden layers
LR = 3e-4

NUM_EPISODE = 500   # the number of episode
LEN_EPISODE = 2000   # the length of each episode
MEMORY_CAPACITY = NUM_EPISODE * LEN_EPISODE     # memory
memory = np.zeros((MEMORY_CAPACITY, N_STATES + N_ACTIONS))
                      # learning rate
NUM_EPOCH = 100000
BATCH_SIZE = 128

memory = np.load('dataset.npy')
np.random.shuffle(memory)       # shuffle dataset
trainingset = np.copy(memory[:-int(0.2 * MEMORY_CAPACITY)])     # 80% for training
testset = np.copy(memory[-int(0.2 * MEMORY_CAPACITY):])         # 20% for testing
trainingloss = []
testloss = []
initialNet = utils.Actor_no_gs().to(device)     # NN without gumbel softmax activation function
optimizer = optim.Adam(initialNet.parameters(), lr = LR)
criterion = nn.MSELoss()
for epoch in range(NUM_EPOCH):
    sample_index = np.random.choice(len(trainingset), BATCH_SIZE)
    b_memory = trainingset[sample_index, :]
    b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
    b_s = b_s.to(device)
    b_ap = Variable(torch.FloatTensor(b_memory[:, -N_ACTIONS:]))
    b_ap = b_ap.to(device)
    b_net = initialNet(b_s)
    loss = criterion(b_net, b_ap)
    if epoch % 100 == 0:
        trainingloss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    test_s = Variable(torch.FloatTensor(testset[:, :N_STATES]))
    test_s = test_s.to(device)
    test_ap = Variable(torch.FloatTensor(testset[:, -N_ACTIONS:]))
    test_ap = test_ap.to(device)
    test_net = initialNet(test_s)
    tloss = criterion(test_net, test_ap)
    if epoch % 100 == 0:
        testloss.append(tloss.item())

with open('Training-loss.txt', 'a+') as f:
    for item in trainingloss:
        f.write(str(item)+'\n')
with open('Test-loss.txt', 'a+') as f:
    for item in testloss:
        f.write(str(item)+'\n')
torch.save(initialNet.state_dict(), 'initialized-actor.pkl')
print('Training model complete. Model saved.')
