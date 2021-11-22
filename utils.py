import numpy as np
import random as rd
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
epsilon = 0.05  # action exploration: randomly select one link with probability epsilon


# 2 links
NUM_OF_LINKS=2
MAX_DEADLINE=5
d_max = [5, 5]
# LAMBDA = [0.75, 0.75]
LAMBDA = [1.2, 0.3]
# INIT_P = [0.6, 0.6]
INIT_P = [0.99, 0.5]
wt = [1, 1]
wt_pnt = [1, 1]      # weight of penalty

# # 5 links
# NUM_OF_LINKS=5
# MAX_DEADLINE=10
# d_max = [10, 10, 10, 10, 10]
# # LAMBDA = [0.75, 0.75]
# LAMBDA = [0.3, 0.3, 0.3, 0.3, 0.3]
# INIT_P = [0.6, 0.6, 0.6, 0.6, 0.6]
# wt = [1, 1, 1, 1, 1]
# wt_pnt = [1, 1, 1, 1, 1]      # weight of penalty








# NUM_OF_LINKS=6
# MAX_DEADLINE=10
# # d_max = [10, 10, 10, 10, 10, 10]
# # d_max = [5, 7, 10, 5, 7, 10]
# # d_max = [2, 6, 10, 2, 6, 10]
# # LAMBDA = [0.1, 0.15, 0.25, 0.1, 0.15, 0.25]
# # LAMBDA = [0.45, 0.15, 0.12, 0.1, 0.1, 0.08] # arrival rate = 1, new arrival packets share the same max deadline
# # INIT_P = [0.8, 0.9, 1.0, 0.8, 0.9, 1.0]
# # INIT_P = 0.66
# # LAMBDA = [0.15, 0.225, 0.375, 0.15, 0.225, 0.375]

# d_max = [3, 3, 10, 10, 10, 10]
# LAMBDA = [0.15, 0.15, 0.3, 0.3, 0.3, 0.3]
# # LAMBDA = [0.05, 0.05, 0.35, 0.35, 0.35, 0.35]
# # LAMBDA = [0.1, 0.1, 0.6, 0.3, 0.2, 0.2]
# px = 0.60
# INIT_P = [0.9, 0.9, px, px, px, px]

# HIDDEN_SIZE = 256    # the dim of hidden layers
HIDDEN_SIZE = 64    # the dim of hidden layers
LR = 1e-4            # learning rate of RL algorithm

Buffer = np.zeros((NUM_OF_LINKS, MAX_DEADLINE+2), dtype=np.float)   # 1 <= d <= d_max, Buffer[l][0]=0, Buffer[l][d_max+1]=0
Deficit = np.zeros((NUM_OF_LINKS), dtype=np.float)
totalArrival = np.zeros((NUM_OF_LINKS), dtype=np.float)   #TA: total arrival packets from beginning
totalDelivered = np.zeros((NUM_OF_LINKS), dtype=np.float) #TD: total delivered packets
p_next = np.zeros((NUM_OF_LINKS), dtype=np.float)
e = np.zeros((NUM_OF_LINKS), dtype=np.int)  # e[l]: earliest deadline of link l
Arrival = np.zeros((NUM_OF_LINKS, MAX_DEADLINE+1), dtype=np.float)
sumArrival = np.zeros((NUM_OF_LINKS), dtype=np.float)
Action = np.zeros((NUM_OF_LINKS, MAX_DEADLINE+2), dtype=np.int)
sumAction = np.zeros((NUM_OF_LINKS), dtype=np.int)
totalBuff = np.zeros((NUM_OF_LINKS), dtype=np.float)

N_ACTIONS = NUM_OF_LINKS
N_STATES = NUM_OF_LINKS * 2    #State s = (Deficit, e)
# N_STATES = NUM_OF_LINKS * (MAX_DEADLINE + 3)    #State s = (Deficit, e, Buffer, p)

# Hyperparameters
clip = 10.0     # gradient clipping
bound = 5.0    # clip input of softmax within [-bound,bound] to avoid nan
POLICY_UPDATE_DELAY = 10      # G: policy update delay
NUM_OF_AGENT=1
polyak = 0.99   # update target networks
# wi = 0.02
# wi = 1 / HIDDEN_SIZE
# wi = 2 / HIDDEN_SIZE
class Actor(nn.Module):
    def __init__(self, ):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(N_STATES, HIDDEN_SIZE)
        self.layer1.weight.data.normal_(0, 0.02)
        self.layer2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer2.weight.data.normal_(0, 0.02)
        self.layer3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer3.weight.data.normal_(0, 0.02)
        # self.layer4 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        # self.layer4.weight.data.normal_(0, 0.02)
        self.action = nn.Linear(HIDDEN_SIZE, N_ACTIONS)
        self.action.weight.data.normal_(0, 0.02)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        # x = self.layer4(x)
        # x = F.relu(x)
        x = self.action(x)
        # x = torch.clamp(x, -bound, bound) # clip input of softmax to avoid nan
        # action_probs = F.softmax(x, dim=-1)
        # return action_probs
        action_binary = F.gumbel_softmax(x, tau=0.01, hard=True, eps=1e-10, dim=-1)
        return action_binary

class Actor_no_gs(nn.Module):
    def __init__(self, ):
        super(Actor_no_gs, self).__init__()
        self.layer1 = nn.Linear(N_STATES, HIDDEN_SIZE)
        self.layer1.weight.data.normal_(0, 0.02)
        self.layer2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer2.weight.data.normal_(0, 0.02)
        self.layer3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer3.weight.data.normal_(0, 0.02)
        # self.layer4 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        # self.layer4.weight.data.normal_(0, 0.02)
        self.action = nn.Linear(HIDDEN_SIZE, N_ACTIONS)
        self.action.weight.data.normal_(0, 0.02)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        # x = self.layer4(x)
        # x = F.relu(x)
        x = self.action(x)
        # action_binary = F.gumbel_softmax(x, tau=1, hard=True, eps=1e-10, dim=-1)
        # return action_binary
        return x

class Critic(nn.Module):
    def __init__(self, ):
        super(Critic, self).__init__()
        # self.layer1 = nn.Linear(N_STATES + 1, HIDDEN_SIZE)    # input: state, action
        self.layer1 = nn.Linear(N_STATES + N_ACTIONS, HIDDEN_SIZE)    # input: state, action
        self.layer1.weight.data.normal_(0, 0.02)
        self.layer2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer2.weight.data.normal_(0, 0.02)
        self.layer3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer3.weight.data.normal_(0, 0.02)
        # self.layer4 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        # self.layer4.weight.data.normal_(0, 0.02)
        self.value = nn.Linear(HIDDEN_SIZE, 1)
        self.value.weight.data.normal_(0, 0.02)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        # x = self.layer4(x)
        # x = F.relu(x)
        state_value = self.value(x)
        return state_value

class MCTS_Actor(nn.Module):
    def __init__(self, ):
        super(MCTS_Actor, self).__init__()
        self.layer1 = nn.Linear(N_STATES, HIDDEN_SIZE)
        self.layer1.weight.data.normal_(0, 0.02)
        self.layer2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer2.weight.data.normal_(0, 0.02)
        self.layer3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer3.weight.data.normal_(0, 0.02)
        self.layer4 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer4.weight.data.normal_(0, 0.02)
        self.action = nn.Linear(HIDDEN_SIZE, N_ACTIONS)
        self.action.weight.data.normal_(0, 0.02)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = F.relu(x)
        x = self.action(x)
        x = torch.clamp(x, -bound, bound) # clip input of softmax to avoid nan
        action_probs = F.softmax(x, dim=-1)
        return action_probs

class MCTS_Critic(nn.Module):
    def __init__(self, ):
        super(MCTS_Critic, self).__init__()
        self.layer1 = nn.Linear(N_STATES + 1, HIDDEN_SIZE)    # input: state, action
        self.layer1.weight.data.normal_(0, 0.02)
        self.layer2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer2.weight.data.normal_(0, 0.02)
        self.layer3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer3.weight.data.normal_(0, 0.02)
        self.layer4 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer4.weight.data.normal_(0, 0.02)
        self.value = nn.Linear(HIDDEN_SIZE, 1)
        self.value.weight.data.normal_(0, 0.02)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = F.relu(x)
        state_value = self.value(x)
        return state_value

# find the link l which should be active in current time slot
def AMIX_ND(Deficit, e, totalBuff):
    ND_ActiveLink = 0
    # remember: localDeficit = Deficit is wrong!
    localDeficit = Deficit.copy()
    ND = []
    # prob[l]: probability of link l to be active
    prob = np.zeros((NUM_OF_LINKS), dtype=np.float)
    # only links with nonempty buffer should be taken into account
    for l in range(NUM_OF_LINKS):
        #totalBuff = 0
        if totalBuff[l] == 0:
            localDeficit[l] = 0

    while True:
        maxDeficit = 0
        maxDificitLink = 0
        # find the link with maximal deficit
        for l in range(NUM_OF_LINKS):
            if maxDeficit < localDeficit[l]:
                maxDeficit = localDeficit[l]
                maxDificitLink = l
        if maxDeficit > 0:
            # find all the links with the same maximal deficit (nonzero), then choose the one with smallest e
            for l in range(NUM_OF_LINKS):
                if localDeficit[l] == maxDeficit:
                    if e[l] < e[maxDificitLink]:
                        maxDificitLink = l
            ND.append(maxDificitLink)
            for l in range(NUM_OF_LINKS):
                if e[l] >= e[maxDificitLink]:
                    localDeficit[l] = 0 # delete the dominated links
        else:
            break

    k = len(ND)
    # if all deficit=0, then return the link with smallest e
    if k == 0:
        # if all buffers are empty, then no link should be active
        ND_ActiveLink = -1
        prob = np.zeros((NUM_OF_LINKS), dtype=np.float)
        # if np.min(e) == MAX_DEADLINE+1: # e[l] initialized as MAX_DEADLINE+1
        #     ND_ActiveLink = -1
        #     prob = np.zeros((NUM_OF_LINKS), dtype=np.float)
        # else:
        #     ND_ActiveLink = np.argmin(e)
        #     prob[ND_ActiveLink] = 1
    # if one link dominates all other links, then active_prob = 1
    elif k == 1:
        ND_ActiveLink = ND[0]
        prob[ND_ActiveLink] = 1
    else:
        r = 1
        for i in range(k-1):
            prob[ND[i]] = min(r, 1 - Deficit[ND[i+1]] / Deficit[ND[i]])
            r = r - prob[ND[i]]
        prob[ND[k-1]] = r
        # torch.multinomial(prob, 1).item()
        start = 0
        randnum = rd.randint(1, 1000000)
        for i in range(k):
            start = start + 1000000*prob[ND[i]]
            if randnum <= start:
                ND_ActiveLink = ND[i]
                break
    return ND_ActiveLink, prob

def generate_state(Deficit, e):  #generate 1-D state
    arr1 = np.array(Deficit)
    arr2 = np.array(e)
    result = np.concatenate((arr1, arr2))
    result = torch.FloatTensor(result)
    return result

def generate_state_1(Deficit, e, Buffer, p):  #generate 1-D state
    #arr1 = np.array(Buffer)[:, 1:MAX_DEADLINE+1]
    arr1 = np.array(Deficit)
    arr2 = np.array(e)
    arr3 = Buffer.flatten()
    arr4 = np.array(p)
    result = np.concatenate((arr1, arr2, arr3, arr4))
    result = torch.FloatTensor(result)
    return result

def generate_state_2(Deficit, e, totalBuff, p):  #generate 1-D state
    #arr1 = np.array(Buffer)[:, 1:MAX_DEADLINE+1]
    arr1 = np.array(Deficit)
    arr2 = np.array(e)
    # arr3 = Buffer.flatten()
    arr3 = np.array(totalBuff)
    arr4 = np.array(p)
    result = np.concatenate((arr1, arr2, arr3, arr4))
    result = torch.FloatTensor(result)
    return result

def store_state_actionprobs(s, action_probs, memory, memory_counter):
    transition = np.hstack((s, action_probs))
    memory[memory_counter, :] = transition
    memory_counter += 1
    return memory_counter

# non-i.i.d. with Poisson distribution
def ARR_POISSON(LAMBDA, Arrival, d_max):
    Arrival.fill(0)
    for l in range(NUM_OF_LINKS):
        # Arrival[l][MAX_DEADLINE] = np.random.poisson(LAMBDA[l])
        Arrival[l][d_max[l]] = np.random.poisson(LAMBDA[l])
    return Arrival

def init_state(Buffer, Deficit, totalArrival, totalDelivered, p_next, Action, sumAction, e):
    Buffer.fill(0)
    Deficit.fill(0)
    totalArrival.fill(0)
    totalDelivered.fill(0)
    p_next.fill(0)
    Action.fill(0)
    sumAction.fill(0)
    e.fill(MAX_DEADLINE+1)

def update_dynamics(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Arrival, sumArrival, Action, sumAction, totalBuff, currentActiveLink, LAMBDA, INIT_P, d_max):
    # update Action
    Action.fill(0)
    if currentActiveLink != -1:
        elstDeadline = e[currentActiveLink]
        Action[currentActiveLink][elstDeadline] = 1
    # total departure num on link l
    sumAction.fill(0)
    for l in range(NUM_OF_LINKS):
        for d in range(1, MAX_DEADLINE+1):
            sumAction[l] = sumAction[l] + Action[l][d]
    # new arrival packets
    Arrival = ARR_POISSON(LAMBDA, Arrival, d_max)
    # total arrival num on link l
    sumArrival.fill(0)
    for l in range(NUM_OF_LINKS):
        for d in range(1, MAX_DEADLINE+1):
            sumArrival[l] = sumArrival[l] + Arrival[l][d]
    # update total arrival packets from beginning
    for l in range(NUM_OF_LINKS):
        totalArrival[l] += sumArrival[l]
        totalDelivered[l] += sumAction[l]
    # update buffer
    for l in range(NUM_OF_LINKS):
        for d in range(1, MAX_DEADLINE+1):
            Buffer[l][d] = max(Buffer[l][d+1] + Arrival[l][d] - Action[l][d+1], 0)
    # # update totalBuff
    totalBuff.fill(0)
    for l in range(NUM_OF_LINKS):
        for d in range(1, MAX_DEADLINE+1):
            totalBuff[l] = totalBuff[l] + Buffer[l][d]
    # update deficit
    for l in range(NUM_OF_LINKS):
        # Deficit[l] = max(Deficit[l] + sumArrival[l] * INIT_P - sumAction[l], 0)
        Deficit[l] = max(Deficit[l] + sumArrival[l] * INIT_P[l] - sumAction[l], 0)
    # update the earliest deadline on link l
    e.fill(MAX_DEADLINE+1)      #initial earliest deadline should be MAX_DEADLINE+1
    for l in range(NUM_OF_LINKS):
        for d in range(1, MAX_DEADLINE+1):
            if Buffer[l][d] > 0:
                e[l] = d
                break
    for l in range(NUM_OF_LINKS):
        if totalArrival[l] == 0:
            p_next[l] = 0
        else:
            p_next[l] = totalDelivered[l] / totalArrival[l] #next delivery ratio

def update_dynamics_take_action(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Arrival, sumArrival, Action, sumAction, totalBuff, currentActiveLink, LAMBDA, INIT_P, d_max):
    # update Action
    Action.fill(0)
    if currentActiveLink != -1:
        elstDeadline = e[currentActiveLink]
        Action[currentActiveLink][elstDeadline] = 1
    # total departure num on link l
    sumAction.fill(0)
    for l in range(NUM_OF_LINKS):
        for d in range(1, MAX_DEADLINE+1):
            sumAction[l] = sumAction[l] + Action[l][d]
    # new arrival packets
    Arrival.fill(0)
    # total arrival num on link l
    sumArrival.fill(0)
    # update total arrival packets from beginning
    for l in range(NUM_OF_LINKS):
        totalArrival[l] += sumArrival[l]
        totalDelivered[l] += sumAction[l]
    # update buffer
    for l in range(NUM_OF_LINKS):
        for d in range(1, MAX_DEADLINE+1):
            Buffer[l][d] = max(Buffer[l][d+1] + Arrival[l][d] - Action[l][d+1], 0)
    # update deficit
    for l in range(NUM_OF_LINKS):
        Deficit[l] = max(Deficit[l] + sumArrival[l] * INIT_P[l] - sumAction[l], 0)
    # update the earliest deadline on link l
    e.fill(MAX_DEADLINE+1)      #initial earliest deadline should be MAX_DEADLINE+1
    for l in range(NUM_OF_LINKS):
        for d in range(1, MAX_DEADLINE+1):
            if Buffer[l][d] > 0:
                e[l] = d
                break
    for l in range(NUM_OF_LINKS):
        if totalArrival[l] == 0:
            p_next[l] = 0
        else:
            p_next[l] = totalDelivered[l] / totalArrival[l] #next delivery ratio

def update_dynamics_new_arrival(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Arrival, sumArrival, Action, sumAction, totalBuff, currentActiveLink, LAMBDA, INIT_P, d_max):
    # update Action
    Action.fill(0)
    # total departure num on link l
    sumAction.fill(0)
    # new arrival packets
    Arrival = ARR_POISSON(LAMBDA, Arrival, d_max)
    # total arrival num on link l
    sumArrival.fill(0)
    for l in range(NUM_OF_LINKS):
        for d in range(1, MAX_DEADLINE+1):
            sumArrival[l] = sumArrival[l] + Arrival[l][d]
    # update total arrival packets from beginning
    for l in range(NUM_OF_LINKS):
        totalArrival[l] += sumArrival[l]
        totalDelivered[l] += sumAction[l]
    # update buffer
    for l in range(NUM_OF_LINKS):
        for d in range(1, MAX_DEADLINE+1):
            Buffer[l][d] = max(Buffer[l][d+1] + Arrival[l][d] - Action[l][d+1], 0)
    # # update totalBuff
    totalBuff.fill(0)
    for l in range(NUM_OF_LINKS):
        for d in range(1, MAX_DEADLINE+1):
            totalBuff[l] = totalBuff[l] + Buffer[l][d]
    # update deficit
    for l in range(NUM_OF_LINKS):
        Deficit[l] = max(Deficit[l] + sumArrival[l] * INIT_P[l] - sumAction[l], 0)
    # update the earliest deadline on link l
    e.fill(MAX_DEADLINE+1)      #initial earliest deadline should be MAX_DEADLINE+1
    for l in range(NUM_OF_LINKS):
        for d in range(1, MAX_DEADLINE+1):
            if Buffer[l][d] > 0:
                e[l] = d
                break
    for l in range(NUM_OF_LINKS):
        if totalArrival[l] == 0:
            p_next[l] = 0
        else:
            p_next[l] = totalDelivered[l] / totalArrival[l] #next delivery ratio

def AMIX_ND_DEF(Deficit, e, totalBuff):
    ND_ActiveLink = 0
    # remember: localDeficit = Deficit is wrong!
    localDeficit = Deficit.copy()
    ND = []
    # prob[l]: probability of link l to be active
    prob = np.zeros((NUM_OF_LINKS), dtype=np.float)
    # only links with nonempty buffer should be taken into account
    for l in range(NUM_OF_LINKS):
        #totalBuff = 0
        if totalBuff[l] == 0:
            localDeficit[l] = 0

    while True:
        maxDeficit = 0
        maxDificitLink = 0
        # find the link with maximal deficit
        for l in range(NUM_OF_LINKS):
            if maxDeficit < localDeficit[l]:
                maxDeficit = localDeficit[l]
                maxDificitLink = l
        if maxDeficit > 0:
            # find all the links with the same maximal deficit (nonzero), then choose the one with smallest e
            for l in range(NUM_OF_LINKS):
                if localDeficit[l] == maxDeficit:
                    if e[l] < e[maxDificitLink]:
                        maxDificitLink = l
            ND.append(maxDificitLink)
            for l in range(NUM_OF_LINKS):
                if e[l] >= e[maxDificitLink]:
                    localDeficit[l] = 0 # delete the dominated links
        else:
            break

    k = len(ND)
    # if all deficit=0, then return the link with smallest e
    if k == 0:
        # if all buffers are empty, then no link should be active
        if np.min(e) == MAX_DEADLINE+1: # e[l] initialized as MAX_DEADLINE+1
            ND_ActiveLink = -1
            prob = np.zeros((NUM_OF_LINKS), dtype=np.float)
        else:
            ND_ActiveLink = np.argmin(e)
            prob[ND_ActiveLink] = 1
    # if one link dominates all other links, then active_prob = 1
    elif k == 1:
        ND_ActiveLink = ND[0]
        prob[ND_ActiveLink] = 1
    else:
        # r = 1
        # for i in range(k-1):
        #     prob[ND[i]] = min(r, 1 - Deficit[ND[i+1]] / Deficit[ND[i]])
        #     r = r - prob[ND[i]]
        # prob[ND[k-1]] = r
        sum_def = 0
        for i in range(k):
            sum_def += Deficit[ND[i]]
        for i in range(k):
            prob[ND[i]] = Deficit[ND[i]] / sum_def

        start = 0
        randnum = rd.randint(1, 1000000)
        for i in range(k):
            start = start + 1000000*prob[ND[i]]
            if randnum <= start:
                ND_ActiveLink = ND[i]
                break
    return ND_ActiveLink, prob

def AMIX_ND_DEF_ED(Deficit, e, totalBuff):
    ND_ActiveLink = 0
    # remember: localDeficit = Deficit is wrong!
    localDeficit = Deficit.copy()
    ND = []
    # prob[l]: probability of link l to be active
    prob = np.zeros((NUM_OF_LINKS), dtype=np.float)
    # only links with nonempty buffer should be taken into account
    for l in range(NUM_OF_LINKS):
        #totalBuff = 0
        if totalBuff[l] == 0:
            localDeficit[l] = 0

    while True:
        maxDeficit = 0
        maxDificitLink = 0
        # find the link with maximal deficit
        for l in range(NUM_OF_LINKS):
            if maxDeficit < localDeficit[l]:
                maxDeficit = localDeficit[l]
                maxDificitLink = l
        if maxDeficit > 0:
            # find all the links with the same maximal deficit (nonzero), then choose the one with smallest e
            for l in range(NUM_OF_LINKS):
                if localDeficit[l] == maxDeficit:
                    if e[l] < e[maxDificitLink]:
                        maxDificitLink = l
            ND.append(maxDificitLink)
            for l in range(NUM_OF_LINKS):
                if e[l] >= e[maxDificitLink]:
                    localDeficit[l] = 0 # delete the dominated links
        else:
            break

    k = len(ND)
    # if all deficit=0, then return the link with smallest e
    if k == 0:
        # if all buffers are empty, then no link should be active
        if np.min(e) == MAX_DEADLINE+1: # e[l] initialized as MAX_DEADLINE+1
            ND_ActiveLink = -1
            prob = np.zeros((NUM_OF_LINKS), dtype=np.float)
        else:
            ND_ActiveLink = np.argmin(e)
            prob[ND_ActiveLink] = 1
    # if one link dominates all other links, then active_prob = 1
    elif k == 1:
        ND_ActiveLink = ND[0]
        prob[ND_ActiveLink] = 1
    else:
        def_ed = np.zeros((NUM_OF_LINKS), dtype=np.float)
        for i in range(k):
            def_ed[ND[i]] = Deficit[ND[i]] / e[ND[i]]
        sum_def_ed = 0
        for i in range(k):
            sum_def_ed += def_ed[ND[i]]
        for i in range(k):
            prob[ND[i]] = def_ed[ND[i]] / sum_def_ed

        start = 0
        randnum = rd.randint(1, 1000000)
        for i in range(k):
            start = start + 1000000*prob[ND[i]]
            if randnum <= start:
                ND_ActiveLink = ND[i]
                break
    return ND_ActiveLink, prob

def perform_func(findex, wt, p_next, INIT_P, wt_pnt):   # performance metric
    performance = 0
    if findex == 1:
        performance = np.inner(wt, p_next)     # case 1
    elif findex == 2:
        performance = np.inner(wt, np.log(p_next+1e-8))     # case 2
    elif findex == 3:
        for idx in range(NUM_OF_LINKS):
            performance += wt[idx] * (10 * np.power(p_next[idx], 2) + p_next[idx])    # case 3
            # performance += wt[idx] * (np.power(p_next[idx], 2) + 2 * p_next[idx])    # case 3
    elif findex == 4:                           # case 1 with penalty-1
        performance = np.inner(wt, p_next)
        for idx in range(NUM_OF_LINKS):
            # performance += np.max(wt) * np.min([0, p_next[idx] - INIT_P[idx]])
            performance += wt[idx] * np.min([0, p_next[idx] - INIT_P[idx]])
    elif findex == 5:                           # case 1 with penalty-1
        performance = np.inner(wt, p_next)
        for idx in range(NUM_OF_LINKS):
            performance += wt_pnt[idx] * np.min([0, p_next[idx] - INIT_P[idx]])
    elif findex == 6:
        performance = np.min(p_next / np.array(INIT_P))
    # elif findex == 5:                           # case 1 with penalty-2
    #     performance = np.inner(wt, p_next)
    #     for idx in range(NUM_OF_LINKS):
    #         performance += wt[idx] * np.min([0, p_next[idx] - INIT_P[idx]])
    # elif findex == 5:                           # case 2 with penalty
    #     performance = np.inner(wt, np.log(p_next+1e-8))
    #     for idx in range(NUM_OF_LINKS):
    #         performance += wt[idx] * np.min([0, p_next[idx] - INIT_P[idx]])
    # elif findex == 6:                           # case 3 with penalty
    #     for idx in range(NUM_OF_LINKS):
    #         performance += wt[idx] * (10 * np.power(p_next[idx], 2) + p_next[idx]) + wt[idx] * np.min([0, p_next[idx] - INIT_P[idx]])
    # elif findex == 4:
    #     for idx in range(NUM_OF_LINKS):
    #         performance += wt[idx] * p_next[idx] * np.power(2, 30 * np.min([0, p_next[idx] - INIT_P[idx]]))   # case 4
    else:
        performance = 0
    return performance
