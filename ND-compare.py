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
wt = utils.wt

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
LR = utils.LR                      # learning rate

p_current = p_next.copy()    #current delivery ratio
NUM_EPISODE = 10  # the number of episode
LEN_EPISODE = 10000   # the length of each episode



def run_ND(func):
    result = torch.zeros((LEN_EPISODE, NUM_EPISODE))
    for i_episode in range(NUM_EPISODE):
        utils.init_state(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Action, sumAction)   # initialization
        global p_current
        p_current.fill(0)
        for len in range(LEN_EPISODE):  #length of each episode
            # currentActiveLink, action_probs = utils.AMIX_ND(np.multiply(Deficit, wt), e, totalBuff)
            currentActiveLink, action_probs = utils.AMIX_ND(Deficit, e, totalBuff)
            # utils.update_dynamics(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Arrival, 
            #                     sumArrival, Action, sumAction, totalBuff, a, LAMBDA, INIT_P, d_max
            #                     )                                                   # update dynamics
            utils.update_dynamics_take_action(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Arrival, 
                                            sumArrival, Action, sumAction, totalBuff, currentActiveLink, LAMBDA, INIT_P, d_max
                                            )
            # p_next here is the delivery ratio after taking action
            performance = 0
            if func == 1:
                performance = np.inner(wt, p_next)     # case 1
            elif func == 2:
                for idx in range(NUM_OF_LINKS):
                    performance += wt[idx] * p_next[idx] * np.power(2, 30 * np.min([0, p_next[idx] - INIT_P[idx]]))   # case 2
            elif func == 3:
                performance = np.inner(wt, np.log(p_next))     # case 3
            elif func == 4:
                performance = 0
            elif func == 5:
                for idx in range(NUM_OF_LINKS):
                    # performance += wt[idx] * (np.power(p_next[idx], 2) + 2 * p_next[idx])    # case 5
                    performance += wt[idx] * (10 * np.power(p_next[idx], 2) + p_next[idx])    # case 5
            else:
                performance = 0

            result[len][i_episode] = round(performance, 3)
            utils.update_dynamics_new_arrival(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Arrival, 
                                            sumArrival, Action, sumAction, totalBuff, currentActiveLink, LAMBDA, INIT_P, d_max
                                            )
            p_current = p_next.copy()

    result = result.sum(-1, keepdim = True)
    result = result / NUM_EPISODE
    res = result.detach().numpy()

    with open('ND-'+str(func)+'.txt', 'a+') as f:
        for x in res:
            f.write(str(x.item())+'\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=int, default='1')
    args = parser.parse_args()
    run_ND(func=args.func)