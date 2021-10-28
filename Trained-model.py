from numpy.lib.npyio import savetxt
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

PP = []     # record the actual delivery ratio at each step

def run_trained_model(model, findex):
    initialNet = utils.Actor().to(device)
    initialNet.load_state_dict(torch.load(model+'.pkl'))

    result = torch.zeros((LEN_EPISODE, NUM_EPISODE))
    for i_episode in range(NUM_EPISODE):
        utils.init_state(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Action, sumAction)   # initialization
        global p_current
        p_current.fill(0)
        # s = utils.generate_state(Deficit, e, Buffer[:,1:MAX_DEADLINE+1], p_next)    # current state
        s = utils.generate_state(Deficit, e)    # current state

        for len in range(LEN_EPISODE):  #length of each episode
            dist = initialNet(s.to(device)).detach().cpu()
            a = torch.multinomial(dist, 1).item()
            # utils.update_dynamics(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Arrival, 
            #                     sumArrival, Action, sumAction, totalBuff, a, LAMBDA, INIT_P, d_max
            #                     )                                                   # update dynamics
            utils.update_dynamics_take_action(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Arrival, 
                                            sumArrival, Action, sumAction, totalBuff, a, LAMBDA, INIT_P, d_max
                                            )
            # p_next here is the delivery ratio after taking action
            performance = utils.perform_func(findex, wt, p_next, INIT_P)

            if i_episode == 0:
                PP.append(p_next.copy())        # record actual delivery ratio only once

            result[len][i_episode] = round(performance, 6)

            utils.update_dynamics_new_arrival(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Arrival, 
                                            sumArrival, Action, sumAction, totalBuff, a, LAMBDA, INIT_P, d_max
                                            )
            # s_ = utils.generate_state(Deficit, e, Buffer[:,1:MAX_DEADLINE+1], p_next)
            s_ = utils.generate_state(Deficit, e)
            s = s_.clone().detach()
            p_current = p_next.copy()

    result = result.sum(-1, keepdim = True)
    result = result / NUM_EPISODE
    res = result.detach().numpy()

    with open(model+'-'+str(findex)+'.txt', 'a+') as f:
        for x in res:
            f.write(str(x.item())+'\n')
    np.savetxt(model+'-'+str(findex)+'delivery ratio.txt', PP)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='initialized-actor')
    parser.add_argument('--findex', type=int, default='1')
    args = parser.parse_args()
    run_trained_model(model=args.model, findex=args.findex)