import torch
import numpy as np
import random as rd
import utils

device = utils.device

INIT_P = utils.INIT_P
LAMBDA = utils.LAMBDA
NUM_OF_LINKS = utils.NUM_OF_LINKS
MAX_DEADLINE = utils.MAX_DEADLINE
d_max = utils.d_max
wt = utils.wt
wt_pnt = utils.wt_pnt

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
# LR = utils.LR                      # learning rate

p_current = p_next.copy()    #current delivery ratio
NUM_EPISODE = 10  # the number of episode
LEN_EPISODE = 10000   # the length of each episode

PP = []     # record the actual delivery ratio at each step

def run_ND(findex, adjust):
    result = torch.zeros((LEN_EPISODE, NUM_EPISODE))
    for i_episode in range(NUM_EPISODE):
        utils.init_state(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Action, sumAction)   # initialization
        global p_current
        p_current.fill(0)
        for len in range(LEN_EPISODE):  #length of each episode
            if adjust == 0:
                currentActiveLink, action_probs = utils.AMIX_ND(Deficit, e, totalBuff)
            else:
                currentActiveLink, action_probs = utils.AMIX_ND(np.multiply(Deficit, wt), e, totalBuff)
            
            # utils.update_dynamics(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Arrival, 
            #                     sumArrival, Action, sumAction, totalBuff, a, LAMBDA, INIT_P, d_max
            #                     )                                                   # update dynamics
            utils.update_dynamics_take_action(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Arrival, 
                                            sumArrival, Action, sumAction, totalBuff, currentActiveLink, LAMBDA, INIT_P, d_max
                                            )
            # p_next here is the delivery ratio after taking action
            performance = utils.perform_func(findex, wt, p_next, INIT_P, wt_pnt)

            if i_episode == 0:
                PP.append(p_next.copy())        # record actual delivery ratio only once
            result[len][i_episode] = round(performance, 6)
            utils.update_dynamics_new_arrival(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Arrival, 
                                            sumArrival, Action, sumAction, totalBuff, currentActiveLink, LAMBDA, INIT_P, d_max
                                            )
            p_current = p_next.copy()

    result = result.sum(-1, keepdim = True)
    result = result / NUM_EPISODE
    res = result.detach().numpy()

    with open('ND-f-'+str(findex)+'-adjust-'+str(adjust)+'.txt', 'a+') as f:
        for x in res:
            f.write(str(x.item())+'\n')
    np.savetxt('ND-f-'+str(findex)+'-adjust-'+str(adjust)+'-delivery ratio.txt', PP)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--findex', type=int, default='1')
    parser.add_argument('--adjust', type=int, default='0')
    args = parser.parse_args()
    run_ND(findex=args.findex, adjust=args.adjust)