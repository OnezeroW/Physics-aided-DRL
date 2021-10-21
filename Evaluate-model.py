import numpy as np
import random as rd
import torch
import utils

INIT_P = 0.73
P_STEP = 0.01
# INIT_P = utils.INIT_P
LEN_EPISODE = 50000     # length of episode to find the maximum reachable delivery ratio
P = []
PD = []

device = utils.device

NUM_OF_LINKS = utils.NUM_OF_LINKS
MAX_DEADLINE = utils.MAX_DEADLINE
LAMBDA = utils.LAMBDA

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

p_current = p_next.copy()    #current delivery ratio

while INIT_P <= 0.8:
    INIT_P = round(INIT_P, 2)
    utils.init_state(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Action, sumAction)   # initialization
    initialNet = utils.Actor().to(device)
    initialNet.load_state_dict(torch.load('EnsembleQ-p-0.73.pkl'))
    p_current.fill(0)
    s = utils.generate_state(Deficit, e, Buffer[:,1:MAX_DEADLINE+1], p_next)    # current state
    # result = torch.zeros((LEN_EPISODE, NUM_EPISODE))
    for index in range(LEN_EPISODE):
        dist = initialNet(s.to(device)).detach().cpu()
        a = torch.multinomial(dist, 1).item()
        # currentActiveLink, action_probs = utils.AMIX_ND(Deficit, e, totalBuff)
        utils.update_dynamics(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Arrival,
                              sumArrival, Action, sumAction, totalBuff, a, LAMBDA, INIT_P
                              )
        s_ = utils.generate_state(Deficit, e, Buffer[:,1:MAX_DEADLINE+1], p_next)
        r = np.min(p_next)
        p_current = p_next.copy()
        s = s_.clone().detach()
    P.append(INIT_P)
    PD.append(Deficit.copy())
    INIT_P += P_STEP
print(PD)
PD = np.transpose(PD)
for link in range(NUM_OF_LINKS):
    with open('link'+str(link+1)+'.txt', 'a+') as f:
        for x in PD[link]:
            f.write(str(x)+'\n')
with open('qos'+'.txt', 'a+') as f:
    for x in P:
        f.write(str(x)+'\n')
    #     print('index=', index, ', p=', np.min(p_next))
    # print('index=', index, ', Deficit=', Deficit, ', p=', np.min(p_next), flush=True)
    # avg_Deficit = np.mean(Deficit)
    # if avg_Deficit > 1.0:
    #     print('Maximum delivery ratio = ', round(INIT_P - P_STEP, 2), flush=True)
    #     break
    # else:
    #     print(INIT_P)
    #     INIT_P += P_STEP

# utils.init_state(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Action, sumAction)   # initialization
# for index in range(LEN_EPISODE):
#     currentActiveLink, action_probs = utils.AMIX_ND(Deficit, e, totalBuff)
#     utils.update_dynamics(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Arrival,
#                             sumArrival, Action, sumAction, totalBuff, currentActiveLink, LAMBDA, INIT_P
#                             )
#     if index % 100 == 0:
#         print('index=', index, ', Deficit=', Deficit, ', p=', p_next, flush=True)
# print('index=', index, ', Deficit=', Deficit, ', p=', np.min(p_next), flush=True)