import wandb
import torch
from torch import tensor
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random as rd
import torch.optim as optim
import utils

device = utils.device
# Hyperparameters
POLICY_UPDATE_DELAY = utils.POLICY_UPDATE_DELAY      # G: policy update delay
NUM_OF_AGENT = utils.NUM_OF_AGENT
polyak = utils.polyak   # update target networks
clip = utils.clip       # gradient clipping to avoid nan value

# STEP_PER_EPOCH = 100
# NUM_OF_EPOCH = 200
# LEN_EPISODE = STEP_PER_EPOCH * NUM_OF_EPOCH
LEN_EPISODE = 10000

INIT_P = utils.INIT_P
LAMBDA = utils.LAMBDA
NUM_OF_LINKS = utils.NUM_OF_LINKS
MAX_DEADLINE = utils.MAX_DEADLINE
d_max = utils.d_max
wt = utils.wt
wt_pnt = utils.wt_pnt
epsilon = utils.epsilon

N_ACTIONS = utils.N_ACTIONS
N_STATES = utils.N_STATES

Buffer = utils.Buffer
Deficit = utils.Deficit
totalArrival = utils.totalArrival
totalDelivered = utils.totalDelivered
e = utils.e
Arrival = utils.Arrival
sumArrival = utils.sumArrival
Action = utils.Action
sumAction = utils.sumAction
totalBuff = utils.totalBuff
p_next = utils.p_next
p_current = p_next.copy()    #current delivery ratio

# MEMORY_CAPACITY = NUM_OF_AGENT  # H*K
MEMORY_CAPACITY = 1000
BATCH_SIZE = 64
memory_counter = 0
memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + N_ACTIONS + 1))

def store_transition(memory, s, a, r, s_):
    # global memory
    global memory_counter
    global MEMORY_CAPACITY
    transition = np.hstack((s, a, [r], s_))
    # replace the old memory with new memory
    index = memory_counter % MEMORY_CAPACITY
    memory[index, :] = transition
    memory_counter += 1

def reward_func(rindex, wt, p_next, p_current, INIT_P, a, wt_pnt):  # a is an int value
    if rindex == 1:
        r = wt[a] * p_next[a]    # case 1
    elif rindex == 2:
        r = wt[a] * p_next[a]
        for idx in range(NUM_OF_LINKS):
            r += wt[idx] * np.min([0, p_next[idx] - INIT_P[idx]])    # case 2
    elif rindex == 3:
        r = wt[a] * (p_next[a] - p_current[a])    # case 3
    elif rindex == 4:
        r = wt[a] * (p_next[a] - p_current[a])
        for idx in range(NUM_OF_LINKS):
            r += wt[idx] * np.min([0, p_next[idx] - INIT_P[idx]])    # case 4
    elif rindex == 5:
        r = wt[a] * np.log(p_next[a]+1e-8)
    elif rindex == 6:
        r = wt[a] * (np.log(p_next[a]+1e-8) - np.log(p_current[a]+1e-8))
    elif rindex == 7:
        r = wt[a] * (10 * np.power(p_next[a], 2) + p_next[a])
    elif rindex == 8:
        r = wt[a] * (10 * np.power(p_next[a], 2) + p_next[a] - 10 * np.power(p_current[a], 2) - p_current[a])
    elif rindex == 9:
        r = wt[a] * p_next[a]
        for idx in range(NUM_OF_LINKS):
            if idx != a:
                # r += np.max(wt) * np.min([0, p_next[idx] - INIT_P[idx]])    # performace metric 1 with penalty
                r += wt[idx] * np.min([0, p_next[idx] - INIT_P[idx]])
    elif rindex == 10:
        r = wt[a] * (p_next[a] - p_current[a])
        for idx in range(NUM_OF_LINKS):
            if idx != a:
                # r += np.max(wt) * np.min([0, p_next[idx] - INIT_P[idx]])    # performace metric 1 with penalty
                r += wt[idx] * np.min([0, p_next[idx] - INIT_P[idx]])
    elif rindex == 11:
        r = wt[a] * p_next[a]
        for idx in range(NUM_OF_LINKS):
            if idx != a:
                r += wt_pnt[idx] * np.min([0, p_next[idx] - INIT_P[idx]])    # performace metric 1 with penalty
    elif rindex == 12:
        r = wt[a] * (p_next[a] - p_current[a])
        for idx in range(NUM_OF_LINKS):
            if idx != a:
                r += wt_pnt[idx] * np.min([0, p_next[idx] - INIT_P[idx]])    # performace metric 1 with penalty
    elif rindex == 13:
        r = np.min(p_next / np.array(INIT_P))
    elif rindex == 14:
        r = np.min(p_next / np.array(INIT_P)) - np.min(p_current / np.array(INIT_P))
    # elif rindex == 5:
    #     r = np.inner(wt, p_next - p_current) - wt[a] * np.min([0, p_current[a] - INIT_P[a]])
    # elif rindex == 6:
    #     r = wt[a] * (10 * np.power(p_next[a], 2) + p_next[a])
    #     for idx in range(NUM_OF_LINKS):
    #         r += wt[idx] * np.min([0, p_next[idx] - INIT_P[idx]])
    # elif rindex == 7:
        # r = wt[a] * p_next[a] * np.power(2, -30 * np.min([0, p_next[a] - INIT_P[a]]))
    else:
        r = 0
    return r

def run_exp(seed=0, NUM_Q=2, NUM_MIN=2, adaeq=0, tp=0.001, lr=3e-4, alpha=0, gamma=0.99, rscale=1, rindex=0, init=0, explore=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    actor = utils.Actor().to(device)
    if init == 1:
        actor.load_state_dict(torch.load('initialized-actor.pkl'))     # load initial NN model trained from base policy

    q_net_list, q_target_net_list = [], []
    for q_i in range(NUM_Q):    # N: number of Q networks
        new_q_net = utils.Critic().to(device)
        q_net_list.append(new_q_net)
        new_q_target_net = utils.Critic().to(device)
        new_q_target_net.load_state_dict(new_q_net.state_dict())
        q_target_net_list.append(new_q_target_net)
    optimizerA = optim.Adam(actor.parameters(), lr=lr)
    optimizerQ_list = []
    for q_i in range(NUM_Q):
        optimizerQ_list.append(optim.Adam(q_net_list[q_i].parameters(), lr=lr))
    mse_criterion = nn.MSELoss()

    wandb.watch(actor, log='all')
    for q_i in range(NUM_Q):
        wandb.watch(q_net_list[q_i], log='all')


    utils.init_state(Buffer, Deficit, totalArrival, totalDelivered,
                        p_next, e, Action, sumAction
                        )   # initialization
    p_current = p_next.copy()
    # p_current.fill(0)
    s = utils.generate_state(Deficit, e)

    for collect_t in range(10):
        utils.init_state(Buffer, Deficit, totalArrival, totalDelivered,
                        p_next, e, Action, sumAction
                        )   # initialization
        p_current = p_next.copy()
        # p_current.fill(0)
        s = utils.generate_state(Deficit, e)

        for t in range(100):
            s_normalized = torch.sigmoid(s)
            dist = actor(s_normalized.to(device)).detach().cpu()
            # dist = actor(s.to(device)).detach().cpu()   # dist is action_binary, of which only the a^{th} entry is 1.
            try:
                a = torch.multinomial(dist, 1).item()
            except:
                print("ERROR! ", dist)
            # log_prob = torch.log(dist[a])   # no use
            # a = rd.randint(0, NUM_OF_LINKS-1)       # random action

            utils.update_dynamics_take_action(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Arrival, 
                                sumArrival, Action, sumAction, totalBuff, a, LAMBDA, INIT_P, d_max
                                )
            r = reward_func(rindex, wt, p_next, p_current, INIT_P, a, wt_pnt)
            utils.update_dynamics_new_arrival(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Arrival, 
                                sumArrival, Action, sumAction, totalBuff, a, LAMBDA, INIT_P, d_max
                                )
            s_ = utils.generate_state(Deficit, e)
            # store_transition(memory, s, a, r, s_)
            store_transition(memory, s, dist, r, s_)
            p_current = p_next.copy()   #current delivery ratio
            s = s_.clone().detach()

    eval_interval = 2000
    eval_step = 0
    N_TRAIN = 5
    for train_t in range(N_TRAIN):
        utils.init_state(Buffer, Deficit, totalArrival, totalDelivered,
                        p_next, e, Action, sumAction
                        )   # initialization
        p_current = p_next.copy()
        # p_current.fill(0)
        s = utils.generate_state(Deficit, e)
        EVAL_INT = LEN_EPISODE
        for len in range(LEN_EPISODE):  #length of each episode
            # if (train_t+1) % N_TRAIN == 0 and (len+1) % EVAL_INT == 0:
            #     torch.save(actor.state_dict(), 'actor-r-'+str(rindex)+'-gamma-'+str(gamma)+'-init-'+str(init)+'.pkl')
            eval_step += 1
            if eval_step % eval_interval == 0:
                torch.save(actor.state_dict(), 'actor-r-'+str(rindex)+'-gamma-'+str(gamma)+'-init-'+str(init)+'-eval-'+str(eval_step)+'.pkl')

            s_normalized = torch.sigmoid(s)
            dist = actor(s_normalized.to(device)).detach().cpu()
            # dist = actor(s.to(device)).detach().cpu()
            wandb.log({'step': train_t*LEN_EPISODE+len, 'state': s, 'normalized_state': s_normalized, 'prob_vector': dist})

            if explore == 1:                    # exploration with probability epsilon
                exploration = rd.uniform(0,1)
                if exploration < epsilon:
                    dist = torch.zeros(NUM_OF_LINKS)
                    rdint = rd.randint(0, NUM_OF_LINKS-1)
                    dist[rdint] = 1

            try:
                a = torch.multinomial(dist, 1).item()
            except:
                print("ERROR! ", dist)

            utils.update_dynamics_take_action(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Arrival, 
                                sumArrival, Action, sumAction, totalBuff, a, LAMBDA, INIT_P, d_max
                                )
            r = reward_func(rindex, wt, p_next, p_current, INIT_P, a, wt_pnt)
            wandb.log({'step': train_t*LEN_EPISODE+len, 'state': s, 'action': a, 'reward': r})
            utils.update_dynamics_new_arrival(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Arrival, 
                                sumArrival, Action, sumAction, totalBuff, a, LAMBDA, INIT_P, d_max
                                )
            s_ = utils.generate_state(Deficit, e)
            # store_transition(memory, s, a, r, s_)
            store_transition(memory, s, dist, r, s_)
            # print('n_train=', train_t, ', step=', len, ', \n', s, a, r, s_)
            p_current = p_next.copy()   #current delivery ratio
            s = s_.clone().detach()

            # sample batch transitions
            sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
            b_memory = memory[sample_index, :]
            b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
            b_s = b_s.to(device)
            b_a = Variable(torch.FloatTensor(b_memory[:, N_STATES:N_STATES+N_ACTIONS]))
            b_a = b_a.to(device)
            b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+N_ACTIONS:N_STATES+N_ACTIONS+1]))
            b_r = b_r.to(device)
            b_s_ = Variable(torch.FloatTensor(b_memory[:, N_STATES+N_ACTIONS+1:2*N_STATES+N_ACTIONS+1]))
            b_s_ = b_s_.to(device)

            # update_a = actor(b_s_)     # next state -> next action -> log prob
            s_normalized = torch.sigmoid(b_s_)
            update_a = actor(s_normalized)

            # select M out of N q_nets
            sample_idxs = np.random.choice(NUM_Q, NUM_MIN, replace=False)
            q_prediction_next_list = []
            for sample_idx in sample_idxs:                      # use truncated rollout to get a better estimation of q_prediction_next. 09/08/2021
                sa_normalized = torch.sigmoid(torch.cat((b_s_, update_a), 1))
                q_prediction_next = q_target_net_list[sample_idx](sa_normalized)
                # q_prediction_next = q_target_net_list[sample_idx](torch.cat((b_s_, update_a), 1))
                q_prediction_next_list.append(q_prediction_next)
            q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
            min_q, min_indices = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            # y_q = b_r + gamma * (min_q - update_log_prob * alpha)     # alpha is a SAC entropy hyperparameter       ***modified on 07/18/2021
            y_q = b_r + gamma * min_q
            # if i == 0 or i == 5:
            #     print('len=', len, '\n target_q=', y_q.transpose(0,1), '\n min_q=', min_q.transpose(0,1), flush=True)
            q_prediction_list = []
            for q_i in range(NUM_Q):
                sa_normalized = torch.sigmoid(torch.cat((b_s, b_a), 1))
                q_prediction = q_net_list[q_i](sa_normalized)
                # q_prediction = q_net_list[q_i](torch.cat((b_s, b_a), 1))
                q_prediction_list.append(q_prediction)

            q_prediction_cat = torch.cat(q_prediction_list, dim=1)
            y_q = y_q.expand((-1, NUM_Q)) if y_q.shape[1] == 1 else y_q
            q_loss_all = mse_criterion(q_prediction_cat, y_q.detach()) * NUM_Q   # total loss function of N q_nets
            print('critic loss: ', q_loss_all)
            wandb.log({'step': train_t*LEN_EPISODE+len, 'critic loss': q_loss_all})
            # test_s = np.array([1,1,1,1])
            # test_a = np.array([0,1])
            # test_input = np.concatenate((test_s, test_a))
            # test_input = torch.FloatTensor(test_input)
            # test_input = test_input.to(device)
            # print('\n before update (Q): ', q_net_list[0](test_input), q_net_list[1](test_input))
            for q_i in range(NUM_Q):
                optimizerQ_list[q_i].zero_grad()
            q_loss_all.backward()
            for q_i in range(NUM_Q):
                torch.nn.utils.clip_grad_value_(q_net_list[q_i].parameters(), clip)  # gradient clipping
            for q_i in range(NUM_Q):
                optimizerQ_list[q_i].step()
            # print('\n after update (Q): ', q_net_list[0](test_input), q_net_list[1](test_input))

            for q_i in range(NUM_Q):
                for target_param, param in zip(q_target_net_list[q_i].parameters(), q_net_list[q_i].parameters()):
                    target_param.data.copy_(
                        target_param.data * polyak + param.data * (1 - polyak)
                    )
            
            # update policy/actor every POLICY_UPDATE_DELAY steps
            if len % POLICY_UPDATE_DELAY == 0:
                # dist_tilda = actor(b_s).cpu()     # current state -> action -> log prob
                # a_tilda = torch.zeros(BATCH_SIZE, dtype=torch.int)
                # log_prob_tilda = torch.zeros(BATCH_SIZE)
                # for j in range(BATCH_SIZE):
                #     a_tilda[j] = torch.multinomial(dist_tilda[j], 1).item()
                #     print(len, 'a_tilda_grad', dist_tilda[j], torch.multinomial(dist_tilda[j], 1))
                #     log_prob_tilda[j] = torch.log(dist_tilda[j,int(a_tilda[j])])
                # a_tilda = a_tilda.reshape([BATCH_SIZE,1])
                # a_tilda = a_tilda.to(device)
                # log_prob_tilda = log_prob_tilda.reshape([BATCH_SIZE,1])
                # log_prob_tilda = log_prob_tilda.to(device)

                s_normalized = torch.sigmoid(b_s)
                a_tilda = actor(s_normalized)
                # a_tilda = actor(b_s)     # current state -> action -> log prob

                q_a_tilda_list = []
                for sample_idx in range(NUM_Q):
                    sa_normalized = torch.sigmoid(torch.cat((b_s, a_tilda), 1))
                    q_a_tilda = q_net_list[sample_idx](sa_normalized)
                    # q_a_tilda = q_net_list[sample_idx](torch.cat((b_s, a_tilda), 1))
                    q_a_tilda_list.append(q_a_tilda)
                q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
                ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
                # actor_loss = (log_prob_tilda * alpha - ave_q).mean()    # as in REDQ, not acsent but descent
                actor_loss = - ave_q.mean()
                print('actor loss: ', actor_loss)
                wandb.log({'step': train_t*LEN_EPISODE+len, 'actor loss': actor_loss})
                
                # test_s = np.array([1,1,1,1])
                # test_input = torch.FloatTensor(test_s)
                # test_input = test_input.to(device)
                # print('\n before update (P): ', actor(test_input))
                optimizerA.zero_grad()

                # actor_loss.retain_grad()
                actor_loss.backward()
                
                # actor.requires_grad_(True)
                for name, params in actor.named_parameters():
                    print('-->name:', name, '-->grad_requirs:', params.requires_grad, '-->grad_value:', params.grad)
                # wandb.log({'step': train_t*LEN_EPISODE+len, 'actor gradients': actor.named_parameters()})
                
                torch.nn.utils.clip_grad_value_(actor.parameters(), clip)  # gradient clipping
                
                optimizerA.step()
                # print('\n after update (P): ', actor(test_input))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--m', type=int, default=2)
    parser.add_argument('--adaeq', type=int, default=0)
    parser.add_argument('--tp', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--rscale', type=float, default=1)
    parser.add_argument('--rindex', type=int, default=1)
    parser.add_argument('--init', type=int, default=0)
    parser.add_argument('--explore', type=int, default=0)
    args = parser.parse_args()
    run_exp(seed=args.seed, NUM_Q=args.n, NUM_MIN=args.m, adaeq=args.adaeq, tp=args.tp, lr=utils.LR, 
            alpha=args.alpha, gamma=args.gamma, rscale=args.rscale, rindex=args.rindex, init=args.init, explore = args.explore)     # LR defined in utils.py
