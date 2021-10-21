import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random as rd
import torch.optim as optim
import utils

device = utils.device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

N_ACTIONS = utils.N_ACTIONS
N_STATES = utils.N_STATES

Buffer = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS, MAX_DEADLINE+2), dtype=np.float)
Deficit = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS), dtype=np.float)
Arrival = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS, MAX_DEADLINE+1), dtype=np.float)
sumArrival = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS), dtype=np.float)     #total arrival packets at current time slot
totalArrival = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS), dtype=np.float)   #TA: total arrival packets from beginning
totalDelivered = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS), dtype=np.float) #TD: total delivered packets
Action = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS, MAX_DEADLINE+2), dtype=np.float)
sumAction = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS), dtype=np.int)
totalBuff = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS), dtype=np.float)
e = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS), dtype=np.int)    # e[l]: earliest deadline of link l
p_current = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS), dtype=np.float)    #current delivery ratio
p_next = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS), dtype=np.float)   #next delivery ratio
s = torch.zeros(NUM_OF_AGENT, N_STATES)
s_ = torch.zeros(NUM_OF_AGENT, N_STATES)

MEMORY_CAPACITY = NUM_OF_AGENT  # H*K ------------------0710 updated--------------------
memory_counter = 0
memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 3))

# used to store all visited states (no more than 1000)
TOTAL_MEMORY_CAPACITY = 1000
total_memory_counter = 0
total_memory_is_full = False
total_memory = np.zeros((TOTAL_MEMORY_CAPACITY, N_STATES))
savedBuffer = np.zeros((TOTAL_MEMORY_CAPACITY, NUM_OF_LINKS, MAX_DEADLINE+2))
savedDeficit = np.zeros((TOTAL_MEMORY_CAPACITY, NUM_OF_LINKS))
savedTotalArrival = np.zeros((TOTAL_MEMORY_CAPACITY, NUM_OF_LINKS))
savedTotalDelivered = np.zeros((TOTAL_MEMORY_CAPACITY, NUM_OF_LINKS))

def store_transition(memory, s, a, r, s_, log_prob):
    # global memory
    global memory_counter
    global MEMORY_CAPACITY
    transition = np.hstack((s, [a, r], s_, [log_prob]))
    # replace the old memory with new memory
    index = memory_counter % MEMORY_CAPACITY
    memory[index, :] = transition
    memory_counter += 1

def store_total_transition(
    total_memory, s_, 
    savedBuffer, savedDeficit, savedTotalArrival, savedTotalDelivered, 
    Buffer, Deficit, totalArrival, totalDelivered):
    global total_memory_counter
    global total_memory_is_full
    global TOTAL_MEMORY_CAPACITY
    if total_memory_counter >= TOTAL_MEMORY_CAPACITY:
        total_memory_is_full = True 
    transition = np.hstack((s_))
    # replace the old memory with new memory
    index = total_memory_counter % TOTAL_MEMORY_CAPACITY
    total_memory[index, :] = transition
    savedBuffer[index] = Buffer.copy()
    savedDeficit[index] = Deficit.copy()
    savedTotalArrival[index] = totalArrival.copy()
    savedTotalDelivered[index] = totalDelivered.copy()
    total_memory_counter += 1

def run_exp(seed=0, NUM_Q=2, NUM_MIN=2, adaeq=0, tp=0.001, lr=3e-4, alpha=0, gamma=0.99, rscale=1, rindex=0, init=0):
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
    Q_estimation_error = []

    for agent in range(NUM_OF_AGENT):
        utils.init_state(Buffer[agent], Deficit[agent], totalArrival[agent], totalDelivered[agent],
                         p_next[agent], e[agent], Action[agent], sumAction[agent]
                         )   # initialization
        p_current[agent].fill(0)

    for agent in range(NUM_OF_AGENT):
        # s[agent] = utils.generate_state(Deficit[agent], e[agent], Buffer[agent][:,1:MAX_DEADLINE+1], p_next[agent])
        s[agent] = utils.generate_state(Deficit[agent], e[agent])

    EVAL_INT = LEN_EPISODE
    for len in range(LEN_EPISODE):  #length of each episode
        if (len+1) % EVAL_INT == 0:
            # torch.save(actor.state_dict(), 'EnsembleQ-N'+str(NUM_Q)+'M'+str(NUM_MIN)+'-'+str(adaeq)+'-'+str(seed)
            #            +'-'+str(tp)+'-'+str(lr)+'-'+str(alpha)+'-'+str(gamma)+'-'+str(rscale)+'-'+str(len+1)+'.pkl'
            #            )
            torch.save(actor.state_dict(), 'EnsembleQ-actor-1-r'+str(rindex)+'-init-'+str(init)+'.pkl')
        for agent in range(NUM_OF_AGENT):
            dist = actor(s[agent].to(device)).detach().cpu()
            try:
                a = torch.multinomial(dist, 1).item()
            except:
                print("ERROR! ", dist)
            log_prob = torch.log(dist[a])   # no use

            # utils.update_dynamics(Buffer[agent], Deficit[agent], totalArrival[agent], totalDelivered[agent], p_next[agent], e[agent], Arrival[agent], 
            #                     sumArrival[agent], Action[agent], sumAction[agent], totalBuff[agent], a, LAMBDA, INIT_P, d_max
            #                     )
            utils.update_dynamics_take_action(Buffer[agent], Deficit[agent], totalArrival[agent], totalDelivered[agent], p_next[agent], e[agent], Arrival[agent], 
                                sumArrival[agent], Action[agent], sumAction[agent], totalBuff[agent], a, LAMBDA, INIT_P, d_max
                                )
            # r = rscale * np.min(p_next[agent])   # reward scaling

            if rindex == 1:
                r = np.inner(wt, p_next[agent] - p_current[agent]) - wt[a] * np.min([0, p_current[agent][a] - INIT_P[a]])     # case 1-1
            elif rindex == 2:
                r = wt[a] * p_next[agent][a]
                for idx in range(NUM_OF_LINKS):
                    r += wt[idx] * np.min([0, p_next[agent][idx] - INIT_P[idx]])    # case 1-2
            elif rindex == 3:
                r = wt[a] * (p_next[agent][a] - p_current[agent][a])
                for idx in range(NUM_OF_LINKS):
                    r += wt[idx] * np.min([0, p_next[agent][idx] - INIT_P[idx]])    # case 1-3
            elif rindex == 5:
                r = wt[a] * (10 * np.power(p_next[agent][a], 2) + p_next[agent][a])
                for idx in range(NUM_OF_LINKS):
                    r += wt[idx] * np.min([0, p_next[agent][idx] - INIT_P[idx]])    # case 5-2
            else:
                r = 0
            # r = wt[a] * p_next[agent][a] * np.power(2, -30 * np.min([0, p_next[agent][a] - INIT_P[a]]))   # case 2

            utils.update_dynamics_new_arrival(Buffer[agent], Deficit[agent], totalArrival[agent], totalDelivered[agent], p_next[agent], e[agent], Arrival[agent], 
                                sumArrival[agent], Action[agent], sumAction[agent], totalBuff[agent], a, LAMBDA, INIT_P, d_max
                                )
            # s_[agent] = utils.generate_state(Deficit[agent], e[agent], Buffer[agent][:,1:MAX_DEADLINE+1], p_next[agent])
            s_[agent] = utils.generate_state(Deficit[agent], e[agent])
            store_transition(memory, s[agent], a, r, s_[agent], log_prob)
            store_total_transition(
                total_memory, s_[agent], 
                savedBuffer, savedDeficit, savedTotalArrival, savedTotalDelivered, 
                Buffer[agent], Deficit[agent], totalArrival[agent], totalDelivered[agent])
            p_current[agent] = p_next[agent].copy()   #current delivery ratio
            s[agent] = s_[agent].clone().detach()

        # sample batch transitions
        #sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = memory[:, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_s = b_s.to(device)
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
        b_a = b_a.to(device)
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
        b_r = b_r.to(device)
        b_s_ = Variable(torch.FloatTensor(b_memory[:, N_STATES+2:2*N_STATES+2]))
        b_s_ = b_s_.to(device)
        b_log = Variable(torch.FloatTensor(b_memory[:, 2*N_STATES+2:2*N_STATES+3]), requires_grad=True) # no use
        b_log = b_log.to(device)

        update_a = torch.zeros(NUM_OF_AGENT, dtype=torch.int)
        update_log_prob = torch.zeros(NUM_OF_AGENT)
        for i in range(POLICY_UPDATE_DELAY):    # G: policy update delay
            update_dist = actor(b_s_).cpu()     # next state -> next action -> log prob
            for j in range(NUM_OF_AGENT):
                try:
                    update_a[j] = torch.multinomial(update_dist[j], 1).item()     # use cpu for sampling
                except:
                    print("ERROR! ", len, i, j, update_dist[j])
                update_log_prob[j] = torch.log(update_dist[j,int(update_a[j])])
            update_a = update_a.reshape([NUM_OF_AGENT,1])
            update_a = update_a.to(device)
            update_log_prob = update_log_prob.reshape([NUM_OF_AGENT,1])
            update_log_prob = update_log_prob.to(device)

            # select M out of N q_nets
            sample_idxs = np.random.choice(NUM_Q, NUM_MIN, replace=False)
            q_prediction_next_list = []
            for sample_idx in sample_idxs:                      # use truncated rollout to get a better estimation of q_prediction_next. 09/08/2021
                q_prediction_next = q_target_net_list[sample_idx](torch.cat((b_s_, update_a), 1))
                q_prediction_next_list.append(q_prediction_next)
            q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
            min_q, min_indices = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            y_q = b_r + gamma * (min_q - update_log_prob * alpha)     # alpha is a SAC entropy hyperparameter       ***modified on 07/18/2021
            # if i == 0 or i == 5:
            #     print('len=', len, '\n target_q=', y_q.transpose(0,1), '\n min_q=', min_q.transpose(0,1), flush=True)
            q_prediction_list = []
            for q_i in range(NUM_Q):
                q_prediction = q_net_list[q_i](torch.cat((b_s, b_a), 1))
                q_prediction_list.append(q_prediction)

            q_prediction_cat = torch.cat(q_prediction_list, dim=1)
            y_q = y_q.expand((-1, NUM_Q)) if y_q.shape[1] == 1 else y_q
            q_loss_all = mse_criterion(q_prediction_cat, y_q.detach()) * NUM_Q   # total loss function of N q_nets
            for q_i in range(NUM_Q):
                optimizerQ_list[q_i].zero_grad()
            q_loss_all.backward()
            for q_i in range(NUM_Q):
                torch.nn.utils.clip_grad_value_(q_net_list[q_i].parameters(), clip)  # gradient clipping
            for q_i in range(NUM_Q):
                optimizerQ_list[q_i].step()

            for q_i in range(NUM_Q):
                for target_param, param in zip(q_target_net_list[q_i].parameters(), q_net_list[q_i].parameters()):
                    target_param.data.copy_(
                        target_param.data * polyak + param.data * (1 - polyak)
                    )

        dist_tilda = actor(b_s).cpu()     # current state -> action -> log prob
        a_tilda = torch.zeros(NUM_OF_AGENT, dtype=torch.int)
        log_prob_tilda = torch.zeros(NUM_OF_AGENT)
        for j in range(NUM_OF_AGENT):
            a_tilda[j] = torch.multinomial(dist_tilda[j], 1).item()
            log_prob_tilda[j] = torch.log(dist_tilda[j,int(a_tilda[j])])
        a_tilda = a_tilda.reshape([NUM_OF_AGENT,1])
        a_tilda = a_tilda.to(device)
        log_prob_tilda = log_prob_tilda.reshape([NUM_OF_AGENT,1])
        log_prob_tilda = log_prob_tilda.to(device)

        q_a_tilda_list = []
        for sample_idx in range(NUM_Q):
            q_a_tilda = q_net_list[sample_idx](torch.cat((b_s, a_tilda), 1))
            q_a_tilda_list.append(q_a_tilda)
        q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
        ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
        ## actor_loss = (ave_q - alpha * b_log).mean()    # as in A2C-parallel-q.py
        actor_loss = (log_prob_tilda * alpha - ave_q).mean()    # as in REDQ, not acsent but descent
        
        optimizerA.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_value_(actor.parameters(), clip)  # gradient clipping
        optimizerA.step()

    #     if (len+1) % STEP_PER_EPOCH == 0:
    #         epoch = len // STEP_PER_EPOCH
    #         epoch_exp_error = get_redq_true_estimate_value(q_net_list, actor, NUM_Q, n_sa_pair=10, n_eval=10, max_ep_len=500, 
    #                                                         gamma=gamma, rscale=rscale, INIT_P=INIT_P)
    #         print('len=', len, ', epoch_exp_error=', epoch_exp_error, flush=True)
    #         Q_estimation_error.append(epoch_exp_error)

    #         if adaeq == 1:
    #             if epoch % 5 == 0:
    #                 if epoch_exp_error > tp and NUM_MIN < 10:
    #                     lower_bound = NUM_MIN
    #                     NUM_MIN = np.random.randint(lower_bound, 11)
    #                 elif epoch_exp_error < tp and NUM_MIN > 3:
    #                     upper_bound = NUM_MIN
    #                     NUM_MIN = np.random.randint(2, upper_bound)
    #                 else :
    #                     NUM_MIN = NUM_MIN

    # # with open('EnsembleQ-N'+str(NUM_Q)+'M'+str(NUM_MIN)+'-'+str(adaeq)+'-'+str(seed)+'-'+str(tp)+
    # #           '-'+str(lr)+'-'+str(alpha)+'-'+str(gamma)+'-'+str(rscale)+'-Qerror'+'.txt', 'a+') as f:
    # with open('EnsembleQ-p-'+str(INIT_P)+'-Qerror'+'.txt', 'a+') as f:
    #     for x in Q_estimation_error:
    #         f.write(str(x)+'\n')

def get_redq_true_estimate_value(q_net_list, actor, NUM_Q, n_sa_pair, n_eval, max_ep_len, gamma, rscale, INIT_P):
    # Return estimate and true value (MC simulation) of a set of samples.
    # max_ep_len = 500
    true_return_list = []       # true Q value
    estimate_return_list = []   # Q estimate
    sample_index = np.random.choice(TOTAL_MEMORY_CAPACITY if total_memory_is_full else total_memory_counter, n_sa_pair, replace=True)
    agent = 0   # A-C agent for test
    temp_Buffer = Buffer[agent].copy()
    temp_Deficit = Deficit[agent].copy()
    temp_totalArrival = totalArrival[agent].copy()
    temp_totalDelivered = totalDelivered[agent].copy()
    temp_totalBuff = totalBuff[agent].copy()
    temp_e = e[agent].copy()
    temp_p_current = p_current[agent].copy()
    temp_p_next = p_next[agent].copy()
    temp_Arrival = Arrival[agent].copy()
    temp_sumArrival = sumArrival[agent].copy()
    temp_Action = Action[agent].copy()
    temp_sumAction = sumAction[agent].copy()
    
    for idx_pair in sample_index:
        true_return_list_sa = []    # Q(s,a) estimated by multiple MC rollout episodes from (s,a)
        s = Variable(torch.FloatTensor(total_memory[idx_pair, :]))
        s = s.to(device)
        utils.init_state(temp_Buffer, temp_Deficit, temp_totalArrival, temp_totalDelivered, temp_p_next, temp_e, temp_Action, temp_sumAction)   # initialization
        temp_Buffer = savedBuffer[idx_pair].copy()
        temp_Deficit = savedDeficit[idx_pair].copy()
        temp_totalArrival = savedTotalArrival[idx_pair].copy()
        temp_totalDelivered = savedTotalDelivered[idx_pair].copy()
        
        dist = actor(s.to(device)).detach().cpu()
        a = torch.multinomial(dist, 1).item()
        q_prediction_list = []
        det_action = torch.tensor([a])
        for q_i in range(NUM_Q):
            q_prediction = q_net_list[q_i](torch.cat((s[None,:].to(device), det_action[None,:].to(device)), 1))  # pay attention to torch.cat()
            q_prediction_list.append(q_prediction)
        q_prediction_list_mean = torch.cat(q_prediction_list, 1).mean(dim=1).reshape(-1, 1) # Copy 154 line
        estimate_return_list.append(q_prediction_list_mean)

        for idx_eval in range(n_eval):
            utils.update_dynamics(temp_Buffer, temp_Deficit, temp_totalArrival, temp_totalDelivered, temp_p_next, temp_e, temp_Arrival, 
                                temp_sumArrival, temp_Action, temp_sumAction, temp_totalBuff, a, LAMBDA, INIT_P
                                )                                                   # update dynamics
            temp_s_ = utils.generate_state(temp_Deficit, temp_e, temp_Buffer[:,1:MAX_DEADLINE+1], temp_p_next)
            r_true = rscale * np.min(temp_p_next) # reward scaling
            temp_p_current = temp_p_next.copy()   #current delivery ratio
            temp_s = temp_s_.clone().detach()

            ep_ret_true = r_true
            ep_len_true = 1
            while not (ep_len_true == max_ep_len):
                dist = actor(temp_s.to(device)).detach().cpu()
                a = torch.multinomial(dist, 1).item()
                utils.update_dynamics(temp_Buffer, temp_Deficit, temp_totalArrival, temp_totalDelivered, temp_p_next, temp_e, temp_Arrival, 
                                temp_sumArrival, temp_Action, temp_sumAction, temp_totalBuff, a, LAMBDA, INIT_P
                                )                                                   # update dynamics
                temp_s_ = utils.generate_state(temp_Deficit, temp_e, temp_Buffer[:,1:MAX_DEADLINE+1], temp_p_next)
                r_true = rscale * np.min(temp_p_next) # reward scaling
                temp_p_current = temp_p_next.copy()   #current delivery ratio
                temp_s = temp_s_.clone().detach()
                ep_ret_true = ep_ret_true + r_true * (gamma ** ep_len_true) # discounted MC return
                ep_len_true = ep_len_true  + 1
            true_return_list_sa.append(ep_ret_true)
        true_return_list_sa_avg = np.mean(true_return_list_sa)
        true_return_list.append(true_return_list_sa_avg)
    estimate_return_list_array = torch.cat(estimate_return_list, 1).detach().cpu().numpy().reshape(-1)
    true_return_list_array = np.array(true_return_list)

    expected_true_value = abs(np.mean(true_return_list_array))
    exp_error = np.mean(estimate_return_list_array-true_return_list_array)
    exp_error = np.true_divide(exp_error, expected_true_value)
    std_error = np.std(estimate_return_list_array-true_return_list_array)
    std_error = np.true_divide(std_error, expected_true_value)

    print(
        'estimate_return_list_array:\n', estimate_return_list_array,
        '\n true_return_list_array:\n', true_return_list_array, 
        '\n exp_error:\n', exp_error,
        '\n expected_true_value:\n', expected_true_value,
        flush=True)
    return exp_error

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--m', type=int, default=2)
    parser.add_argument('--adaeq', type=int, default=0)
    parser.add_argument('--tp', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--rscale', type=float, default=1)
    # parser.add_argument('--p', type=float, default=0.73)
    parser.add_argument('--rindex', type=int, default=1)
    parser.add_argument('--init', type=int, default=0)
    args = parser.parse_args()
    run_exp(seed=args.seed, NUM_Q=args.n, NUM_MIN=args.m, adaeq=args.adaeq, tp=args.tp, lr=args.lr, 
            alpha=args.alpha, gamma=args.gamma, rscale=args.rscale, rindex=args.rindex, init=args.init)
