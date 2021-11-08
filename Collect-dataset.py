import utils
import numpy as np

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

NUM_EPISODE = 500   # the number of episode
LEN_EPISODE = 2000   # the length of each episode

N_ACTIONS = utils.N_ACTIONS
N_STATES = utils.N_STATES

MEMORY_CAPACITY = NUM_EPISODE * LEN_EPISODE     # memory
memory_counter = 0
memory = np.zeros((MEMORY_CAPACITY, N_STATES + N_ACTIONS))
buffer_counter = 0
savedBuffer = np.zeros((MEMORY_CAPACITY, NUM_OF_LINKS, MAX_DEADLINE+2))
savedDeficit = np.zeros((MEMORY_CAPACITY, NUM_OF_LINKS))
savedTotalArrival = np.zeros((MEMORY_CAPACITY, NUM_OF_LINKS))
savedTotalDelivered = np.zeros((MEMORY_CAPACITY, NUM_OF_LINKS))

for i_episode in range(NUM_EPISODE):
    utils.init_state(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Action, sumAction)   # initialization
    if buffer_counter != 0:
        i_buffer = np.random.randint(0, buffer_counter)
        Buffer = savedBuffer[i_buffer].copy()
        Deficit = savedDeficit[i_buffer].copy()
        totalArrival = savedTotalArrival[i_buffer].copy()
        totalDelivered = savedTotalDelivered[i_buffer].copy()

    for index in range(LEN_EPISODE):
        # s = utils.generate_state(Deficit, e, Buffer[:,1:MAX_DEADLINE+1], p_next)    # current state
        s = utils.generate_state(Deficit, e)    # current state
        currentActiveLink, action_probs = utils.AMIX_ND(Deficit, e, totalBuff)      # take action according to ND algorithm
        memory_counter = utils.store_state_actionprobs(s, action_probs, memory, memory_counter)     # store (state, probs) pair in memory
        utils.update_dynamics(Buffer, Deficit, totalArrival, totalDelivered, p_next, e, Arrival, 
                              sumArrival, Action, sumAction, totalBuff, currentActiveLink, LAMBDA, INIT_P, d_max
                              )                                                     # update dynamics
        savedBuffer[buffer_counter, :] = Buffer
        savedDeficit[buffer_counter, :] = Deficit
        savedTotalArrival[buffer_counter, :] = totalArrival
        savedTotalDelivered[buffer_counter, :] = totalDelivered
        buffer_counter += 1

np.save('dataset.npy', memory)     # save dataset