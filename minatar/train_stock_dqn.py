################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian(ttian@ualberta.ca)                                                                                 #
#                                                                                                              #
# python3 dqn.py -g <game>                                                                                     #
#   -o, --output <directory/file name prefix>                                                                  #
#   -v, --verbose: outputs the average returns every 1000 episodes                                             #
#   -l, --loadfile <directory/file name of the saved model>                                                    #
#   -a, --alpha <number>: step-size parameter                                                                  #
#   -s, --save: save model data every 1000 episodes                                                            #
#   -r, --replayoff: disable the replay buffer and train on each state transition                              #
#   -t, --targetoff: disable the target network                                                                #
#                                                                                                              #
# References used for this implementation:                                                                     #
#   https://pytorch.org/docs/stable/nn.html#                                                                   #
#   https://pytorch.org/docs/stable/torch.html                                                                 #
#   https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html                                   #
################################################################################################################
RANDOM_SEED = 13011

import os
import sys
import copy
import json
import random
random.seed(RANDOM_SEED)
import numpy as np
np.random.seed(RANDOM_SEED)
import torch
torch.manual_seed(RANDOM_SEED)
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import time

import numpy, argparse, logging

from collections import namedtuple, deque
from minatar import Environment
import common
from ewc import EWC
from task_detector import TaskDetector
from Doric import ProgNet

################################################################################################################
# Constants
#
################################################################################################################
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
TARGET_NETWORK_UPDATE_FREQ = 1000
TRAINING_FREQ = 1
NUM_FRAMES = 3000000
#NUM_FRAMES = 100000
FIRST_N_FRAMES = 100000
REPLAY_START_SIZE = 5000
END_EPSILON = 0.1
STEP_SIZE = 0.00025
GRAD_MOMENTUM = 0.95
SQUARED_GRAD_MOMENTUM = 0.95
MIN_SQUARED_GRAD = 0.01
GAMMA = 0.99
EPSILON = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def pad(o):
    p = np.zeros([10, 10, 10 - o.shape[-1]])
    o = np.concatenate([o, p], axis = 2)
    return o


################################################################################################################
# class QNetwork
#
# One hidden 2D conv with variable number of input channels.  We use 16 filters, a quarter of the original DQN
# paper of 64.  One hidden fully connected linear layer with a quarter of the original DQN paper of 512
# rectified units.  Finally, the output layer is a fully connected linear layer with a single output for each
# valid action.
#
################################################################################################################
'''
class QNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):

        super(QNetwork, self).__init__()

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        # Output layer:
        self.output = nn.Linear(in_features=128, out_features=num_actions)

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x):
        # Rectified output from the first conv layer
        x = f.relu(self.conv(x))

        # Rectified output from the final hidden layer
        x = f.relu(self.fc_hidden(x.view(x.size(0), -1)))

        # Returns the output from the fully-connected linear layer
        return self.output(x)
'''






###########################################################################################################
# class replay_buffer
#
# A cyclic buffer of a fixed size containing the last N number of recent transitions.  A transition is a
# tuple of state, next_state, action, reward, is_terminal.  The boolean is_terminal is used to indicate
# whether if the next state is a terminal state or not.
#
###########################################################################################################
transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')
class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque([], maxlen = buffer_size)

    def add(self, *args):
        self.buffer.append(transition(*args))

    def addTransition(self, t):
        self.buffer.append(t)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def shuffle(self):
        random.shuffle(self.buffer)



def randomBufferUnion(buffs, size):
    n = size // len(buffs)
    newBuff = replay_buffer(size)
    for b in buffs:
        samp = b.sample(n)
        for rec in samp:
            newBuff.addTransition(rec)
    newBuff.shuffle()
    return newBuff










################################################################################################################
# get_state
#
# Converts the state given by the environment to a tensor of size (in_channel, 10, 10), and then
# unsqueeze to expand along the 0th dimension so the function returns a tensor of size (1, in_channel, 10, 10).
#
# Input:
#   s: current state as numpy array
#
# Output: current state as tensor, permuted to match expected dimensions
#
################################################################################################################
def get_state(s):
    s = (torch.tensor(pad(s), device=device).permute(2, 0, 1)).unsqueeze(0).float()
    return s








################################################################################################################
# world_dynamics
#
# It generates the next state and reward after taking an action according to the behavior policy.  The behavior
# policy is epsilon greedy: epsilon probability of selecting a random action and 1 - epsilon probability of
# selecting the action with max Q-value.
#
# Inputs:
#   t : frame
#   replay_start_size: number of frames before learning starts
#   num_actions: number of actions
#   s: current state
#   env: environment of the game
#   policy_net: policy network, an instance of QNetwork
#
# Output: next state, action, reward, is_terminated
#
################################################################################################################
def world_dynamics(t, replay_start_size, num_actions, s, env, policy_net):

    # A uniform random policy is run before the learning starts
    if t < replay_start_size:
        action = torch.tensor([[random.randrange(num_actions)]], device=device)
    else:
        # Epsilon-greedy behavior policy for action selection
        # Epsilon is annealed linearly from 1.0 to END_EPSILON over the FIRST_N_FRAMES and stays 0.1 for the
        # remaining frames
        epsilon = END_EPSILON if t - replay_start_size >= FIRST_N_FRAMES \
            else ((END_EPSILON - EPSILON) / FIRST_N_FRAMES) * (t - replay_start_size) + EPSILON

        if numpy.random.binomial(1, epsilon) == 1:
            action = torch.tensor([[random.randrange(num_actions)]], device=device)
        else:
            # State is 10x10xchannel, max(1)[1] gives the max action value (i.e., max_{a} Q(s, a)).
            # view(1,1) shapes the tensor to be the right form (e.g. tensor([[0]])) without copying the
            # underlying tensor.  torch._no_grad() avoids tracking history in autograd.
            with torch.no_grad():
                action = policy_net(s).max(1)[1].view(1, 1)

    # Act according to the action and observe the transition and reward
    reward, terminated = env.act(action)

    # Obtain s_prime
    s_prime = get_state(env.state())

    return s_prime, action, torch.tensor([[reward]], device=device).float(), torch.tensor([[terminated]], device=device)








################################################################################################################
# train
#
# This is where learning happens. More specifically, this function learns the weights of the policy network
# using huber loss.
#
# Inputs:
#   sample: a batch of size 1 or 32 transitions
#   policy_net: an instance of QNetwork
#   target_net: an instance of QNetwork
#   optimizer: centered RMSProp
#
################################################################################################################
def train(sample, policy_net, target_net, optimizer, ewc = None):
    # Batch is a list of namedtuple's, the following operation returns samples grouped by keys
    batch_samples = transition(*zip(*sample))

    # states, next_states are of tensor (BATCH_SIZE, in_channel, 10, 10) - inline with pytorch NCHW format
    # actions, rewards, is_terminal are of tensor (BATCH_SIZE, 1)
    states = torch.cat(batch_samples.state)
    next_states = torch.cat(batch_samples.next_state)
    actions = torch.cat(batch_samples.action)
    rewards = torch.cat(batch_samples.reward)
    is_terminal = torch.cat(batch_samples.is_terminal)

    # Obtain a batch of Q(S_t, A_t) and compute the forward pass.
    # Note: policy_network output Q-values for all the actions of a state, but all we need is the A_t taken at time t
    # in state S_t.  Thus we gather along the columns and get the Q-values corresponds to S_t, A_t.
    # Q_s_a is of size (BATCH_SIZE, 1).
    Q_s_a = policy_net(states).gather(1, actions)

    # Obtain max_{a} Q(S_{t+1}, a) of any non-terminal state S_{t+1}.  If S_{t+1} is terminal, Q(S_{t+1}, A_{t+1}) = 0.
    # Note: each row of the network's output corresponds to the actions of S_{t+1}.  max(1)[0] gives the max action
    # values in each row (since this a batch).  The detach() detaches the target net's tensor from computation graph so
    # to prevent the computation of its gradient automatically.  Q_s_prime_a_prime is of size (BATCH_SIZE, 1).

    # Get the indices of next_states that are not terminal
    none_terminal_next_state_index = torch.tensor([i for i, is_term in enumerate(is_terminal) if is_term == 0], dtype=torch.int64, device=device)
    # Select the indices of each row
    none_terminal_next_states = next_states.index_select(0, none_terminal_next_state_index)

    Q_s_prime_a_prime = torch.zeros(len(sample), 1, device=device)
    if len(none_terminal_next_states) != 0:
        Q_s_prime_a_prime[none_terminal_next_state_index] = target_net(none_terminal_next_states).detach().max(1)[0].unsqueeze(1)

    # Compute the target
    target = rewards + GAMMA * Q_s_prime_a_prime

    # Huber loss
    loss = f.smooth_l1_loss(target, Q_s_a)
    if ewc is None:
        loss = f.smooth_l1_loss(target, Q_s_a)
    else:
        loss = f.smooth_l1_loss(target, Q_s_a) + ewc.penalty(policy_net)

    # Zero gradients, backprop, update the weights of policy_net
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()








################################################################################################################
# dqn
#
# DQN algorithm with the option to disable replay and/or target network, and the function saves the training data.
#
# Inputs:
#   env: environment of the game
#   replay_off: disable the replay buffer and train on each state transition
#   target_off: disable target network
#   output_file_name: directory and file name prefix to output data and network weights, file saved as
#       <output_file_name>_data_and_weights
#   store_intermediate_result: a boolean, if set to true will store checkpoint data every 1000 episodes
#       to a file named <output_file_name>_checkpoint
#   load_path: file path for a checkpoint to load, and continue training from
#   step_size: step-size for RMSProp optimizer
#
#################################################################################################################
def dqn(env, replay_off, target_off, output_file_name, model, store_intermediate_result=False, load_path=None, step_size=STEP_SIZE):
    in_channels = 10
    num_actions = 6

    # Instantiate networks, optimizer, loss and buffer
    policy_net = model.to(device)
    replay_start_size = 0
    if not target_off:
        target_net = copy.deepcopy(policy_net).to(device)
        target_net.load_state_dict(policy_net.state_dict())

    if not replay_off:
        r_buffer = replay_buffer(REPLAY_BUFFER_SIZE)
        replay_start_size = REPLAY_START_SIZE

    optimizer = optim.RMSprop(policy_net.parameters(), lr=step_size, alpha=SQUARED_GRAD_MOMENTUM, centered=True, eps=MIN_SQUARED_GRAD)

    # Set initial values
    e_init = 0
    t_init = 0
    policy_net_update_counter_init = 0
    avg_return_init = 0.0
    data_return_init = []
    frame_stamp_init = []

    # Load model and optimizer if load_path is not None
    if load_path is not None and isinstance(load_path, str):
        checkpoint = torch.load(load_path)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

        if not target_off:
            target_net.load_state_dict(checkpoint['target_net_state_dict'])

        if not replay_off:
            r_buffer = checkpoint['replay_buffer']

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        e_init = checkpoint['episode']
        t_init = checkpoint['frame']
        policy_net_update_counter_init = checkpoint['policy_net_update_counter']
        avg_return_init = checkpoint['avg_return']
        data_return_init = checkpoint['return_per_run']
        frame_stamp_init = checkpoint['frame_stamp_per_run']

        # Set to training mode
        policy_net.train()
        if not target_off:
            target_net.train()

    # Data containers for performance measure and model related data
    data_return = data_return_init
    frame_stamp = frame_stamp_init
    avg_return = avg_return_init

    # Train for a number of frames
    t = t_init
    e = e_init
    policy_net_update_counter = policy_net_update_counter_init
    t_start = time.time()
    while t < NUM_FRAMES:
        # Initialize the return for every episode (we should see this eventually increase)
        G = 0.0

        # Initialize the environment and start state
        env.reset()
        s = get_state(env.state())
        is_terminated = False
        while(not is_terminated) and t < NUM_FRAMES:
            # Generate data
            s_prime, action, reward, is_terminated = world_dynamics(t, replay_start_size, num_actions, s, env, policy_net)

            sample = None
            if replay_off:
                sample = [transition(s, s_prime, action, reward, is_terminated)]
            else:
                # Write the current frame to replay buffer
                r_buffer.add(s, s_prime, action, reward, is_terminated)

                # Start learning when there's enough data and when we can sample a batch of size BATCH_SIZE
                if t > REPLAY_START_SIZE and len(r_buffer.buffer) >= BATCH_SIZE:
                    # Sample a batch
                    sample = r_buffer.sample(BATCH_SIZE)

            # Train every n number of frames defined by TRAINING_FREQ
            if t % TRAINING_FREQ == 0 and sample is not None:
                if target_off:
                    train(sample, policy_net, policy_net, optimizer)
                else:
                    policy_net_update_counter += 1
                    train(sample, policy_net, target_net, optimizer)

            # Update the target network only after some number of policy network updates
            if not target_off and policy_net_update_counter > 0 and policy_net_update_counter % TARGET_NETWORK_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            G += reward.item()

            t += 1

            # Continue the process
            s = s_prime

        # Increment the episodes
        e += 1

        # Save the return for each episode
        data_return.append(G)
        frame_stamp.append(t)

        # Logging exponentiated return only when verbose is turned on and only at 1000 episode intervals
        avg_return = 0.90 * avg_return + 0.10 * G
        if e % 10 == 0:
            print("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                         str(numpy.around(avg_return, 2)) + " | Frame: " + str(t)+" | Time per frame: " +str((time.time()-t_start)/t) )
            logging.info("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                         str(numpy.around(avg_return, 2)) + " | Frame: " + str(t)+" | Time per frame: " +str((time.time()-t_start)/t) )

        # Save model data and other intermediate data if the corresponding flag is true
        if store_intermediate_result and e % 1000 == 0:
            torch.save({
                        'episode': e,
                        'frame': t,
                        'policy_net_update_counter': policy_net_update_counter,
                        'policy_net_state_dict': policy_net.state_dict(),
                        'target_net_state_dict': target_net.state_dict() if not target_off else [],
                        'optimizer_state_dict': optimizer.state_dict(),
                        'avg_return': avg_return,
                        'return_per_run': data_return,
                        'frame_stamp_per_run': frame_stamp,
                        'replay_buffer': r_buffer if not replay_off else []
            }, output_file_name + "_checkpoint")

    # Print final logging info
    logging.info("Avg return: " + str(numpy.around(avg_return, 2)) + " | Time per frame: " + str((time.time()-t_start)/t))

    # Write data to file
    torch.save({
        'returns': data_return,
        'frame_stamps': frame_stamp,
        'policy_net_state_dict': policy_net.state_dict()
    }, output_file_name + "_data_and_weights")






def dqnContinual(env, envName, model, replay_off, target_off, output_file_name, contBuffer=None, step_size=STEP_SIZE, target = None, ewc = None, modelDir = None):
    in_channels = 10
    num_actions = 6
    policy_net = model.to(device)
    replay_start_size = 0
    ld = {"rewards": [], "episodes": 0, "steps": NUM_FRAMES, "losses": [], "buffer_size": 0}
    if not target_off:
        #target_net = QNetwork(in_channels, num_actions).to(device)             # CHANGED.
        target_net = target.to(device)
        target_net.load_state_dict(policy_net.state_dict())
    policy_net = policy_net.to(device)

    if not replay_off:
        if contBuffer is None:
            r_buffer = replay_buffer(REPLAY_BUFFER_SIZE)
            replay_start_size = REPLAY_START_SIZE
        else:
            r_buffer = contBuffer
            replay_start_size = REPLAY_START_SIZE

    optimizer = optim.RMSprop(policy_net.parameters(), lr=step_size, alpha=SQUARED_GRAD_MOMENTUM, centered=True, eps=MIN_SQUARED_GRAD)

    # Set initial values
    e_init = 0
    t_init = 0
    policy_net_update_counter_init = 0
    avg_return_init = 0.0
    data_return_init = []
    frame_stamp_init = []

    # Data containers for performance measure and model related data
    data_return = data_return_init
    frame_stamp = frame_stamp_init
    avg_return = avg_return_init

    # Train for a number of frames
    t = t_init
    e = e_init
    policy_net_update_counter = policy_net_update_counter_init
    t_start = time.time()
    while t < NUM_FRAMES:
        # Initialize the return for every episode (we should see this eventually increase)
        G = 0.0

        # Initialize the environment and start state
        env.reset()
        s = get_state(env.state())
        is_terminated = False
        while(not is_terminated) and t < NUM_FRAMES:
            # Generate data
            s_prime, action, reward, is_terminated = world_dynamics(t, replay_start_size, num_actions, s, env, policy_net)

            sample = None
            if replay_off:
                sample = [transition(s, s_prime, action, reward, is_terminated)]
            else:
                # Write the current frame to replay buffer
                r_buffer.add(s, s_prime, action, reward, is_terminated)

                # Start learning when there's enough data and when we can sample a batch of size BATCH_SIZE
                if t > REPLAY_START_SIZE and len(r_buffer.buffer) >= BATCH_SIZE:
                    # Sample a batch
                    sample = r_buffer.sample(BATCH_SIZE)

            # Train every n number of frames defined by TRAINING_FREQ
            if t % TRAINING_FREQ == 0 and sample is not None:
                if target_off:
                    lo = train(sample, policy_net, policy_net, optimizer, ewc=ewc)
                    ld["losses"].append(lo)
                else:
                    policy_net_update_counter += 1
                    lo = train(sample, policy_net, target_net, optimizer, ewc=ewc)
                    ld["losses"].append(lo)

            # Update the target network only after some number of policy network updates
            if not target_off and policy_net_update_counter > 0 and policy_net_update_counter % TARGET_NETWORK_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            G += reward.item()

            t += 1

            # Continue the process
            s = s_prime

        # Increment the episodes
        e += 1
        ld["episodes"] += 1

        # Save the return for each episode
        data_return.append(G)
        frame_stamp.append(t)
        ld["rewards"].append(G)

        # Logging exponentiated return only when verbose is turned on and only at 1000 episode intervals
        avg_return = 0.90 * avg_return + 0.10 * G
        if e % 10 == 0:
            print("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                         str(numpy.around(avg_return, 2)) + " | Frame: " + str(t)+" | Time per frame: " +str((time.time()-t_start)/t) )
            logging.info("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                         str(numpy.around(avg_return, 2)) + " | Frame: " + str(t)+" | Time per frame: " +str((time.time()-t_start)/t) )

    # Print final logging info
    logging.info("Avg return: " + str(numpy.around(avg_return, 2)) + " | Time per frame: " + str((time.time()-t_start)/t))

    # Write data to file
    #torch.save({
    #    'returns': data_return,
    #    'frame_stamps': frame_stamp,
    #    'policy_net_state_dict': policy_net.state_dict()
    #}, output_file_name + "_data_and_weights")
    if not replay_off:
        ld["buffer_size"] = r_buffer.buffer_size
    if modelDir is not None:
        with open(os.path.join(modelDir, "log_%s.json" % envName), 'w') as jsonfile:
            json.dump(ld, jsonfile)
    return policy_net





def trainTaskDetector(buf, td, env, ep = 40000, bs = 128):
    for _ in range(ep):
        sample = buf.sample(bs)
        batch_samples = transition(*zip(*sample))
        x = torch.cat(batch_samples.state)
        td.trainStep(x, env)






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", "-g", type=str)
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--checkpoint_file", type=str)
    parser.add_argument("--alpha", "-a", type=float, default=STEP_SIZE)
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--mode", help="Run mode", choices=["single", "base", "replay", "prog", "prog_td", "ewc"], default="base")
    parser.add_argument("--save_dir", help="Specify path to save model.", type=str, default="./data/models/")
    parser.add_argument("--load_up", help="Number of tasks to pre-load in order.", type=int, default=0)
    parser.add_argument("--single_env", help="Specify if you want to run just one env at a time.", choices = ["", "asterix", "breakout", "freeway", "seaquest", "space_invaders"], default="")
    #parser.add_argument("--prog_file", help="Progressive neural net file.", type=str, default=None)
    #parser.add_argument("--prog_size", help="Number of cols saved in prog_file.", type=int, default=0)
    args = parser.parse_args()

    envs = ["asterix", "breakout", "freeway", "seaquest", "space_invaders"]

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    if args.output:
        file_name = args.output
    else:
        file_name = os.getcwd() + "/" + args.mode
        if args.mode == "single":
            file_name += "_" + args.game
    load_file_path = None
    if args.checkpoint_file:
        load_file_path = args.checkpoint_file

    print("Cuda availability:  %s." % str(torch.cuda.is_available()))
    print("Using device:  %s." % device)
    recsStored = 0


    if args.mode == "single":
        model = common.QNetwork(10, 6).to(device)
        dqn(env, False, False, file_name, model, args.save, load_file_path, args.alpha)
        torch.save(model.state_dict(), os.path.join(args.save_dir, "single_model_%s.pt" % env))
    elif args.mode == "base":
        model = common.QNetwork(10, 6).to(device)
        target = common.QNetwork(10, 6).to(device)
        for e in envs:
            env = Environment(e)
            print("Training on %s." % e)
            dqnContinual(env, e, model, False, False, file_name, step_size=args.alpha, target=target, modelDir=args.save_dir)
            target.load_state_dict(model.state_dict())
            torch.save(model.state_dict(), os.path.join(args.save_dir, "base_nr_model_%s.pt" % e))
    elif args.mode == "replay":
        model = common.QNetwork(10, 6).to(device)
        target = common.QNetwork(10, 6).to(device)
        buf = replay_buffer(REPLAY_BUFFER_SIZE)
        oldBufs = []
        for e in envs:
            env = Environment(e)
            print("Training on %s." % e)
            dqnContinual(env, e, model, False, False, file_name, contBuffer=buf, step_size=args.alpha, target=target, modelDir=args.save_dir)
            target.load_state_dict(model.state_dict())
            torch.save(model.state_dict(), os.path.join(args.save_dir, "base_r_model_%s.pt" % e))
            oldBufs.append(buf)
            recsStored += len(buf.buffer)
            buf = randomBufferUnion(oldBufs, REPLAY_BUFFER_SIZE)
    elif args.mode == "prog":
        prognet = ProgNet(colGen=common.QGenerator(10, 6))
        prognetTarget = ProgNet(colGen=common.QGenerator(10, 6))
        model = common.ProgWrapper(prognet)
        target = common.ProgWrapper(prognetTarget)
        for e in envs:
            env = Environment(e)
            print("Training on %s." % e)
            model.switch(e)
            target.switch(e)
            model = dqnContinual(env, e, model, False, False, file_name, step_size=args.alpha, target=target, modelDir=args.save_dir)
            target.load_state_dict(model.state_dict())
            torch.save(prognet.state_dict(), os.path.join(args.save_dir, "prognet_model_%s.pt" % e))
    elif args.mode == "prog_td":
        prognet = ProgNet(colGen=common.QGenerator(10, 6))
        prognetTarget = ProgNet(colGen=common.QGenerator(10, 6))
        td = TaskDetector(common.DetGen(device), args.save_dir)
        model = common.ProgWrapper(prognet)
        target = common.ProgWrapper(prognetTarget)
        if args.load_up > 0 and args.load_up < len(envs):
            lp = os.path.join(args.save_dir, "prognet_model_%s.pt" % envs[args.load_up - 1])
            common.loadPrognet(prognet, envs[:args.load_up], lp)
            common.loadPrognet(prognetTarget, envs[:args.load_up], lp)
            print("Loaded:   %s." % lp)
        if args.single_env != "":
            envs = [args.single_env]
        for e in envs:
            buf = replay_buffer(REPLAY_BUFFER_SIZE)
            env = Environment(e)
            print("Training on %s." % e)
            model.switch(e)
            target.switch(e)
            td.addTask(e)
            model = dqnContinual(env, e, model, False, False, file_name, contBuffer=buf, step_size=args.alpha, target=target, modelDir=args.save_dir)
            target.load_state_dict(model.state_dict())
            print("Training TD on %s." % e)
            trainTaskDetector(buf, td, e)
            td.expelDetector(e)
            torch.save(prognet.state_dict(), os.path.join(args.save_dir, "prognet_model_%s.pt" % e))
    elif args.mode == "ewc":
        model = common.QNetwork(10, 6).to(device)
        target = common.QNetwork(10, 6).to(device)
        ewc = EWC(device = device)
        for i, e in enumerate(envs):
            if i > 0:
                print("Sampling old tasks for EWC.")
                sam = buf.sample(256)
                bsam = transition(*zip(*sam))
                oldDS = [torch.cat(bsam.state)]
                print("Archiving for EWC.")
                ewc.archive(model, oldDS)
            buf = replay_buffer(REPLAY_BUFFER_SIZE)
            env = Environment(e)
            print("Training on %s." % e)
            if i > 0:
                dqnContinual(env, e, model, False, False, file_name, contBuffer=buf, step_size=args.alpha, target=target, ewc = ewc, modelDir=args.save_dir)
            else:
                dqnContinual(env, e, model, False, False, file_name, contBuffer=buf, step_size=args.alpha, target=target, ewc = None, modelDir=args.save_dir)
            target.load_state_dict(model.state_dict())
            torch.save(model.state_dict(), os.path.join(args.save_dir, "ewc_model_%s.pt" % e))
    print("Records stored:  %d." % recsStored)
    print("Done.")







if __name__ == '__main__':
    main()
