
RANDOM_SEED = 13011

import os
import sys
import json
from collections import defaultdict
import random
random.seed(RANDOM_SEED)
import numpy as np
np.random.seed(RANDOM_SEED)
import torch
torch.manual_seed(RANDOM_SEED)
import argparse

from minatar import Environment
import common
from task_detector import TaskDetector
from Doric import ProgNet



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def pad(o):
    p = np.zeros([10, 10, 10 - o.shape[-1]])
    o = np.concatenate([o, p], axis = 2)
    return o



def getObs(s):
    s = (torch.tensor(pad(s), device=device).permute(2, 0, 1)).unsqueeze(0).float()
    return s



def printCM(cm, tasks):
    print("          Predicted:")
    print("         ", end = '')
    dotsLen = 0
    for task in tasks:
        s = task + "   "
        print(s, end = '')
        dotsLen += len(s)
    print()
    print("         ", end = '')
    for _ in range(dotsLen):
        print('-', end = '')
    print()
    for taskNum in range(len(tasks)):
        print("Task " + str(taskNum) + ":  |", end = '')
        for predNum in range(len(tasks)):
            print("{:8.3f}".format(cm[(tasks[predNum], tasks[taskNum])]), end = "   ")
        print("|")
    print("         ", end = '')
    for _ in range(dotsLen):
        print('-', end = '')
    print()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="Run mode", choices=["base", "replay", "prog", "prog_td", "ewc"], default="base")
    parser.add_argument("--save_dir", help="Specify path to save model.", type=str, default="./data/models/")
    parser.add_argument("--episodes", help="Episodes to run per env.", type=int, default=300)
    parser.add_argument("--display", help="Specify 1 to display using matplotlib.", type=int, choices = [0, 1], default=0)
    args = parser.parse_args()
    envs = ["asterix", "breakout", "freeway", "seaquest", "space_invaders"]
    logDict = {"episodes": args.episodes}
    print("Cuda availability:  %s." % str(torch.cuda.is_available()))
    print("Using device:  %s." % device)
    if args.mode == "base":
        model = common.QNetwork(10, 6).to(device)
        model.load_state_dict(torch.load(os.path.join(args.save_dir, "base_nr_model_%s.pt" % envs[-1])))
    elif args.mode == "replay":
        model = common.QNetwork(10, 6).to(device)
        model.load_state_dict(torch.load(os.path.join(args.save_dir, "base_r_model_%s.pt" % envs[-1])))
    elif args.mode == "prog":
        prognet = ProgNet(colGen=common.QGenerator(10, 6))
        common.loadPrognet(prognet, envs, os.path.join(args.save_dir, "prognet_model_%s.pt" % envs[-1]))
        model = common.ProgWrapper(prognet)
    elif args.mode == "prog_td":
        prognet = ProgNet(colGen=common.QGenerator(10, 6))
        common.loadPrognet(prognet, envs, os.path.join(args.save_dir, "prognet_model_%s.pt" % envs[-1]))
        td = TaskDetector(common.DetGen(device), args.save_dir)
        td.loadAll(names = envs)
        model = common.ProgTD(prognet, td)
    elif args.mode == "ewc":
        model = common.QNetwork(10, 6).to(device)
        model.load_state_dict(torch.load(os.path.join(args.save_dir, "ewc_model_%s.pt" % envs[-1])))
    model.to(device)
    model.eval()
    dqn = common.DQN(model)
    with torch.no_grad():
        cm = defaultdict(lambda: 0.0)
        for e in envs:
            print("Testing %s on %s." % (args.mode, e))
            logDict["rewards_%s" % e] = []
            logDict["steps_%s" % e] = []
            env = Environment(e)
            if args.mode == "prog":
                model.switch(e)
            envReward = 0.0
            for ep in range(args.episodes):
                env.reset()
                obs = getObs(env.state())
                terminal = False
                epReward = 0.0
                t = 0
                while not terminal:
                    action = dqn.act(obs)
                    if args.mode == "prog_td":
                        dtask = model.getLastCol()
                        cm[(dtask, e)] += 1
                    reward, terminal = env.act(action)
                    if args.display == 1:
                        env.display_state(50)
                    nobs = getObs(env.state())
                    obs = nobs
                    epReward += reward
                    t += 1
                logDict["rewards_%s" % e].append(epReward)
                logDict["steps_%s" % e].append(t)
                envReward += epReward
                print("   Env: %s;   Ep: %d;   Steps: %d;   R: %f." % (e, ep, t, epReward))
            print("Env %s total reward:  %f." % (e, envReward))
            print("Env %s avg reward:  %f." % (e, envReward / args.episodes))
            if args.display == 1:
                env.close_display()
    printCM(cm, envs)
    logPath = os.path.join(args.save_dir, "%s_test_log.json" % (args.mode))
    with open(logPath, 'w') as jsonfile:
        json.dump(logDict, jsonfile)




if __name__ == '__main__':
    main()
