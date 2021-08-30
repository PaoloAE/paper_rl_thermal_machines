import time
import tempfile
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.join('..','lib'))
import plotting

"""
This module contains support and extra functions.
"""

class MeasureDuration:
    """ Used to measure the duration of a block of code.

    to use this:
    with MeasureDuration() as m:
        #code to measure
    """
    def __init__(self, what_str=""):
        self.start = None
        self.end = None
        self.what_str = what_str
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        print(f"Time:  {self.duration()}  for {self.what_str}")
    def duration(self):
        return str((self.end - self.start)) + ' s'

@dataclass
class SacTrainState:
    """
    Data class. It is used internally by sac.SacTrain and sac_tri.SacTrain to save the internal
    training state. When saving and loading a training session, it is pickled and unpickled.
    """
    device = None
    save_data_dir = None
    env_params = None
    training_hyperparams = None
    log_info = None
    state = None
    log_session = None
    steps_done = None
    memory = None
    running_reward = None
    running_loss = None
    actions = None

class LogSession(object):
    """
    Data object used to store the location of all the logging and training state data.
    It is used internally by sac.SacTrain and sac_tri.SacTrain to handle training state saving 
    and logging.
    """
    def __init__(self, log_dir, state_dir, log_running_reward, log_running_loss, log_actions,
                running_reward_file, running_loss_file, actions_file):
        self.log_dir = log_dir
        self.state_dir = state_dir
        self.log_running_reward = log_running_reward
        self.log_running_loss = log_running_loss
        self.log_actions = log_actions
        self.running_reward_file = running_reward_file
        self.running_loss_file = running_loss_file
        self.actions_file = actions_file

def test_policy(env_class, env_params, policy, gamma, is_tri, steps=2000, env_state = None, suppress_show=False,
                actions_to_plot=400, save_policy_to_file_name=None, actions_ylim=None):
    """
    Function to test the performance of a given policy. It creates a new instance of the environment, eventually
    at a given initial state, and performs a given number of steps recording the reward and computing the running 
    return weighed by some given gamma (not necessarily the same gamma as training). It then returns the running 
    return, and eventually plots the return and the last actions takes. It can also save the chosen actions to file
    Works both for sac and sac_tri (so both for the continuous and discrete + continuous case)

    Args:
        env_class: class of the environment
        env_params(dict): dictionary of parameters to initialize the environment
        policy: the policy to test, i.e. a function taking a state as input, and outputting an action
        gamma (float): the discount factor used to compute the average return
        is_tri (bool): if the environment also has discrete actions (True) 
        steps (int): number of steps to perform on the environment
        env_state: initial state of the environment. If None, it will be chosen by env_class
        suppress_show (bool): if False, it will plot the running return and the last chosen actions
        actions_to_plot (int): how many of the last actions to show in the plot
        save_policy_to_file_name (str): if specified, it will save the chosen actions to this file
        actions_ylim ((float,float)): y_lim for the plot of the chosen actions

    Returns:
        (float): final value of the running return

    """
    #create an instance of the environment
    env = env_class(env_params)
    state = env.reset()
    #if env_state was specfified, we load it
    if env_state is not None:
        env.set_current_state(env_state)
        state = env_state
    #initialize variables to compute the running reward without bias, and to save the actions
    running_reward = 0.
    o_n = 0.
    running_rewards = []
    actions = []
    #loop to interact with the environment 
    for i in range(steps):
        act = policy(state)
        state,ret,_,_ =  env.step(act)
        o_n += (1.-gamma)*(1.-o_n)
        running_reward += (1.-gamma)/o_n*(ret - running_reward)
        running_rewards.append([i,running_reward])
        if is_tri:
            actions.append([i] + [act[0]] + list(act[1])) 
        else:
            actions.append([i] +  list(act))

    #if necessary, saves the chosen actions to file
    if save_policy_to_file_name is not None:
        f_actions_name = save_policy_to_file_name
        Path(f_actions_name).parent.mkdir(parents=True, exist_ok=True)
    else:
        f_actions_name = None

    #if we need to plot the rewards and actions
    if not suppress_show:
        #save data to a temp file in order to cal the plotting functions which loads data from files
        f_running_rewards = tempfile.NamedTemporaryFile()
        f_running_rewards_name = f_running_rewards.name
        f_running_rewards.close()
        if f_actions_name is None:
            f_actions = tempfile.NamedTemporaryFile()
            f_actions_name = f_actions.name
            f_actions.close()
        np.savetxt(f_running_rewards_name, np.array(running_rewards))
        np.savetxt(f_actions_name, np.array(actions))
        #plot the files
        plotting.plot_sac_logs(Path(f_running_rewards_name).parent.name, running_reward_file=f_running_rewards_name,
            running_loss_file=None, actions_file=f_actions_name, actions_per_log=1,
            plot_to_file_line = None, suppress_show=False, save_plot = False, extra_str="",is_tri=is_tri,
            actions_to_plot=actions_to_plot,actions_ylim=actions_ylim)
    return running_reward




