from __future__ import print_function
import os
import numpy as np
import torch
import torch.optim as optim
import pickle
import shutil
import sys
import warnings
from itertools import chain
from datetime import datetime
from pathlib import Path
from copy import deepcopy

sys.path.append(os.path.join('..','lib'))
import plotting
import core_tri
import sac_tri_envs
import extra

"""
This mudule contains the objects used to train quantum thermal machine environments with 1 continuous action
and one discrete action that can be 0,1,2. All torch tensors that are not integers are torch.float32.
It was written starting from the code:
J. Achiam, Spinning Up in Deep Reinforcement Learning, https://github.com/openai/spinningup (2018).
"""

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents with continuous and discrete actions.

    Args:
        obs_dim (int): number of continuous parameters of observation space.
        act_dim (int): number of continuous parameters of action space.
        size (int): size of the buffer.
        device (torch.device): which torch device to use.
    """

    def __init__(self, obs_dim, act_dim, size, device):  
        self.obs_buf = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.obs2_buf = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.tri_act_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, tri_act, act, rew, next_obs):
        """
        stores a transition into the buffer. All args are torch.float32.

        Args:
            obs (torch.tensor): the initial state
            tri_act (torch.tensor): the discrete action (0,1,2)
            act (torch.tensor): the continuous action
            rew (torch.tensor): the rewards
            next_obs (torch.tensor): the next state        
        """
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.tri_act_buf[self.ptr] = tri_act
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size):
        """
        Return a random batch of experience from the buffer.
        The batch index is the leftmost index.

        Args:
            batch_size (int): size of batch
        """
        idxs = torch.randint(0, self.size, size=(batch_size,), device=self.device)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     tri_act=self.tri_act_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs])
        return batch

def state_to_tensor(state, device):
    """ Coverts a numpy state to a torch.tensor with the right dimension """
    return torch.as_tensor(state, device=device, dtype=torch.float32).view(-1)

def action_to_tensor(state, device):
    """ Coverts a numpy action to a torch.tensor with the right dimension """
    return torch.as_tensor(state, device=device, dtype=torch.float32)

class SacTrain(object):
    """
    Main class to train the RL agent on a quantum thermal machine environment
    with 1 continuous action, and 1 discrete action (0,1,2).
    This class can create a new training session, or load an existing one.
    It takes care of logging and of saving the training session all in 1 folder.

    Usage:
        After initialization either
        - call initialize_new_train() to initialize a new training session
        - call load_train() to load an existing training session
    """

    #define some constants defining the filestructure of the logs and saved state.
    PARAMS_FILE_NAME = "params.txt"
    S_FILE_NAME = "s.dat"
    POLICY_NET_FILE_NAME = "policy_net.dat"
    TARGET_NET_FILE_NAME = "target_net.dat"
    STATE_FOLDER_NAME = "state"
    SAVE_DATA_DIR = os.path.join("..", "data")
    SAVED_LOGS_FOLDER = "logs"
    RUNNING_REWARD_FILE_NAME = "running_reward.txt"
    RUNNING_LOSS_FILE_NAME = "running_loss.txt"
    ACTIONS_FILE_NAME = "actions.txt"
    SAVED_POLICY_DIR_NAME = "saved_policies"

    #internal variables used during training.
    zero_float = None
    one_float = None
    two_float = None

    #Methods that can be called:

    def initialize_new_train(self, env_class, env_params, training_hyperparams, log_info):
        """ Initializes a new training session. Should be called right after initialization.

        Args:
            env_class (gym.Env): class representing the quantum thermal machine environment to learn
            env_params (dict): parameters used to initialize env_class. See specific env requirements.
            training_hyperparameters (dict): dictionary with training hyperparameters. Must contain the following
                "BATCH_SIZE" (int): batch size
                "LR" (float): learning rate
                "ALPHA_START" (float): initial value of SAC temperature
                "ALPHA_END" (float): final value of SAC temperature
                "ALPHA_DECAY" (float): exponential decay of SAC temperature
                "REPLAY_MEMORY_SIZE" (int): size of replay buffer
                "POLYAK" (float): polyak coefficient
                "LOG_STEPS" (int): save logs and display training every number of steps
                "GAMMA" (float): RL discount factor
                "HIDDEN_SIZES" tuple(int): size of hidden layers 
                "SAVE_STATE_STEPS" (int): saves complete state of trainig every number of steps
                "INITIAL_RANDOM_STEPS" (int): number of initial uniformly random steps
                "UPDATE_AFTER" (int): start minimizing loss function after initial steps
                "UPDATE_EVERY" (int): performs this many updates every this many steps
                "USE_CUDA" (bool): use cuda for computation
            log_info (dict): specifies logging info. Must contain
                    "log_running_reward": log running reward 
                    "log_running_loss": log running loss
                    "log_actions" (bool): log chosen actions
                    "extra_str" (str): extra string to append to training folder
        """
        
        #initialize a SacTrainState to store the training state 
        self.s = extra.SacTrainState()

        #save input parameters
        self.s.save_data_dir = self.SAVE_DATA_DIR
        self.s.env_params = env_params
        self.s.training_hyperparams = training_hyperparams
        self.s.log_info = log_info 
 
        #setup the torch device
        if self.s.training_hyperparams["USE_CUDA"]:
            if torch.cuda.is_available():
                self.s.device = torch.device("cuda")
            else:
                warnings.warn("Cuda is not available. Will use cpu instead.")
                self.s.device = torch.device("cpu")
        else:
            self.s.device = torch.device("cpu")

        #create environment
        self.env = env_class(self.s.env_params)

        #add the environment name to the env_params dictionary
        self.s.env_params["env_name"] = self.env.__class__.__name__

        #set the training steps_done to zero
        self.s.steps_done = 0

        #reset the environment and save the initial state
        self.s.state = state_to_tensor(self.env.reset(), self.s.device)

        #initialize logging session
        self.s.log_session = self.initialize_log_session()

        #setup the memory replay buffer
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space[1].shape[0]
        self.s.memory = ReplayBuffer(obs_dim, act_dim, self.s.training_hyperparams["REPLAY_MEMORY_SIZE"],self.s.device)

        # #initialize the NNs
        self.initialize_nns()

        # #setup the optimizer
        self.create_optimizer()

    def load_train(self, log_folder, specific_state_folder = None, no_train=False):
        """
        Loads a training session that had been previously saved. The training
        sessions are saved as folders numbered as "0", "1",... By default, the latest
        one is loaded.
        
        Args:
            log_folder (str): folder of the training session
            specific_state_folder (str): can load a specific save. If None, loads the latest one.
            no_train (bool): if False, it creates a new logging folder where all new saves and loggings
                are located. If true, it doesn't create a new folder, but cannot train anymore.        
        """

        #construct the location of the actual saved data
        save_dir_path = os.path.join(log_folder, self.STATE_FOLDER_NAME)
        if specific_state_folder is not None:
            save_dir_path = os.path.join(save_dir_path,specific_state_folder) 
        else:
            #must find the latest folder if not specificed
            path = Path(save_dir_path)
            folders = [dir.name for dir in path.iterdir() if dir.is_dir()]
            index = int(folders[0])
            for folder in folders:
                index = max(index, int(folder))
            save_dir_path = os.path.join(save_dir_path, str(index)) 

        #load self.s
        with open(os.path.join(save_dir_path, self.S_FILE_NAME), 'rb') as input:
            self.s = pickle.load(input)

        #load the environment
        env_method = self.return_env_class_from_name()
        self.env = env_method(self.s.env_params)
        try:
            self.env.set_current_state(self.s.state.cpu().numpy())
        except:
            self.env.set_current_state(self.s.state)

        #create the nns
        self.initialize_nns()
        #load the policy net
        self.ac.load_state_dict(torch.load(os.path.join(save_dir_path, self.POLICY_NET_FILE_NAME)))
        #load the targer net
        self.ac_targ.load_state_dict(torch.load(os.path.join(save_dir_path, self.TARGET_NET_FILE_NAME)))

        #i stop here if i don't want to further optimize the model
        if no_train:
            self.s.log_session.log_dir = log_folder
        else:
            #load the optimizer
            self.create_optimizer()
            #now that everything is loaded, i create a new LogSession, and copy in the old logs
            self.s.log_session = self.initialize_log_session(reset_running_vars = False)
            for file in Path(os.path.join(save_dir_path, self.SAVED_LOGS_FOLDER)).iterdir():
                shutil.copy(str(file), os.path.join(self.s.log_session.log_dir, file.name))

    def train(self, steps, output_plots = True):
        """
        Runs "steps" number of training steps. Takes care of saving and logging. It can be called multiple
        times and it will keep training the same model.

        Args:
            steps (int): number of training steps to perform
            output_plots (bool): if true, it will output a plot with all the running logs every LOG_STEPS.
        """
        for _ in range(steps):
            
            #choose an action (random uniform for first INITIAL_RANDOM_STEPS, then according to policy )
            if self.s.steps_done > self.s.training_hyperparams["INITIAL_RANDOM_STEPS"]:
                tri_a, a = self.get_action(self.s.state)
            else:
                tri_a = action_to_tensor(self.env.action_space[0].sample(),self.s.device)
                a = action_to_tensor(self.env.action_space[1].sample(),self.s.device)

            #perform the action on environment
            o2_np, r, _, _ = self.env.step( (int(np.round(tri_a.cpu().numpy())), a.cpu().numpy()) )
            o2 = state_to_tensor(o2_np,self.s.device)
            
            # Store experience to replay buffer
            self.s.memory.store(self.s.state, tri_a, a, r, o2)
            
            #move to the next state
            self.s.state = o2

            #increase the step counter
            self.s.steps_done += 1

            # Perform NN parameters updates
            if self.s.steps_done > self.s.training_hyperparams["UPDATE_AFTER"] and \
                self.s.steps_done % self.s.training_hyperparams["UPDATE_EVERY"] == 0:
                for _ in range(self.s.training_hyperparams["UPDATE_EVERY"]):
                    #collect a batch of experience to use for training
                    batch = self.s.memory.sample_batch(self.s.training_hyperparams["BATCH_SIZE"])
                    #perform the update using the batch
                    q_loss, pi_loss = self.update(data=batch)
                    #update logging: running loss
                    self.s.running_loss[0] += (1.-self.s.training_hyperparams["GAMMA"])*(q_loss - self.s.running_loss[0])
                    self.s.running_loss[1] += (1.-self.s.training_hyperparams["GAMMA"])*(pi_loss - self.s.running_loss[1])

            #update logging: reward and action
            self.s.running_reward += (1.-self.s.training_hyperparams["GAMMA"])*(r - self.s.running_reward)
            self.s.actions.append([self.s.steps_done] + tri_a.view(-1).tolist() + a.tolist() ) 
            
            #if its time to log
            if self.s.steps_done % self.s.training_hyperparams["LOG_STEPS"] == 0 :
                #update log files
                self.update_log_files()
                
                #plot the logs
                if output_plots:
                    self.plot_logs()
            
            #if it's time to save the full training state
            if self.s.steps_done % self.s.training_hyperparams["SAVE_STATE_STEPS"] == 0:
                self.save_full_state()

    def save_full_state(self):
        """
        Saves the full state to file, so that training can continue exactly from here by loading the file.
        The saved session is placed in a folder inside STATE_FOLDER_NAME, named using an ascending index
         0, 1, ... Largest index is the most recent save.
         """
        #folder where the session is saved
        path_location = os.path.join(self.s.log_session.state_dir, str(len(list(Path(self.s.log_session.state_dir).iterdir()))))
        #create the folder to save the state
        Path(path_location).mkdir(parents=True, exist_ok=True)
        #save self.s state object
        with open(os.path.join(path_location, self.S_FILE_NAME), 'wb') as output: 
            pickle.dump(self.s, output, pickle.HIGHEST_PROTOCOL)
        #save policy_net params
        torch.save(self.ac.state_dict(), os.path.join(path_location, self.POLICY_NET_FILE_NAME))
        #save target_net params
        torch.save(self.ac_targ.state_dict(), os.path.join(path_location, self.TARGET_NET_FILE_NAME))
        #copy over the logging folder 
        saved_logs_path = os.path.join(path_location, self.SAVED_LOGS_FOLDER)
        Path(saved_logs_path).mkdir(parents=True, exist_ok=True)
        for file in Path(self.s.log_session.log_dir).iterdir():
            if not file.is_dir() :
                shutil.copy(str(file), os.path.join(saved_logs_path, file.name))

    def evaluate_current_policy(self, deterministic, steps=1000, suppress_show=False, gamma=None,actions_to_plot=400,
                                save_policy_to_file_name = None,actions_ylim=None):
        """
        creates a copy of the environment, and evaluates the current policy in a deterministic or probabilistic way.
        It return the final running rewards, and plots the running rewards and actions.

        Args:
            deterministic(bool): if the chosen actions should be deterministic or not
            steps(int): how many steps of the environment to do to evaluate the running return
            suppress_show(bool): if True, it will not plot anything
            gamma(float): the exponential average factor to compute the return
                It doesn't have to coincide with the one used for training
            actions_to_plot(int): how many of the latest actions to show in the plots
            save_policy_to_file_name(str): if specified, it will save the chosen actions to file, so they can be plotted
            actions_ylim(tuple(float,float)): the y_lim for plotting the actions

        Returns:
            (float): the final value of the running return
        """
        #if gamma is not specified, it will use the one used during training
        if gamma is None:
            gamma = self.s.training_hyperparams["GAMMA"]
        #if necessary, we create the path to save the actions
        if save_policy_to_file_name is not None:
            save_policy_to_file_name = os.path.join(self.s.log_session.log_dir, self.SAVED_POLICY_DIR_NAME, save_policy_to_file_name)
        #define an auxillary necessary for torch to numpy compatibility
        def action_to_numpy(act):
            return ( int(np.round(act[0].cpu().numpy())), act[1].cpu().numpy() )
        #evaluates the policy
        return extra.test_policy(self.return_env_class_from_name(), self.s.env_params,
                     lambda o: action_to_numpy(self.get_action(torch.as_tensor(o,dtype=torch.float32,device=self.s.device),
                     deterministic=deterministic)), gamma, True, steps=steps, env_state = self.s.state.cpu().numpy(),
                     suppress_show=suppress_show,actions_to_plot=actions_to_plot,save_policy_to_file_name=save_policy_to_file_name,
                     actions_ylim=actions_ylim)

    #Methods that should only be used internally:

    def initialize_log_session(self, reset_running_vars = True):
        """
        creates a folder, named with the current time and date, for logging and saving a training session,
        and saves all the physical parameters and hyperparameters in file PARAMS_FILE_NAME. 
        
        Args:
            reset_running_vars (bool): wether to reset the logged data

        Raises:
            Exception: if the folder for logging already exists
        
        Returns:
            log_session (extra.LogSession): info used by this class to do logging and saving state in the right place
        """
        #reset the running variables
        if reset_running_vars:
            self.s.running_reward = 0.
            self.s.running_loss = np.zeros(2, dtype=np.float32)
            self.s.actions =[]

        #create folder for logging
        now = datetime.now()
        log_dir = os.path.join(self.s.save_data_dir, now.strftime("%Y_%m_%d-%H_%M_%S") + self.s.log_info["extra_str"] )
        Path(log_dir).mkdir(parents=True, exist_ok=False)
            
        #create a file with all the environment params and hyperparams
        param_str = ""
        for name, value in chain(self.s.env_params.items(), self.s.training_hyperparams.items()):
            param_str += f"{name}:\t{value}\n"
        param_file = open(os.path.join(log_dir, self.PARAMS_FILE_NAME),"w") 
        param_file.write(param_str)
        param_file.close()
        
        #create files for logging
        running_reward_file = os.path.join(log_dir, self.RUNNING_REWARD_FILE_NAME)
        running_loss_file = os.path.join(log_dir, self.RUNNING_LOSS_FILE_NAME)
        actions_file = os.path.join(log_dir, self.ACTIONS_FILE_NAME)

        #create folder for saving the state
        state_dir = os.path.join(log_dir, self.STATE_FOLDER_NAME)
        Path(state_dir).mkdir(parents=True, exist_ok=True)
        
        return extra.LogSession(log_dir, state_dir, self.s.log_info["log_running_reward"], self.s.log_info["log_running_loss"],
                             self.s.log_info["log_actions"], running_reward_file, running_loss_file, actions_file)                
   
    def initialize_nns(self):
        """ Initializes the NNs for the soft actor critic method """
        #create the main NNs
        self.ac = core_tri.MLPActorCritic(self.env.observation_space, self.env.action_space,
                                     hidden_sizes=self.s.training_hyperparams["HIDDEN_SIZES"]).to(self.s.device)
        #create the target NNs
        self.ac_targ = deepcopy(self.ac)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (saved for convenience)
        self.q_params = chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Count and print number of variables 
        var_counts = tuple(core_tri.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    def create_optimizer(self):
        """ Setup the ADAM optimizer for pi and q"""
        self.pi_optimizer = optim.Adam(self.ac.pi.parameters(), lr=self.s.training_hyperparams["LR"])
        self.q_optimizer = optim.Adam(self.q_params, lr=self.s.training_hyperparams["LR"]) 
    
    def current_alpha(self):
        """ returns the current value of alpha, which decreases exponentially """
        return self.s.training_hyperparams["ALPHA_END"] + \
            (self.s.training_hyperparams["ALPHA_START"] - self.s.training_hyperparams["ALPHA_END"]) * \
            np.exp(-1. * self.s.steps_done / self.s.training_hyperparams["ALPHA_DECAY"])

    def get_action(self, o, deterministic=False):
        """ Returns an on-policy action based on the state passed in.
        This computation does not compute the gradients.

        Args:
            o (torch.Tensor): state from which to compute action
            deterministic (bool): wether the action should be sampled or deterministic  
        """
        return self.ac.act(o, deterministic)

    def is_zero(self,tens):
        """ Checks which elements of tens are nearly zero.
        Args:
            tens (torch.Tensor(float32)): some input tensor
        
        Returns:
            (torch.Tensor(bool)): tensor with the same shape of tens with True where the element was
                close to 0
        """
        if self.zero_float is None:
            self.zero_float = torch.tensor(0., dtype = torch.float32, device = self.s.device)
        return torch.isclose(tens,self.zero_float) 

    def is_one(self,tens):
        """ Checks which elements of tens are nearly one
        Args:
            tens (torch.Tensor(float32)): some input tensor
        
        Returns:
            (torch.Tensor(bool)): tensor with the same shape of tens with True where the element was
                close to 1
        """
        if self.one_float is None:
            self.one_float = torch.tensor(1., dtype = torch.float32, device = self.s.device)
        return torch.isclose(tens,self.one_float) 

    def is_two(self,tens):
        """ Checks which elements of tens are nearly two
        Args:
            tens (torch.Tensor(float32)): some input tensor
        
        Returns:
            (torch.Tensor(bool)): tensor with the same shape of tens with True where the element was
                close to 2
        """
        if self.two_float is None:
            self.two_float = torch.tensor(2., dtype = torch.float32, device = self.s.device)
        return torch.isclose(tens,self.two_float) 

    def compute_loss_q(self, data):
        """
        Compute the loss function of the q-value functions given a batch of data. This function
        is used to find the gradient of the loss using backprop.

        Args:
            data(dict): dictionary with the following
                "obs" (torch.Tensor): batch of states
                "tri_act" (torch.Tensor): batch f discrete actions
                "act" (torch.Tensor): batch of continuous actions
                "rew" (torch.Tensor): batch of rewards
                "obs2" (torch.Tensor): batch of next states

        Returns:
            (torch.Tensor): the sum of the loss function for both q-values
        """
        #unpack the batched data
        o, tri_a, a, r, o2 = data['obs'], data['tri_act'], data['act'], data['rew'], data['obs2']

        #value Q(s,d,u), for each d=0,1,2. Gradients will be computed respect to this.
        q1_vals = self.ac.q1(o,a)
        q2_vals = self.ac.q2(o,a)

        #Select the q1 and q2 values that correspond to the tri_a in the batch
        is_zero_vec = self.is_zero(tri_a).view(-1,1)
        is_one_vec = self.is_one(tri_a).view(-1,1)
        is_two_vec = self.is_two(tri_a).view(-1,1)
        mask = torch.cat([is_zero_vec,is_one_vec,is_two_vec], dim=1)
        q1_vals = q1_vals[mask]
        q2_vals = q2_vals[mask]

        # Bellman backup for Q functions. The gradient is not taken respect to the target
        with torch.no_grad():
            # Target actions come from *current* policy
            _, a2_given_0, a2_given_1, a2_given_2, p, logp_0, logp_1, logp_2 =  self.ac.pi(o2)

            #target Q-values (this could be made more efficient). Average over discrete actions is computed explicitly, while
            #it is sampled from the discrete action.
            q1_pi_targ = p[:,0] * self.ac_targ.q1(o2, a2_given_0)[:,0] + p[:,1] * self.ac_targ.q1(o2, a2_given_1)[:,1] +\
                            p[:,2]* self.ac_targ.q1(o2, a2_given_2)[:,2]
            q2_pi_targ = p[:,0] * self.ac_targ.q2(o2, a2_given_0)[:,0] + p[:,1] * self.ac_targ.q2(o2, a2_given_1)[:,1] +\
                            p[:,2]* self.ac_targ.q2(o2, a2_given_2)[:,2]
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            logp_a2 = p[:,0] * logp_0 + p[:,1]*logp_1 + p[:,2]*logp_2
            backup = r + self.s.training_hyperparams["GAMMA"] * (q_pi_targ - self.current_alpha() * logp_a2)
 
        # MSE loss against Bellman backup
        loss_q1 = ((q1_vals - backup)**2).mean()
        loss_q2 = ((q2_vals - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    def compute_loss_pi(self, data):
        """
        Compute the loss function for the policy given a batch of data. This function
        is used to find the gradient of the loss using backprop.

        Args:
            data(dict): dictionary with the following
                "obs" (torch.Tensor): batch of states

        Returns:
            (torch.Tensor): the loss function for the policy
        """
        o = data['obs']
        _, a_given_0, a_given_1, a_given_2, p, logp_0, logp_1, logp_2 = self.ac.pi(o)

        #compute the average over the discrete states
        q1_pi = p[:,0] * self.ac.q1(o, a_given_0)[:,0] + p[:,1] * self.ac.q1(o, a_given_1)[:,1] +\
                p[:,2] * self.ac.q1(o, a_given_2)[:,2]
        q2_pi = p[:,0] * self.ac.q2(o, a_given_0)[:,0] + p[:,1] * self.ac.q2(o, a_given_1)[:,1] +\
                p[:,2] * self.ac.q2(o, a_given_2)[:,2]
        q_pi = torch.min(q1_pi, q2_pi)
        logp_a = p[:,0] * logp_0 + p[:,1]*logp_1 + p[:,2]*logp_2

        # Entropy-regularized policy loss
        loss_pi = (self.current_alpha() * logp_a - q_pi).mean()
        return loss_pi

    def update(self, data):
        """
        Performs an update of the parameters of both Q and Pi.

        Args:
            data (dict): batch of experience drawn from replay buffer. See compute_loss_q for details
        
        Return:
             (loss_q(float), loss_pi(float)): the numerical value of the loss functions on the data batch
        """
        #update of the two value functions
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data) 
        loss_q.backward()
        self.q_optimizer.step()

        #update of the policy function
        # Freeze Q-networks since they will not be updated
        for p in self.q_params:
            p.requires_grad = False

        # optimze the policy params
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so they can be optimized at the next step
        for p in self.q_params:
            p.requires_grad = True

        # Update target networks by polyak averaging
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # in-place operations
                p_targ.data.mul_(self.s.training_hyperparams["POLYAK"])
                p_targ.data.add_((1 - self.s.training_hyperparams["POLYAK"]) * p.data)

        return loss_q.item(), loss_pi.item()

    def update_log_files(self):
        """ updates all the log files with the current running reward, current running losses, and actions"""
        if self.s.log_session.log_running_reward:
            self.append_log_line(f"{self.s.running_reward}", self.s.log_session.running_reward_file, self.s.steps_done)
        if self.s.log_session.log_running_loss:
            self.append_log_line(np.array_str(self.s.running_loss,999999).replace("[", "").replace("]","")
                                ,self.s.log_session.running_loss_file, self.s.steps_done)
        if self.s.log_session.log_actions: 
            f=open(self.s.log_session.actions_file,'ab')
            np.savetxt(f, self.s.actions)
            f.close()
            self.s.actions = []

    def append_log_line(self, data, file, count):
        """appends count and data to file as plain text """
        file_object = open(file, 'a')
        file_object.write(f"{count}\t{data}\n")
        file_object.close()

    def plot_logs(self):
        """ displays a plot of the current running reward, losses and actions """
        plotting.plot_sac_logs(self.s.log_session.log_dir, True, running_reward_file=self.s.log_session.running_reward_file,
            running_loss_file=self.s.log_session.running_loss_file, actions_file=self.s.log_session.actions_file,
                plot_to_file_line = None, suppress_show=False, save_plot = False, extra_str="")

    def return_env_class_from_name(self):
        """
        Return the class to create a new environment, given the string
        of the environment class name in self.s.env_params['env_name'].
        Looks in sac_tri_envs for the environment class.

        Raises:
            NameError: if env_name doesn't exist

        Returns:
            Returns the class to create the environment
        """
        if hasattr(sac_tri_envs, self.s.env_params['env_name']):
            return getattr(sac_tri_envs, self.s.env_params['env_name'])
        else:
            raise NameError(f"Environment named {self.s.env_params['env_name']} not found in sac_tri_envs")






