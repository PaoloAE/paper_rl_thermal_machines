from __future__ import print_function
import gym
import numpy as np
import dataclasses

"""
This module contains gym.Env environments that can be trained using sac.SacTrain. We implemented a single bath
diagonal qubit (operated e.g. as a heater), and the coherent qubit fridge.
These environments, besides being proper gym.Env, MUST satisfy these additional requirements:
    1) __init__ must accept a single dict with all the parameters necessary to define the environment.
    2) implement set_current_state(state). Functions that takes a state as input, and sets the environment to that state
"""

class SuperconductingQubitRefrigerator(gym.Env):
    """
    Gym.Env representing a refrigerator based on a qubit where the sigma_x component is fixed, and the
    sigma_z prefactor is the only continuous controllable parameter. See 
    jupyter/superconducting_qubit_refrigerator.ipynb and sec IVb of the manuscript for additional info.
    The equation for the evolution are derived from the Lindblad equation given in
    https://doi.org/10.1103/PhysRevB.100.035407
    The reward of this environment is the cooling power out of bath 1. So one must specify
    inverse temperatures such that b0 <= b1.
    
    Args:
        env_params is a dictionary that must contain the following: 
        "g0" (float): g of bath 0
        "g1" (float): g of bath 1
        "b0" (float): inverse temperature \beta of bath 0
        "b1" (float): inverse temperature \beta of bath 1
        "q0" (float): quality factor of bath 0
        "q1" (float): quality factor of bath 1
        "e0" (float): E_0
        "delta" (float): \Delta
        "w0" (float): resonance frequency of bath 0
        "w1" (float): resonance frequency of bath 1
        "min_u" (float): minimum value of action u
        "max_u" (float): maximum value of action u
        "dt" (float): timestep \Delta t
        "reward_extra_coeff" (float): the reward is multiplied by this factor
    """

    @dataclasses.dataclass
    class State:
        """
        Data object representing the state of the environment. We use as state a full 
        description of the density matrix, given by rh0_ee, Re[rho_ge], Im[rho_ge], and the 
        last chosen action. In Python, we denote these with (p, re_p, im_p, u).
        """
        p: float = 0.
        re_p: float = 0.
        im_p: float = 0.
        u: float = 0.
        
    #constant variables used for the computation in step()
    c_mat = np.array([[0.,0.,1.],
                    [-1j,1j,0.],
                    [1.,1.,0.]])
    c_mat_inv = np.linalg.inv(c_mat)

    def __init__(self, env_params):
        super(SuperconductingQubitRefrigerator, self).__init__()
        self.load_env_params(env_params)

    def load_env_params(self, env_params):
        """
        Initializes the environment
        
        Args:
            env_params: environment parameters as passed to __init__()

        Raises:
            assert: temperature of bath 0 must be greater or equal than bath 1
        """

        #load all the parameters 
        self.g0 = env_params["g0"]
        self.g1 = env_params["g1"]
        self.b0 = env_params["b0"]
        self.b1 = env_params["b1"]
        self.q0 = env_params["q0"]
        self.q1 = env_params["q1"]
        self.e0 = env_params["e0"]
        self.delta = env_params["delta"]        
        self.w0 = env_params["w0"]
        self.w1 = env_params["w1"]
        self.min_u = env_params["min_u"]
        self.max_u = env_params["max_u"]
        self.dt = env_params["dt"]
        self.reward_extra_coeff = env_params["reward_extra_coeff"]
        self.state = self.State()

        #check if the temperatures are such that bath 1 is colder than bath 0
        assert self.b0 <= self.b1

        #set the observation and action spaces
        self.observation_space = gym.spaces.Box( low=np.array([0.,-0.5,-0.5, self.min_u], dtype=np.float32),
                                            high=np.array([1.,0.5,0.5,self.max_u], dtype=np.float32), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([self.min_u],dtype=np.float32),
                                            high=np.array([self.max_u],dtype=np.float32), dtype=np.float32)

        #reset the state of the environment
        self.reset_internal_state_variables()

    def reset(self):
        """ resets the state of the environment """
        self.reset_internal_state_variables()
        return self.current_state()
    
    def step(self, action):
        """ Evolves the state for a timestep depending on the chosen action

        Args:
            action (type specificed by self.action_space): the action to perform on the environment

        Raises:
            Exception: action out of bound

        Returns:
            state(np.Array): new state after the step
            reward(float): the reward, i.e. the average cooling power
                from both baths during the current timestep
            end(bool): whether the episode ended (these environments never end)
            additional_info: required by gym.Env, but we don't use it
        """
        
        #check if action in range
        if not self.action_space.contains(action):
            raise Exception(f"Action {action} out of bound")

        #load action
        new_u = action[0]
        
        #compute the effect of the quench in the control
        dtheta = self.theta(new_u) - self.theta(self.state.u)
        rotation = np.array([ [np.cos(2*dtheta), -np.sin(2*dtheta)],
                         [np.sin(2*dtheta), np.cos(2*dtheta)  ]])
        vec1 = np.array([ self.state.p-0.5, self.state.re_p ])
        vec2 = np.array([0.5, 0.])
        temp_p, temp_re_p  = rotation @ vec1 + vec2

        #compute the effect of the constant part of duration dt
        de = self.de(new_u)
        init_state = np.array([temp_p, temp_re_p,self.state.im_p])
        b_element = self.s0(-de) + self.s1(-de)
        s_tot = self.s0(de) + self.s1(de) + b_element
        a = np.array([[ -s_tot, 0., 0.  ],
                    [0., -0.5*s_tot, -de],
                    [0., de, -0.5*s_tot  ]  ])
        b = np.array([b_element, 0., 0.])
        a_eigen = np.array([0.5*(-2.*1j*de-s_tot),0.5*(2*1j*de-s_tot),-s_tot ])
        exp_mat = np.diag(np.exp(a_eigen*self.dt))
        a_inv_b = np.linalg.inv(a) @ b
        sol = np.real( self.c_mat @ exp_mat @ self.c_mat_inv @ ( init_state + a_inv_b ) - a_inv_b )

        #compute the integral of the state between 0 and dt
        exp_mat_integrated = np.diag(  (np.expm1(a_eigen*self.dt))/a_eigen   )
        sol_integrated = np.real( self.c_mat @ exp_mat_integrated @ self.c_mat_inv @ ( init_state + a_inv_b ) - a_inv_b*self.dt )
        p_integrated = sol_integrated[0]

        #compute the reward as heat flowing out of reservoir 1 (which is the cold one)
        reward = de*self.s1(-de) -de*p_integrated*(self.s1(de)+self.s1(-de))/self.dt
        reward *= self.reward_extra_coeff
      
        #update the state
        self.state.p, self.state.re_p, self.state.im_p = sol
        self.state.u = new_u

        return self.current_state(), reward, False, {}
    
    def render(self):
        """ Required by gym.Env. Prints the current state."""
        print(self.state)

    def current_state(self):
        """ Returns the current state as the type specificed by self.observation_space"""
        return np.array([self.state.p, self.state.re_p, self.state.im_p, self.state.u] , dtype=np.float32)

    def set_current_state(self, state):
        """ 
        Allows to set the current state of the environment. This function must be implemented in order
        for sac.SacTrain.load_full_state() to properly load a saved training session.

        Args:
            state (type specificed by self.observation_space): state of the environment
        """
        self.state.p, self.state.re_p, self.state.im_p, self.state.u = state

    def reset_internal_state_variables(self):
        """ Sets the initial values for the state """        
        #set initial population to average temperature and choose random action b
        avg_b = 2./(1./self.b0 + 1./self.b1 )
        random_u =  self.action_space.sample()[0]

        #set the 4 state variables
        self.state.p = self.peq( self.de(random_u) ,avg_b)
        self.state.re_p = 0.
        self.state.im_p = 0.
        self.state.u = random_u

    def de(self,u):
        """
        Returns the instantaneous energy gap of the qubit 

        Args:
            u (float): value of the control
            
        Returns:
            de (float): instantaneous energy gap of the qubit  
        """
        return 2. * self.e0 * np.sqrt(self.delta**2 + u**2)

    def theta(self,u):
        """
        Variable parameterizing the instantaneous eigenstates of the qubit. See
        https://doi.org/10.1103/PhysRevB.100.035407 for details.

        Args:
            u (float): value of the control

        Returns:
        theta (float): value of the angle parameterizing the instantaneous eigenstates
            of the qubit
        """
        return 0.5*np.arctan(self.delta/u)

    def peq(self, de, b):
        """
        Thermal equilibrium probability of being in the excited state

        Args:
            de (float): energy gap of the qubit
            b (float): inverse temperature

        Returns:
            peq (float): thermal equilibrium probability of being in the excited state
        """
        return 1. / (1. + np.exp(b*de) )
    
    def s0(self, de):
        """
        Noise power spectrum of bath 0. See Eq. (13) of the manuscript

        Args:
            de (float): energy gap of the qubit

        Returns:
        s0 (float): noise power spectrum of bath 0
        """
        return 0.5 * self.g0 * de / (1-np.exp(-self.b0*de)) / ( 1 + self.q0**2 * ( de/self.w0 - self.w0/de )**2 )

    def s1(self, de):
        """
        Noise power spectrum of bath 1. See Eq. (13) of the manuscript

        Args:
            de (float): energy gap of the qubit

        Returns:
        s1 (float): noise power spectrum of bath 1
        """
        return 0.5 * self.g1 * de / (1-np.exp(-self.b1*de)) / ( 1 + self.q1**2 * ( de/self.w1 - self.w1/de )**2 )

    def square_policy(self, state, u0, u1):
        """
        WARNING: this function is only used for testing.
        It represents a policy that applies a square policy alternating between u0
        and u1 at each dt step.

        Args:
            state: environment state
            u0 (float): one value of u in the square cycle
            u1 (float): other value of u in the square cycle

        Returns:
            action: the action to perform on the environment
        """
        last_u = state[3]
        d0 = np.abs(last_u-u0)
        d1 = np.abs(last_u-u1)
        if d0<d1:
            return np.array([u1], dtype=np.float32)
        else:
            return np.array([u0], dtype=np.float32)

    def trapezoid_policy(self, state, u0, u1, inter_steps=10, wait_steps=10):
        """
        WARNING: this function is only used for testing. It should not be trusted and it is quite hacky.
        It represents a trapezoidal policy.

        Args:
            state: environment state
            u0 (float): smallest value of u in the trapezoidal cycle
            u1 (float): largerst value of u in the trapezoidal cycle
            inter_steps (int): how many steps necessary to ramp from u0 to u1 (and viceversa)
            wait_steps (int): number of steps to spend at u0 and u1

        Returns:
            action: the action to perform on the environment
        """
        assert u0<u1
        inter_steps += 2
        if not "_trapez_up" in dir(self):
            self._trapez_up = True
            self._trapez_waited_steps = 0
        last_q = state[3]
        q_vals = np.linspace(u0,u1,inter_steps)
        nearest_index = np.argmin(np.abs(q_vals-last_q))
        #if at bottom
        if nearest_index == 0:
            self._trapez_waited_steps += 1
            #if should start going  up again
            if self._trapez_waited_steps >= wait_steps:
                self._trapez_waited_steps = 0
                self._trapez_up = True
                q_ind = 1
            else:
                #if I should wait
                q_ind = 0
        #if at top
        elif nearest_index == inter_steps -1:
            self._trapez_waited_steps += 1
            #if should start going  up again
            if self._trapez_waited_steps >= wait_steps:
                self._trapez_waited_steps = 0
                self._trapez_up = False
                q_ind = inter_steps-2
            else:
                #if I should wait
                q_ind = inter_steps-1
        else:
            #if i'm doing the ramp
            if self._trapez_up:
                q_ind =nearest_index +1
            else:
                q_ind =nearest_index -1
        return np.array([q_vals[q_ind]])

class TwoLevelHeater(gym.Env):
    """
    Gym.Env representing a two level system coupled to a single bath, where the rewards is the
    power dissipated into the environment (a heater). Same model as the one studied in Sec. IVa of
    the manuscript, but operated as a heater with a single bath.
    Just used for testing. The optimal solution for this setup is derived in:
    https://doi.org/10.1088/1367-2630/ab4dca .

    Args:
        env_params is a dictionary that must contain the following: 
        "g0" (float): \Gamma of the bath
        "b0" (float): inverse temperature \beta of the bath
        "min_u" (float): minimum value of action u
        "max_u" (float): maximum value of action u
        "e0" (float): E_0
        "dt" (float): timestep \Delta t
        "reward_extra_coeff" (float): the reward is multiplied by this factor
    """
    def __init__(self, env_params):
        super(TwoLevelHeater, self).__init__()
        self.load_env_params(env_params)

    def load_env_params(self, env_params):
        """
        Initializes the environment
        
        Args:
            env_params: environment parameters as passed to __init__()
        """
        #load the environment parameters
        self.g0 = env_params["g0"]
        self.b0 = env_params["b0"]
        self.min_u = env_params["min_u"]
        self.max_u = env_params["max_u"]
        self.e0 = env_params["e0"]
        self.dt = env_params["dt"]
        self.reward_extra_coeff = env_params["reward_extra_coeff"]

        #set the observation and action spaces
        self.observation_space = gym.spaces.Box( low=np.array([0., -1.],dtype=np.float32),
                                            high=np.array([1.,1.],dtype=np.float32), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([self.min_u],dtype=np.float32),
                                            high=np.array([self.max_u],dtype=np.float32), dtype=np.float32)
 
        #reset the state of the environment
        self.reset_internal_state_variables()

    def reset(self):
        """ resets the state of the environment """
        self.reset_internal_state_variables()
        return self.current_state()
            
    def step(self, action):
        """ Evolves the state for a timestep depending on the chosen action

        Args:
            action (type specificed by self.action_space): the action to perform on the environment

        Raises:
            Exception: action out of bound

        Returns:
            state(np.Array): new state after the step
            reward(float): the reward, i.e. the average heat flux absorbed
                from both baths during the current timestep
            end(bool): whether the episode ended (these environments never end)
            additional_info: required by gym.Env, but we don't use it
        """

        #check if action in bound
        if not self.action_space.contains(action):
            raise Exception(f"Action {action} out of bound")

        #load the action
        eps = self.de(action[0])
        self.chosen_peq = self.peq(eps,self.b0)
        
        #do time evolution
        prev_p = self.p
        self.p = (prev_p - self.chosen_peq)*np.exp(-self.g0*self.dt) + self.chosen_peq
        #the minus sign in the reward is because it's a heater and not an engine
        reward = - self.reward_extra_coeff* eps*(self.p - prev_p)/self.dt
        
        return self.current_state(), reward, False, {}
    
    def render(self):
        """ Required by gym.Env. Prints the current state."""
        state = self.current_state()
        print(f"p: {state[0]};   p-peq = {state[1]}")

    def set_current_state(self, state):
        """ 
        Allows to set the current state of the environment. This function must be implemented in order
        for sac.SacTrain.load_full_state() to properly load a saved training session.

        Args:
            state (type specificed by self.observation_space): state of the environment
        """
        self.p = state[0]
        self.chosen_peq = self.p - state[1]
    
    def current_state(self):
        """ Returns the current state as the type specificed by self.observation_space"""
        return np.array([self.p, self.p - self.chosen_peq] , dtype=np.float32)
           
    def reset_internal_state_variables(self):
        """ sets initial values for the state """
        #Random initial action
        random_u = self.action_space.sample()[0]
        eps = self.de(random_u)
        #set the initial state to this thermal state
        self.p = self.peq(eps, self.b0)
        #choose another random action
        random_action = self.action_space.sample()
        #perform a step to properly initialize the state
        self.step(random_action)

    def peq(self, eps, b):
        """
        Equilibrium probability of being in excited state at energy gap eps and inverse temperature b

        Args:
            eps (float): energy gap of the qubit
            b (float): inverse temperature

        Returns:
            peq (float): equilibrium probability of being in excited state
        """
        return 1. / (1. + np.exp(b*eps) )

    def de(self, u):
        """
        Energy gap of the qubit.

        Args:
            u (float): value of the control
        
        Returns:
            de (float): energy gap of the qubit
        """
        return self.e0 * u

