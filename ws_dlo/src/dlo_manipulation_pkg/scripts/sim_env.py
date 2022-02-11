#!/usr/bin/env python

# main node of the simulation
# bridge between the Unity Simulator and the controller scripts, based on 'mlagents' and 'gym'

import numpy as np
from matplotlib import pyplot as plt
import time
import sys
import rospy
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import gym
from gym_unity.envs import UnityToGymWrapper

from controller_ours import Controller


class Environment(object):
    def __init__(self):
        self.project_dir = rospy.get_param("project_dir")
        self.env_dim = rospy.get_param("env/dimension")

        engine_config_channel = EngineConfigurationChannel()
        env_params_channel = EnvironmentParametersChannel()
        env_file = self.project_dir + "env_dlo/env_" + self.env_dim
        # env_file = None
        unity_env = UnityEnvironment(file_name=env_file, seed=1, side_channels=[engine_config_channel, env_params_channel])
        engine_config_channel.set_configuration_parameters(width=640, height=480, time_scale=5.0) # set the simulation speed ('time_scale' times of real time)
        self.env = UnityToGymWrapper(unity_env)
        
        self.controller = Controller()
        self.control_input = np.zeros((12, ))

    
    # -------------------------------------------------------------------
    def mainLoop(self):
        # state = self.env.reset()

        # the first second in unity is not stable, so we do nothing in the first second
        for k in range(10):
            state, reward, done, _ = self.env.step(self.control_input)
               
        k = 1
        while not rospy.is_shutdown():
            self.control_input = self.controller.generateControlInput(state)
            state, reward, done, _ = self.env.step(self.control_input)

            if done or k % 300 == 0: # Time up (30s), the env and the controller are reset. Next case with different desired shapes.
                self.controller.reset(state)
                self.env.reset()

            k += 1





# --------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        env = Environment()
        env.mainLoop()

    except rospy.ROSInterruptException:
        print("program interrupted before completion.")