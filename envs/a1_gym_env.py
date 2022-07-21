
import time
from pybullet_utils import bullet_client

import numpy as np
import pybullet as pb
import pybullet_data
from gym import Env, spaces

from robots.a1 import  A1

class A1GymEnv(Env):
    def __init__(self, task, is_render, args, log, motor_control_mode="Torque" ) -> None:

         # set mode to pb.DIRECT ,providing the fastest, non-visual connection
        
        self.task = task
        self.is_render = is_render
        self.args = args
        self.log = log

        if self.is_render:
            self.client = bullet_client.BulletClient(connection_mode=pb.GUI)
        else:
            self.client = bullet_client.BulletClient()

        self.client.setGravity(0, 0, -9.8)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.robot = A1(self.client, motor_control_mode, args, log)
        self.planeId = self.client.loadURDF("plane.urdf")
        

        self.init_pos, self.int_ori = self.robot.GetBasePosition(), self.robot.GetTrueBaseOrientation(),

        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()
        print("-"*50)
        print(f"In A1GymEnv.__init__ init_pos {self.init_pos}, init_ori {self.int_ori}")

        self.reset()

    def reset(self):
        self.robot.reset()
        self._last_action = np.zeros(self.action_space.shape)
        self.task.reset(self)

        return self._get_observation()

    def step(self, action):

        self._last_action = action
        self._act(action)
        observation = self._get_observation()

        reward = self.task.reward(self)
        done = self.task.done(self)
        #print(f"observation , len {len(observation)}, type{type(observation)}, {observation}")
        #print()
        #exit(0)
        return observation, reward, done, {}

    def _get_observation(self):
        return self.robot.get_obs()
    
    def _act(self, action):
        self.robot.take_action(action)
        self.task.update(self)
        
    def _build_action_space(self):
        # All limits defined according to urdf file
        joint_limits = self.robot._joint_limits
        print(f"Motor Control mode {self.robot.motor_control_mode}")
        if self.robot.motor_control_mode == "Torque":
            OBSERVED_TORQUE_LIMIT = 5.7
            high = [OBSERVED_TORQUE_LIMIT]*self.robot.num_joints
            low = - high
            action_space = spaces.Box(low, high, dtype=np.float32)
        elif self.robot.motor_control_mode == "Position":
            high = joint_limits[:, 1]
            low = joint_limits[:, 0]
            # print(high)
            # print(low)
            action_space = spaces.Box(low, high, dtype=np.float32)
        elif self.robot.motor_control_mode == "Velocity":
            high = joint_limits[:, 3]
            low = - high
            action_space = spaces.Box(low, high, dtype=np.float32)
        else:
            raise ValueError
        return action_space

    def _build_observation_space(self,):

        high = []
        low = []

        trunk_pos_limit_high = [100] * 3
        trunk_pos_limit_low = [-100] * 3
        if self.robot._incl_current_pos:
            high.extend(trunk_pos_limit_high)
            low.extend(trunk_pos_limit_low)
        
        # trunk_ori_limit_high = [2*np.pi] * 3    # in euler angles
        # trunk_ori_limit_low = - trunk_ori_limit_high

        trunk_ori_limit_high = [1] * 4
        trunk_ori_limit_low = [-1] * 4
        high.extend(trunk_ori_limit_high)
        low.extend(trunk_ori_limit_low)

        motor_angle_limit_high  = [np.pi] * 12
        motor_angle_limit_low  = [-np.pi] * 12
        high.extend(motor_angle_limit_high)
        low.extend(motor_angle_limit_low)

        trunk_lin_vel_high = [10] * 3
        trunk_lin_vel_low = [-10] * 3
        high.extend(trunk_lin_vel_high)
        low.extend(trunk_lin_vel_low)

        trunk_ang_vel_high = [200 * np.pi] * 3
        trunk_ang_vel_low = [-200 * np.pi] * 3
        high.extend(trunk_ang_vel_high)
        low.extend(trunk_ang_vel_low)

        # link_ang_vel_high = [200] * 3 * 12
        # link_ang_vel_low = [-200] * 3 * 12
        # high.extend(link_ang_vel_high)
        # low.extend(link_ang_vel_low)

        motor_vel_low = [-100] * 12
        motor_vel_high = [100] * 12
        high.extend(motor_vel_high)
        low.extend(motor_vel_low)

        high=np.array(high)
        low=np.array(low)
        observation_space = spaces.Box(low, high, dtype=np.float64)
        return observation_space

    @property
    def last_action(self):
        return self._last_action