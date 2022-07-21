from collections import OrderedDict
from typing import Union

import numpy as np
#import pybullet as pb
from robots import constants
from utils import to_log, MapToMinusPiToPi

class A1(object):
    link_name2id = {
        "trunk" : -1,
        "imu_link" : 0,
        "FR_hip" : 1,
        "FR_upper_shoulder" : 2,
        "FR_upper" : 3,
        "FR_lower" : 4,
        "FR_toe" : 5,
        "FL_hip" : 6,
        "FL_upper_shoulder" : 7,
        "FL_upper" : 8,
        "FL_lower" : 9,
        "FL_toe" : 10,
        "RR_hip" : 11,
        "RR_upper_shoulder" : 12,
        "RR_upper" : 13,
        "RR_lower" : 14,
        "RR_toe" : 15,
        "RL_hip" : 16,
        "RL_upper_shoulder" : 17,
        "RL_upper" : 18,
        "RL_lower" : 19,
        "RL_toe" : 20,
    }

    def __init__(
        self,
        client,
        motor_control_mode,
        args, 
        log,
        self_collision_enabled : bool = False,
        on_rack : bool = False,
        incl_current_pos : bool = False,
        ) -> None:
        
        self.client = client
        self.args = args
        self.log = log
        self._self_collision_enabled = self_collision_enabled
        self._on_rack = on_rack
        self._incl_current_pos = incl_current_pos
        self.motor_control_mode = motor_control_mode
        
        self._observation_noise_stdev=constants.SENSOR_NOISE_STDDEV
        self._motor_direction = constants.JOINT_DIRECTIONS
        self._motor_offset = constants.JOINT_OFFSETS
        self.robotid = self._load_urdf()
        self.joint_id2name, self.joint_ids = self._select_joints()
        self.link_ids = self._select_links()
        self.num_joints = len(self.joint_ids)
        self.num_links = len(self.link_ids)
        self._joint_limits = self.get_joint_limits()

        self._log_general()
        self.disable_motor()

        print(f"robotid {self.robotid}")
        print(f"joint ids {self.joint_ids}")
        print(f"link ids {self.link_ids}")

        self.reset()        
        pass
    
    def _load_urdf(self):
        self.init_position = self._get_default_init_position() # base
        self.init_orientation = self._get_default_init_orientation() # base
        if self._self_collision_enabled:
            robotid = self.client.loadURDF(
                fileName=constants.URDF_FILEPATH, 
                basePosition=self.init_position, 
                baseOrientation=self.init_orientation, 
                flags=self.client.URDF_USE_SELF_COLLISION)
        else:
            robotid = self.client.loadURDF(
                fileName=constants.URDF_FILEPATH, 
                basePosition=self.init_position, 
                baseOrientation=self.init_orientation)
        
        return robotid

    def _get_default_init_position(self):
        if self._on_rack:
            return constants.INIT_RACK_POSITION
        else:
            return constants.INIT_POSITION
    
    def _get_default_init_orientation(self):
        # The Laikago URDF assumes the initial pose of heading towards z axis,
        # and belly towards y axis. The following transformation is to transform
        # the Laikago initial orientation to our commonly used orientation: heading
        # towards -x direction, and z axis is the up direction.
        init_orientation = self.client.getQuaternionFromEuler([0., 0., 0.])
        return init_orientation

    def _select_joints(self, incl_fixed=False):
        FIXED_JOINT_TYPE = 4

        number_of_joints = self.client.getNumJoints(self.robotid)
        joint_id2name = OrderedDict()

        for joint_number in range(number_of_joints):
            joint_info = self.client.getJointInfo(self.robotid, joint_number)

            if not incl_fixed and joint_info[2] == FIXED_JOINT_TYPE:
                continue
            else:
                joint_id2name[joint_number] = joint_info[1]

        return joint_id2name, joint_id2name.keys()
    
    def _select_links(self, only_mesh=True):
        MESH_GEOMENTRY_TYPE = 5

        shape_info = self.client.getVisualShapeData(self.robotid)
        link_ids = []

        if not only_mesh:
            link_ids = list(range(len(shape_info)))
        else:
            for link in shape_info:
                if link[2] == MESH_GEOMENTRY_TYPE and link[1] != -1:
                    link_ids.append(link[1])
        
        return link_ids

    def _set_initial_measurements(self):

        # Inverse of global base orientation
        base_raw_orientation = self.client.getBasePositionAndOrientation(self.robotid)[1]
        _, self._init_base_orientation_inv = self.client.invertTransform(position=[0, 0, 0], 
            orientation=base_raw_orientation)
        # print(f"_init_base_orientation_inv {type(self._init_base_orientation_inv)} {len(self._init_base_orientation_inv)}")
        # print(f"base_raw_orientation {type(base_raw_orientation)} {len(base_raw_orientation)}")

        # _init_link_orientations_inv = []
        # for link in self._link_states:
        #     orientation=link[1]
        #     orientations_inv = self.client.invertTransform(position=[0, 0, 0],
        #     orientation=orientation)
        #     _init_link_orientations_inv.append(orientations_inv)
        
        # self._init_link_orientations_inv = _init_link_orientations_inv


    def reset(self):
        self.client.resetBasePositionAndOrientation(self.robotid, self.init_position, self.init_orientation)
        self.client.resetBaseVelocity(self.robotid, [0, 0, 0], [0, 0, 0])

        self.update_states()
        self._set_initial_measurements()

        self._last_action = None

    def take_action(self, action):
        max_force = self._joint_limits[:, 2]
        self._last_action = action
        if self.motor_control_mode == "Torque":
            self.client.setJointMotorControlArray(self.robotid, 
                                    jointIndices=self.joint_ids,
                                    controlMode = self.client.TORQUE_CONTROL,
                                    forces=max_force)
        elif self.motor_control_mode == "Position":
            self.client.setJointMotorControlArray(self.robotid, 
                                    self.joint_ids,
                                    controlMode=self.client.POSITION_CONTROL,
                                    targetPositions=action,
                                    forces=max_force)
        elif self.motor_control_mode == "Velocity":
            controlMode = self.client.VELOCITY_CONTROL
            self.client.setJointMotorControlArray(self.robotid, 
                                    self.joint_ids,
                                    controlMode=self.client.VELOCITY_CONTROL,
                                    targetVelocities=action,
                                    forces=max_force)
        else:
            raise ValueError
        
        self.client.stepSimulation()
        self.update_states()

    def get_obs(self, incl_current_pos : Union[bool, None]=None):
        
        observation = []

        if incl_current_pos is not None:
            self._incl_current_pos = incl_current_pos

        if self._incl_current_pos:
            trunk_pos = self.GetBasePosition() # tuple(3)
            observation.extend(trunk_pos)
        
        trunk_ori = self.GetBaseOrientation() # tuple(4), quat
        observation.extend(trunk_ori)
        #print(f"Observation trunk pos + ori, len {len(observation)}")
        
        
        joint_angles = self.GetMotorAngles() # list(self.num_joints)
        observation.extend(joint_angles)
        #print(f"Observation +joint angles, len {len(observation)}")

        trunk_vel = self.GetBaseVelocity() # tuple(3)
        trunk_ang_vel = self.GetBaseRollPitchYawRate() # tuple(3)
        observation.extend(trunk_vel)
        observation.extend(trunk_ang_vel)
        #print(f"Observation +joint trunk lin and ang vel , len {len(observation)}")

        # for link_id, link in enumerate(link_states):
        #     link_ang_vel = link[-1] # tuple(3), worldLinkAngularVelocity
        #     observation.extend(link_ang_vel)

        # link_ang_vels = self.GetRawLinkRollPitchYawRates() # tuple(3)
        # observation.extend(link_ang_vels)

        motor_vels = self.GetMotorVelocities()
        observation.extend(motor_vels)

        #print(f"Observation + link ang vel , len {len(observation)}")
        return np.asarray(observation)

    def update_states(self):
        self._link_states = self.get_link_states()
        self._joint_states = self.get_joint_states()
        
    def get_link_states(self):
        link_states = self.client.getLinkStates(self.robotid, self.link_ids, computeLinkVelocity=1)
        # print(f"link states type {type(link_states)}, len {len(link_states)}")
        to_log(self.log, "Link States", link_states,header="link_states")

        return link_states
    
    def get_joint_states(self):
        joint_states = self.client.getJointStates(self.robotid, self.joint_ids)
        to_log(self.log, "Joint States", joint_states, header="joint_states")

        return joint_states
    
    def get_joint_limits(self):
        joint_limits = []
        
        for joint_id in self.joint_ids:
            joint_limits.append(self.client.getJointInfo(self.robotid, joint_id)[8:12]) # list(4)
        
        joint_limits = np.array(joint_limits)
        to_log(self.log, desc="Joint Limits", msg=joint_limits, is_tabular=False)
        return joint_limits


    def _log_general(self):
        
        joint_infos = [self.client.getJointInfo(self.robotid, joint_number) 
                    for joint_number in range(self.client.getNumJoints(self.robotid))]
        to_log(self.log, "Joint Informations", joint_infos, header="joint_infos")

        shape_info = self.client.getVisualShapeData(self.robotid)
        to_log(self.log, "Visual Shape info", shape_info,header="shape_infos")
    
    def disable_motor(self):
        if self.motor_control_mode == "Torque":
            self.client.setJointMotorControlArray(self.robotid,
                                    self.joint_ids,
                                    controlMode=self.client.VELOCITY_CONTROL,
                                    forces=np.zeros(len(self.joint_ids)).tolist())



    def GetBasePosition(self):
        """Get the position of quadruped's base.

        Returns:
        The position of quadruped's base.
        """
        base_position , _ = self.client.getBasePositionAndOrientation(self.robotid)
        return base_position
    
    def GetBaseVelocity(self):
        """Get the linear velocity of quadruped's base.

        Returns:
        The velocity of quadruped's base.
        """
        velocity = self.client.getBaseVelocity(self.robotid)[0]
        to_log(self.log, desc="Base Linear Velocity", msg=velocity, is_tabular=False)
        return velocity

    def GetBaseOrientation(self):
        """Get the orientation of quadruped's base, represented as quaternion.

        This function mimicks the noisy sensor reading and adds latency.
        Returns:
        The orientation of quadruped's base polluted by noise and latency.
        """
        true_orientation = self.GetTrueBaseOrientation()
        return self._AddSensorNoise(
            true_orientation,
            self._observation_noise_stdev[3])

    def GetTrueBaseOrientation(self):
        """Get the orientation of quadruped's base, represented as quaternion.
        
        Computes the relative orientation relative to the robot's initial_orientation.

        Returns:
        The orientation of quadruped's base.
        """
        _, base_orientation = self.client.getBasePositionAndOrientation(self.robotid)
        # _, _init_orientation_inv = pb.invertTransform(position=[0, 0, 0], 
        #     orientation=self._get_default_init_orientation())
        _, base_orientation = self.client.multiplyTransforms(
            positionA=[0, 0, 0],
            orientationA=base_orientation,
            positionB=[0, 0, 0],
            orientationB=self._init_base_orientation_inv)
        return base_orientation

    def GetBaseRollPitchYaw(self):
        """Get quadruped's base orientation in euler angle in the world frame.

        This function mimicks the noisy sensor reading and adds latency.
        Returns:
        A tuple (roll, pitch, yaw) of the base in world frame polluted by noise
        and latency.
        """
        roll_pitch_yaw = self.GetTrueBaseRollPitchYaw()
        return self._AddSensorNoise(
            roll_pitch_yaw, 
            self._observation_noise_stdev[3])

    def GetTrueBaseRollPitchYaw(self):
        """Get quadruped's base orientation in euler angle in the world frame.

        Returns:
        A tuple (roll, pitch, yaw) of the base in world frame.
        """
        orientation = self.GetTrueBaseOrientation()
        roll_pitch_yaw = self.client.getEulerFromQuaternion(orientation)
        return np.asarray(roll_pitch_yaw)

    def GetBaseRollPitchYawRate(self):
        """Get the rate of orientation change of the quadruped's base in euler angle.

        This function mimicks the noisy sensor reading and adds latency.
        Returns:
        rate of (roll, pitch, yaw) change of the quadruped's base polluted by noise
        and latency.
        """
        return self._AddSensorNoise(
            self.GetTrueBaseRollPitchYawRate(),
            self._observation_noise_stdev[4])

    def GetTrueBaseRollPitchYawRate(self):
        """Get the rate of orientation change of the quadruped's base in euler angle.

        Returns:
        rate of (roll, pitch, yaw) change of the quadruped's base.
        """
        angular_velocity = self.client.getBaseVelocity(self.robotid)[1]
        orientation = self.GetTrueBaseOrientation()
        angular_velocity_local = self.TransformAngularVelocityToLocalFrame(angular_velocity, orientation)
        to_log(self.log, "Base Angular Velocity", angular_velocity_local, is_tabular=False)
        return angular_velocity_local

    def TransformAngularVelocityToLocalFrame(self, angular_velocity, orientation):
        """Transform the angular velocity from world frame to robot's frame.

        Args:
        angular_velocity: Angular velocity of the robot in world frame.
        orientation: Orientation of the robot represented as a quaternion.

        Returns:
        angular velocity of based on the given orientation.
        """
        # Treat angular velocity as a position vector, then transform based on the
        # orientation given by dividing (or multiplying with inverse).
        # Get inverse quaternion assuming the vector is at 0,0,0 origin.
        _, orientation_inversed = self.client.invertTransform([0, 0, 0], orientation)
        # Transform the angular_velocity at neutral orientation using a neutral
        # translation and reverse of the given orientation.
        relative_velocity, _ = self.client.multiplyTransforms(
            [0, 0, 0], orientation_inversed, angular_velocity,
            self.client.getQuaternionFromEuler([0, 0, 0]))
        return np.asarray(relative_velocity)
    
    def GetMotorAngles(self):
        """Gets the motor angles.

        This function mimicks the noisy sensor reading and adds latency. The motor
        angles that are delayed, noise polluted, and mapped to [-pi, pi].

        Returns:
        Motor angles polluted by noise and latency, mapped to [-pi, pi].
        """
        motor_angles = self._AddSensorNoise(
            self.GetTrueMotorAngles(),
            self._observation_noise_stdev[0])
        return MapToMinusPiToPi(motor_angles)

    def GetTrueMotorAngles(self):
        """Gets the motor angles at the current moment, mapped to [-pi, pi].

        Returns:
        Motor angles, mapped to [-pi, pi].
        """

        motor_angles = []

        # for joint in self._joint_states:
        #     joint_angle = map_to_minus_pi_to_pi(joint[0]) # scalar, jointPosition
        #     motor_angles.append(joint_angle)
        motor_angles = [state[0] for state in self._joint_states]
        motor_angles = np.multiply(np.asarray(motor_angles) - np.asarray(self._motor_offset), self._motor_direction)
        #print(f"Motor ANgles {len(motor_angles)} {motor_angles}")
        return motor_angles
    
    def GetMotorVelocities(self):
        """Get the velocity of all eight motors.

        This function mimicks the noisy sensor reading and adds latency.
        Returns:
        Velocities of all eight motors polluted by noise and latency.
        """
        return self._AddSensorNoise(
            self.GetTrueMotorVelocities(),
            self._observation_noise_stdev[1])

    def GetTrueMotorVelocities(self):
        """Get the velocity of all eight motors.

        Returns:
        Velocities of all eight motors.
        """
        motor_velocities = [state[1] for state in self._joint_states]
        motor_velocities = np.multiply(motor_velocities, self._motor_direction)
        return motor_velocities
    
    # def GetTrueLinkOrientations(self):
    #     """Get the orientation of quadruped's links, represented as quaternion.
        
    #     Computes the relative orientation relative to the robot's initial_orientation.

    #     Returns:
    #     The orientations of quadruped's links.
    #     """
    #     link_orientations = []
    #     for index, link in enumerate(self._link_states):
    #         _init_link_orientations_inv = self._init_link_orientations_inv[index]
    #         #print(f"_init_link_orientations_inv {type(_init_link_orientations_inv)} {len(_init_link_orientations_inv)}")
    #         orientation = link[1]

    #         print(f"orientation {type(orientation)} {len(orientation)}")
    #         print(f"_init_link_orientations_inv {type(_init_link_orientations_inv)} {len(_init_link_orientations_inv)}")
    #         _, orientation = pb.multiplyTransforms(
    #             positionA=[0, 0, 0],
    #             orientationA=orientation,
    #             positionB=[0, 0, 0],
    #             orientationB=_init_link_orientations_inv)
    #         link_orientations.append(orientation)
    #     return link_orientations

    # def GetTrueLinkRollPitchYawRates(self):
    #     """Get the rate of orientation change of the quadruped's base in euler angle.

    #     Returns:
    #     rate of (roll, pitch, yaw) change of the quadruped's links.
    #     """
    #     ang_velocities = []
    #     link_orientations = self.GetTrueLinkOrientations()

    #     for index, link in enumerate(self._link_states):
    #         link_ang_vel = link[-1]#, link[-1] # tuple(4), tuple(3);
    #         link_ang_vel = self.TransformAngularVelocityToLocalFrame(link_ang_vel, link_orientations[index])
    #         ang_velocities.extend(link_ang_vel)
        
    #     return np.asarray(ang_velocities)
    
    def GetRawLinkRollPitchYawRates(self):

        ang_velocities = []
        for link in self._link_states:
            ang_velocities.extend(link[-1])
        
        to_log(self.log, "Link Raw Angular Velocity", np.asarray(ang_velocities).reshape(12, 3), header="link_angular_velocities")
        return np.asarray(ang_velocities)

    def _AddSensorNoise(self, sensor_values, noise_stdev):
        if noise_stdev <= 0:
            return sensor_values
        observation = sensor_values + np.random.normal(scale=noise_stdev, size=sensor_values.shape)
        return observation