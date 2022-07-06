import pybullet as pb
import pybullet_data
import numpy as np
from tabulate import tabulate

import os
import time
import logging
import argparse
from collections import OrderedDict

class A1Env():

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

    def __init__(self, args, log) -> None:

         # set mode to pb.DIRECT ,providing the fastest, non-visual connection
        self.client = pb.connect(pb.GUI)

        pb.setGravity(0, 0, -9.8, physicsClientId=self.client)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.planeId = pb.loadURDF("plane.urdf")
        self.robotid = pb.loadURDF("a1/a1.urdf",basePosition=[0,0,0.5])

        self.init_pos, self.int_ori = pb.getBasePositionAndOrientation(self.robotid)
        self.joint_id2name, self.joint_ids = self._select_joints()
        self.link_ids = self._select_links()

        self.args = args
        self.log = log

        self._log_general()
        print("-"*50)
        print(f"init_pos {self.init_pos}, init_ori {self.int_ori}")
        print(f"robotid {self.robotid}")
        print(f"control joint ids {self.joint_ids}")
        print(f"link ids {self.link_ids}")
        self.disable_motor()

    def step(self, action):
        pos, ori = pb.getBasePositionAndOrientation(self.robotid)
        print(f"pos type{type(pos)}, length{len(pos)}")
        print(f"ori type{type(ori)}, length{len(ori)}")
        
        self.take_action(action)
        observation = self.get_obs()
        print(f"observation , len {len(observation)}, type{type(observation)}")
        # exit(0)
        pb.stepSimulation()
        # exit(0)

    def train(self):
        print()
        print("Training...")
        # Action space from urdf file
        low = np.zeros(12)
        high = [20,55,55,20,55,55,20,55,55,20,55,55]

        for _ in range(300):
            action = np.random.uniform(low, high)
            print("Action is ", action.tolist())
            self.step(action.tolist())
            time.sleep(0.02)
        print("-"*50)

    def take_action(self, action):
        pb.setJointMotorControlArray(self.robotid, 
                                    self.joint_ids,
                                    pb.TORQUE_CONTROL,
                                    forces=action)

    def get_obs(self, incl_current_pos=True):
        
        observation = []

        joint_states =  self.get_joint_states()
        link_states = self.get_link_states()

        trunk_pos, trunk_ori = pb.getBasePositionAndOrientation(self.robotid) # tuple(3), tuple(4)

        if incl_current_pos:
            observation.extend(trunk_pos)
        observation.extend(trunk_ori)
        # print(f"Observation trunk pos + ori, len {len(observation)}")
        
        for joint in joint_states:
            joint_angle = joint[0]
            observation.append(joint_angle)
        
        # print(f"Observation +joint angle, len {len(observation)}")

        trunk_vel, trunk_ang_vel = pb.getBaseVelocity(self.robotid) # tuple(3), tuple(3)
        observation.extend(trunk_vel)
        observation.extend(trunk_ang_vel)

        # print(f"Observation +joint angle, len {len(observation)}")

        for link_id, link in enumerate(link_states):
            link_ang_vel = link[-1] # tuple(3)
            observation.extend(link_ang_vel)
        
        return observation

    def get_link_states(self):
        
        link_states = pb.getLinkStates(self.robotid, self.link_ids, computeLinkVelocity=1)
        print(f"link states type {type(link_states)}, len {len(link_states)}")
        to_log(self.log, "Link States", link_states,header="link_states")

        return link_states
    
    def get_joint_states(self):
        joint_states = pb.getJointStates(self.robotid, self.joint_ids)
        to_log(self.log, "Joint States", joint_states, header="joint_states")

        return joint_states

    def _select_joints(self, incl_fixed=False):
        FIXED_JOINT_TYPE = 4

        number_of_joints = pb.getNumJoints(self.robotid)
        joint_id2name = OrderedDict()

        for joint_number in range(number_of_joints):
            joint_info = pb.getJointInfo(self.robotid, joint_number)

            if not incl_fixed and joint_info[2] == FIXED_JOINT_TYPE:
                continue
            else:
                joint_id2name[joint_number] = joint_info[1]

        return joint_id2name, joint_id2name.keys()
    
    def _select_links(self, only_mesh=True):
        MESH_GEOMENTRY_TYPE = 5

        shape_info = pb.getVisualShapeData(self.robotid)
        link_ids = []

        if not only_mesh:
            link_ids = list(range(len(shape_info)))
        else:
            for link in shape_info:
                if link[2] == MESH_GEOMENTRY_TYPE and link[1] != -1:
                    link_ids.append(link[1])
        
        return link_ids

    def _log_general(self):
        
        joint_infos = [pb.getJointInfo(self.robotid, joint_number) 
                    for joint_number in range(pb.getNumJoints(self.robotid))]
        to_log(self.log, "Joint Informations", joint_infos, header="joint_infos")

        shape_info = pb.getVisualShapeData(self.robotid)
        to_log(self.log, "Visual Shape info", shape_info,header="shape_infos")
    
    def disable_motor(self):
        pb.setJointMotorControlArray(self.robotid,
                                self.joint_ids,
                                controlMode=pb.VELOCITY_CONTROL,
                                forces=np.zeros(len(self.joint_ids)).tolist())

def to_log(logger, desc="", msg="", level="DEBUG", is_tabular=True, header="firstrow"):

    headers = {
        "joint_infos": ["jointIndex","jointName","jointType", "qIndex", "uIndex", "flags", 
                        "jointDamping", "jointFriction", "jointLowerLimit", "jointUpperLimit", 
                        "jointMaxForce", "jointMaxVelocity", "linkName", "jointAxis", "parentFramePos", 
                        "parentFrameOrn", "parentIndex"
                        ],
        "joint_states": ["jointPosition", "jointVelocity", "jointReactionForces", 
                        "appliedJointMotorTorque"
                        ],
        "shape_infos" : ["objectUniqueId", "linkIndex", "visualGeometryType", "dimensions", 
                        "meshAssetFileName", "localVisualFrame position", "localVisualFrame orientation",
                        "rgbaColor", "textureUniqueId"
                        ],
        "link_states" : ["linkWorldPosition", "linkWorldOrientation", "localInertialFramePosition", 
                        "localInertialFrameOrientation", "worldLinkFramePosition", "worldLinkFrameOrientation",
                        "worldLinkLinearVelocity", "worldLinkAngularVelocity",
                        ]
    }

    if is_tabular:
        logger.debug(f"{desc} : \n{tabulate(msg, headers=headers[header])}")
    else:
        logger.debug(f"{desc} : {msg}")

def parse_arguements():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--log', "-l", default="debug", type=str, help='set log level')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')


    args = parser.parse_args()
    args.log = args.log.upper()
    return args

if __name__ == "__main__":
    args = parse_arguements()

    logging.basicConfig(filename="a1_env.log", filemode='w', level=os.environ.get("LOGLEVEL", args.log))
    log = logging.getLogger(__name__)
    
    env = A1Env(args, log)
    env.train()
