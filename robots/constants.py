import numpy as np

URDF_FILEPATH = "a1/a1.urdf"
NUM_MOTORS = 12

INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.4]

JOINT_DIRECTIONS = np.ones(12)

HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0.0
KNEE_JOINT_OFFSET = 0.0
NUM_LEGS = 4


JOINT_OFFSETS = np.array(
    [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * NUM_LEGS)

SENSOR_NOISE_STDDEV = (0.0, 0.0, 0.0, 0.0, 0.0)

