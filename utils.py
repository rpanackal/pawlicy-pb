from tabulate import tabulate
import copy
import math

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
                        ],
        "link_angular_velocities" : ["dR", "dP", "dY"]
    }

    if is_tabular:
        if header in {"joint_infos", "shape_infos"}:
            logger.info(f"{desc} : \n{tabulate(msg, headers=headers[header])}")
        else:
            logger.debug(f"{desc} : \n{tabulate(msg, headers=headers[header])}")
    else:
        logger.debug(f"{desc} : {msg}")


def map_to_minus_pi_to_pi(angle):
    """Maps a list of angles to [-pi, pi].

    Args:
      angles: A list of angles in rad.

    Returns:
      A list of angle mapped to [-pi, pi].
    """
    mapped_angle = math.fmod(angle, 2 * math.pi)
    if mapped_angle >= math.pi:
        mapped_angle -= 2 * math.pi
    elif mapped_angle < -math.pi:
        mapped_angle += 2 * math.pi
    return mapped_angle

def MapToMinusPiToPi(angles):
    """Maps a list of angles to [-pi, pi].

    Args:
      angles: A list of angles in rad.

    Returns:
      A list of angle mapped to [-pi, pi].
    """
    mapped_angles = copy.deepcopy(angles)
    for i in range(len(angles)):
        mapped_angles[i] = math.fmod(angles[i], 2 * math.pi)
        if mapped_angles[i] >= math.pi:
            mapped_angles[i] -= 2 * math.pi
        elif mapped_angles[i] < -math.pi:
            mapped_angles[i] += 2 * math.pi
    return mapped_angles