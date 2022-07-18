import argparse
import logging
import os
import inspect
from typing import Union

from envs.a1_gym_env import A1GymEnv
from learning import utils
from learning.trainer import Trainer
from tasks.walk_along_x import WalkAlongX


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
SAVE_DIR = os.path.join(currentdir, "agents")

def build_env(args, log, enable_rendering=False):
    """ Builds the gym environment needed for RL

    Args:
        randomise_terrain: Whether to randomize terrain or not
        motor_control_mode: Position, Torque or Hybrid
        enable_rendering: Whether to configure pybullet in GUI mode or DIRECT mode
        robot_on_rack: Whether robot is on rack or not
    """
    # gym_config = LocomotionGymConfig()
    # gym_config.enable_rendering = enable_rendering
    # gym_config.motor_control_mode = MOTOR_CONTROL_MODE_MAP[args.motor_control_mode]
    # gym_config.reset_time = 2
    # gym_config.num_action_repeat = 10
    # gym_config.enable_action_interpolation = True
    # gym_config.enable_action_filter = True
    # gym_config.enable_clip_motor_commands = False
    # gym_config.robot_on_rack = False
    # gym_config.randomise_terrain = args.randomise_terrain

    if args.visualize:
        enable_rendering = not(enable_rendering)

    task = WalkAlongX()

    env = A1GymEnv(task=task, is_render=enable_rendering, args=args, log=log)

    return env
def parse_arguements():

    boolparse = lambda x : bool(x)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--log_lvl', "-ll", dest="log", default="debug", type=str, help='set log level')
    parser.add_argument('--mode', "-m", dest="mode", default="test", choices=["train", "test"], type=str, help='to set to training or testing mode')
    parser.add_argument('--max_episode_steps', "-mes", dest="max_episode_steps", default=1000, type=int, help='maximum steps per episode')
    parser.add_argument('--visualize', "-v", dest="visualize", action="store_true", help='To flip rendering behaviour')
    parser.add_argument("--randomise_terrain", "-rt", dest="randomise_terrain", default=False, type=bool, help="to setup a randommized terrain")
    parser.add_argument("--motor_control_mode", "-mcm", dest="motor_control_mode",  default="position", choices=["position", "torque", "hybrid"], type=str, help="to set motor control mode")

    parser.add_argument('--author', "-au", dest="author", default="rpanackal", type=str, help='name of author')
    parser.add_argument('--exp_suffix', "-s", dest="exp_suffix", default="", type=str, help='appends to experiment name')
    parser.add_argument('--total_timesteps', "-tts", dest="total_timesteps", default=int(1e5), type=int, help='total number of training steps')
    
    parser.add_argument('--total_num_eps', "-tne", dest="total_num_eps", default=20, type=int, help='total number of test episodes')
    parser.add_argument('--load_exp_name', "-l", dest="load_exp_name", default="sac_rpanackal_tns100000", type=str, help='name of experiment to be tested')


    args = parser.parse_args()
    args.log = args.log.upper()
    args.motor_control_mode = args.motor_control_mode.capitalize()
    return args

def main():

    args = parse_arguements()
    
    logging.basicConfig(filename="a1_env.log", filemode='w', level=os.environ.get("LOGLEVEL", args.log))
    log = logging.getLogger(__name__)
    
    # Training
    if args.mode == "train":
        env = build_env(args, log, enable_rendering=False)

        # Train the agent
        local_trainer = Trainer(env, "SAC", args)
        _, hyperparameters = utils.read_hyperparameters("SAC", 1, {"learning_starts": 2000})
        model = local_trainer.train(hyperparameters)

        # Save the model after training
        local_trainer.save_model(SAVE_DIR)

    # Testing
    if args.mode == "test":
        test_env = build_env(args, log, enable_rendering=True)
        Trainer(test_env, "SAC", args).test(SAVE_DIR)

if __name__ == "__main__":
    main()
