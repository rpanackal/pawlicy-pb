import os

from gym.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.wrappers.normalize_actions_wrapper import NormalizeActionWrapper

class Trainer:
    """
    The trainer class provides some basic methods to train an agent using different algorithms
    available in stable_baselines3

    Args:
        env: The gym environment to train on.
        algorithm: The algorithm to use.
        max_episode_steps: The no. of steps per episode
    """
    def __init__(self, env, algorithm, args):
        self._env = env
        self._algorithm = algorithm
        self._max_episode_steps = args.max_episode_steps
        self._total_timesteps = args.total_timesteps

        self._args = args 
        self.setup_env(self._max_episode_steps)

    def train(self, hyperparameters, total_timesteps=None, eval_env=None):
        """
        Trains an agent to use the environment to maximise the rewards while performing
        a specific task. This will tried out with multiple other algorithms later for
        benchmarking purposes.

        Args:
            env: The gym environment to train the agent on.
            algorithm: The algorithm to use
            hyperparameters: The hyperparameters to use
            n_timesteps: The number of timesteps to train
            eval_env: The gym environment used for evaluation.
        """
        if total_timesteps is not None:
            print("I was here")
            self._total_timesteps = total_timesteps

        # Check which algorithm to use
        if self._algorithm == "SAC":
            self._model = SAC(env=self._env, verbose=1, **hyperparameters)

        # Train the model (check if evaluation is needed)
        if eval_env is not None:
            self._model.learn(self._total_timesteps, log_interval=20, eval_env=eval_env, eval_freq=100)
        else:
            self._model.learn(self._total_timesteps, log_interval=20)

        # Return the trained model
        return self._model

    def save_model(self, save_path):
        """
        Saves the trained model. Also saves the replay buffer

        Args:
            model: The trained agent
        """
        if save_path is None:
            raise ValueError("No path specified to save the trained model.")
        else:
            exp_name = f"{self._algorithm}_{self._args.author}_tns{self._total_timesteps}"
            if self._args.exp_suffix:
                exp_name = f"{exp_name}_{self._args.exp_suffix}"
            
            # Create the directory to save the models in.
            os.makedirs(save_path, exist_ok=True)
            self._model.save(os.path.join(save_path, exp_name, "model"))
            self._model.save_replay_buffer(os.path.join(save_path, exp_name, "replay_buffer"))
            #self._model.save(os.path.join(save_path, f"{self._algorithm}_emrald"))
            #self._model.save_replay_buffer(os.path.join(save_path, f"{self._algorithm}_replay_buffer_emrald"))

    def test(self, model_path=None):
        """
        Tests the agent

        Args:
            env: The gym environment to test the agent on.
        """
        if model_path is not None:
            # self._model = SAC.load(os.path.join(model_path, f"{self._algorithm}_emrald"))
            # self._model.load_replay_buffer(os.path.join(model_path, f"{self._algorithm}_replay_buffer_emrald"))
            self._model = SAC.load(os.path.join(model_path, self._args.load_exp_name, "model"))
            self._model.load_replay_buffer(os.path.join(model_path, self._args.load_exp_name, "replay_buffer"))

        for i in range(self._args.total_num_eps):
            done = False
            obs = self._env.reset()
            while not done:
                action, _states = self._model.predict(obs, deterministic=True)
                obs, reward, done, info = self._env.step(action)
                if done:
                    obs = self._env.reset()

    def setup_env(self, max_episode_steps):
        """
        Modifies the environment to suit to the needs of stable_baselines3.

        Args:
            max_episode_steps: The number of steps per episode
        """
        self._max_episode_steps = max_episode_steps
        # Normalize the action space
        self._env = NormalizeActionWrapper(self._env)
        # Set the number of steps for each episode
        self._env = TimeLimit(self._env, self._max_episode_steps)
        # To monitor training stats
        self._env = Monitor(self._env)
        check_env(self._env)
        # a simple vectorized wrapper
        self._env = DummyVecEnv([lambda: self._env])
        # Normalizes the observation space and rewards
        self._env = VecNormalize(self._env, norm_obs=True, norm_reward=True)

    @property
    def model(self):
        return self._model

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    @max_episode_steps.setter
    def max_episode_steps(self, value):
        self._max_episode_steps = value
