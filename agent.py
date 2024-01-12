import gymnasium as gym
from logging import Logger
import numpy as np
from typing import Callable


def get_egreedy_action(
    q_values: np.ndarray,
    e: float,
    action_n: int
) -> int:
    # if e==0 return greedy action (maximum q-value)
    max_action_idx = np.argmax(q_values).item()
    if e == 0:
        return max_action_idx
    # else return epsilon-greedy action
    policy = np.ones(action_n) * (e / action_n)
    policy[max_action_idx] += 1 - e
    return np.random.choice(np.arange(action_n), p=policy).item()

LearningFunc = Callable[[gym.Env, np.ndarray, int, int, float, float], np.ndarray]

def SARSA(
    env: gym.Env,
    Q: np.ndarray,
    episode_n: int,
    t_len: int = 500,
    gamma: float = 0.99,
    alpha: float = 0.1
) -> np.ndarray:
    total_rewards = np.zeros(episode_n)
    action_n = env.action_space.n.item()
    for i in range(episode_n):
        e = 1.0 / (i + 1)
        s = env.reset()[0]
        a = get_egreedy_action(Q[s], e, action_n)
        for j in range(t_len):
            ns, r, term, trun, _ = env.step(a)
            na = get_egreedy_action(Q[ns], e, action_n)
            Q[s,a] += alpha * (r + gamma * Q[ns,na] - Q[s,a])
            total_rewards[i] += r
            s = ns
            a = na
            if term or trun:
                break
    return total_rewards

def Q_learning(
    env: gym.Env,
    Q: np.ndarray,
    episode_n: int,
    t_len: int = 500,
    gamma: float = 0.99,
    alpha: float = 0.1
) -> np.ndarray:
    total_rewards = np.zeros(episode_n)
    action_n = env.action_space.n.item()
    for i in range(episode_n):
        e = 1.0 / (i + 1)
        s = env.reset()[0]
        for j in range(t_len):
            a = get_egreedy_action(Q[s], e, action_n)
            ns, r, term, trun, _ = env.step(a)
            Q[s,a] += alpha * (r + gamma * Q[ns].max() - Q[s,a])
            total_rewards[i] += r
            s = ns
            if term or trun:
                break
    return total_rewards

class LearningAlgorithm:
    learning_func: dict[str, LearningFunc] = {}

    def __init__(self):
        self.register('sarsa', SARSA)
        self.register('qlearning', Q_learning)

    def register(self, algorithm: str, f: LearningFunc) -> None:
        self.learning_func[algorithm] = f

    def get_learning_function(self, algorithm: str) -> LearningFunc:
        if algorithm not in self.learning_func.keys():
            raise NotImplementedError
        return self.learning_func[algorithm]

class Agent:
    def __init__(self, env: gym.Env, logger: Logger) -> None:
        # init the agent
        self.env = env
        self.log = logger
        self.st_n = env.observation_space.n.item()
        self.ac_n = env.action_space.n.item()
        self.learning_algorithm = LearningAlgorithm()
        self.reset()

    def learn(
            self,
            algorithm: str = 'qlearning',
            episode_n: int = 1000,
            t_len: int = 500,
            gamma: float = 0.99,
            alpha: float = 0.1
    ) -> np.ndarray:
        """Learn the agent on :attr:`env` environment with ``algorithm`` learning algorithm (Q-learning (``algorithm='qlearning'``)
        and SARSA (``algorithm='sarsa'``) algorithms are implemented, default is Q-learning"""
        # get learning function
        f = self.learning_algorithm.get_learning_function(algorithm)
        # learn the agent
        self.log.info(f'starting agent learning with {algorithm} algorithm')
        self.total_rewards = f(self.env, self.Q, episode_n = episode_n, t_len = t_len, gamma = gamma, alpha = alpha)
        # set trained flag to true
        self._is_trained = True
        # average reward for last 100 episodes
        rmean = np.mean(self.total_rewards[episode_n-100:]).item()
        self.log.info(f'agent learning done, average reward for the last 100 episodes is {rmean}')
        return self.total_rewards

    def act(self, episode_n: int = 100) -> float | None:
        # if agent is not trained return None
        if not self._is_trained:
            return None
        # init rewards array
        rewards = np.zeros(episode_n)
        Q = self.Q
        # run the agent for episode_n episodes
        for i in range(episode_n):
            # reset environment
            s = self.env.reset()[0]
            # go through episode
            while (True):
                # get greedy action (e = 0)
                a = get_egreedy_action(Q[s], 0, self.ac_n)
                # do step
                s, r, term, trun, _ = self.env.step(a)
                # append episode reward
                rewards[i] += r
                if term or trun:
                    break
        # return average reward
        return np.mean(rewards).item()

    def reset(self) -> None:
        # reset the agent
        self.total_rewards = None
        self.Q = np.zeros((self.st_n, self.ac_n))
        self._is_trained = False

    def is_trained(self) -> bool:
        return self._is_trained

    def get_rewards_ma(self, n: int = 100) -> np.ndarray | None:
        # calculate the moving average of rewards with period n
        if self.total_rewards is None:
            return None
        size = self.total_rewards.shape[0]
        rewards_ma = np.zeros(size)
        if n <= size:
            for i in range(n,size+1):
                rewards_ma[i-1] = np.mean(self.total_rewards[i-n:i])
        return rewards_ma

    def save(self, filename: str) -> None:
        # save agent's Q-function to file
        np.save(filename, self.Q)
        self.log.info(f'save agent\'s Q-function to {filename}.npy')

    def load(self, filename: str) -> bool:
        # load agent's Q-function from file
        try:
            self.log.info(f'loading agent\'s Q-function from file {filename}.npy')
            self.Q = np.load(f'{filename}.npy')
        except IOError:
            self.log.error(f'{filename}.npy not found or corrupted')
            return False
        self._is_trained = True
        return self._is_trained

