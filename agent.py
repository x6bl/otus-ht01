import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import yaml
from typing import Callable


def get_egreedy_action(
    q_values: np.ndarray,
    e: float,
    action_n: int
) -> int:
    max_action_idx = np.argmax(q_values).item()
    if e == 0:
        return max_action_idx
    policy = np.ones(action_n) * (e / action_n)
    policy[max_action_idx] += 1 - e
    return np.random.choice(np.arange(action_n), p=policy).item()

type LearningFunc = Callable[[gym.Env, np.ndarray, int, int, float, float], np.ndarray]

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

def avg_reward(
    r: np.ndarray,
    n: int = 100
) -> np.ndarray:
    size = r.shape[0]
    res = np.zeros(size)
    if n <= size:
        for i in range(n,size+1):
            res[i-1] = np.average(r[i-n:i])
    return res

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
    def __init__(self, env: gym.Env) -> None:
        self.env = env
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
        f = self.learning_algorithm.get_learning_function(algorithm)
        self.total_rewards = f(self.env, self.Q, episode_n = episode_n, t_len = t_len, gamma = gamma, alpha = alpha)
        self._is_trained = True
        return self.total_rewards

    def act(self) -> None:
        pass

    def reset(self) -> None:
        self.total_rewards = None
        self.Q = np.zeros((self.st_n, self.ac_n))
        self._is_trained = False

    def is_trained(self) -> bool:
        return self._is_trained

