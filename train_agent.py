import gymnasium as gym
import logging
import matplotlib.pyplot as plt
import numpy as np
import yaml
from agent import Agent


def main() -> None:
    # load config
    with open('.config.yml', 'r') as cf:
        config = yaml.safe_load(cf)
    # setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(config['logger']['train_log_path'])
    fh.setLevel(logging.INFO)
    fh.setFormatter(logger_formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logger_formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    # Taxi environment
    env = gym.make('Taxi-v3')
    # agent parameters
    fname = config['agent']['qf_save_to']
    # parameters for SARSA and Q-Learning
    ma_p = config['taxi_env']['rewards_ma_period']
    episode_n = config['taxi_env']['episode_n']
    tr_len = config['taxi_env']['tr_len']
    g = config['taxi_env']['gamma']
    a = config['taxi_env']['alpha']
    # learn agent with Q-Learning algorithm (default)
    agent = Agent(env, logger)
    tr = agent.learn(episode_n = episode_n, t_len = tr_len, gamma = g, alpha = a)
    ar = agent.get_rewards_ma(n = ma_p)
    # draw rewards plot
    avx = np.arange(ma_p - 1, episode_n)
    plt.figure(figsize=(12.8, 8))
    plt.plot(tr, color='blue', label='Rewards')
    if ar is not None:
        plt.plot(avx, ar[ma_p-1:], color='red', label=f'Rewards Moving Average (period={ma_p})')
    plt.title('Agent Learning Dynamics for Taxi-v3 Environment (Q-Learning)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.show()
    # Save agent Q-function
    agent.save(fname)


if __name__ == '__main__':
    main()

