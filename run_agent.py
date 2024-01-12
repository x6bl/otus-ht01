import gymnasium as gym
import logging
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
    fh = logging.FileHandler(config['logger']['run_log_path'])
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
    agent = Agent(env, logger)
    # reset agent
    agent.reset()
    # load agent Q-function
    if agent.load(fname):
        # run agent
        n = 100
        logger.info(f'run the agent for {n} episodes')
        reward = agent.act(episode_n = n)
        if reward is None:
            logger.warning('agent is not trained.')
        else:
            logger.info(f'average reward is {reward}')


if __name__ == '__main__':
    main()

