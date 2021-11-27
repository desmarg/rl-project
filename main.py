import argparse

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed


def run(args):
    env = rlcard.make(args["env"], config={"seed": 42})
    num_episodes = 1

    set_seed(42)

    agent = RandomAgent(num_actions=env.num_actions)
    env.set_agents([agent for _ in range(env.num_players)])

    for episode in range(num_episodes):

        trajectories, player_wins = env.run(is_training=False)
        print("\nEpisode {}".format(episode))
        print(trajectories)


if __name__ == '__main__':
    args = {
        "env": "leduc-holdem"
    }

    run(args)