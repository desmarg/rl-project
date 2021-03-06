import os
import argparse

import numpy as np
import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger

import matplotlib.pyplot as plt


def train(args):
    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    # Initialize the agent and use random agents as opponents
    if args.algorithm == 'dqn':
        # from dqn_agent_modified import DQNAgentModified
        # agent = DQNAgentModified(num_actions=env.num_actions,
        #                          state_shape=env.state_shape[0],
        #                          mlp_layers=[64, 128, 64],
        #                          device=device)
        from rlcard.agents import DQNAgent
        agent = DQNAgent(num_actions=env.num_actions,
                                 state_shape=env.state_shape[0],
                                 mlp_layers=[64, 64],
                                 device=device)
    elif args.algorithm == 'nfsp':
        from rlcard.agents import NFSPAgent
        agent = NFSPAgent(num_actions=env.num_actions,
                          state_shape=env.state_shape[0],
                          hidden_layers_sizes=[64, 64],
                          q_mlp_layers=[64, 64],
                          device=device)
    elif args.algorithm == 'sarsa':
        from sarsa_agent import SarsaAgent
        agent = SarsaAgent(num_actions=env.num_actions,
                                 state_shape=env.state_shape[0],
                                 mlp_layers=[64, 64],
                                 device=device)

    competitor = RandomAgent(num_actions=env.num_actions)

    agents = [agent, competitor]
    # for _ in range(env.num_players):
    #     agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    # Start training
    with Logger(args.log_dir) as logger:

        rewards = []
        episodes = []

        for episode in range(args.num_episodes):

            if args.algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)



            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for ts in trajectories[0]:
                (state, action, reward, next_state, done) = ts
                next_action = agent.step(next_state)
                ts = (state, action, reward, next_state, next_action, done)
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                if episode % 100 == 0:
                    print("\nEpisode: {}".format(episode))
                tourney = tournament(env, args.num_games)
                rewards.append(tourney[0])
                episodes.append(episode)
                # logger.log_performance(env.timestep, tourney[0])
                # logger.log_performance(env.timestep, tourney[1])

        # Plot the learning curve
        # logger.plot(args.algorithm)

        # with open("modified_dqn_vs_random.txt", "w") as file:
        #     for element in rewards:
        #         file.write(element + "\n")

        plt.plot(episodes, rewards, label="Sarsa Agent", linewidth=0.6)
        plt.plot(episodes, np.multiply(-1, np.array(rewards)), label="Random Agent", linewidth=0.6)

        plt.xlabel("Episode number")
        plt.ylabel("Average payoff (reward) over 2000 games")
        plt.legend()
        plt.title("Sarsa Agent vs. Random Agent")
        plt.show()

    # Save model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN example in RLCard")
    parser.add_argument('--env', type=str, default='leduc-holdem')
    parser.add_argument('--algorithm', type=str, default='sarsa', choices=['dqn', 'nfsp', 'sarsa'])
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=5000)
    parser.add_argument('--num_games', type=int, default=2000)
    parser.add_argument('--evaluate_every', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default='experiments/leduc_holdem_dqn_result/')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)