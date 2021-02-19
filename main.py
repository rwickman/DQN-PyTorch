import argparse, gym

from dqn import DQNActor
from trainer import Trainer

def main(args):
    if args.env == "CP":
        env = gym.make('CartPole-v0')
    else:
        env = gym.make('MountainCar-v0')

    dqn_actor = DQNActor(args)
    trainer = Trainer(args, dqn_actor, env)
    trainer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=int, default=10000,
            help="Maximum size of replay memory.")
    parser.add_argument("--epsilon", type=float, default=1.0,
            help="Initial epsilon value used for epsilon-greedy.")
    parser.add_argument("--epsilon_decay", type=int, default=1000,
            help="Epsilon decay step.")
    parser.add_argument("--min_epsilon", type=float, default=0.05,
            help="Minimum epsilon value used for epsilon-greedy.")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="Parameter learning rate.")
    parser.add_argument("--gamma", type=float, default=0.90,
            help="Gamma value used for future reward discount.")
    parser.add_argument("--batch_size", type=int, default=32,
            help="Batch size of samples.")
    parser.add_argument("--episodes", type=int, default=100000,
            help="Number of episodes to train the model.")
    parser.add_argument("--target_update_step", type=int, default=200,
            help="Number of steps to wait before updating policy.")
    parser.add_argument("--load_model", action="store_true",
            help="Load model paramters.")
    parser.add_argument("--model", default="./model",
            help="Path to model parameters.")
    parser.add_argument("--env", default="CP",
            help="Environment to train (CP for Cart Pole or MC for Mountain Car).")
    


    main(parser.parse_args())