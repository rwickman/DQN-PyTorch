import argparse, gym

from dqn import DQNActor
from trainer import Trainer

def main(args):
    env = gym.make(args.env)
    dqn_actor = DQNActor(args,  env.action_space.n, env.observation_space.shape[0])
    trainer = Trainer(args, dqn_actor, env)
    trainer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem_cap", type=int, default=10000,
        help="Maximum size of replay memory.")
    parser.add_argument("--epsilon", type=float, default=1.0,
        help="Initial epsilon value used for epsilon-greedy.")
    parser.add_argument("--epsilon_decay", type=int, default=2048,
        help="Epsilon decay step.")
    parser.add_argument("--min_epsilon", type=float, default=0.05,
        help="Minimum epsilon value used for epsilon-greedy.")
    parser.add_argument("--lr", type=float, default=3e-4,
        help="Parameter learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="Gamma value used for future reward discount.")
    parser.add_argument("--batch_size", type=int, default=32,
        help="Batch size of samples.")
    parser.add_argument("--episodes", type=int, default=100000,
        help="Number of episodes to train the model.")
    parser.add_argument("--target_update_step", type=int, default=128,
        help="Number of steps to wait before updating policy.")
    parser.add_argument("--load", action="store_true",
        help="Load model paramters.")
    parser.add_argument("--model", default="./model",
        help="Path to model parameters.")
    parser.add_argument("--env", default="Pong-v0",
        help="Environment to train (e.g., CartPole-v0 or MountainCar-v0).")
    parser.add_argument("--img_dim", type=int, default=84,
        help="The width (and height) of input image.")
    parser.add_argument("--n_frames", type=int, default=4,
        help="Number of input frames.")
    parser.add_argument("--eps", type=float, default=1e-6,
        help="Epsilon used for proportional priority.")
    parser.add_argument("--per_alpha", type=float, default=0.9,
        help="Alpha used for proportional priority.")
    parser.add_argument("--per_beta", type=float, default=0.4,
        help="Beta used for proportional priority.")
    parser.add_argument("--grad_norm", type=float, default=1.0,
        help="Max gradient norm.")
    parser.add_argument("--use_grad_norm", action="store_true",
        help="Clip the gradient.")
    parser.add_argument("--save_dir", default="models",
        help="Save directory.")
    parser.add_argument("--update_steps", type=int, default=8,
        help="How many episode timesteps elapsed before updating.")

    



    main(parser.parse_args())
