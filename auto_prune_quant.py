import os
import sys
import time
import torch
import logging
import numpy as np
import torch.backends.cudnn as cudnn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from RL.agent import Agent
from RL.agent import Memory
from RL.environment import RL_env
from parameter import parse_args
from prune_quant.quant_utils import load_fp32_model
from utils.net_info import get_num_hidden_layer
from utils.iscx2016vpn_training_utils import create_data_loaders_iscx2016vpn

logging.disable(30)
torch.backends.cudnn.deterministic = True


def search(env, agent, update_timestep, max_timesteps, max_episodes,
           log_interval=10, solved_reward=None, random_seed=None):

    env_name        = "prune_quant_env"
    running_reward  = 0
    avg_length      = 0
    time_step       = 0
    memory          = Memory()

    if random_seed:
        print(f"Random Seed: {random_seed}")
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)


    print("-" * 20, "Start searching for the pruning and quantization policies", "--" * 20)
    
    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        start_time = time.time()

        for t in range(max_timesteps):

            time_step += 1

            # Running policy_old:
            action = agent.select_action(state, memory)
            state, reward, done = env.step(action, t + 1)

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                print("-*" * 10, "start training the RL agent", "-*" * 10)
                agent.update(memory)
                memory.clear_memory()
                time_step = 0
                print("-*" * 10, "start search the pruning policies", "-*" * 10)

            running_reward += reward

            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if (i_episode % log_interval) != 0 and running_reward / (i_episode % log_interval) > (solved_reward):
            print("########## Solved! ##########")
            torch.save(agent.policy.state_dict(), 
                       os.path.join("RL", f"rl_solved_{env_name}.pth"))
            break

        # save every 50 episodes
        if i_episode % 50 == 0:
            torch.save(agent.policy.state_dict(),
                       os.path.join("RL", f"rl_{env_name}.pth"))
            torch.save(agent.policy.actor.state_dict(),
                       os.path.join("RL", f"rl_actor_{env_name}.pth"))
            torch.save(agent.policy.critic.state_dict(),
                       os.path.join("RL", f"rl_critic_{env_name}.pth"))

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print(f"Episode {i_episode} \t Avg length: {avg_length} \t Avg reward: {running_reward} \t Elapsed time {(time.time() - start_time):.3f}")
            running_reward = 0
            avg_length = 0



if __name__ == "__main__":

    # Set hyperparameters
    args            = parse_args()
    device          = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size      = args.data_bsize
    n_worker        = args.n_worker
    train_ratio     = 0.65
    val_ratio       = 0.15
    dataset         = os.path.join('data', 'datasets', 'iscx2016vpn-pytorch')


    # Load dataset
    if args.dataset == "iscx2016vpn":
        train_loader, val_loader, test_loader, classes = create_data_loaders_iscx2016vpn(dataset, batch_size, n_worker, train_ratio, val_ratio)
        example_inputs  = (next(iter(test_loader))[0]).to(device)
    else:
        raise NotImplementedError(f"Did not implement for this \"{args.dataset}\" dataset")


    # Load baseline model
    path_own_model  = os.path.join('networks', 'pretrained_models', 'iscx2016vpn', 'CNN1D_TrafficClassification_best_model_without_aux.pth')
    net             = load_fp32_model(path=path_own_model, input_ch=example_inputs.shape[1], num_classes=classes, device=device)
    cudnn.benchmark = True
    n_layer         = get_num_hidden_layer(net, args.model)
    net.to(device)


    # RL search
    env   = RL_env(net, n_layer, args.dataset, train_loader, val_loader, test_loader, classes, 
                   args.prune_ratio, args.state_dim, example_inputs, args.max_timesteps, args.model, device)
    
    betas = (0.9, 0.999)

    agent = Agent(state_dim=args.state_dim, action_dim=n_layer, action_std=args.action_std, lr=args.lr,
                  betas=betas, gamma=args.gamma, K_epochs=args.K_epochs, eps_clip=args.eps_clip, seed=args.seed)
    
    search(env,agent, update_timestep=args.update_timestep, max_timesteps=args.max_timesteps, max_episodes=args.max_episodes,
           log_interval=args.log_interval, solved_reward=args.solved_reward, random_seed=args.seed)
