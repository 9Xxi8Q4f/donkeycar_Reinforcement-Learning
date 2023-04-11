import math
import gym
import gym_donkeycar
import ddpg_ 
import json
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
import numpy as np
#* CONF TO SIMULATOR PATH
PATH_TO_APP = "/home/tinrafiq/Documents/DonkeySimLinux"
exe_path = f"{PATH_TO_APP}/donkey_sim.x86_64"
port = 9091
headless = False

conf = { "exe_path" : exe_path, "port" : port, "headless" : headless }

#TODO : shape tensorboard and output graphs
env = gym.make("donkey-generated-roads-v0", conf = conf)
obs = env.reset()

agent_ = ddpg_.Agent(input_dims= obs.shape, min_mem_size=130, max_size=10000, batch_size=128)
#* stats
show_preview = False
aggregate_stats_every = 10 #* save model in every
ep_rewards = [0]
ep_average_rewards = [0]
average_error = [0]
MODEL_NAME = "ddpg_"
ep = 0

#* load model, params and episode rewards
load_weights = False
if load_weights == True:

    params_ = json.load(open("parameters_ddpg/episode_370_params.json"))
    agent_.memory.mem_cntr = params_["mem_cntr"]
    agent_.memory.mem_cntr_ = params_["mem_cntr_"]
    agent_.memory.total_mem_cnt = params_["total_mem_cnt"]
    ep = params_["episode"]
    agent_.noise = params_["noise"]

    actor_eval = f"weights_/episode_{ep}/actor_eval/"
    actor_target = f"weights_/episode_{ep}/actor_target/"
    critic_eval = f"weights_/episode_{ep}/critic_eval/"
    critic_target = f"weights_/episode_{ep}/critic_target/"

    agent_.load_models(actor_eval=actor_eval, actor_target=actor_target, \
        critic_eval=critic_eval, critic_target=critic_target)

    agent_.memory.load_data()

    with open(f"parameters_ddpg/episode_{ep}_rewards.json", 'r') as rew:
        ep_rewards = json.load(rew)
    with open(f"parameters_ddpg/episode_{ep}_av_rewards.json", 'r') as av:
        ep_average_rewards = json.load(av)
    with open(f"parameters_ddpg/episode_{ep}_average_error.json", 'r') as err:
        average_error = json.load(err)

load_checkpoint = False
if load_checkpoint:
        n_steps = 0
        while n_steps <= agent_.batch_size:

            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent_.remember(observation, action, reward, observation_, done)
            n_steps += 1
        agent_.learn()
        agent_.load_models()
        evaluate = True
else:
        evaluate = False

try:
    for episode in range(ep +1,2000):
        #* episodes returns
        #! env.reward also is episode reward !!
        episode_reward = 0
        observation = env.reset()
        shape_y = observation.shape[0]
        step = 1
        error = 0

        while True: #* if it's done block will break

            #! normally shape is (X,) 
            observation = observation.reshape((1,shape_y))
            action = agent_.choose_action(observation, evaluate)
            action = np.array(action)
            if action[0,0] < -1.0: action[0,0] = -1.0
            if action[0,0] > 1.0 : action[0,0] = 1.0
            if action[0,1] < -1.0: action[0,1] = -1.0
            if action[0,1] > 1.0 : action[0,1] = 1.0

            action = np.array([action[0,0],(action[0,1]+1.5)/15.0])

            new_observation, reward, done, info = env.step(action)
            error += math.fabs(info["cte"])
            episode_reward += reward

            #* Every step we update replay memory and train main network
            agent_.remember(observation, action, reward, new_observation, done)
            # print("Reward : ", reward, "    CTE : ", info["cte"])
            agent_.learn()

            observation = new_observation
            step += 1
            # print("Step Reward: ", reward)
            #* episode done info
            if done:
                break

        av_error = error/step
        average_error.append(av_error)
        ep_rewards.append(episode_reward)
        average_reward = episode_reward/step
        ep_average_rewards.append(average_reward)
        print("Episode Over: ", episode, "   Reward: ", episode_reward, "   Av. Reward: ", average_reward, "   Av. Error: ", av_error)
        # agent_.tensorboard.update_stats(reward_avg=average_reward, reward_total=episode_reward, error_avg=av_error)
        #* save things in checkpoints
        if episode % aggregate_stats_every == 0 or episode == 1:
            #* save current parameters as json
            dictionary = {"episode":episode, "mem_cntr":agent_.memory.mem_cntr, "mem_cntr_": agent_.memory.mem_cntr_, "noise": agent_.noise, "total_mem_cnt": agent_.memory.total_mem_cnt}
            with open(f'parameters_ddpg/episode_{episode}_params.json', 'x') as outfile:
                json.dump(dictionary, outfile)

            #* save the weights manually
            agent_.save_models(actor_eval = f"weights_/episode_{episode}/actor_eval/", actor_target=f"weights_/episode_{episode}/actor_target/", \
                critic_eval = f"weights_/episode_{episode}/critic_eval/", critic_target=f"weights_/episode_{episode}/critic_target/")
            agent_.memory.save_data()

            #* save the episode results as json
            with open(f"parameters_ddpg/episode_{episode}_rewards.json", 'x') as f:
                # indent=2 is not needed but makes the file human-readable 
                # if the data is nested
                json.dump(ep_rewards, f, indent=2) 
            
            with open(f"parameters_ddpg/episode_{episode}_av_rewards.json", 'x') as f:
                # indent=2 is not needed but makes the file human-readable 
                # if the data is nested
                json.dump(ep_average_rewards, f, indent=2) 

            with open(f"parameters_ddpg/episode_{episode}_average_error.json", 'x') as f:
                # indent=2 is not needed but makes the file human-readable 
                # if the data is nested
                json.dump(average_error, f, indent=2) 

except KeyboardInterrupt:
    # You can kill the program using ctrl+c
    pass

    # Exit the scene
env.close()
