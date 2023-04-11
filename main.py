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

#* stats
show_preview = False
aggregate_stats_every = 10 #* save model in every
ep_rewards = [0]
ep_average_rewards = [0]
<<<<<<< HEAD
average_error = [0]
MODEL_NAME = "ddpg_"
=======
MODEL_NAME = "ddqn_"
>>>>>>> 7d4588d93ca9125b7dff7a650a8e40591b1c5788
ep = 0
evaluate = False

#* env output shape handling for neural networks
env = gym.make("donkey-generated-roads-v0", conf = conf)
obs, reward, done, info = env.reset()
info = np.array([info["cte"], info["speed"], info["forward_vel"], info["vel"][0], info["vel"][1], info["vel"][2]])

#* agent set up
agent_ = ddpg_.Agent(input_dims= obs.shape, scaler_dims = info.shape, min_mem_size=100, max_size=10000, batch_size=64)

def load_weights(load_weights = False):
    #* load model, params and episode rewards
    if load_weights == True:

<<<<<<< HEAD
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
=======

#* load model, params and episode rewards
load_weights = True
if load_weights == True:

    params_ = json.load(open("parameters/episode_690_params.json"))
    # episode = params_["episode"] + 1 #* 501
    agent_.epsilon = params_["epsilon"]
    agent_.memory.mem_cntr = params_["mem_cntr"]
    agent_.memory.mem_cntr_ = params_["mem_cntr_"]
    ep = params_["episode"]
>>>>>>> 7d4588d93ca9125b7dff7a650a8e40591b1c5788

        agent_.load_models(actor_eval=actor_eval, actor_target=actor_target, \
            critic_eval=critic_eval, critic_target=critic_target)

        agent_.memory.load_data()

        with open(f"parameters_ddpg/episode_{ep}_rewards.json", 'r') as rew:
            ep_rewards = json.load(rew)
        with open(f"parameters_ddpg/episode_{ep}_av_rewards.json", 'r') as av:
            ep_average_rewards = json.load(av)
        with open(f"parameters_ddpg/episode_{ep}_average_error.json", 'r') as err:
            average_error = json.load(err)

    with open(f"parameters/episode_{ep}_av_rewards.json", 'r') as av:
        ep_rewards = json.load(av)

try:
    for episode in range(ep +1,2000):
        #* episodes returns
        #! env.reward also is episode reward !!
        episode_reward = 0
        observation, reward, done, info = env.reset()
        shape_y = observation.shape[0]
<<<<<<< HEAD
        info = np.array([info["cte"], info["speed"], info["forward_vel"], info["vel"][0], info["vel"][1], info["vel"][2]])
        step = 1
        error = 0
=======
        step = 1

>>>>>>> 7d4588d93ca9125b7dff7a650a8e40591b1c5788

        while True: #* if it's done block will break

            #! normally shape is (X,) 
            observation = observation.reshape((1,shape_y))
            info = info.reshape((1, info.shape[0]))
            action = agent_.choose_action(observation, info, evaluate)
            action = np.array(action)

            if action[0,0] < -1.0: action[0,0] = -1.0
            if action[0,0] > 1.0 : action[0,0] = 1.0
            if action[0,1] < -1.0: action[0,1] = -1.0
            if action[0,1] > 1.0 : action[0,1] = 1.0

            action = np.array([action[0,0],(action[0,1]+1.5)/25.0])


            new_observation, reward, done, new_info = env.step(action)

            error += math.fabs(new_info["cte"])

            new_info = np.array([new_info["cte"], new_info["speed"], 
            new_info["forward_vel"], new_info["vel"][0],
             new_info["vel"][1], new_info["vel"][2]])

            #* Every step we update replay memory and train main network
            agent_.remember(observation, action, reward, new_observation, done, info, new_info)
            # print("Reward : ", reward, "    CTE : ", info["cte"])
            agent_.learn()

            observation = new_observation
<<<<<<< HEAD
            info = new_info
=======
            step += 1
>>>>>>> 7d4588d93ca9125b7dff7a650a8e40591b1c5788

            episode_reward += reward
            step += 1
            #* episode done info
            if done:
                break

<<<<<<< HEAD
        av_error = error/step
        average_error.append(av_error)
        ep_rewards.append(episode_reward)
        average_reward = episode_reward/step
        ep_average_rewards.append(average_reward)
        print("Episode Over: ", episode, "   Reward: ", episode_reward, "   Av. Reward: ", average_reward, "   Av. Error: ", av_error)
        # agent_.tensorboard.update_stats(reward_avg=average_reward, reward_total=episode_reward, error_avg=av_error)
        #* save things in checkpoints
        if episode % aggregate_stats_every == 0 or episode == 1:
=======

        ep_rewards.append(episode_reward)
        average_reward = episode_reward/step
        ep_average_rewards.append(average_reward)


        #* save things in checkpoints
        if episode % aggregate_stats_every == 0 or episode == 1:
            # average_reward = sum(ep_rewards[-aggregate_stats_every:])/len(ep_rewards[-aggregate_stats_every:])
            min_reward = min(ep_rewards[-aggregate_stats_every:])
            max_reward = max(ep_rewards[-aggregate_stats_every:])
            

            agent_.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward)

            #* save all model all at once
            # agent_.save_model(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

>>>>>>> 7d4588d93ca9125b7dff7a650a8e40591b1c5788
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
<<<<<<< HEAD
            
            with open(f"parameters_ddpg/episode_{episode}_av_rewards.json", 'x') as f:
=======
        
            with open(f"parameters/episode_{episode}_av_rewards.json", 'x') as f:
>>>>>>> 7d4588d93ca9125b7dff7a650a8e40591b1c5788
                # indent=2 is not needed but makes the file human-readable 
                # if the data is nested
                json.dump(ep_average_rewards, f, indent=2) 

<<<<<<< HEAD
            with open(f"parameters_ddpg/episode_{episode}_average_error.json", 'x') as f:
                # indent=2 is not needed but makes the file human-readable 
                # if the data is nested
                json.dump(average_error, f, indent=2) 
=======
>>>>>>> 7d4588d93ca9125b7dff7a650a8e40591b1c5788

except KeyboardInterrupt:
    # You can kill the program using ctrl+c
    pass

    # Exit the scene
env.close()