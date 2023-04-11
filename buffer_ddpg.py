import numpy as np

class ReplayBuffer:

    def __init__(self, max_size, min_size, input_shape, n_actions, number_of_values):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.mem_cntr_ = 0
        self.total_mem_cnt = 0

        self.min_size = min_size
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.info = np.zeros((self.mem_size, *number_of_values))
        self.new_info = np.zeros((self.mem_size, *number_of_values))

    def store_transition(self, state, action, reward, state_, done, info, new_info):
        
        if self.mem_size > self.mem_cntr:
            index = self.mem_cntr % self.mem_size
            self.mem_cntr += 1

        elif self.mem_size == self.mem_cntr:
            index = self.mem_cntr_ % self.mem_size
            self.mem_cntr_ +=1
            if self.mem_cntr_ == self.mem_cntr:
                self.mem_cntr_ = 0


        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.info[index] = info
        self.new_info[index] = new_info

        self.total_mem_cnt += 1

        self.total_mem_cnt +=1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        infos = self.info[batch]
        new_infos = self.new_info[batch]

        return states, actions, rewards, states_, dones, infos, new_infos

    def save_data(self):
        np.save("data_/states.npy", self.state_memory[:self.mem_cntr])
        np.save("data_/states_.npy", self.new_state_memory[:self.mem_cntr])
        np.save("data_/actions.npy", self.action_memory[:self.mem_cntr])
        np.save("data_/rewards.npy", self.reward_memory[:self.mem_cntr])
        np.save("data_/terminal.npy", self.terminal_memory[:self.mem_cntr])
        np.save("data_/info.npy", self.info[:self.mem_cntr])
        np.save("data_/new_info.npy", self.new_info[:self.mem_cntr])
    
    def load_data(self):
        self.state_memory[:self.mem_cntr] = np.load("data_/states.npy")
        self.new_state_memory[:self.mem_cntr] = np.load("data_/states_.npy")
        self.action_memory[:self.mem_cntr] = np.load("data_/actions.npy")
        self.reward_memory[:self.mem_cntr] = np.load("data_/rewards.npy")
        self.terminal_memory[:self.mem_cntr] = np.load("data_/terminal.npy")
        self.info[:self.mem_cntr] = np.load("data_/info.npy")
        self.new_info[:self.mem_cntr] = np.load("data_/new_info.npy")


