import PER_DDQN as buffer
import numpy as np
import time
import os
import keras
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
MODEL_NAME = "ddqn_"

class DDQNAgent(object):

    """
    * init values for agent
    * set buffer and networks

    * gamma: is the discount factor
    * alpha: is the learning rate
    * epsilon: is the exploring rate
    """

    def __init__(self, alpha=0.001, gamma=0.99, n_actions=27, epsilon= 1.0, batch_size=64,
                 input_dims = None, epsilon_dec=0.9995,  epsilon_end=0.02,
                 mem_size=20000, min_mem_size = 1000, fname='ddqn_model.h5', replace_target=100):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        #? self.model_file = fname
        self.replace_target = replace_target

        # * Action Space setting Up
        steering_actions = [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]
        # throttle_actions = steering_actions.copy()
        throttle_actions = [0, 0.5, 0.1]
        self.discrete_action_space = np.array(np.meshgrid(steering_actions,throttle_actions)).T.reshape(-1, 2)

        self.n_actions = len(self.discrete_action_space)

        #* Set up Buffer
        self.memory = buffer.ReplayBuffer(mem_size, min_mem_size, input_dims[0], n_actions,
                                   discrete=True)

        #* Network set up for eval and target networks
        self.network = QNetwork(fc1_dims=1024, fc2_dims=1024, fc3_dims=512, input_dims = input_dims, 
        n_actions=27, learning_rate = 0.01, name='q_net', chkpt_dir='./')
        
        self.q_eval = self.network.build_ddqn()
        self.q_target = self.network.build_ddqn()

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
 
    #* Adds step's data to a memory replay array
    #* (observation space, action, reward, new observation space, done)    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
 
    def get_action(self, observation):
        #* get action based on epsilon value
        if np.random.random() > self.epsilon:
            # print("GETTING Q VALUES")
            qs_ = self.get_qs(observation)
            # print("Q Values : ", qs_)
            action_index = np.argmax(qs_)
            # print("action_index : " , action_index)
            action = self.discrete_action_space[action_index]

        else:

            action_index = np.random.randint(0,len(self.discrete_action_space))
            action = self.discrete_action_space[action_index]

        return action, action_index

    def train(self):

        #* if memory size is smaller than min size, do nothing
        if (self.memory.mem_cntr) < self.memory.min_size:
            return
        
        #* and ELSE:
        #* sample minibatch and get states vs..
        state, action, reward, new_state, done, importance, sample_indices = \
                            self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        #* get the q values of current states by main network
        q_pred = self.q_eval.predict(state)

        #! for abs error
        target_old = np.array(q_pred)

        #* get the q values of next states by target network
        q_next = self.q_target.predict(new_state) #! target_val

        #* get the q values of next states by main network
        q_eval = self.q_eval.predict(new_state) #! target_next

        #* get the actions with highest q values
        max_actions = np.argmax(q_eval, axis=1)

        #* we will update this dont worry
        q_target = q_pred

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        #* new_q = reward + DISCOUNT * max_future_q
        q_target[batch_index, action_indices] = reward + \
                    self.gamma*q_next[batch_index, max_actions.astype(int)]*done

        #* error
        error = target_old[batch_index, action_indices]-q_target[batch_index, action_indices]
        # absolute_errors = np.abs(error)
        self.memory.set_priorities(sample_indices, error)

        #* epsilon is the beta here
        # importance = importance**(1-self.epsilon)

        #* now we fit the main model (q_eval)
        _ = self.q_eval.fit(state, q_target, verbose=0)

        #* If counter reaches set value, update target network with weights of main network
        #* it will update it at the very beginning also


        if self.memory.mem_cntr % self.replace_target == 0:
            self.update_network_parameters()
            print("Target replaced")
        if self.memory.mem_cntr_ != 0 and self.memory.mem_cntr_ % self.replace_target == 0:
            self.update_network_parameters()
            print("Target replaced")


        self.epsilon_dec_func()

    def update_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())

    #* Saves entire model
    def save_model(self, model_file):
        self.q_eval.save(model_file)

    #* Loads entire model
    def load_model(self):
        self.q_eval = load_model(self.model_file)
            # if we are in evaluation mode we want to use the best weights for
            # q_target
        if self.epsilon == 0.0:
            self.update_network_parameters()

    #* compute q values
    def get_qs(self, state):
        return self.q_eval.predict(state)

    #*dec epsilon value
    def epsilon_dec_func(self):
        #* Decay epsilon
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    #* Saves weights only
    def save_weights_all(self, eval_file, target_file):
        self.q_eval.save_weights(eval_file)
        self.q_target.save_weights(target_file)

    #* Loads weights only
    def load_weights_all(self, eval_file, target_file):
        self.q_eval.load_weights(eval_file)
        self.q_target.load_weights(target_file)
        self.epsilon_dec_func()

class QNetwork(keras.Model):

    def __init__(self, fc1_dims=1024, fc2_dims=1024, fc3_dims=512, input_dims = None, n_actions=25,
     learning_rate = 0.01, name='q_net', chkpt_dir='./'):
        super(QNetwork, self).__init__()
        
        self.learning_rate = learning_rate

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_ddqn.h5')

    def build_ddqn(self):
        model = Sequential()

        model.add(Dense(self.fc1_dims, input_shape = self.input_dims))
        model.add(Activation("relu"))
        #model.add(Dense(self.fc2_dims))
        #model.add(Activation("relu"))
        #model.add(Dense(self.fc3_dims))
        #model.add(Activation("relu"))
        model.add(Dense(self.n_actions))

        model.compile(loss = "mse", optimizer= Adam(learning_rate = self.learning_rate), metrics=["accuracy"])

        return model

class ModifiedTensorBoard(TensorBoard):

    # *Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, MODEL_NAME)

    # *Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model
        
        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter 

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        pass

    #* Overrided, saves logs with our step number
    #* (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    #* Overrided
    #* We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    #* Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    #* Custom method for saving own metrics
    #* Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    #* Added because of version
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

