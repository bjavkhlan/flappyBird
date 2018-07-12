import random
import os
import numpy as np
from collections import deque
import keras
from keras.layers import Dense, add
from keras.models import Sequential, Model
from keras.optimizers import Adam

import sys
sys.path.append("game/")
import wrapped_flappy_bird as game

class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup      = "flappyBird_weight.h5"
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=2048)
        self.learning_rate      = 0.001
        self.gamma              = 0.95
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.99
        self.model              = self._build_model()
        self.target_model       = self._build_model()
        self.update_target_model()

    def _build_model(self):
        
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        
        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model

    def save_model(self):
            self.model.save(self.weight_backup)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
            
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


class FlappyBird:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes          = 25000
        self.env               = game.GameState()
        self.state_size        = 3
        self.action_size       = 2
        self.agent             = Agent(self.state_size, self.action_size)
        self.score_file        = open('score.txt', 'w')
    

    def fly(self):
        try:
            for e in range(self.episodes):
                state, _, _, _ = self.env.frame_step(0)
                state =  np.array([(state[0].astype('float32')+50)/288, (state[1].astype('float32')+200)/512, (state[2].astype('float32')+10)/20])
                state = state.reshape(1, state.shape[0])
                done = False
                score = 0
                while not done:
                    action = self.agent.act(state)
                    next_state, reward, done, score = self.env.frame_step(action)
                    next_state =  np.array([(next_state[0].astype('float32')+50)/288, (next_state[1].astype('float32')+200)/512, (next_state[2].astype('float32')+10)/20])
                    next_state = next_state.reshape(1, next_state.shape[0])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    
                print("Episode {}# Score: {}".format(e, score))
                self.score_file.write(str(score)+"\n")
                self.agent.update_target_model()
                if self.sample_batch_size < len(self.agent.memory):
                    self.agent.replay(self.sample_batch_size)
        finally:
            self.agent.save_model()
            self.score_file.flush()
if __name__ == "__main__":
    bird = FlappyBird()
    bird.fly()
