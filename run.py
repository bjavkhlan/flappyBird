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
        self.learning_rate      = 0.001
        self.model              = self._build_model()

    def _build_model(self):
        
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        
        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
        return model

    def act(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

class FlappyBird:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes          = 100
        self.env               = game.GameState()
        self.state_size        = 3
        self.action_size       = 2
        self.agent             = Agent(self.state_size, self.action_size)
        self.score_file        = open('run_score.txt', 'w')
    

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
                    state, _, done, score = self.env.frame_step(action)
                    state =  np.array([(state[0].astype('float32')+50)/288, (state[1].astype('float32')+200)/512, (state[2].astype('float32')+10)/20])
                    state = state.reshape(1, state.shape[0])
                
                print("Episode {}# Score: {}".format(e, score))
                self.score_file.write(str(score)+"\n")
        finally:
            self.score_file.flush()

if __name__ == "__main__":
    bird = FlappyBird()
    bird.fly()
