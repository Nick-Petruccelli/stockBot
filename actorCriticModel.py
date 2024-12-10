import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Dense, Input
from keras.optimizers import Adam
import numpy as np
import os
import matplotlib.pyplot as plt

N_EPISODES = 780

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
keras.config.disable_interactive_logging()
class ActorCritic():
    def __init__(self, alpha, beta, n_outputs, input_size, gamma=.99):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_outputs = n_outputs
        self.n_inputs = input_size
        self.actor, self.critic, self.policy = self.make_actor_critic_model()


    def make_actor_critic_model(self):
        input = Input(shape=(self.n_inputs))
        delta = Input(shape=[1])
        layer1 = Dense(32, activation='relu')(input)
        layer2 = Dense(16, activation='relu')(layer1)
        layer3 = Dense(8, activation='relu')(layer2)
        layer4 = Dense(4, activation='relu')(layer3)
        probs = Dense(self.n_outputs, activation='softmax')(layer4)
        values = Dense(1, activation='linear')(layer4)

        actor = keras.Model(inputs=[input, delta], outputs=[probs])
        actor.compile(optimizer=Adam(learning_rate=self.alpha), loss=keras.losses.BinaryCrossentropy(from_logits=True))

        critic = keras.Model(inputs=[input], outputs=[values])
        critic.compile(optimizer=Adam(learning_rate=self.beta), loss='mean_squared_error')

        policy = keras.Model(inputs=[input], outputs=[probs])

        return actor, critic, policy

    def choose_action(self, obs):
        state = obs[np.newaxis]
        probs = self.policy.predict(state)[0]
        action = np.random.choice([0, 1], p=probs)
        return action

    def learn(self, state, action, reward, new_state, done):
        state = state[np.newaxis, :]
        new_state = new_state[np.newaxis, :]
        new_critic_value = self.critic.predict(new_state)
        critic_value = self.critic.predict(state)
        target = reward + self.gamma*new_critic_value*(1-int(done))
        delta = target - critic_value
        actions = np.zeros([1, self.n_outputs])
        actions[np.arange(1), action] = 1.0
        self.actor.fit([state, delta], actions, verbose=0)
        self.critic.fit(state, target, verbose=0)

    def train(self, env):
        rewards_per_episode = []
        episode = 0
        while True:
            print(f"---Episode {episode} start---")
            obs = env.start_next_episode()
            if len(obs) == 0:
                break
            reward_total = 0
            while True:
                action = self.choose_action(obs)
                new_obs, reward, done, _ = env.step(action)
                reward_total += reward
                if done:
                    rewards_per_episode.append(reward_total)
                    break
                self.learn(obs, action, reward, new_obs, done)
                obs = new_obs
            print(f"---Episode {episode}: score {reward_total}---")
            episode += 1
        plt.plot(rewards_per_episode)
        plt.savefig("plots/ActorCriticResult.png")
        self.actor.save("models/actor.keras")
        self.critic.save("models/critic.keras")
        self.policy.save("models/policy.keras")

class SumLayer(keras.Layer):
        def call(self, x):
                return keras.backend.sum(x)
