import tensorflow as tf
import tensorflow.keras as keras
from collections import deque
import numpy as np
import matplotlib.pyplot as plt


class DQN():
    def __init__(self,n_outputs, input_size, batch_size=32, discount_factor=.95, optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss_fn=keras.losses.MeanSquaredError()):
        self.n_outputs = n_outputs
        self.n_inputs = input_size
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.model = keras.models.Sequential([
                keras.layers.Dense(32, activation="elu", input_size=input_size),
                keras.layers.Dense(16, activation="elu"),
                keras.layers.Dense(8, activation="elu"),
                keras.layers.Dense(4, activation="elu"),
                keras.layers.Dense(self.n_outputs)])
        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        self.replay_buffer = deque(maxlen=2000)

    def take_step(self, env, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, info = env.step(action)
        self.replay_buffer.append((state, action, reward, next_state, done))
        return next_state, reward, done, info

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_outputs)
        else:
            q_values = self.model.predict(state[np.newaxis])
            return np.argmax(q_values[0])

    def sample_buffer(self):
        indecies = np.random.randint(len(self.replay_buffer), size=self.batch_size)
        batch = [self.replay_buffer[i] for i in indecies]
        states, actions, rewards, next_states, dones = [np.array([experience[field_index] for experience in batch]) for field_index in range(5)]
        return states, actions, rewards, next_states, dones

    def training_step(self, episode):
        experiences = self.sample_buffer()
        states, actions, rewards, next_states, dones = experiences
        next_q_values = self.model.predict(next_states)
        best_next_actions = np.argmax(next_q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, self.n_outputs).numpy()
        next_best_q_values = (self.target_model.predict(next_states) * next_mask).sum(axis=1)
        target_q_values = (rewards + (1 - dones) * self.discount_factor * next_best_q_values)
        mask = tf.one_hot(actions, self.n_outputs)
        with tf.GradientTape() as tape:
            all_q_values = self.model(states)
            q_values = tf.reduce_sum(all_q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_q_values, q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        if episode % 100 == 0:
            self.target_model.set_weights(self.model.get_weights())

    def train(self, env, n_episodes=600, max_steps=200):
        rewards_per_episode = []
        epsilon_denominator = int(n_episodes * .8)
        for episode in range(n_episodes):
            obs = env.reset()
            reward_total = 0
            for step in range(max_steps):
                epsilon = max(1 - episode / epsilon_denominator, .01)
                obs, reward, done, info = self.take_step(env, obs, epsilon)
                reward_total += reward
                if done:
                    rewards_per_episode.append(reward_total)
                    break
            if episode > 50:
                self.training_step(episode)
        plt.plot(rewards_per_episode)
        plt.savefig("plots/DoubleDQNResult.png")
        self.model.save_weights("lunerLanderModel.keras")
