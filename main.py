from stockEnv import StocksEnv
from dqnModel import DQN
from actorCriticModel import ActorCritic
import tensorflow as tf

if __name__ == "__main__":
    past_days_looked_at = 7
    env = StocksEnv(past_days_looked_at=past_days_looked_at)
    #n_inputs = past_days*3(ie open, close , volume) + curBalance + curStocksHeld + day + month
    n_inputs = [past_days_looked_at * 3 + 4]
    model = DQN(n_outputs=2, input_size=n_inputs)
    #model.train(env)

    env = StocksEnv(past_days_looked_at=past_days_looked_at)
    actor_critic = ActorCritic(.00001, .00005, n_outputs=2, input_size=n_inputs)
    actor_critic.train(env)
