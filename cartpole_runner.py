import gymnasium as gym
import numpy as np
import pickle

rng = np.random.default_rng()   # random number generator (explore/exploit choice)

class Discretizer:
    def __init__(self,
                 pos_buckets:int=10,
                 vel_buckets:int=10,
                 ang_pos_buckets:int=10,
                 ang_vel_buckets:int=10):
        self.pos_space = np.linspace(-2.4, 2.4, pos_buckets)
        self.vel_space = np.linspace(-4, 4, vel_buckets)
        self.ang_pos_space = np.linspace(-.2095, .2095, ang_pos_buckets)
        self.ang_vel_space = np.linspace(-4, 4, ang_vel_buckets)

    def discretize(self, t:tuple) -> tuple:
        return (
            np.digitize(t[0], self.pos_space),
            np.digitize(t[1], self.vel_space),
            np.digitize(t[2], self.ang_pos_space),
            np.digitize(t[3], self.ang_vel_space)
        )

class QTable:
    def __init__(self,
                 pos_buckets:int=10,
                 vel_buckets:int=10,
                 ang_pos_buckets:int=10,
                 ang_vel_buckets:int=10,
                 learning_rate:float=0.1,
                 discount_factor:float=0.99):
        self.table = np.zeros((
            pos_buckets+1,
            vel_buckets+1,
            ang_pos_buckets+1,
            ang_vel_buckets+1,
            2
        ))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def set(self, state:tuple, action:int, value):
        self.table[state[0], state[1], state[2], state[3], action] = value

    def get(self, state:tuple, action:int):
        return self.table[state[0], state[1], state[2], state[3], action]

    def max_action_value_at(self, state):
        return np.max(self.table[state[0], state[1], state[2], state[3],:])

    def max_action_at(self, state):
        return np.argmax(self.table[state[0], state[1], state[2], state[3],:])

    def learn(self, state, action, reward, new_state):
        new_value = (self.get(state, action)
                     + self.learning_rate
                     * (reward + self.discount_factor * self.max_action_value_at(new_state) - self.get(state, action)))  # Bellman function
        self.set(state, action, new_value)

def load_q_table() -> QTable:
    f = open('cartpole.pkl', 'rb')
    Q_table = pickle.load(f)  # load model
    f.close()
    return Q_table

def save_q_table(Q_table:QTable):
    f = open('cartpole.pkl', 'wb')
    pickle.dump(Q_table, f)
    f.close()

def run_single_episode(env, discretizer:Discretizer, Q_table:QTable, epsilon:float=0.0, is_training:bool=False, max_episode_length:int=5000) -> int:
    state = discretizer.discretize(env.reset()[0])  # Starting position, starting velocity always 0
    terminated = False                              # will be True when failed
    rewards = 0                                     # increased by each successful step

    while not terminated and rewards < max_episode_length:
        action = env.action_space.sample() if (is_training and rng.random() < epsilon) else Q_table.max_action_at(state)

        new_state, reward, terminated, _, _ = env.step(action)
        new_state = discretizer.discretize(new_state)

        if is_training:
            Q_table.learn(state, action, reward, new_state)

        state = new_state
        rewards += reward

        if not is_training and rewards % 100 == 0:
            print(f'Steps done: {rewards}')

    if not is_training:
        print(f'{"Success! " if not terminated else "Failed at step: "}{rewards}')
    return rewards


def run(is_training=True, render=False):

    env = gym.make('CartPole-v1', render_mode='human' if render else None)          # Create an environment to explore

    discretizer = Discretizer()         # Discretize position, velocity, pole angle, and pole angular velocity
    if not is_training:
        run_single_episode(env=env, discretizer=discretizer, Q_table=load_q_table())    # run one episode using a trained model
        return                                                                          # and exit

    Q_table = QTable()                  # create a new Q-table to train
    epsilon = 1                         # exploration rate, 1 = 100% random actions, 0 = no random actions
    epsilon_linear_decay = 0.00001      # epsilon decay - linearly decreases exploration rate
    mean_rewards_quality_pass = 2500    # minimal avg steps of last 100 episodes to finish the learning

    rewards_per_episode = []

    i = 0

    while is_training:
        rewards = run_single_episode(
            env=env,
            discretizer=discretizer,
            Q_table=Q_table,
            epsilon=epsilon,
            is_training=is_training,
            max_episode_length=2 * mean_rewards_quality_pass    # must be bigger than mean_rewards_quality_pass!
        )

        rewards_per_episode.append(rewards)

        if i%100 == 0:
            mean_rewards = np.mean(rewards_per_episode)
            rewards_per_episode = []
            print(f'Episode: {i} {rewards}  Epsilon: {epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')
            if mean_rewards > mean_rewards_quality_pass:
                break

        epsilon = max(epsilon - epsilon_linear_decay, 0.01)
        i+=1

    env.close()
    save_q_table(Q_table)   # the model has been trained - dump Q_table to a file

if __name__ == '__main__':
    run(is_training=True, render=False) # train the model (you can do it just once, or play around with different params) and save it

    run(is_training=False, render=True) # load the model and test!