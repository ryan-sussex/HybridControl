import gym
import numpy as np

# Initialize environment and parameters
env = gym.make('MountainCarContinuous-v0')
epsilon = 0.1  # Exploration rate
alpha = 0.1    # Learning rate
gamma = 0.99   # Discount factor
n_bins = 20  # Number of bins for discretization
action_bins = 10  # Number of bins for action discretization

q_table_shape = (n_bins, n_bins, action_bins)  # Shape of Q-table
q_table = np.zeros(q_table_shape)  # Initialize Q-table

# Discretize the action space
actions = np.linspace(env.action_space.low, env.action_space.high, action_bins)

def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(actions[:, 0])  # Explore, take the first element for continuous action
    else:
        state_idx = discretize_state(state)
        action_idx = np.argmax(q_table[state_idx])  # Exploit
        return actions[action_idx, 0]  # Return the corresponding continuous action

def discretize_state(state):
    # Discretize the continuous state space for Q-learning
    bins = [np.linspace(-1.2, 0.6, n_bins), np.linspace(-0.07, 0.07, n_bins)]
    state_idx = []
    for i in range(len(state)):
        state_idx.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(state_idx)

def process_state(state):
    if isinstance(state, dict):
        # Extract the numerical values if state is a dictionary
        state = np.array(list(state.values()))
    return state

for episode in range(1000):
    state = process_state(env.reset())
    state = discretize_state(state)
    done = False
    total_reward = 0
    
    while not done:
        action = choose_action(state, epsilon)
        action_idx = np.digitize(action, actions[:, 0]) - 1
        next_state, reward, done, _ = env.step([action])
        next_state = process_state(next_state)
        next_state = discretize_state(next_state)
        
        # Q-learning update
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + gamma * q_table[next_state][best_next_action]
        td_error = td_target - q_table[state][action_idx]
        q_table[state][action_idx] += alpha * td_error
        
        state = next_state
        total_reward += reward
        
        if done:
            print(f"Episode: {episode}, Total Reward: {total_reward}")

env.close()
