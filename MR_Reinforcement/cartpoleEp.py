
import gym
env = gym.make("CartPole-v1")

for episode in range(1, 11):
    score = 0
    state = env.reset()
    done = False
  
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, _, info = env.step(action)
        score += reward
        
    print('Episode:', episode, 'Score:', score)
env.close()
