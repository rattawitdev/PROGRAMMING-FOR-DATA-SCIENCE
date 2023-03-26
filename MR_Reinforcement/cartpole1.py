
import gym
env = gym.make("CartPole-v1")
env.reset()
for i in range(200):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, _, info = env.step(action)
    print("step", i, "action",action, observation, reward, done, info)
env.close()
