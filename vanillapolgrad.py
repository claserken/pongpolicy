import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
from tensorflow.keras.models import load_model


render = True
gamma = 0.99

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector -- changed, now just outputs 80x80 array"""
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I #I.astype(np.float).ravel() 
a
def make_model():
    model = tf.keras.Sequential()
    
    model.add(layers.Conv2D(10, (3, 3), strides=(2, 2), padding='same',
                                     input_shape=(80,80, 1)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    # model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

#[0,0,0,0,0,1] --> [0.98,0.99,1]
#[0,0,0,0,0,1,0,0,0,0,0,1] --> [...,0.98,0.99,1,...,0.98,0.99,1]
#[-1, -1.5]

#log probabilities for each frame: [-1, -0.2832, -0.7]  called game_log_array
#reward for each frame: [0,0,0,0,0,1] or [0,0,0,0,0,0,-1] reward_array
#discount_rewards(reward_array): [0.94, 0.95, 0.96, 0.97,...,1]
#weighted_log_sum: dot_product([-1, -0.2832, -0.7,...], [0.94, 0.95, 0.96, 0.97,...,1])


def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r) # [0] * len(r)
  running_add = 0
  for t in reversed(range(0, len(r))):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r 
  
def discountify(r):
    """ take 1D float array of rewards and compute discounted reward """
    running_add = 0
    for power, t in enumerate(reversed(range(0, len(r)))):
        running_add += (gamma **power) * r[t]
    return running_add 

optimizer = tf.keras.optimizers.Adam(1e-4)

# model = make_model()

modelLocation = "./model2"


if os.path.isdir(modelLocation):
    model=load_model(modelLocation)
    print('LOADED MODEL')
else:
    model = make_model()
    print('MAKING NEW MODEL')


env = gym.make("Pong-v0")
observation = env.reset() # Shape (210, 160, 3)
episode_number = 0
batch_size = 4 # every how many episodes to do a param update?
reward_sum = 0
prev_x = None

win_list = []
rolling_win_count = 0


opt = tf.keras.optimizers.Adam(1e-4)

game_log_array = []  
reward_array = []

while True:
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        weighted_log_sum = 0

        while True:
            if render: env.render()

            cur_x = prepro(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros(shape=(80,80))
            prev_x = cur_x

            aprob = model(x.reshape(1,80,80,1), training=True)
            
            action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

            # game_log_sum = tf.math.log(aprob if action==2 else (1-aprob))
            game_log_array.append(tf.math.log(aprob if action==2 else (1-aprob)))
            observation, reward, done, info = env.step(action)

            reward_sum += reward
            reward_array.append(reward)
                
            if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
                 # Win count calculation
                normalized_reward = (reward + 1) / 2
                win_list.append(normalized_reward)
                rolling_win_count += normalized_reward
                if(len(win_list) > 200):
                    rolling_win_count -= win_list[0]
                    win_list = win_list[-200:]
                    
                print (('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))

                # Did this have no effect the whole time?
                
                # if reward == 1:
                #     reward *= 3
                # else:
                #     reward /= 3

                discounted_rewards = discount_rewards(reward_array)

                for i in range(len(discounted_rewards)):
                   weighted_log_sum += discounted_rewards[i] * game_log_array[i]                
                
                game_log_array = []  
                reward_array = []

            if done: # an episode finished  
                episode_number += 1
                    
                if episode_number % batch_size == 0: 
                    model.save(modelLocation)
                    print('MODEL SAVED')
                env.reset()

                if episode_number % batch_size == 0:
                    print (('resetting env. episode reward total was %f. weighted log sum: %f' % (reward_sum, weighted_log_sum)))
                    print("Rolling Win Avg: " + str(100.0*rolling_win_count/len(win_list)) + '%')
                    reward_sum = 0
                    # print(weighted_log_sum)
                    # print(aprob)
                    weighted_log_sum *= -1
                    break
                
    gradients = tape.gradient(weighted_log_sum, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    # for grad, var in zip(gradients, model.trainable_variables):
    #     print(grad)
    #     var.assign_add(0.01*grad)