import sys
import numpy as np
import os
import re
import string
import time
import random
import argparse
import nltk
from collections import defaultdict
import cPickle as pickle
from copy import deepcopy
from bidict import bidict
from replay import ReplayBuffer

cur_directory = os.path.dirname(os.path.abspath(__file__))
replace_punct = re.compile('[^\s\w_]+')

def softmax_select(x, alpha):
    exp = np.exp(alpha * x)
    return exp / np.sum(exp)

def clean_string(sentence):
    sentence = replace_punct.sub('', sentence)
    sentence = ' '.join(sentence.split())
    return sentence

def make_one_hot(one, size):
    ret = [0.0] * size
    ret[one] = 1.0
    return ret

def contains_nums(sentence):
    return any(x.isdigit() for x in sentence)

def build_lookup():
    corpus = nltk.corpus.brown.sents(categories=['news'])
    corpus = [' '.join(x) for x in corpus]
    sentences = filter(lambda x: len(x) <= 100 and not contains_nums(x), [clean_string(sentence.lower()) for sentence in corpus])
    max_len = max([len(s) for s in sentences])
    train_vocab = set(reduce(lambda x,y: x+y, [list(x) for x in sentences]))
    vocab_lookup = bidict({char : id for id, char in enumerate(train_vocab)})
    return max_len, sentences, vocab_lookup


if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')

    # parse arguments
    parser = argparse.ArgumentParser(description = "Deeply Reinforced Language Model")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode with verbose printing")
    parser.add_argument("--gpu", action='store_true', help="Indicator if running code on machine with GPU")
    args = parser.parse_args()

    max_len, sentences, vocab_lookup = build_lookup()

    print max_len
    print sorted(vocab_lookup.iteritems())
    print len(sentences)

    vocab_size = len(vocab_lookup.keys())
    print 'Vocabulary size = {0}'.format(vocab_size)
    print 'Maximum sentence length = {0}'.format(max_len)
    print 'Total number of sentences for training = {0}'.format(len(sentences))

    #### DRLM Model ####
    from keras.models import Model
    from keras.layers import Input, Dense, merge
    from keras.layers.recurrent import LSTM
    from keras.optimizers import Adam, SGD
    from keras.layers.advanced_activations import LeakyReLU

    #### DRLM parameters ####
    debug = args.debug
    replay_limit = 1000000
    replay_mem = ReplayBuffer(replay_limit) # Replay memory will be stored as a list of tuples
    alpha = 0.1 # TODO: See if there is a better setting of this value
        
    learning_rate = 0.001 # 10x the learning rate of Nature DQN
    hidden_size = 100
    gamma = 0.99
    num_epochs = 2
    batch_size = 32
    consumption = "cpu" if not args.gpu else "gpu"
    activ = lambda: LeakyReLU(alpha=0.0001) # Paper uses tanh
    #########################

    st_inp = Input(shape=(max_len, 1)) # State (sentence) input will be a tensor of shape (batch x max_len x 1)
    st_h1 = LSTM(hidden_size, activation=activ(), unroll=True, consume_less=consumption, input_shape=(max_len, 1))(st_inp)
    st_h2 = Dense(hidden_size, activation=activ())(st_h1)

    act_inp = Input(shape=(vocab_size,)) # Action will be an integer corresponding to a character
    act_h1 = Dense(hidden_size, activation=activ())(act_inp)

    merge = merge([st_h2, act_h1], mode='dot', dot_axes=1)
    state_embed = Model(input=[st_inp], output=st_h2)
    action_embed = Model(input=[act_inp], output=act_h1)
    drlm = Model(input=[st_inp, act_inp], output=merge)
    opt = Adam(lr=learning_rate)
    drlm.compile(optimizer=opt, loss='mse')
    ####################

    for epoch in xrange(num_epochs):
        print 'Starting Epoch {0}'.format(epoch + 1)
        random.shuffle(sentences)
        for episode, sentence in enumerate(sentences): # One sentences => one episode
            episode = episode + 1
            curr_state = np.empty((100,1))
            curr_state.fill(-1)
            agent_sentence = []
            termination = len(sentence)
            reward_sum = 0.0
            loss_sum = 0.0
            for step in xrange(termination):

                # If we have collected enough transitions, perform a gradient descent update step
                if replay_mem.size() >= batch_size: 
                    states, actions, rewards, terms, next_states = replay_mem.sample_batch(batch_size)
                    actions = np.array([make_one_hot(x, vocab_size) for x in actions])
   
                    def get_replay_target(target_state, reward, terminal, gam):                         
                        if terminal:
                            return reward
                        else:
                            target_states = np.repeat(target_state[np.newaxis,:,:], vocab_size, axis=0)
                            target_actions = np.identity(vocab_size)
                            q_vals = drlm.predict([target_states, target_actions], batch_size=1, verbose=0).flatten()
                            max_q = np.max(q_vals)
                            return reward + gam * max_q

                    for i in xrange(batch_size):
                        target = np.array([get_replay_target(next_states[i], rewards[i], terms[i], gamma)])
                        loss_sum += drlm.fit([np.array([states[i]]), np.array([actions[i]])], target, batch_size=1, nb_epoch=1, verbose=0, shuffle=True).history['loss'][0]

                ## Select an action based on the current state according to a Boltzmann policy
                
                #Setup Numpy arrays for Q-value feedforward computation
                state_inps = np.repeat(curr_state[np.newaxis,:,:], vocab_size, axis=0)
                action_inps = np.identity(vocab_size)
                        
                # Get Q-values for each action and perform softmax selection
                q_vals = drlm.predict([state_inps, action_inps], batch_size=1, verbose=0)
                q_vals = q_vals.flatten()

                # Compute Boltzmann Q-policy
                soft_select = softmax_select(q_vals, alpha)
                if np.isnan(soft_select).any():
                    soft_select = (1.0 / vocab_size) * np.ones((vocab_size,))
                    print 'Using uniform action selection'

                # Select action according to distribution
                action = np.random.choice(vocab_size, replace=False, p=soft_select)

                # +1.0 reward for correct character choice and -1.0 otherwise
                reward = 1.0 if vocab_lookup[sentence[step]] == action else -1.0

                # Update to next state
                prev_state = curr_state
                curr_state[step, :] = action

                # If the next transition will send us to a termminal state
                if step+1 == termination:
                    terminal = True
                else:
                    terminal = False

                # Add experience tuple to replay memory
                replay_mem.add(prev_state, action, reward, terminal, curr_state)

                # Bookkeeping
                reward_sum += reward
                agent_sentence.append(vocab_lookup.inv[action])

            print 'Completed episode {0}/{1}'.format(episode, len(sentences))
            print 'Total reward = {0}/{1}'.format(reward_sum, len(sentence))
            print 'True sentence = {0}'.format(sentence)
            print 'Agent sentence = {0}'.format(''.join(agent_sentence))
            print 'Average episode loss = {0}'.format(loss_sum / len(sentence))
            print 'Replay memory = {0}/{1}'.format(replay_mem.size(), replay_limit)
            print '\n'
