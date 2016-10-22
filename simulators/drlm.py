import argparse
from collections import defaultdict
import HTMLParser
import numpy as np
import operator
import os
#import pdb; pdb.set_trace()
try:
    import cPickle as pickle
except:
    import pickle
import random
import re
import socket
import sys
import string
import time
from copy import deepcopy
from bidict import bidict

curDirectory = os.path.dirname(os.path.abspath(__file__))

def build_lookup(file_obj):
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    sentences = [''.join([i if ord(i) < 128 else ' ' for i in sentence]).translate(replace_punctuation).strip().split() for sentence in file_obj.readlines()]
    max_len = max([len(s) for s in sentences])
    train_vocab = set([y for x in sentences for y in x])
    vocab_lookup = bidict({word : id for id, word in enumerate(train_vocab)})
    return max_len, vocab_lookup


def vectorize(sentence, size, mapping):
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    sentence = ''.join([i if ord(i) < 128 else ' ' for i in sentence])
    sentence = sentence.translate(replace_punctuation)
    vec = np.zeros((size,)) # Create a vocabulary size column vector
    for index in [mapping[x.strip()] for x in sentence.split()]: # For each word index
        vec[index] += 1.0 # Increment count by 1
    return vec

def softmax_select(x, alpha):
    exp = np.exp(alpha * x)
    return exp / np.sum(exp)

def compute_bleu(cand, ref):
    return np.sum(np.clip(cand, 0, ref)) / np.sum(cand)

if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')

    # parse arguments
    parser = argparse.ArgumentParser(description = "Deeply Reinforced Language Model")
    parser.add_argument("--train", type = str, help = "File path to training corpus", required = True)
    parser.add_argument("--test", type = str, help = "File path to testing corpus", required = True)
    args = parser.parse_args()

    startTime = time.time()

    with open(os.path.join(curDirectory, args.train), 'rb') as train_data:
        maxStep, train_lookup = build_lookup(train_data)

    vocab_size = len(train_lookup.keys())
    print 'Vocabulary size = {0}'.format(vocab_size)
    print 'Maximum sentence length = {0}'.format(maxStep)
    
    numEpisode = 0
    numStep = 0
    rewardSum = 0
    totalEpisodes = 4000
    prev_step = None
    avg_reward = []
    metric_freq = 200
    STOP = 'STOP'
    START = 'START'
    start_id = train_lookup[START]
    stop_id = train_lookup[STOP]
    
    #### DRLM parameters ####
    replay_limit = 1000000
    replay_mem = [] # Replay memory will be stored as a list of tuples
    alpha = 1.0 # TODO: See if there is a better setting of this value
        
    learning_rate = 0.0025 # 10x the learning rate of Nature DQN
    rand_exp = 1000 # Number of episodes to take under random policy
    hidden_size = 100
    gamma = 0.99
    num_epochs = 1
    batch_size = 64 # Value left unspecified in original paper
    activ = 'tanh' # Paper uses tanh
    #########################

    #### DRLM Model ####
    from keras.models import Model
    from keras.layers import Input, Dense, merge
    from keras.optimizers import Adam, SGD

    st_inp = Input(shape=(vocab_size,)) # State (sentence) input will be a BOW vector
    st_h1 = Dense(hidden_size, activation=activ)(st_inp)
    st_h2 = Dense(hidden_size, activation=activ)(st_h1)

    act_inp = Input(shape=(1,)) # Action will be an integer corresponding to word in the vocabulary
    act_h1 = Dense(hidden_size, activation=activ)(act_inp)

    merge = merge([st_h2, act_h1], mode='dot', dot_axes=1)
    state_embed = Model(input=[st_inp], output=st_h2)
    action_embed = Model(input=[act_inp], output=act_h1)
    drlm = Model(input=[st_inp, act_inp], output=merge)
    opt = Adam(lr=learning_rate)
    drlm.compile(optimizer=opt, loss='mse')
    ####################

    results = []
    train_sentences = []
    with open(os.path.join(curDirectory, args.train), 'rb') as train_data:
        for sentence in train_data:
            train_sentences.append((sentence, vectorize(sentence, vocab_size, train_lookup)))

    for epoch in xrange(num_epochs):
        print 'Starting epoch {0}'.format(epoch+1)
        for data, sentence in train_sentences:
            print 'Training on sentence: {0}'.format(data)
            state = vectorize(START, vocab_size, train_lookup) # Always start with the START symbol
            real_state = [START]
            
            for t in xrange(maxStep):
                #print 'DRLM agent sentence: {0}'.format(real_state)
                if prev_step: # If this is not the first step in the episode
                    s, a, r = prev_step # Retrieve the previous state and action taken
                    replay_mem.append((s, a, r, state, False))  # Add transition to replay memory using current state and actions
                    while len(replay_mem) >= replay_limit: # If we have reached the capacity of replay memory
                        replay_mem.pop(0) # Remove least recent transitions until there is space for one more transition

                    if len(replay_mem) > batch_size:
                        sampled = np.random.choice(len(replay_mem), size=batch_size, replace=False)
                        state_inps = np.array([replay_mem[x][0] for x in sampled])
                        action_inps = np.array([replay_mem[x][1] for x in sampled])
                    
                        def get_replay_target(sars):
                            s, a, r, s_prime, terminal = sars
                            
                            if terminal:
                                return r
                            else:                        
                                target_states = np.repeat(s_prime[np.newaxis,:], vocab_size, axis=0)
                                target_actions = np.array([[x] for x in xrange(vocab_size)])
                                q_vals = drlm.predict([target_states, target_actions], batch_size=1, verbose=0).flatten()
                                max_q = np.max(q_vals)
                                return r + gamma * max_q

                        loss = 0.0
                        for index, r in enumerate(sampled):
                            target = np.array([get_replay_target(replay_mem[r])])
                            loss += drlm.fit([np.array([state_inps[index]]), np.array([action_inps[index]])], target, batch_size=1, nb_epoch=1, verbose=0, shuffle=True).history['loss'][0]
                        print 'Loss = {0}'.format(loss / batch_size)
                        #print
        
                if prev_step: # If the last word selected by the agent was the STOP symbol
                    if prev_step[1] == stop_id:
                        # We have reached a terminal state

                        # Remove last transition in replay memory and change the terminal bit
                        s, a, r, s_prime, term = replay_mem.pop()
                        replay_mem.append((s, a, r, s_prime, True))

                        numEpisode += 1
                        #avg_reward.append(rewardSum)
                        #while len(avg_reward) > metric_freq:
                        #    avg_reward.pop(0)
                
                        print 'Completed episode {0}/{1}'.format(numEpisode, totalEpisodes)
                        print 'Total reward = {0}, # of steps = {1}'.format(rewardSum, numStep)
                        print 'Current replay memory = {0}/{1}'.format(len(replay_mem), replay_limit)
            
                        #if numEpisode % metric_freq == 0:
                        #    print 'Average reward over the last {0} episodes = {1}'.format(metric_freq, np.mean(avg_reward))
                        #    results.append(np.mean(avg_reward))
                        #    print 'Result values collected thus far: {0}'.format(results)
                        #    #print 'Current replay memory = {0}/{1}'.format(len(replay_mem), replay_limit)
                        #    print
            
                        rewardSum = 0
                        numStep = 0
                        prev_step = None
                        break

                    else:
                        # We are not in a terminal state
                        
                        if numEpisode > rand_exp:
                            #Setup Numpy arrays for Q-value feedforward computation
                            state_inps = np.repeat(state[np.newaxis,:], vocab_size, axis=0)
                            action_inps = np.array([[x] for x in xrange(vocab_size)])
                        
                            # Get Q-values for each action and perform softmax selection
                            q_vals = drlm.predict([state_inps, action_inps], batch_size=1, verbose=0)
                            q_vals = q_vals.flatten()
                            soft_select = softmax_select(q_vals, alpha) # Compute Boltzmann Q-policy
                        else:
                            soft_select = (1.0 / vocab_size) * np.ones((vocab_size,))
                    
                        if numStep < maxStep:
                            action = np.random.choice(vocab_size, replace=False, p=soft_select)  # Select action according to distribution
                        else:
                            action = stop_id
                        
                        if action == stop_id: # Dropped STOP symbol so compute BLEU score
                            reward = compute_bleu(state, sentence)
                        else:
                            reward = 0.0
                        
                        prev_step = state, action, reward  # Keep track of transition
                        state[action] += 1 # Update state BOW vector according to selected action/word
                        real_state.append(train_lookup.inv[action])
                        numStep += 1
                    
                else:
                    # We are not in a terminal state

                    if numEpisode > rand_exp:
                        #Setup Numpy arrays for Q-value feedforward computation
                        state_inps = np.repeat(state[np.newaxis,:], vocab_size, axis=0)
                        action_inps = np.array([[x] for x in xrange(vocab_size)])
                        
                        # Get Q-values for each action and perform softmax selection
                        q_vals = drlm.predict([state_inps, action_inps], batch_size=1, verbose=0)
                        q_vals = q_vals.flatten()
                        soft_select = softmax_select(q_vals, alpha) # Compute Boltzmann Q-policy
                    else:
                        soft_select = (1.0 / vocab_size) * np.ones((vocab_size,))

                    if numStep < maxStep:
                        action = np.random.choice(vocab_size, replace=False, p=soft_select)  # Select action according to distribution
                    else:
                        action = stop_id
                        
                    if action == stop_id: # Dropped STOP symbol so compute BLEU score
                        reward = compute_bleu(state, sentence)
                    else:
                        reward = 0.0
                        
                    prev_step = state, action, reward  # Keep track of transition
                    state[action] += 1 # Update state BOW vector according to selected action/word
                    real_state.append(train_lookup.inv[action])
                    numStep += 1
            sys.exit(0)
