#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The goal of this assignment is to apply reinforcement learning methods to a
simple card game that we call Easy21. This exercise is similar to the Blackjack
example in Sutton and Barto 5.3 – please note, however, that the rules of the
card game are different and non-standard.
"""

import random

import numpy as np

# Q1: Implementation of Easy21
"""
You should write an environment that implements the game Easy21. Specifically,
write a function, named ``step``, which takes as input a state ``s`` (dealer’s
first card 1–10 and the player’s sum 1–21), and an action a (``hit`` or 
``stick``), and returns a sample of the next state s′ (which may be terminal
if the game is finished) and reward ``r``.
"""


def _draw():
    value = random.randrange(1, 11)  # Draw random sample from [1,10]
    color = np.random.choice([1, -1], p=[1.0/3, 2.0/3])  # Black: 1, Red: -1
    return value * color


def _dealer_move(dealer_first_card):
    if dealer_first_card >= 17:
        return dealer_first_card
    else:
        return _dealer_move(dealer_first_card + _draw())


def step(s, a):
    # s = (dealer's first card, player's sum)
    # a = 'hit' or 'stick'
    dealer_first_card = s[0]
    player_sum = s[1]
    if a == 'hit':
        player_sum += _draw()
        if player_sum > 21 or player_sum < 1:
            reward = -1
            is_terminal = True
        elif player_sum > dealer_first_card:
            reward = 1
            is_terminal = False
        elif player_sum == dealer_first_card:
            reward = 0
            is_terminal = False
        else:
            reward = -1
            is_terminal = False
        return (dealer_first_card, player_sum), reward, is_terminal
    elif a == 'stick':
        dealer_sum = _dealer_move(dealer_first_card)
        if dealer_sum > 21 or dealer_sum < 1:
            reward = 1
        if player_sum > dealer_sum:
            reward = 1
        elif player_sum == dealer_sum:
            reward = 0
        else:
            reward = -1
        is_terminal = True
        return (dealer_first_card, player_sum), reward, is_terminal

# Q2: Monte-Carlo Control in Easy21
"""
Apply Monte-Carlo control to Easy21. Initialise the value function to zero.
Use a time-varying scalar step-size of αt = 1/N(st,at) and an ε-greedy
exploration strategy with εt = N0/(N0 + N(st)), where N0 = 100 is a constant,
N(s) is the number of times that state s has been visited, and N(s,a) is the
number of times that action a has been selected from state s. Feel free to
choose an alternative value for N0, if it helps producing better results. Plot
the optimal value function V ∗ (s) = maxa Q∗ (s, a) using similar axes to the
following figure taken from Sutton and Barto’s Blackjack example.
"""

def monte_carlo_episode(state, ):
    N0 = 100
    N_s = 0  # Number of times state s has been visited
    N_s_a = 0  # Number of times action a has been selected from state s
    value_function = 0


if __name__ == '__main__':
    print(step((10, 10), 'hit'))
