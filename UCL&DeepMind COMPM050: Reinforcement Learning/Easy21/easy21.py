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


def draw():
    value = random.randrange(1, 11)  # Draw random sample from [1,10]
    color = np.random.choice([1, -1], p=[1.0/3, 2.0/3])  # Black: 1, Red: -1
    return value * color


def dealer_move(dealer_first_card):
    if dealer_first_card >= 17:
        return dealer_first_card
    else:
        return dealer_move(dealer_first_card + draw())


def step(s, a):
    # s = (dealer's first card, player's sum)
    # a = 'hit' or 'stick'
    dealer_first_card = s[0]
    player_sum = s[1]
    if a == 'hit':
        player_sum += draw()
        if player_sum > 21 or player_sum < 1:
            return (dealer_first_card, player_sum), -1
    elif a == 'stick':
        dealer_sum = dealer_move(dealer_first_card)
        if dealer_sum > 21 or dealer_sum < 1:
            return (dealer_first_card, player_sum), 1
    if player_sum > dealer_sum:
        return (dealer_first_card, player_sum), 1
    elif player_sum == dealer_sum:
        return (dealer_first_card, player_sum), 0
    else:
        return (dealer_first_card, player_sum), -1


if __name__ == '__main__':
    print(step((10, 10), 'hit'))
