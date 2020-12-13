"""
The goal of this assignment is to apply reinforcement learning methods to a
simple card game that we call Easy21. This exercise is similar to the Blackjack
example in Sutton and Barto 5.3 – please note, however, that the rules of the
card game are different and non-standard.
"""

from typing import Tuple
import numpy as np

# Q1: Implementation of Easy21
"""
You should write an environment that implements the game Easy21. Specifically,
write a function, named ``step``, which takes as input a state ``s`` (dealer’s
first card 1–10 and the player’s sum 1–21), and an action a (``hit`` or 
``stick``), and returns a sample of the next state s′ (which may be terminal
if the game is finished) and reward ``r``.
"""


def _draw() -> int:
    """Draw a card 1-10 and a colour (black/red) according to a
    pdf and return an associated value (Black: 1, Red: -1)"""
    black_probability = 1/3
    red_probability = 2/3

    value = np.random.randint(1, 11)
    color = np.random.choice([1, -1], p=[black_probability, red_probability])
    return value * color


def _dealer_move(dealer_sum) -> int:
    """Dealer will always stick at any sum >= 17 and hit otherwise"""
    if dealer_sum > 21 or dealer_sum < 1:
        return dealer_sum
    if dealer_sum >= 17:
        return dealer_sum
    else:
        return _dealer_move(dealer_sum + _draw())


def step(s, a) -> Tuple[Tuple[int, int], int, bool]:
    """
    Given a state and an action return a tuple:
    (next_state, reward, is_terminal) where is_terminal
    describes whether it's the end of an episode.

    Args:
        s: State tuple (dealer's first card, player's sum)
        a: Action 'hit' or 'stick'

    Returns:
        Tuple with the next state, reward and a boolean describing
        if the end of episode
    """
    dealer_first_card = s[0]
    player_sum = s[1]
    if a == 'hit':
        player_sum += _draw()
        if player_sum > 21 or player_sum < 1:
            reward = -1
            is_terminal = True
        else:
            reward = 0
            is_terminal = False
    elif a == 'stick':
        dealer_sum = _dealer_move(dealer_first_card)
        if dealer_sum > 21 or dealer_sum < 1:
            reward = 1
        elif player_sum > dealer_sum:
            reward = 1
        elif player_sum == dealer_sum:
            reward = 0
        else:
            reward = -1
        is_terminal = True
    return (dealer_first_card, player_sum), reward, is_terminal


def draw_initial_state() -> Tuple[int, int]:
    dealer = np.random.randint(1, 11)
    player = np.random.randint(1, 11)
    return dealer, player
