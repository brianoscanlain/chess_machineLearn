#!/home/brian/miniconda3/bin/python3.7

'''
This script is focused on running a neural network for 
learning how to play chess!

We will start with pytorch.
First we will learn from game states without introducing
temporal learning (multi-action and multiple future states).

We have established a method to serialize chess-board 
states from 2-Million game pool, all of which are one 
of a win, loss or draw. 

While there are multiple moves per game, this raises the
question on how best we associate game moves and the 
final game outcome. With some playing around, I have 
observed higher training accuracies when the moves 
associated with win/loose games are assigned neutral
reward for the first fraction of the game timeline
and the remaining (mature) fraction with the game
result.

Reward functions - as mentioned by the previous 
paragraph, the moves specific to each game can
be assigned rewards in a number of ways. Here are
the functions which I have defined, and considered:
+ constant  -  every move in a game is assigned the
               game result.
+ linear    -  the first 50% of game moves are 
               prescribed a neutral result (i.e. 0).
               The remaining moves are assigned the
               game result.
+ concave   -  the first 25% of games are assigned 
               neutral result, and remaining 75% are
               assigned game result.
+ convex    -  the first 75% of games are assigned 
               neutral result, and remaining 25% are
               assigned game result.

The advantages/disadvantages of each reward function
are not easily predicted. One big motivator for me was 
that "constant" function confusingly issues three 
rewards (1 for win, -1 for loss, and 0 for draw) for 
the starting board state. This is confusing, and to
my intuition, will play havoc with our machine learning
training.



Soon to follow, we will implement temporal learning 
framework, like SARSA, among others!

Brian Scanlon, 15 January, 2020
'''