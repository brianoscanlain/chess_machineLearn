#!/$HOME/miniconda3/bin/python3.7

'''let's try to build a chess engine'''

1. we need data  (pgn = portable game notation, plenty of)

2. we need to build a chess engine:
   + state representation (as some array of numbers 
     (so as to appeal to CNN's))
   + import of pgn's
   + list of legal actions
   + extract results (WIN/LOOSE/DRAW)
   + is game over?
   + which player's turn?

3. need to build CNN for learning from the older data.

4. Incorporate reinforcement learning with temporal
   learning capability (predicting next move and beyond)


   '''


Resources:
==========

PGN: -- https://en.wikipedia.org/wiki/Portable_Game_Notation
-------
It is an ascii format for recording chess games (board state
and associated game metadata). It was  devised circa 1993 by Steven 
Edwards, and today is supported by many chess programs.

1. set of tag pairs container [] (data describing:
                                                  Event,
                                                  Site,
                                                  Date,
                                                  Round,
                                                  White,
                                                  Black,
                                                  Result)

    NOTE 1: result important, has three possible values:
        '1-0'      (White won)
        '0-1'      (Black won)
        '1/2-1/2'  (draw)
    NOTE 2: two players, distinguished by their assigned 
         colour (black or white). The PGN notation takes
         'white' as a primary player and 'black' comes 
         after. 
     NOTE 3: White player performs the first move

2. There exists a chess library for python, which provides
   an engine and import (or interpreter) for PGN files.
   https://python-chess.readthedocs.io/


CHESS     https://en.wikipedia.org/wiki/Chess
----
+ Two player game
+ 8x8  position board
+ believed to be a derivation of an Indian game, sometime
  before 600 A.D>
+ Game is observable by both players (no occlusions of game
  state components)
+ 32 pieces, 16 per player:
    1 king
    1 Queen
    2 rooks
    2 knights
    2 bishops
    8 pawns  

+ win objective is to achieve a check-mate on the opponents 
  King, which is a scenario where the king is placed under an
  inescapable threat of capture.

+ Rules book https://en.wikipedia.org/wiki/FIDE




pyhton3.7 -m pip install python-chess


