#!/$HOME/miniconda3/bin/python3.7

'''let's try to build a chess engine'''

1. we need data  (pgn = portable game notation, plenty of)
   --KingBase2018-pgn (429MB) offers over 2 Million games.
    https://archive.org/downloads/KingBase2018/
   --KingBase2019-pgn (458MB)
    https://archive.org/downloads/KingBase2019/


   There also exists other game (and board state) formats
   for example the Forsyth-Edwards Notation (FEN) is a 
   standard notation.

2. we need to build a chess engine:
   + state representation (as some array of numbers 
     (so as to appeal to CNN's))
   + import of pgn's
   + list of legal actions
   + extract results (WIN/LOOSE/DRAW)
   + is game over?
   + which player's turn?
  --we can use python-chess module to grealy reduce 
    development of a chess engine.

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

2. Movetext 
   This describes the moves of the game, number indicators,
   followed by one or three "."; one if next move is white's
   move, three if it is black's turn, using Standard Algebraic
   Notation (SAN)

   SAN -- x = capture
          " " = single-letter character, representing the board
                piece (K (king), Q (queen), R (rook), B (bishop),
                       N (knight)). Pawn is usually not
                       represented, or can be represented by P.
          "  " = two-character algebraic name of the final square
                 the piece moved to. 
      
      -- When abiguity arises, the initial starting position
         of the piece can be provided.
      -- "O-O" = kingside castling
      -- "O-O-O" = queenside castling
      --"=" = pawn promotions
      --"#" = checking move

3. Comments and annotations:
   + comments are contained within {}
   + Annotations from commentators are enclosed in ()
   (these need to be ignored).


---------------------------------------------------------                
              (BLACK PLAYER)
    __________________________________
    8|   |   |   |   |   |   |   |   |
    ----------------------------------
    7|   |   |   |   |   |   |   |   |
    ----------------------------------
    6|   |   |   |   |   |   |   |   |
    ----------------------------------
    5|   |   |   |   |   |   | g5|   |
    ----------------------------------
    4|   |   |   |   |   |   |   |   |
    ----------------------------------
    3|   |   | c3|   |   |   |   |   |
    ----------------------------------
    2|   | b2|   |   |   |   |   |   |
    ----------------------------------
    1|   |   |   |   |   |   |   |   |
    ----------------------------------
       a   b   c   d   e   f   g   h

               (WHITE PLAYER)

---------------------------------------------------------


3. There exists a chess library for python, which provides
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
  
  flags:
  ------
  Castling (rights and achieved status) Kingside & queenside
  "en passant" available or not?

+ win objective is to achieve a check-mate on the opponents 
  King, which is a scenario where the king is placed under an
  inescapable threat of capture.

+ Rules book https://en.wikipedia.org/wiki/FIDE

+ Castling - a move invloving a player's king and either player's
             original rooks. It is the only move in chess in which a 
             player moves two pieces in the same move, and it is the
             only move where a piece (other than a Knight) can jump
             over another.

             It involves moving the king two squares towards a rook
             on the players first rank, then moving the rook to
             the square over which the king crossed.

             Requirements:
             1. King and the chosen rook are on the player's 
                first rank
             2. Neither the King nor the chosen have have been
                previously moved.
             3. There are no pieces between the king and the 
                chosen rook.
             4. That the king is not currently in check.
             5. That the king does not pass through a square that is 
                attacked by an enemy piece.
             6. That the king does not end up in check.


pyhton3.7 -m pip install python-chess






state representation:
location specific:
+ 6*2 types of pieces (rnbqkpRNBKQP)
+ 1*2 en passant
+ blank space
=> 15 discrete values per square of chess board

meta flags:
+ 2*1 player turn (black or white)
+ 4*2 castling rights available (queenside & kingside)
+ check

2**4 = 16, so 16x16x4
+
16x16x1 to store the meta flags.




