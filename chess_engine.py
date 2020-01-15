#!$HOME/miniconda3.bin/python3.7
import numpy as np
import chess #  !pip install python-chess
import chess.pgn as cgn


class game(object):
  def __init__(self):
    game.state =new_game_board()



    def new_game_state(self):
        #here we generate a new game, and    
        FEN_new_game = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
      return serialize_FEN(pgn_new_game)








def serialize_FEN(FEN):
  num_list = ['1','2','3','4','5','6','7','8']
  value_dic = {'r':1, 'n':2, 'b':3, 'k':4, 'q':5, 'p':6,\
               'R':15, 'N':14, 'B':13, 'K':12, 'Q':11, 'P':10 }
  col_dic = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7}
  state = np.zeros((6,8,8))
  temp_state = np.zeros((1,8,8))
  temp_meta1 = np.zeros((1,8,8))
  temp_meta2 = np.zeros((1,8,8))
  #FEN = '4r3/6P1/2p2P1k/1p6/pP2p1R1/P1B5/2P2K2/3r4 b - - 0 45'
  r_i = 0
  for line in FEN.split('/'):
    c_i = 0
    #check if last line:
    if len(line.split(' ')) == 1:
      lastLine = False
    else:
      lastLine = True
      line,turnP,castle,enP,hT,fT = line.split(' ')[:6]
    #parse information
    for c in line:
      if c in num_list:
        #skip number of spaces
        c_i += num_list.index(c)   #we are aware of additional +1 few lines below
      elif c in value_dic.keys():
        temp_state[0,r_i,c_i] = value_dic[c]
      c_i += 1
    r_i += 1
    if lastLine: #parse meta data:
      #save which player's turn it is:
      if turnP.lower() == 'w':
        temp_meta1[0,:,:] = 1
      #save castling rights status:
      if castle == '-':
        pass
      else:
        for c in castle:
          if c == 'k':
            temp_meta2[0,0,6] = 1
          elif c == 'q':
            temp_meta2[0,0,2] = 1
          elif c == 'K':
            temp_meta2[0,7,2] = 1
          elif c == 'Q':
            temp_meta2[0,7,6] = 1
      #save en Passant location (increment temp_state location +1)
      if enP == '-':
        pass
      else:
        c_i = col_dic[enP[0]]
        r_i = num_list.index(enP[1])+1
        temp_meta2[0,r_i,c_i] = 1
      #we skip information on the number of moves.
  #Now we prepare the final output:
  state[5,:,:] = temp_meta2
  state[4,:,:] = temp_meta1
  state[3,:,:] = temp_state//(2**3)
  temp_state = temp_state%(2**3)
  state[2,:,:] = temp_state//(2**2)
  temp_state = temp_state%(2**2)
  state[1,:,:] = temp_state//(2**1)  
  temp_state = temp_state%(2**1)
  state[0,:,:] = temp_state//(2**0)        
  return state,int(hT),int(fT)


def retrieveFEN(state,hT=0,fT=0):
  value_hashmap = {1:'r', 2:'n', 3:'b', 4:'k', 5:'q', 6:'p',\
             15:'R', 14:'N', 13:'B', 12:'K', 11:'Q', 10:'P'}
  #player turn:
  if state[4,:,:].all() == 1 #boolean if white's turn to play
    playerTurn = 'w'
  else:
    playerTurn = 'b'
  #castling:
  castleFlag=''
  if state[5,0,6] == 1:
    castleFlag += 'k'
  if state[5,0,2] == 1:
    castleFlag += 'q'
  if state[5,7,2] == 1:
    castleFlag += 'K'
  if state[5,7,6] == 1:
    castleFlag += 'Q'
  if len castleFlag == 0:
    castleFlag = '-'
  #en Passant:
  enPas=''
  for ir in range(1,7):
    for ic in range(1,7):
      if state[5,ir,ic] == 1:
        enPas += {2:'b',3:'c',4:'d',5:'e',6:'f',7:'g'}[ic+1] 
       enPas += str(ir+1)
  if len enPas == 0:
    enPas = '-'
  #Let's merge the first four pages:
  FEN=''
  state2D=np.zeros((1,8,8))
  for i in range(4):
    state2D += (state[i,:,:])*(2**i)
  #scan each location and determine unit:
  for ir in range(8):
    consec_blanks = 0
    for ic in range(8):
      val = state2D[1,ir,ic]
      if val == 0:
        consec_blanks += 1
      else:
        if consec_blanks > 0:
          FEN += str(consec_blanks)
          consec_blanks = 0
        FEN += value_hasmap[val]
      if ic == 7:
        if consec_blanks > 0:
          FEN += str(consec_blanks)
        if ir < 6: #we dont print after last rank
          FEN += '/'
  #append meta data:
  FEN + ' {} {} {} {} {}'.format(playerTurn,castleFlag,enPas,hT,fT)
  return FEN


def parse_pgn(file_name_list):
  games = []
  total_games = 0
  for file in file_name_list:
    print("opening file {}".format(file.split('/')[-1]))
    with open(file) as pgn:
      game_counter = 0
      while True:
        game_counter += 1
        total_games += 1
        try:
          game = cgn.pgn.read_game(pgn)
        except:
          print('{} games retrieved')
          break
        board = game.board()
        #parse each move:
        for move in game.mainline_moves():


        board = game.board()
  print('{} PGN files parsed. Total of {}games retrieved'.format(\
    len(file_name_list),total_games))






