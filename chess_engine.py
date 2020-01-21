#!/home/brian/miniconda3/bin/python3.7
import pandas as pd
import numpy as np
import chess #  !pip install python-chess
import chess.pgn as cgn
from itertools import count
import time
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
# class game(object):
#   def __init__(self):
#     game.state =new_game_board()



#     def new_game_state(self):
#         #here we generate a new game, and    
#         FEN_new_game = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
#       return serialize_FEN(pgn_new_game)








def serialize_FEN(FEN):
  if type(FEN) is not str:
    print('FEN type not known:{}'.type(FEN))
    return None
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
        r_i = num_list.index(enP[1])
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
  return state.astype('float32'),int(hT),int(fT)


def deserialize(state,hT=0,fT=0):
  value_hashmap = {1:'r', 2:'n', 3:'b', 4:'k', 5:'q', 6:'p',\
             15:'R', 14:'N', 13:'B', 12:'K', 11:'Q', 10:'P'}
  #player turn:
  if state[4,:,:].all() == 1: #boolean if white's turn to play
    playerTurn = 'w'
  else:
    playerTurn = 'b'
  #castling:
  castleFlag=''
  if state[5,7,2] == 1:
    castleFlag += 'K'
  if state[5,7,6] == 1:
    castleFlag += 'Q'
  if state[5,0,6] == 1:
    castleFlag += 'k'
  if state[5,0,2] == 1:
    castleFlag += 'q'
  if len(castleFlag) == 0:
    castleFlag = '-'
  #en Passant:
  enPas=''
  for ir in range(1,7):
    for ic in range(1,7):
      if state[5,ir,ic] == 1:
        enPas += {2:'b',3:'c',4:'d',5:'e',6:'f',7:'g'}[ic+1] 
        enPas += str(ir+1)
  if len(enPas)== 0:
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
      val = state2D[0,ir,ic]
      if val == 0:
        consec_blanks += 1
      else:
        if consec_blanks > 0:
          FEN += str(consec_blanks)
          consec_blanks = 0
        FEN += value_hashmap[val]
      if ic == 7:
        if consec_blanks > 0:
          FEN += str(consec_blanks)
        if ir < 7: #we dont print after last rank
          FEN += '/'
  #append meta data:
  FEN += ' {} {} {} {} {}'.format(playerTurn,castleFlag,enPas,hT,fT)
  return FEN


def test():
  print('Testing the serialization of FEN,' + \
    ' and subsequenst deserialization for data loss')
  FEN = []
  FEN.append('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
  FEN.append('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1')
  FEN.append('rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2')
  FEN.append('rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2')
  FEN.append('r6k/2R5/6R1/pp1Ppp2/8/Pn2B1Pr/4KP2/8 w - - 0 1')
  FEN.append('fen = 6R1/1r3rp1/1kp2pR1/p1p1p3/P3P3/1PKP1P2/2P5/8 w - - 10 44')
  FEN.append('rnbqkb1r/pp1ppppp/2p2n2/8/2P5/2N2N2/PP1PPPPP/R1BQKB1R b KQkq - 1 3')
  FEN.append('r1bqkbnr/pp1p1ppp/2n1p3/2p5/4PP2/3P4/PPP1B1PP/RNBQK1NR b KQkq - 0 4')
  FEN.append('rnbqk2r/pp2ppbp/3p1np1/8/2PNP3/2N5/PP2BPPP/R1BQK2R b KQkq - 1 7')
  FEN.append('5rk1/1br1nppp/1p1qpn2/pP1p4/3P1PP1/P2BP3/5N1P/RN1QK2R w KQ - 3 17')
  FEN.append('3q1r1k/p2n2pp/2n2b2/2P5/3pQ3/1B1N3P/P1PB2P1/3R2K1 b - - 10 33')
  #FEN.append('')
  for i in range(len(FEN)):
    st,ht,ft = serialize_FEN(FEN[i])
    fen_est = retrieveFEN(st,ht,ft)
    if fen_est == FEN[i]:
      print('Test {}: Pass'.format(i+1))
    else:
      print('Test {}: fails...'.format(i+1))
      print('\toriginal FEN: ' + FEN[i])
      print('\testimate FEN: ' + fen_est)
      for ic,co,ce in zip(count(),FEN[i],fen_est):
        if co != ce:
          print('Difference at char {}; {} != {}'.format(ic, ce,co))



  

def parse_pgn_bulk(file_name_list,reward_function='constant',ID=0):
  if reward_function not in ['constant','linear','convex','concave']:
    print('reward function not recognised: {}'.format(reward_function))
    return -1    
  #load all games, then each move. We will save each board state,
  #and append the win value also.
  '''
  There are different reward functions we can apply. We can 
  associate the moves of a winning game ranging from all win
  to partial win-partial draw. The fraction of the draw values
  are controlled by specifying the reward function:

  constant, all moves (start to end) are considered winning
  linear, first half of moves are considered draw, with remaining\
          moves assigned maximum reward.
  inv power, only last 25% of moves are considered winning\
  exp
  '''
  result_hashmap = {'1-0':1,'1/2-1/2':0,'0-1':-1} # white wins =1, draw = 0. black wins = -1
  state = []
  game_number = []
  move_number = []
  game_result = []
  #
  total_games = 0
  tic = time.time()
  for file in file_name_list:
    print("opening file {}, {:.2f} sec elapsed".format(\
      file.split('/')[-1],time.time()-tic))
    with open(file) as pgn:
      game_counter = 0
      while True:
        move_counter = 0
        #print stats:
        if total_games%100000 == 0 and total_games > 0:
          print('{} games processed, {} sec elapsed.'.format(\
            total_games,time.time()-tic))
          #break  #just for testing
        try:
          game = cgn.read_game(pgn)
        except:
          print('{} games retrieved, {:.2f} sec elapsed.'.format(total_games,time.time()-tic))
          break
        #game loaded, let's parse:
        try:
          result = result_hashmap[game.headers['Result']]
          #print(result)
        except:
          print('game: {} unknown game result: {}'.format(\
            game_counter,game.headers['Result']))
        board = game.board()
        #record initial  state:
        game_number.append(total_games)
        move_number.append(move_counter) #we will define state_numer=-1 as a starting board
        state.append(board.fen())
        #parse each move:
        for move in game.mainline_moves():
          board.push(move)
          move_counter += 1
          game_number.append(total_games)
          move_number.append(move_counter)
          state.append(board.fen())
        #prepare results:
        num_board_state = move_counter + 1
        if (reward_function == 'constant') or (result == 0):
          game_result.extend([result]*num_board_state)
        elif reward_function == 'linear':
          game_result.extend(np.array(result*np.arange(0,num_board_state)\
            /(num_board_state-1)).round().astype('int8').tolist())
        elif reward_function == 'convex':
          game_result.extend(np.array([((x/(num_board_state-1))**(2.5))*result \
            for x in np.arange(num_board_state)]).round().astype(\
            'int8').tolist())
        elif reward_function == 'concave':
          game_result.extend(np.array([((x/(num_board_state-1))**(0.5))*result \
            for x in np.arange(num_board_state)]).round().astype(\
            'int8').tolist())
        game_counter += 1
        total_games += 1
  #Save the data:
  # game = pd.DataFrame({'game_number':game_number,
  #               'move_number':move_number,
  #               'result':game_result,
  #               'fen':state})
  #game.to_csv('{}.csv'.format(output_filename))
  return ID,game_number,move_number, game_result, state



def main(parallel=False,numProcs=16):
  tic=time.time()
  pgn_files = glob('*.pgn')
  print('{} pgn files found'.format(len(pgn_files)))
  #let's build a training set of games:
  reward = 'constant'
  if int(len(pgn_files)) < numProcs:
    numProc = len(pgn_files)
  else:
    numProc = numProcs
  print(numProc)
  if parallel != True:
    _,game_number, move_number, game_result,state = \
    parse_pgn_bulk(pgn_files,reward_function = reward) 
  else:
    pool = ProcessPoolExecutor(numProc)
    futures=[]
    OUTPUT={}
    for i,file in zip(range(numProc),pgn_files):
      file_list = [file]
      futures.append(pool.submit(parse_pgn_bulk,file_list,reward,i))
    for x in as_completed(futures):
      result = x.result()
      OUTPUT[result[0]] = result[1:]
    for i in range(numProc):
      if i == 0:
        game_number = OUTPUT[i][0]
        move_number = OUTPUT[i][1]
        game_result = OUTPUT[i][2]
        state       = OUTPUT[i][3]
      else:
        game_number.extend((np.array(OUTPUT[i][0])+game_number[-1]).tolist())
        move_number.extend(OUTPUT[i][1])
        game_result.extend(OUTPUT[i][2])
        state.extend(OUTPUT[i][3])
    game = pd.DataFrame({'game_number':game_number,
      'move_number':move_number,
      'result':game_result,
      'fen':state})
    print('number of games is {}'.format(len(game)))
    print('{:.2f} seconds elapsed'.format(time.time()-tic))
  game.to_csv('{}{}.csv'.format('training_data_',reward))



      


      
      

if __name__ == '__main__':
  run_in_parallel = True
  number_of_processors = 16
  main(run_in_parallel,number_of_processors)





