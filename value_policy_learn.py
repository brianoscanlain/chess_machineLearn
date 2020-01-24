#!/home/brian/miniconda3/bin/python3.7
import os
import random
import numpy as np
import math
import time
import torch.nn.parallel
import torch.optim as optim
import torch.backends.cudnn as cudnn
#from apex import amp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from chess_engine import serialize_FEN


net_config = dict({
    'lr': 0.001,
    'dropout': 0.3, #when not training (i.e. net.eval()), this --> 0 automatically
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels_init': 6,  #descriptor of number of channels of state tensor
    'num_channels_during_nn': 20, #num of channels to grow tensor within nn
    'ngpu':2
})

game_config = dict({
    'name': 'chess',
    'board_size' : [8,8],
    'number_of_actions': 20,
    'playerTurn': True,
    'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
})


train_config = dict({
    'numIter': 'chess',
    'num_epochs' : [8,8],
})




# custom weights initialization called on  netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)




class gameMetaData():
  #Simple container for game meta data. We will have to \
  #perform the game parameters 
  def __init__(self,game_config):
    self.num_actions = game_config['number_of_actions']
    self.board_size = game_config['board_size']
    self.playerTurn = game_config['playerTurn']
    self.canon_fen = game_config['fen']
    self.fen = game_config['fen']
    self.gameOver = False
    self.outcome = None
    self.reward = 0
    self.actions = [] 
  
  def updateOptions(self,num_actions,playerTurn):
    #lesser version of updateFEN()
    self.num_actions = num_actions
    self.playerTurn = playerTurn
    
  def updateFEN(self,fen):
    self.fen = fen 
    game=chess.Board(fen)
    self.gameOver = game.is_game_over()
    if self.gameOver:
      self.actions = []
      self.outcome = game.result()
      self.reward = {'1-0':1, '1/2-1/2':0, '0-1':-1}[game.result()]
    else:
      self.actions = [a.xboard() for a in game.legal_moves]
    self.num_actions = len(self.actions)
    self.playerTurn = game.turn
    #update canonical:
    if self.playerTurn:
      self.canon_fen = fen
    else: 
      self.canon_fen = fen[:fen.rfind('b')] + 'w' + fen[fen.rfind('b'):]

  def retrieveNextState(self,a):
    game=chess.Board(self.fen)
    if game.is_legal(chess.Move.from_uci(a)):
      game.push_xboard(a)
      return game.fen()
    else:
      print("error found, action '{}'".format(a)\
      +" is not considered legal for board:\r\n{}".format(self.fen))
      try:
        game.push_xboard(a)
        return game.fen()
      except:
        return None

  def nextState(self,a):
    fen=retrieveNextState(a)
    updateFEN(fen)
  
  def startNewGame(self):
    starting_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    updateFEN(starting_fen)
  
  def getBoardSize(self):
    return self.board_size
  
  def getActionSize(self):
    return self.num_actions
  
  def __len__(self):
    return self.num_actions
  
  def __getitem__(self,index):
    return self.num_actions, self.playerTurn
   
  def __exit__(self, exc_type, exc_value, traceback):
    pass
  
  def __enter__(self):
    return self

    


class NetVIP(nn.Module):
  def __init__(self, game_handle, net_config):
    super(NetVIP, self).__init__()
    # class netVIP(nn.Module):
    #   def __init__(self, game_info_handle, net_config):
    #     super(netVIP, self).__init__()
    # game params
    self.board_x, self.board_y = game_handle.getBoardSize()
    self.action_size = game_handle.getActionSize()
    #self.config = net_config
    self.nc_init = net_config['num_channels_init']
    self.nc_nn = net_config['num_channels_during_nn']
    self.im2Dh = int(np.ceil(np.sqrt(self.board_x*self.board_y*self.nc_init)))
    self.ngpu =  net_config['ngpu']
    self.dropout = net_config['dropout']
    #
    self.fc1 = nn.Linear(512, self.action_size)
    self.fc2 = nn.Linear(512, 1)
    #
    self.main = nn.Sequential(
      #Block 1:
      Reshape2D(), #3D to 2D reshape with padding (bs,nc,h,w) --> (bs,1,X,X), 
      nn.Conv2d(1, self.nc_nn, 3, stride=1, padding=1),
      nn.BatchNorm2d(self.nc_nn),
      nn.ReLU(True),
      #Block 2:
      nn.Conv2d(self.nc_nn, self.nc_nn, 3, stride=1, padding=1),
      nn.BatchNorm2d(self.nc_nn),
      nn.ReLU(True),
      #Block 3:
      nn.Conv2d(self.nc_nn, self.nc_nn, 3, stride=1),
      nn.BatchNorm2d(self.nc_nn),
      nn.ReLU(True),
      #Block 4:
      nn.Conv2d(self.nc_nn, self.nc_nn, 3, stride=1),
      nn.BatchNorm2d(self.nc_nn),
      nn.ReLU(True),        #   (bs,nc,X-4,X-4)
      #decode block 1: #2D to 1D flatten, followed by linear blocks:
      Flatten(),   #(bs,nc=20,16x16) --> (bs,5120)
      #block 5
      nn.Linear(self.nc_nn*(self.im2Dh-4)*(self.im2Dh-4), 4096),
      nn.BatchNorm1d(4096),
      nn.ReLU(),
      nn.Dropout(p=self.dropout),
      #block 6:
      nn.Linear(4096, 2048),
      nn.BatchNorm1d(2048),
      nn.ReLU(),
      nn.Dropout(p=self.dropout),
      #block 6:
      nn.Linear(2048, 1024),
      nn.BatchNorm1d(1024),
      nn.ReLU(),
      nn.Dropout(p=self.dropout),
      #block 7:
      nn.Linear(1024, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Dropout(p=self.dropout),  #(bs,512) tensor returned.
    )
  
  def forward(self, input):
    if input.is_cuda and self.ngpu > 1:
      output1 = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    else:
      output1 = self.main(input)
    #return output1
    pi = self.fc1(output1) #policies ranging possible actions  
    v = self.fc2(output1)  #
    return F.log_softmax(pi,dim=1),torch.tanh(v)
 



class Reshape2D(nn.Module):
  def forward(self,x):
    batch_size=x.shape[0]
    num_channels=x.shape[1]
    img_height=x.shape[2]
    img_width=x.shape[3]
    new_img_dims =  int(np.ceil(np.sqrt(num_channels*img_height*img_width)))
    pad_len = new_img_dims*new_img_dims - num_channels*img_height*img_width
    #perform flattening, padding, and reshaping
    x = x.view(-1,num_channels*img_height*img_width) #flatten to [bs,:]
    x = F.pad(x, (0,pad_len), 'constant', 0)
    x = x.view(-1,1,new_img_dims,new_img_dims)
    return x

class Flatten(nn.Module):
  def forward(self,x):
    batch_size=x.shape[0]
    num_channels=x.shape[1]
    img_height=x.shape[2]
    img_width=x.shape[3]
    #return x.view(batch_size,-1)  # bad way, as it jumbles up the batches
    return x.view(-1,num_channels*img_height*img_width)





def rand_idx_generator(num,length):
  OUT=[]
  while len(OUT) < num:
    i = int( np.floor(np.random.random()*length) )
    if i not in OUT:
      OUT.append(i)
  return OUT




#DEV and testing:
#1 define game config
#2 init game
device='cuda:0'

game = gameMetaData(game_config)
net = NetVIP(game,net_config).to(device)
net.apply(weights_init)
t = torch.zeros(1,6,8,8)

t = t.to(device)
outputs = net(t)
outputs.size()






















class NNetWrapper(object):
  def __init__(self, game,net_config):
    self.cuda = net_config['cuda']
    if self.cuda:
      self.device = 'cuda:0'
    else:
      self.device = 'cpu'
    self.game = game
    self.nnet = NetVIP(self.game,net_config).to(self.device)
    self.batch_size = net_config['batch_size']
    self.num_epocs = net_config['epochs']
    self.loss_valuation = nn.MSELoss()
    self.loss_policy = nn.MSELoss()
    def train(self, examples):
      """
      examples: list of examples, each example is of form (board, pi, v)
      """
      optimizer = optim.Adam(self.nnet.parameters())
      tic=time.time()

      for epoch in range(self.num_epocs):
        
        print('EPOCH ::: {}, time = {:.1f} seconds'.format(epoch+1,tic-time.time()))
        
        self.nnet.train()
        pi_losses = 0
        v_losses = 0
        counts = 0
        
        data_cycles = int(np.floor(len(examples)/self.batch_size))
        
        for i in range(data_cycles): #cycles of batches
          batch_idx = rand_idx_generator(len(examples), self.batch_size)
          states, pis, vs = list(zip(*[examples[b_idx] for b_idx in batch_idx]))
          #boards = torch.FloatTensor(np.array(boards).astype(np.float64))
          #target_pis = torch.FloatTensor(np.array(pis))
          #target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
          # predict
          if self.cuda:
            states, target_pis, target_vs = states.contiguous().cuda(), \
              target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
          
          # retrive neural network prediction:
          out_pi, out_v = self.nnet(boards)
          l_pi = self.loss_policy(pis, out_pi)
          l_v = self.loss_valuation(vs, out_v)
          total_loss = l_pi + l_v
          # record loss
          pi_losses += l_pi.item()
          v_losses += l_v.item()
          counts += states.size(0)
          # compute gradient and do SGD step
          optimizer.zero_grad()
          total_loss.backward()
          optimizer.step()
          # measure elapsed time
          if i%int(data_cycles/4) == 0:
            print('i={}/{}, v_losses={:.6f}, pi_losses={:.6f},  {:.1f} secs elapsed)'.format(\
              i+1,data_cycles, pi/((i+1)*batchSize)*100,time.time()-tic))
          
        #epoch stats:
        epoch_loss_v = v_losses / counts
        epoch_loss_p = pi_losses / counts
        epoch_loss_tot = epoch_loss_p + epoch_loss_v 
        print('epoch {} finished, total loss: {:.4f}'.format(\
          epoch, epoch_loss_tot) + \
          ', val_loss: {:.4f},val_loss: {:.4f}'.format(\
          epoch_loss_v,epoch_loss_p))
      #return updated nnet          
   
  def predict(self, fen):
      #retrieve estimate
      tic = time.time()
      self.nnet.eval()

      # preparing input
      state = torch.stack([torch.from_numpy(serialize_FEN(fen))]) 
      if self.cuda: 
        state = state.contiguous().cuda()
      #retrieve estimates:
      with torch.no_grad():
        pi, v = self.nnet(board)
      return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

  def loss_pi(self, targets, outputs):
      return -torch.sum(targets*outputs)/targets.size()[0]

  def loss_v(self, targets, outputs):
      return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]

  def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
      filepath = os.path.join(folder, filename)
      if not os.path.exists(folder):
          print("Checkpoint Directory does not exist! Making directory {}".format(folder))
          os.mkdir(folder)
      else:
          print("Checkpoint Directory exists! ")
      torch.save({
          'state_dict' : self.nnet.state_dict(),
      }, filepath)

  def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
      # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
      filepath = os.path.join(folder, filename)
      if not os.path.exists(filepath):
          raise("No model in path {}".format(filepath))
      map_location = None if self.cuda else 'cpu'
      checkpoint = torch.load(filepath, map_location=map_location)
      self.nnet.load_state_dict(checkpoint['state_dict'])































mcts_config = dict({
    'numMCTSSims': 1000,
})








class MCTS():
  def __init__(self, game, mcts_config):
    self.game = game  #this game handle provides functionality
    self.config = mcts_config
    self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
    self.Nsa = {}       # stores #times edge s,a was visited
    self.Ns = {}        # stores #times board s was visited
    self.Pi = {}        # stores initial policy (returned by neural net)
    self.Ts = {}        # stores game.getGameEnded ended for board s
    self.Vs = {}  
  
  def getActionProb(self,fen, temp=1):
    """
    This function performs numMCTSSims simulations of MCTS starting from
    canonicalBoard.
    Returns:
        probs: a policy vector where the probability of the ith action is
               proportional to Nsa[(s,a)]**(1./temp)
    """
    for i in range(self.config.numMCTSSims):
      self.search(fen)  #populate Qsa, Nsa, Ns, Pi
    #    

    #
    if temp==0:
      bestA = np.argmax(self.Nsa[fen])
      probs = [0]*len(self.Nsa[fen])
      probs[bestA]=1
      return probs
    #
    counts = [a**(1./temp) for a in self.Nsa[fen]]
    counts_sum = float(sum(counts))
    probs = [x/counts_sum for x in counts]
    return probs

  def search(self, fen_state,nnet):
    #update game engine with current state:
    self.game.updateFEN(fen_state)
    #if game over, return reward:
    if self.game.gameOver:   #reward winning player (1 if player w wins)
      reward = {'1-0':1,'1/2-1/2':0,'0-1':-1}[self.game.result()]
      return reward
    
    if fen_state not in self.Ns.keys(): #unseen state found!
      self.Ns[fen_state] = 1
      self.Nsa[fen_state]=[0]*self.game.num_actions #just simply initializing here
      self.Qsa[fen_state]=[0]*self.game.num_actions #just simply initializing here
      p,v = nnet.predict(serialize_FEN(fen_state))
      self.Pi[fen_state] = p.squeeze().tolist()
      return v
    
    #determine best move (using evolving estimates of QSA, PSA, NSA)
    max_u, best_a = -float("inf"),-1
    for a in self.game.actions:
      u = self.Qsa[fen_state][a] + \
        self.c_puct*self.Pi[fen_state][a]*sqrt(\
          sum(self.Ns[fen_state]))/(1+self.Nsa[fen_state][a])
      if u>max_u:
        max_u = u
        best_a = a
    a = best_a
    
    #perform next move:
    fen_state_future = self.game.retrieveNextState(a)
    game.update(fen_state_future)
    v = search(fen_state_future)
    self.Qsa[fen_state][a] = (self.Nsa[fen_state][a]*self.Qsa[fen_state][a] + v)/\
                             (self.Nsa[fen_state][a]+1) # = v if Nsa = 0
    self.Nsa[fen_state][a] +=1
    return v




def spawnBots(game_config,net_config. train_config):   # formerly known as "policyIterSP"
  game = gameMetaData(game_config)
  nnet=NNetWrapper(game, net_config) #init a game bot
  nnet_alt = NNetWrapper(game, net_config) #init a game bot
  examples = []
  #put bots to work:
  for i in range(net_config.numIters):
    for e in range(net_config.numEpochs):
      examples += spawnBotGame()
    #each iteration, let's train
    self.nnet.train(examples)
    frac_win = pit(new_nnet, nnet)
    if frac_win > threshold:
      nnet = new_nnet
    return nnet









class Coach():
  def __init(self,game_config,net_config,train_config):
    self.game = gameMetaData()
    self.nnet1 = NNetWrapper(game, net_config) #init a game bot
    self.nnet2 = self.nnet.__class__(self.game) #alternative bot
    self.mcts = mcts = MCTS(game, mcts_config)
    self.trainExampleHistory = []
    self.skipFirstSelfPlay = False
    self.tempThreshold

  
  def spawnBotGame(self):   #formerly known as "executeEpisode"
    recorder = []
    self.game.startNewGame()
    #
    while True:
      for _ in range(self.mcts.config.numMCTSims):
        mcts.search(self.game.fen, self.game, self.nnet2) #build Qtables
      recorder.append([s, self.mcts.pi(s), self.game.playerTurn]) #retrieve and store
      a = np.random.choice(len(self.mcts.pi(s)),p=self.mcts.pi(s)) #choose action
      s = self.game.nextState(a) #perform action
      if game.gameOver(s):
        #we now assign the reward. Each action the winner performed gets
        #a +1, while the loosing actions get assigned -1. Draw reward = 0
        recorder = [(x[0], x[1], game.result*((-1)**x[2]))  for x in examples]
        return recorder

  def botBattle(self):
    #Here we will let the two n
    self.game.startNewGame()








