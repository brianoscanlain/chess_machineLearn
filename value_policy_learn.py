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
    'dropout': 0.3, #we can set this --> 0 when we are not training
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
    'playerTurn': 'w',
    'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
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
    self.fen = game_config['fen']
    self.gameOver = False
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
    else:
      self.actions = [a.xboard() for a in game.legal_moves]
    self.num_actions = len(self.actions)
    self.playerTurn = 'w' if game.turn else 'b'
  
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
t = torch.zeros(10,6,8,8)

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



