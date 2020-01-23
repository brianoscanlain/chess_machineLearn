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
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 6,
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
    self.num_channels = net_config['num_channels']
    self.ngpu =  net_config['ngpu']
    #
    self.fc1 = nn.Linear(512, self.action_size)
    self.fc2 = nn.Linear(512, 1)
    #
    self.main = nn.Sequential(
      #Block 1:
      Reshape2D(), #3D to 2D reshape with padding (bs,nc,h,w) --> (bs,1,X,X), 
      nn.Conv2d(1, self.num_channels, 3, stride=1, padding=1),
      nn.BatchNorm2d(self.num_channels),
      nn.ReLU(True),
      # #Block 2:
      # nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1, padding=1),
      # nn.BatchNorm2d(self.num_channels),
      # nn.ReLU(True),
      # #Block 3:
      # nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1),
      # nn.BatchNorm2d(self.num_channels),
      # nn.ReLU(True),
      # #Block 4:
      # nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1),
      # nn.BatchNorm2d(self.num_channels),
      # nn.ReLU(True),           (bs,nc,X-4,X-4)
      #decode block 1:
      # Flatten(),
      # nn.Linear(self.num_channels*16*16, 1024),
      # nn.BatchNorm2d(self.num_channels),
      # F.ReLU(),
      # F.dropout(p=self.dropout, training=self.training),
      # #decode block 2:
      # nn.Linear(1024, 512),
      # nn.BatchNorm2d(512),
      # F.ReLU(),
      # F.dropout(p=self.dropout, training=self.training),
      # #
      # nn.Linear(512, self.action_size),
      # nn.Linear(512, 1)
    )
  
  def forward(self, input):
    if input.is_cuda and self.ngpu > 1:
      output1 = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    else:
      output1 = self.main(input)
    return output1
    #pi = self.fc1(output1) #policy                                                                        # batch_size x action_size
    #v = self.fc2(output1)  #
    #return F.log_softmax(pi,dim=1),torch.tanh(v)
 



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






















class NNetWrapper(NeuralNet):
  def __init__(self, game,ngpu):
    self.nnet = netVP(ngpu).to(device)
    self.board_x, self.board_y = game.board_size
    self.action_size = game.num_actions
    
    if args.cuda:
      self.nnet.cuda()
    
    def train(self, examples):
      """
      examples: list of examples, each example is of form (board, pi, v)
      """
      optimizer = optim.Adam(self.nnet.parameters())
      
      for epoch in range(args.epochs):
        print('EPOCH ::: ' + str(epoch+1))
        self.nnet.train()

        pi_losses = 0
        v_losses = 0
        counts = 0
        
        data_cycles = int(np.floor(len(examples)/args.batch_size))
        
        for i in range(data_cycles):
          batch_idx = rand_idx_generator(len(examples), args.batch_size)
          boards, pis, vs = list(zip(*[examples[i] for i in batch_idx]))
          boards = torch.FloatTensor(np.array(boards).astype(np.float64))
          target_pis = torch.FloatTensor(np.array(pis))
          target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
          # predict
          if args.cuda:
            boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
          # measure data loading time
          
          # compute output
          out_pi, out_v = self.nnet(boards)
          l_pi = self.loss_pi(target_pis, out_pi)
          l_v = self.loss_v(target_vs, out_v)
          total_loss = l_pi + l_v
          # record loss
          pi_losses += l_pi.item()
          v_losses += l_v.item()
          counts += boards.size(0)
          # compute gradient and do SGD step
          optimizer.zero_grad()
          total_loss.backward()
          optimizer.step()
          # measure elapsed time
          batch_time = time.time() - end
          end = time.time()
          batch_idx += 1
          # plot progress
          bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                      batch=batch_idx,
                      size=int(len(examples)/args.batch_size),
                      data=data_time.avg,
                      bt=batch_time.avg,
                      total=bar.elapsed_td,
                      eta=bar.eta_td,
                      lpi=pi_losses.avg,
                      lv=v_losses.avg,
                      )
          bar.next()
        bar.finish()
   
  def predict(self, fen):
      """
      board: np array with board
      """
      # timing
      start = time.time()
      
      # preparing input
      board = torch.stack([torch.from_numpy(serialize_FEN(fen))])
        
      if args.cuda: board = board.contiguous().cuda()
      board = board.view(1, self.board_x, self.board_y)
      self.nnet.eval()
      with torch.no_grad():
        pi, v = self.nnet(board)

      #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
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
      map_location = None if args.cuda else 'cpu'
      checkpoint = torch.load(filepath, map_location=map_location)
      self.nnet.load_state_dict(checkpoint['state_dict'])





def rand_idx_generator(num,length):
  OUT=[]
  while len(OUT) < num:
    i = int( np.floor(np.random.random()*length) )
    if i not in OUT:
      OUT.append(i)
  return OUT
