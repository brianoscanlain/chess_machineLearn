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
               game result
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
import numpy as np
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils import data
from apex import amp
from chess_engine import serialize_FEN
# custom weights initialization called on  netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)






class netVP(nn.Module):
  def __init__(self, game, args):
    # game params
    self.board_x, self.board_y = game.getBoardSize()
    self.action_size = game.getActionSize()
    self.args = args
    
    super(netA, self).__init__()
    self.ngpu = ngpu
    self.fc3 = nn.Linear(512, self.action_size)
    self.fc4 = nn.Linear(512, 1)
    self.main = nn.Sequential(
      #Block 1:
      nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1),
      nn.BatchNorm2d(args.num_channels),
      F.ReLU(),
      #Block 2:
      nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1),
      nn.BatchNorm2d(args.num_channels),
      F.ReLU(),
      #Block 3:
      nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1),
      nn.BatchNorm2d(args.num_channels),
      F.ReLU(),
      #Block 4:
      nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1),
      nn.BatchNorm2d(args.num_channels),
      F.ReLU(),
      #decode block 1:
      Flatten(),
      nn.Linear(args.num_channels*4*4, 1024),
      nn.BatchNorm2d(args.num_channels),
      F.ReLU(),
      F.dropout(p=self.args.dropout, training=self.training),
      #decode block 2:
      nn.Linear(1024, 512),
      nn.BatchNorm2d(512),
      F.ReLU(),
      F.dropout(p=self.args.dropout, training=self.training),
      #
      nn.Linear(512, self.action_size),
      nn.Linear(512, 1)
    )
  
  def forward(self, input):
    if input.is_cuda and self.ngpu > 1:
      output1 = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    else:
      output1 = self.main(input)
    pi = self.fc1(output1) #policy                                                                        # batch_size x action_size
    v = self.fc1(output1)  #
    return F.log_softmax(pi,dim=1),torch.tanh(v)
 




class NetA(nn.Module):
  def __init__(self, ngpu,nc,ndf):
    super(NetA, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
      #Convolution block #1::    input: batch_size,6,8,8
      nn.Conv2d(nc, ndf, 1, 1, 0, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      #Convolution block #2::    input: batch_size,ndf*2,8,8
      nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False),
      nn.BatchNorm2d(ndf * 2),
      nn.LeakyReLU(0.2, inplace=True),
      #Convolution block #3::    input: batch_size,ndf*4,8,8
      nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False),
      nn.BatchNorm2d(ndf * 4),
      nn.LeakyReLU(0.2, inplace=True),
      #Convolution block #4::    input: batch_size,ndf*4,8,8
      nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
      nn.BatchNorm2d(ndf * 8),
      nn.LeakyReLU(0.2, inplace=True),
      #
      #Decoding Block :: input: batch_size,ndf*8,8,8
      nn.Conv2d(ndf * 8, ndf*4, 3, 1, 1, bias=False), #128,64,8,8
      nn.MaxPool2d((2,2)), #128,64,4,4
      Flatten(), #128,1024
      # #nn.view(-1,ndf*4*4*4)
      nn.Linear(ndf*4*4*4,128), #
      nn.Sigmoid(),    
      nn.Linear(128,64),
      nn.Linear(64,1),
      )
   
  def forward(self, input):
    if input.is_cuda and self.ngpu > 1:
      output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    else:
      output = self.main(input)
    return output.squeeze(1)#.view(-1, 1).squeeze(1)
 



class NetB(nn.Module):
  def __init__(self, ngpu,nc,ndf):
    super(NetB, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
      #Convolution block #1::    input: batch_size,6,8,8
      nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
      nn.BatchNorm2d(ndf),
      nn.ReLU(),
      #Convolution block #2::    input: batch_size,ndf,8,8
      nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False),
      nn.BatchNorm2d(ndf * 2),
      nn.ReLU(),
      #Convolution block #3::    input: batch_size,ndf*2,8,8
      nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False),
      nn.BatchNorm2d(ndf * 4),
      nn.ReLU(),
      #Convolution block #4::    input: batch_size,ndf*4,8,8
      nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
      nn.BatchNorm2d(ndf * 8),
      nn.ReLU(),
      #Convolution block #5::    input: batch_size,ndf*8,8,8
      #nn.Conv2d(ndf * 8, ndf*4, 3, 1, 1, bias=False), #128,64,8,8
      #
      #Decoding Block :: input: batch_size,ndf*8,8,8
      nn.MaxPool2d((2,2)), #reduce im H,W to 4x4
      Flatten(), #128,1024
      nn.Linear(ndf*8*4*4,128), 
      nn.Sigmoid(),    
      nn.Linear(128,1),
      #nn.Linear(64,1),
      )
   
  def forward(self, input):
    if input.is_cuda and self.ngpu > 1:
      output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    else:
      output = self.main(input)
    return output.squeeze(1)#.view(-1, 1).squeeze(1)
 


class NetD(nn.Module):
  def __init__(self, ngpu,nc,ndf):
    super(NetD, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
      #Convolution block #1::    input: batch_size,6,8,8
      nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # # state size. (ndf) x 32 x 32
      nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False),
      nn.BatchNorm2d(ndf * 2),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*2) x 16 x 16
      nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False),
      nn.BatchNorm2d(ndf * 4),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*4) x 8 x 8
      nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
      nn.BatchNorm2d(ndf * 8),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*8) x 4 x 4
      nn.Conv2d(ndf * 8, ndf*4, 3, 1, 1, bias=False), #128,64,8,8
      nn.MaxPool2d((2,2)), #128,64,4,4
      #Decoding:
      Flatten(), #128,1024
      # #nn.view(-1,ndf*4*4*4)
      nn.Linear(ndf*4*4*4,128), #6x6 from image dimension
      nn.Sigmoid(),    
      nn.Linear(128,64),
      nn.Linear(64,1),
      )
   
  def forward(self, input):
    if input.is_cuda and self.ngpu > 1:
      output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    else:
      output = self.main(input)
    return output.squeeze(1)#.view(-1, 1).squeeze(1)
 

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




class dataload(object):
  '''This class is custom, and provides batch-sample retrieval of
     data to feed the training. The data is loaded, in a compressed
     format (ASCII strings for each chess move -- FEN format). The
     FEN strings are serialized to a (batch_sizex6x8x8) tensor
     on the fly to minimize file size and memory usage, at the
     expense of CPU computation.

     For my setup, I use the KingBase-2019 dataset, which has 3.6M
     board states, saved as FEN format strings to a single CSV file
     of size 2.9 GB, prepared by accompanying chess_engine.py. 

     The data file is loaded, randomized and split into training
     and validation data. It is stored in memory (~6 GB).

     Usage (see usage example in "train()" function in this script)

     1. initialize the class (CSV filename specified, 
                              split ratio specified)
     
     2. set the phase ('train' for training purposes,
                       'test' for validation purposes)

     3. retrieve batch of data using "get_batch()"
   
     4. repeat steps 2&3 sequentially until you are satisfied that
        the dataset has been represented (e.g. A method I employ is 
        by defining data_cycles in "train()", which specifies the 
        numer ot times to grab data)
  '''
  def __init__(self,csv_filename,train_test_ratio):
    DF = pd.read_csv(csv_filename)
    #drop nans:
    df_len = len(DF)
    DF = DF.dropna()
    # n_nans = df_len - len(DF)
    # if n_nans > 0:
    #   print('{} nan entries removed. {} games loaded'.format(n_nans,len(DF)))
    pivotIdx = int(len(DF)*train_test_ratio) - 1
    DF = DF.reindex(np.random.permutation(DF.index))  
    train = {\
       'result':DF['result'][:pivotIdx].values.tolist(),\
       'fen':DF['fen'][:pivotIdx].values.tolist()}
    test = {\
       'result':DF['result'][pivotIdx:],\
       'fen':DF['fen'][pivotIdx:]}
    # test = test.reindex(range(len(test)))
    # train = train.reindex(range(len(train)))
    self.test_dataset  = test 
    self.train_dataset = train 
    self.x=None
    self.y=None
    self.phase='train'
    self.size=[6,8,8]
    self.len = len(self.train_dataset['fen'])
  
  def set_phase(self,phase):
    #configure for the phase of the runtime: (training, or validation)
    if type(phase) == str:
      phase = phase.lower()
      if phase in ['train','test','validate', 'val']:
        if phase == 'train':
          self.phase = 'train'
          self.len = len(self.train_dataset['fen'])
        else:
          self.phase = 'test'
          self.len = len(self.test_dataset['fen'])
      else:
        print('phase must be: "train" or "test".')  
    else:
      print('phase must be a string.')
  
  def get_batch(self,num_batches=100):
    #Select indx of values:
    x_array=[]#torch.zeros(num_batches,self.size[0],self.size[1],self.size[2])
    y_array=[]#torch.zeros(num_batches)
    if self.phase == 'train':
      idx = rand_idx_generator(num_batches,self.len)
    else:
      idx = rand_idx_generator(num_batches,self.len)
    for i in idx:
      try:
        if self.phase == 'train':
          x = serialize_FEN(self.train_dataset['fen'][i])[0]
          y = self.train_dataset['result'][i]
        else:
          x = serialize_FEN(self.test_dataset['fen'][i])[0]
          y = self.test_dataset['result'][i]
        x_array.append(torch.from_numpy(x).type(\
          torch.float32))
        y_array.append(torch.tensor(y,\
          dtype=torch.float32))
      except:
        t=self.train_dataset['fen'][i]
        #print('error,i = {} fen type is {}, fen = {}'.format(i,type(t),t))
    self.x=torch.stack(x_array)
    self.y=torch.stack(y_array)
    return self.x, self.y
  
  def __len__(self):
    return len(self.x)
  
  def __getitem__(self,index):
    return self.x, self.y
   
  def __exit__(self, exc_type, exc_value, traceback):
    pass
  
  def __enter__(self):
    return self



def train():
  #Config parameters:
  #==================
  nc = 6  #number of channels for the input image (size of 3rd dimension)
  ndf= 16 #scaling parameter outputs following each convolutional layer 
          #This is important for feature engineering
  ngpu = 2
  useGpu=True
  batchSize=64*256#128 #This is tuned so that the GPU memory is being 
                  #utilized but not saturated. (I get ~65% memory usage
                  #spread acorss two 2080ti GPU's)
  lr=0.0002 #learning rate
  precision_lvl='O1' #redundant, was going to implement mixed precision
                     #using NVIDIA 
  n_epochs=250# number of times to train over the data.
  train_valid_ratio=0.7 #ratio for splitting the data into train/test.
  device='cuda:0' #device to run the computation on (GPU or CPU)
  lazy_factor=200 #=1, each epoch should cover every data, and 
                  #=10, each epoch should train on 1/10 of the data, etc
                  #The larger, the less data we train on (...being lazy)
  reward_func = 'convex' #this informs the dataloader what data to load.
  #                      #Possible options are 'convex', 'concave',
  #                      #'linear' and 'constant'. See chess_engine.py
  #                      #for exact details.
  #
  #
  tic = time.time()
  device = torch.device("cuda:0" if useGpu else "cpu")
  #Load data into memory:
  print('Loading data into memory... ({:.2f} sec elapsed)'.format(\
    time.time()-tic))
  chess_data = dataload('training_data_{}.csv'.format(reward_func)\
    ,train_valid_ratio)
  print('Chess data loaded. ({:.2f} sec elapsed)'.format(\
    time.time()-tic))
  #Load the models:
  net = NetB(ngpu,nc,ndf).to(device)
  net.apply(weights_init)
  #Load older model:
  #netD.load_state_dict(torch.load(model_name))
  #define the loss function:
  criterion = nn.MSELoss()#nn.BCEWithLogitsLoss() #nn.BCELoss()
  best_acc = 0.0
  
  # setup optimizer
  optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))
  
  # phase = 'train'
  # net.train(True)
  # chess_data.set_phase(phase)
  # inputs,labels = chess_data.get_batch(batchSize)
  # inputs,labels = inputs.to(device), labels.to(device)
  # outputs = net(inputs)
  # outputs.size()
  num_training_images = len(chess_data.train_dataset['fen'])
  data_cycles = int(num_training_images/batchSize/lazy_factor)
  for epoch in range(n_epochs):
    print('starting Epoch {}/{}, {:.2f} secs elapsed'.format(epoch, \
      n_epochs - 1,time.time()-tic))
    print('-' * 10)
    
    running_loss = 0.0
    running_corrects = 0
    # Each epoch has a training and validation phase
    for i in range(data_cycles):
      
      for phase in ['train', 'val']:
        
        #set train/validation settings:
        if phase == 'train':
          # optimizer = scheduler(optimizer, epoch)
          net.train(True)  # Set model to training mode
        else:
          net.train(False)  # Set model to evaluate mode
        chess_data.set_phase(phase)
        #
        #print('optimizer pass')
        #grab batches of data:
        inputs,labels = chess_data.get_batch(batchSize)
        #print('batch data loaded')
        #pass through:
        inputs, labels = inputs.cuda(), labels.cuda()
        #print('data .cuda() passed')
        inputs,labels = inputs.to(device), labels.to(device)
        #print('batch data loaded to device')
        outputs = net(inputs)
        outputs = outputs.to(device)
        #print('outputs retrieved')
        #return outputs,labels
        loss = criterion(outputs,labels)
        #print('loss criterion pass')
        if phase == 'train':
          optimizer.zero_grad()
          loss.backward()
          #print('loss backward pass')
          optimizer.step()
          #print('optimizer step pass')
        else:
          #statistics:
          running_loss += loss.item()
          running_corrects += torch.sum(torch.round(outputs) == labels).item()
          if i%int(data_cycles/4) == 0:
            print('i={}/{}, acc={:.2f}, ({:.1f} secs elapsed)'.format(\
              i+1,data_cycles, \
              running_corrects/((i+1)*batchSize)*100,time.time()-tic))
          
    #epoch stats:
    epoch_loss = running_loss / (data_cycles*batchSize)
    epoch_acc = running_corrects/ (data_cycles*batchSize)
    print('epoch {} finished, Loss: {:.4f} Acc: {:.4f}%'.format(\
      epoch, epoch_loss, epoch_acc*100))
    if epoch_acc > best_acc:
      print('saving model now.. ({:.2f} seconds elapsed)'.format(time.time()-tic))
      # do checkpointing
      best_acc = epoch_acc
      torch.save(net.state_dict(), 'netB_{}_acc_{}.pth'.format(\
        reward_func,int(best_acc*100) ))




if __name__ == "__main__":
  train()




#Notes:
'''
torch.nn only supports inputs of batch data 
(not single input array)

above, the nn.Conv2d takes in 
(nSamples x nChannels x Height x Width)

the nn.Module is useful class, and allows for encapsulation of 
parameters and easy offloading of workload to GPU's




class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    #1 input image channel, 6 output channels, 3x3 square conv
    self.conv1 = nn.Conv2d(1,6,3)
    self.conv2 = nn.Conv2d(6,16,3)
    #an affine operation: y=Wx+b
    self.fc1 = nn.Linear(16*6*6,120) #6x6 from image dimension
    self.fc2 = nn.Linear(120,84)
    self.fc3 = nn.Linear(84,10)
  
  def forward(self,x):
    #max pooling over a (2,2) window
    x=F.max_pool2d(F.relu(self.conv1(x)), (2,2))
    #if the size is a square you can only specify a single number
    x=F.max_pool2d(F.relu(self.conv2(x)), 2)
    x=x.view(-1, self.num_flat_features(x))
    x=F.relu(self.fc1(x))
    x=F.relu(self.fc2(x))
    x=self.fc3(x)
    return x
  
  def num_flat_features(self,x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
      num_features *= s
    return num_features


class simpleNet(nn.Module):
  def __init__(self):
    super(simpleNet).__init__():

  def forward(self,x):
    x = F.relu(nn.Conv2d(6,16,kernel_size=3))
    x = F.relu(nn.Conv2d(16,16,kernel_size=3))
    x = F.relu(nn.Conv2d(16,32,kernel_size=3))
    x = F.max_pool2d(x)
    
    # 4x4
    x = F.relu(nn.Conv2d(32,32,kernel_size=3))
    x = F.relu(nn.Conv2d(32,32,kernel_size=3))
    x = F.relu(nn.Conv2d(32,64,kernel_size=3))
    x = F.max_pool2d(x)
    
    # 1x1
    x = F.relu(nn.Conv2d(64,64,kernel_size=3))
    x = F.relu(nn.Conv2d(64,64,kernel_size=3))
    x = F.relu(nn.Conv2d(64,128,kernel_size=3))
    x = F.max_pool2d(x)
    
    x = F.relu(nn.Conv2d(128,128,kernel_size=3))
    x = F.relu(nn.Conv2d(128,128,kernel_size=3))
    x = F.relu(nn.Conv2d(128,128,kernel_size=3))
    x = F.max_pool2d(x)

    x = F.view(-1,128)
    x = F.linear(128, 6)
    return F.log_softmax(x,dim=1)



def test():
  net=Net()
  #we can print out the neural network framework:
  print(net)
  
  # we can print out the learnable parameters of the 
  # function:
  params = list(net.parameters())
  print(len(params))
  print(params[0].size()) #conv1's weight parameter
  
  #Let's try a random 32x32 input, and pass it through the net:
  input = torch.randn(1,1,32,32)
  out = net(input)
  print(out)
  
  #Let's now "zero" the gradient buffers of all parameters
  #and backprops with random gradients:
  net.zero_grad()
  out.backward(torch.randnn(1,10))
  
  #Loss Function::
  output = net(input)
  target = torch.randn(10)
  target = target.view(1,-1)
  criterion = nn.MSELoss()
  
  loss = criterion(output,target)
  print(loss)


def test2():
  net = simpleNet():




'''





