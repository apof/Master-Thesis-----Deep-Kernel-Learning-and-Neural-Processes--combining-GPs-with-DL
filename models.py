import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from IPython.display import clear_output
import matplotlib.pyplot as plt
from torch.autograd import Variable
import copy


def var(tensor):
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

class MLPFeatureExtractor(torch.nn.Sequential):
    def __init__(self,data_dim,representation_dim):
        super(MLPFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 100))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(100, 20))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(20, representation_dim))

class Attention(nn.Module):
    def __init__(self, device,hidden_size):
        super(Attention, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.concat_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, hidden_size)

        ## add this model the same same device with the RNN
        self.to(device)

    def forward(self, rnn_outputs, final_hidden_state):
      attn_weights = self.attn(rnn_outputs) # (batch_size, seq_len, hidden_dim)
      attn_weights = torch.bmm(attn_weights, final_hidden_state.unsqueeze(2))
      attn_weights = F.softmax(attn_weights.squeeze(2), dim=1)
      context = torch.bmm(rnn_outputs.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
      attn_hidden = torch.tanh(self.concat_linear(torch.cat((context, final_hidden_state), dim=1)))
      return attn_hidden, attn_weights

class RnnRegressor(nn.Module):
    def __init__(self, device, params_dictionary,index):
        super(RnnRegressor, self).__init__()
        self.params = params_dictionary
        self.device = device

        # Calculate number of directions
        self.num_directions = 2 if self.params.get('bidirectional') == True else 1

        # define an attention model
        # Choose attention model
        self.attention = Attention(self.device,self.params.get('hidden_dim')* self.num_directions)

        ## here store in a list all the dimensions of the layers rnn_output --> linear layers --> labels layer
        self.linear_dims = [self.params.get('hidden_dim') * self.num_directions] + self.params.get('linear_dims')

        # Work with LSTM cell for now
        self.rnn = nn.LSTM

        ## CNN layer
        self.conv1d_layer1 = nn.Conv1d(self.params.get('embedding_size')[index],params_dictionary.get('cnn_reduced_dim_1'),params_dictionary.get('kernel_size'))
        self.conv1d_layer2 = nn.Conv1d(params_dictionary.get('cnn_reduced_dim_1'),params_dictionary.get('cnn_reduced_dim_2'),params_dictionary.get('kernel_size'))
        self.pool_1 = nn.MaxPool1d(3, stride=2)
        self.pool_2 = nn.MaxPool1d(3, stride=2)
        self.activation_func_1 = nn.LeakyReLU(0.1)
        self.activation_func_2 = nn.LeakyReLU(0.1)


        ## define the RNN layer
        self.rnn = self.rnn(params_dictionary['cnn_reduced_dim_1'],
                            self.params.get('hidden_dim'),
                            num_layers=self.params.get('rnn_layers_num'),
                            bidirectional=self.params.get('bidirectional'),
                            dropout=float(self.params.get('dropout')),
                            batch_first=True)
        
        ## the hidden state of the RNN empty for now
        self.hidden = None
        
        # Define set of fully connected layers (Linear Layer + Activation Layer)
        ## this set of layers takes the output of the RNN or the Attention layer and applies a feedforward NN on it
        ## consecutive linear + Relu layers are applied (the final layer does not have a relu activation!)
        self.linears = nn.ModuleList()
        for i in range(0, len(self.linear_dims)-1):
            if self.params.get('dropout') > 0.0:
                self.linears.append(nn.Dropout(p=self.params.get('dropout')))
            linear_layer = nn.Linear(self.linear_dims[i], self.linear_dims[i+1])
            #self.init_weights(linear_layer)
            self.linears.append(linear_layer)
            self.linears.append(nn.Tanh())

        self.to(self.device)

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.params.get('rnn_layers_num') * self.num_directions, batch_size, self.params.get('hidden_dim')).to(self.device),
              torch.zeros(self.params.get('rnn_layers_num') * self.num_directions, batch_size, self.params.get('hidden_dim')).to(self.device))
      

    def forward(self, inp, params_dictionary):

        (inputs,stateful,hidden_state) = inp

        if(params_dictionary.get('cnn_layer') == True):
          inputs = inputs.view(inputs.shape[0], inputs.shape[2], inputs.shape[1])
          with torch.cuda.amp.autocast():

            cnn_processing_x = self.conv1d_layer1(inputs)
            cnn_processing_x = self.activation_func_1(cnn_processing_x)
            #cnn_processing_x = self.pool_1(cnn_processing_x)

            #cnn_processing_x = self.conv1d_layer2(cnn_processing_x)
            #cnn_processing_x = self.activation_func_2(cnn_processing_x)
            #cnn_processing_x = self.pool_2(cnn_processing_x)


          inputs = cnn_processing_x.view(cnn_processing_x.shape[0], cnn_processing_x.shape[2], cnn_processing_x.shape[1])

        batch_size, seq_len, embedding_size = inputs.shape

        embedded_inputs = inputs

        if(stateful == True):
          ## initialise the hidden state of the RNN
          self.hidden = self.init_hidden(batch_size)
        # use the previous hidden state to make the rnn stateful alomg the batches
        else:
          self.hidden = hidden_state


        ## pass the data through the recurrent layer
        rnn_output, self.hidden = self.rnn(embedded_inputs.float(), self.hidden)

        ## Collect last hidden state
        final_state = self.hidden[0].view(self.params.get('rnn_layers_num'), self.num_directions, batch_size, self.params.get('hidden_dim'))[-1]

        # Handle directions if more than one
        final_hidden_state = None
        ## in case we have only one direction
        if self.num_directions == 1:
            final_hidden_state = final_state.squeeze(0)
        ## in case we have 2 directions concatenate these two states
        elif self.num_directions == 2:
            h_1, h_2 = final_state[0], final_state[1]
            final_hidden_state = torch.cat((h_1, h_2), 1)  # Concatenate both states


        ## Attention Layer
        if(self.params.get('attention_layer') == False):
          X = final_hidden_state
        else:
          #rnn_output = rnn_output.permute(1, 0, 2)
          X, attention_weights = self.attention(rnn_output, final_hidden_state)

        # Push through linear layers
        for l in self.linears:
            X = l(X)


        if(self.hidden[0].shape[1] == batch_size ):
          hidden = (var(self.hidden[0].data), var(self.hidden[1].data))
        else:
          hidden = hidden_state

        return X, hidden



class MultiTaskModel(nn.Module):

  def __init__(self, device, params_dictionary):
        super(MultiTaskModel, self).__init__()

        self.params = params_dictionary
        self.device = device

        self.net1 = RnnRegressor(device,params_dictionary,0)
        self.net2 = RnnRegressor(device,params_dictionary,1)

        self.linear_dims = [self.params.get('linear_dims')[-1]*2] + self.params.get('concat_layer')
        self.linear_dims.append(self.params.get('label_size'))

        self.linears = nn.ModuleList()
        for i in range(0, len(self.linear_dims)-1):
            if self.params.get('dropout') > 0.0:
                self.linears.append(nn.Dropout(p=self.params.get('dropout')))
            linear_layer = nn.Linear(self.linear_dims[i], self.linear_dims[i+1])
            #self.init_weights(linear_layer)
            self.linears.append(linear_layer)
            if i == len(self.linear_dims) - 1:
                break
            self.linears.append(nn.Tanh())

        self.to(self.device)


  def multitask_reg(self,reg_rate,reg_type):
    params_1 = []
    params_2 = []
    for p in self.net1.parameters():
      params_1.append(p)
    for p in self.net2.parameters():
      params_2.append(p)

    subtracted_params = []
    added_squares = []
    for i,p1 in enumerate(params_1):
      if(i!=0):
        #print(str(p1.shape) + " " + str(params_2[i].shape))
        subtracted_params.append(torch.subtract(p1, params_2[i]))
        added_squares.append(torch.sqrt(torch.add(torch.square(p1),torch.square(params_2[i]))))

    #for p in subtracted_params:
    #  print(p.shape)

    reg = None
    if(reg_type == 'l2'):
      reg = sum(p.pow(2.0).sum() for p in subtracted_params)
    elif(reg_type == 'l1'):
      reg = sum(torch.norm(p, 1).sum() for p in subtracted_params)

    return reg_rate*reg

  def init_weights(self, layer):
    if type(layer) == nn.Linear:
      torch.nn.init.xavier_uniform_(layer.weight)
      layer.bias.data.fill_(0.01)

  
  def forward(self, inp):

    (inp_1,inp_2,stateful,hidden_state_1,hidden_state_2) = inp

    out_1,hidden_state_1 = self.net1.forward((inp_1,stateful,hidden_state_1),self.params)
    out_2,hidden_state_2 = self.net2.forward((inp_2,stateful,hidden_state_2),self.params)
    concatenated_output = torch.cat((out_1,out_2),dim=1)

    X = concatenated_output

    for l in self.linears:
      X = l(X)

    return X, hidden_state_1, hidden_state_2



