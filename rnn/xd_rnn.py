import torch
import torch.nn as nn

class XD_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
      super(XD_RNN, self).__init__()

      self.hidden_size = hidden_size
      self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
      self.h2o = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        combined = torch.cat(input, hidden);
        hidden = self.i2h(combined);
        output = self.h2o(hidden)
        return output, hidden;


    def get_hidden(self):
        return torch.zeros(1, self.hidden_size)


    

"""
  example code


rnn_model = XD_RNN(input_size=4, hidden_size=4, output_size=2)
# get initial hidden state
hidden = rnn_model.get_hidden()

# rnn model 을 통해 문장을 처리하는 과정
# input : the food is good
_, hidden = rnn_model.forward(input_tensor0, hidden) # the
_, hidden = rnn_model.forward(input_tensor1, hidden) # food
_, hidden = rnn_model.forward(input_tensor2, hidden) # is
output, _ = rnn_model.forward(input_tensor3, hidden) # good

"""






        