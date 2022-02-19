import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from torch import autograd


class conv_block(nn.Module):
    """
    Convolutional block with one convolutional layer
    and ReLU activation function. 
    """
    def __init__(self,ch_in,ch_out,kernel_size,padding = 1, bias = False):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,stride=1,padding=padding,bias=bias),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class convtranspose_block(nn.Module):
    """
    Transposed Convolutional block with one transposed convolutional 
    layer and ReLU activation function. 
    """
    def __init__(self,ch_in,ch_out,kernel_size):
        super(convtranspose_block,self).__init__()
        self.convtrans = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=kernel_size,stride=1,padding=1,bias=False),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.convtrans(x)
        return x


class multi_scale_conv(nn.Module):
  """
  Multi scale convolutional block with 3 convolutional blocks
  with kernel size of 3x3, 5x5 and 7x7. Which is then concatenated
  and fed into a 1x1 convolutional block. 
  """
  def __init__(self,ch_in,ch_out):
        super(multi_scale_conv,self).__init__()
        self.conv3x3 = conv_block(ch_in = ch_in,ch_out = ch_out, kernel_size = 3, padding=1)
        self.conv5x5 = conv_block(ch_in = ch_in,ch_out = ch_out, kernel_size = 5, padding=2)
        self.conv7x7 = conv_block(ch_in = ch_in,ch_out = ch_out, kernel_size = 7, padding=3)
        self.conv1x1 = conv_block(ch_in = ch_out*3,ch_out = ch_out, kernel_size = 1, padding = 0)

  def forward(self,x):
      x1 = self.conv3x3(x)
      x2 = self.conv5x5(x)
      x3 = self.conv7x7(x)
      comb = torch.cat((x1, x2, x3), 1)
      out = self.conv1x1(comb)
      return out


########## ConvVLSTMS ##########
class ConvLSTMCell(nn.Module):
  """
  Basic ConvLSTM Cell 
  """
  def __init__(self,ch_in,ch_hidden, kernel_size, bias):
    super(ConvLSTMCell,self).__init__()

    self.ch_in = ch_in
    self.ch_hidden = ch_hidden
    self.kernel_size = kernel_size
    self.padding = 1
    self.bias = bias

    self.conv = nn.Conv2d(in_channels = self.ch_in + self.ch_hidden, out_channels = 4*self.ch_hidden,
                          kernel_size = self.kernel_size, padding = self.padding, bias = self.bias)

  def forward(self,x,current_state):
    h_current, c_current = current_state

    x1 = self.conv(torch.cat([x, h_current], dim=1))

    cc_i, cc_f, cc_o, cc_g = torch.split(x1, self.ch_hidden, dim=1)
    i = torch.sigmoid(cc_i)
    f = torch.sigmoid(cc_f)
    o = torch.sigmoid(cc_o)
    g = torch.tanh(cc_g)

    c_next = f * c_current + i * g
    h_next = o * torch.tanh(c_next)

    return h_next, c_next
  
  def init_hidden(self, batch_size, image_size):
    height, width = image_size
    return (torch.zeros(batch_size, self.ch_hidden, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.ch_hidden, height, width, device=self.conv.weight.device))



class ConvLSTM(nn.Module):
  """
  Combination of multiple layers of ConvLSTM Cells 
  """
  def __init__(self,ch_in,ch_hidden, kernel_size,num_layers, batch_first=False, bias=True, return_all_layers=False):
    super(ConvLSTM,self).__init__()

    kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
    ch_hidden = self._extend_for_multilayer(ch_hidden, num_layers)
    if not len(kernel_size) == len(ch_hidden) == num_layers:
        raise ValueError('Inconsistent list length.')

    self.ch_in = ch_in
    self.ch_hidden = ch_hidden
    self.kernel_size = kernel_size
    self.num_layers = num_layers
    self.batch_first = batch_first
    self.bias = bias
    self.return_all_layers = return_all_layers

    convlstmcell_list = []
    for i in range(0, self.num_layers):
        cur_input_dim = self.ch_in if i == 0 else self.ch_hidden[i - 1]

        convlstmcell_list.append(ConvLSTMCell(ch_in=cur_input_dim,
                                      ch_hidden=self.ch_hidden[i],
                                      kernel_size=self.kernel_size[i],
                                      bias=self.bias))

    self.convlstmcell_list = nn.ModuleList(convlstmcell_list)

  def forward(self, x, hidden_state=None):
   
    if not self.batch_first:
        # (t, b, c, h, w) -> (b, t, c, h, w)
        x = x.permute(1, 0, 2, 3, 4)

    b, _, _, h, w = x.size()
    # print(x.size())
    # Implement stateful ConvLSTM
    if hidden_state is not None:
        raise NotImplementedError()
    else:
        # Since the init is done in forward. Can send image size here
        hidden_state = self._init_hidden(batch_size=b,
                                          image_size=(h, w))
        

    seq_len = x.size(1)
    cur_layer_input = x

    for layer_idx in range(self.num_layers):

        h, c = hidden_state[layer_idx]
        # print(h,c)
        output_inner = []
        for t in range(seq_len):
            h, c = self.convlstmcell_list[layer_idx](x=cur_layer_input[:, t, :, :, :],
                                              current_state=[h, c])
            output_inner.append(h)

        layer_output = torch.stack(output_inner, dim=1)
        cur_layer_input = layer_output

    return layer_output

  def _init_hidden(self, batch_size, image_size):
      init_states = []
      for i in range(self.num_layers):
          init_states.append(self.convlstmcell_list[i].init_hidden(batch_size, image_size))
      return init_states

  @staticmethod
  def _extend_for_multilayer(param, num_layers):
      if not isinstance(param, list):
          param = [param] * num_layers
      return param



class ConvLSTM_block(nn.Module):
  """
  ConvLSTM block with 3 ConvLSTM layers with ReLU activation based on the 
  architecture proposed on the paper. This class can be used for both forward and 
  backward ConvLSTM blocks. 
  """
  def __init__(self):
        super(ConvLSTM_block,self).__init__()
        self.convlstm1 = ConvLSTM(ch_in = 1, ch_hidden = 32, kernel_size = (3,3), num_layers = 1, batch_first = True, bias = True, return_all_layers = False)
        self.relu1     = nn.ReLU()

        self.convlstm2 = ConvLSTM(ch_in = 32, ch_hidden = 64, kernel_size = (3,3), num_layers = 1, batch_first = True, bias = True, return_all_layers = False)
        self.relu2     = nn.ReLU()

        self.convlstm3 = ConvLSTM(ch_in = 64, ch_hidden = 128, kernel_size = (3,3), num_layers = 1, batch_first = True, bias = True, return_all_layers = False)
        self.relu3     = nn.ReLU()


  def forward(self,x):
      x = x.view(x.shape[0],x.shape[1],1,x.shape[2],x.shape[3])

      out1 = self.convlstm1(x)
      out1 = self.relu1(out1)

      out2 = self.convlstm2(out1)
      out2 = self.relu2(out2)

      out3 = self.convlstm3(out2)
      out3 = self.relu3(out3)
      return out1, out2, out3


class encoder(nn.Module):
    """
    Encoder class of the generator with multiple multi-scale
    convolutional blocks.  
    """
    def __init__(self):
        super(encoder,self).__init__()

        # self.convlstm_forward = ConvLSTM_block()
        # self.convlstm_backward = ConvLSTM_block()

        self.multi_conv1 = multi_scale_conv(1,32)
        self.conv1 = conv_block(ch_in = 32*3 ,ch_out = 32, kernel_size = 3, padding=1)

        self.multi_conv2 = multi_scale_conv(32,64)
        self.conv2 = conv_block(ch_in = 64*3 ,ch_out = 64, kernel_size = 3, padding=1)

        self.multi_conv3 = multi_scale_conv(64,128)
        self.conv3 = conv_block(ch_in = 128*3 ,ch_out = 128, kernel_size = 3, padding=1)

        self.multi_conv4 = multi_scale_conv(128,256)

    def forward(self,x, f1,f2,f3,b1,b2,b3):

        # f1,f2,f3 = self.convlstm_forward(x) 
        # b1,b2,b3 = self.convlstm_backward(x_inv)
        feature_maps = []

        out1 = self.multi_conv1(x)
        out1 = torch.cat((out1, f1, b1), 1)
        out1 = self.conv1(out1)

        out2 = self.multi_conv2(out1)
        out2 = torch.cat((out2, f2, b2), 1)
        out2 = self.conv2(out2)

        out3 = self.multi_conv3(out2)
        out3 = torch.cat((out3, f3, b3), 1)
        out3 = self.conv3(out3)
        
        out4 = self.multi_conv4(out3)
        return out1,out2,out3,out4



class decoder(nn.Module):
    """
    Decoder class of the generator with multiple transposed
    convolutional blocks.  
    """
    def __init__(self):
        super(decoder,self).__init__()

        self.convtrans1 = convtranspose_block(ch_in = 256, ch_out = 128, kernel_size = 3)

        self.convtrans2 = convtranspose_block(ch_in = 128*2, ch_out = 128, kernel_size = 3)
        self.convtrans3 = convtranspose_block(ch_in = 128, ch_out = 64, kernel_size = 3)

        self.convtrans4 = convtranspose_block(ch_in = 64*2, ch_out = 64, kernel_size = 3)
        self.convtrans5 = convtranspose_block(ch_in = 64, ch_out = 32, kernel_size = 3)

        self.convtrans6 = convtranspose_block(ch_in = 32*2, ch_out = 32, kernel_size = 3)
        self.convtrans7 = convtranspose_block(ch_in = 32, ch_out = 1, kernel_size = 3)

    def forward(self,in1,in2,in3,in4):
        out1 = self.convtrans1(in4)

        out2 = torch.cat((out1, in3), 1)
        out2 = self.convtrans2(out2)
        out2 = self.convtrans3(out2)
        
        out3 = torch.cat((out2, in2), 1)
        out3 = self.convtrans4(out3)
        out3 = self.convtrans5(out3)

        out4 = torch.cat((out3, in1), 1)
        out4 = self.convtrans6(out4)
        out4 = self.convtrans7(out4)

        return out4


class Generator(nn.Module):
    """
    Generator model proposed by the model, which takes in frame as a input,
    along the features learned by forward and backward ConvLSTMs and generates 
    a motion artefact free image. Here, skip connections are employed between 
    the encoder and decoder.   
    """
    def __init__(self):
        super(Generator,self).__init__()

        self.encoder = encoder()
        self.decoder = decoder()

    def forward(self,x,f1,f2,f3,b1,b2,b3):
        out1,out2,out3,out4 = self.encoder(x,f1,f2,f3,b1,b2,b3)
        out = self.decoder(out1,out2,out3,out4)

        return out


class Discriminator(nn.Module):
    """
    Discriminator class, which functions as a critic in the adversarial training. 
    The model consists of several convolutional blocks and fully connected layers.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1_1 = conv_block(ch_in = 1 ,ch_out = 64, kernel_size = 3, padding=1, bias = True)
        self.conv1_2 = conv_block(ch_in = 64 ,ch_out = 64, kernel_size = 3, padding=1, bias = True)
        self.maxpool1 = nn.MaxPool2d(2,2)

        self.conv2_1 = conv_block(ch_in = 64 ,ch_out = 128, kernel_size = 3, padding=1, bias = True)
        self.conv2_2 = conv_block(ch_in = 128 ,ch_out = 128, kernel_size = 3, padding=1, bias = True)
        self.maxpool2 = nn.MaxPool2d(2,2)

        self.conv3_1 = conv_block(ch_in = 128,ch_out = 256, kernel_size = 3, padding=1, bias = True)
        self.conv3_2 = conv_block(ch_in = 256 ,ch_out = 256, kernel_size = 3, padding=1, bias = True)
        self.maxpool3 = nn.MaxPool2d(2,2)

        self.fc1   = nn.Linear(12*12*256, 1024)
        self.relu1 = nn.ReLU(True)
        self.fc2   = nn.Linear(1024,1)
        self.relu2 = nn.ReLU(True)  ####

    def forward(self, x):
        out1 = self.conv1_1 (x)
        out1 = self.conv1_2 (out1)
        out1 = self.maxpool1(out1)

        out2 = self.conv2_1 (out1)
        out2 = self.conv2_2 (out2)
        out2 = self.maxpool2(out2)

        out3 = self.conv3_1 (out2)
        out3 = self.conv3_2 (out3)
        out3 = self.maxpool3(out3)

        # print(out3.shape)
        out3 = out3.view(-1, 12*12*256)
        out = self.fc1(out3)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        
        return out

# Vgg16 for perception loss
class Vgg16(torch.nn.Module):
    """
    Vgg16 model trained on the ImageNet dataset. The model is used for 
    the calculation of the perceptual loss. Here, support code was taken from 
    pytorch examples.
    help https://github.com/pytorch/examples
    """
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        # vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        # out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]


def gradient_penalty(netD, real, fake):
    """
    Gradient penalty function for loss calculation of the discriminator.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c,s,h,w = real.shape
    alpha = torch.rand(c, s, h, w)
    alpha = alpha.expand(real.size())
    alpha = alpha.to(device) if torch.cuda.is_available() else alpha

    interpolates = alpha * real + ((1 - alpha) * fake)

    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device) if torch.cuda.is_available() else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
