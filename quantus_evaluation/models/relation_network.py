#numpy
import numpy as np
from numpy import newaxis as na

#pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class RelationNet(nn.Module):

    def __init__(self, vocab_size=82, nb_classes=28, word_embedding_dim=32, device='cuda'):
        """
        Initialize the neural network weights and word embeddings.
        We reimplement the model proposed by Santoro et al. 2017 in: https://papers.nips.cc/paper/7082-a-simple-neural-network-module-for-relational-reasoning.pdf
        In the remainder we refer to this paper as the RN paper.    
        Args:
            - vocab_size:           training set vocabulary size + 2 for UNK and ZERO-PADDING (we use 82)
            - nb_classes:           number of classes of the classification problem (28 for CLEVR dataset)
            - word_embedding_dim:   dimension of the word embeddings
        
        """
        super(RelationNet, self).__init__()
        self.device = device
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=0) # thus idx 0 corresponds to zero vector
        self.lstm = nn.LSTM(word_embedding_dim, 128, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=False) # no dropout

        self.image_encoder = nn.Sequential(nn.Conv2d(3,  24, 3, stride=2, padding=0, bias=True), # output shape floor[(128-3)/2+1] = 63.5->63 (i.e. last pixel in vertical and horizontal direction is ignored)
                                            nn.ReLU(),
                                            nn.BatchNorm2d(24),
                                            nn.Conv2d(24, 24, 3, stride=2, padding=0, bias=True), # output shape floor[(63-3)/2+1]  = 31
                                            nn.ReLU(),
                                            nn.BatchNorm2d(24),
                                            nn.Conv2d(24, 24, 3, stride=2, padding=0, bias=True), # output shape floor[(31-3)/2+1]  = 15
                                            nn.ReLU(),
                                            nn.BatchNorm2d(24),
                                            nn.Conv2d(24, 24, 3, stride=2, padding=0, bias=True), # output shape floor[(15-3)/2+1]  = 7 = d
                                            nn.ReLU(),
                                            nn.BatchNorm2d(24)
                                            )
        
        # -- Object coordinates
        c             = np.arange(7, dtype=np.float32) # raw coordinates
        c_mean, c_std = c.mean(), c.std()
        c_norm        = (c-c_mean)/c_std # normalized coordinates (the origin is now at image center)
        x_coord       = np.repeat(c_norm[na,:,na], 7, axis=0) # shape (H, W, 1)
        y_coord       = np.repeat(c_norm[:,na,na], 7, axis=1) # shape (H, W, 1)
        self.xy_coord = torch.from_numpy(np.concatenate((x_coord,y_coord), axis=2)[na,:,:,:]).clone() # shape (1, H, W, 2), first axis = batch dim, last axis = contains the x, y coordinates
                                                                                                      # (side note: copy_ could be used instead of clone() to avoid gradients propagating)        
        # -- RN 
        self.g1 = nn.Linear(180, 256)
        self.g2 = nn.Linear(256, 256)
        self.g3 = nn.Linear(256, 256)
        self.g4 = nn.Linear(256, 256)
        
        self.f1 = nn.Linear(256, 256)
        self.f2 = nn.Linear(256, 256)
        self.f3 = nn.Linear(256, nb_classes)
     
    def prepare_data(self, data, use_double=False, use_gpu=False):
        if use_double:
            data = data.double()
        if use_gpu:
            data = data.to(self.device)
        return data
          
    def forward(self, image, question, question_len, use_gpu=True, double=False):

        N = image.size(0) # batch size
        T = question.size(1) # length of questions in the batch (including zero-padding)
        assert image.size()==(N, 3, 127, 127), f"shape is {image.size()}"
        assert question.size()==(N, T) 
        assert len(question_len)==N 
        # Note: here above we assume the image has already been truncated to size 127x127 (since the first conv layer would ignore the 128-th pixel anyway)

        # -- LSTM
        self.question_embedding = self.word_embeddings(question) # shape (N, T, embedding_dim)

        packed_question = self.prepare_data(pack_padded_sequence(self.question_embedding, question_len, batch_first=True, enforce_sorted=False), double)
        h_0  = self.prepare_data(torch.zeros(1, N, 128), double, use_gpu)
        c_0  = self.prepare_data(torch.zeros(1, N, 128), double, use_gpu)

        _, (h_T, c_T) = self.lstm(packed_question, (h_0, c_0)) # h_T has shape 1 x N x 128   
        q             = torch.transpose(h_T, 0, 1)  # shape N x 1 x 128, side note: shared data
        assert q.size()==(N, 1, 128)
        q             = q.repeat(1, 49*49, 1)       # copy question to subsequently append to each object pair
        assert q.size()==(N, 49*49, 128)

        # -- CNN
        x = self.image_encoder(image)

        assert x.size()==(N, 24, 7, 7) # shape (N, C, H, W)
        x = x.permute(0, 2, 3, 1)      # put the 24-dimensional image features in the last axis
        assert x.size()==(N, 7, 7, 24) # shape (N, H, W, C)

        # Append 2-dimensional object coordinates to the image features:
        x = torch.cat([x, self.prepare_data(self.xy_coord.repeat(N, 1, 1, 1), double, use_gpu)], 3) 
        assert x.size()==(N, 7, 7, 24+2)

        # Replicate the image objects to form the object pairs:
        x_flat  = x.contiguous().view(-1, 49, 24+2) # shape (N, 49, (24+2)),     side note: shared data, view needs contiguous tensor
        x1      = torch.unsqueeze(x_flat, 2)        # shape (N, 49, 1,  (24+2)), side note: shared data
        assert x1.size()==(N, 49, 1, 24+2)
        x1      = x1.repeat(1, 1, 49, 1)            # shape (N, 49, 49, (24+2)), side note: copied data
        x2      = torch.unsqueeze(x_flat, 1)        # shape (N, 1 , 49, (24+2))   
        assert x2.size()==(N, 1, 49, 24+2)
        x2      = x2.repeat(1, 49, 1, 1)            # shape (N, 49, 49, (24+2))
        xpair   = torch.cat([x1, x2], 3)            # shape (N, 49, 49, (24+2)*2), last axis = object features

        # -- RN
        xpair = xpair.contiguous().view(-1, 49*49, (24+2)*2)
        # Append question to the object pairs:
        xpair = torch.cat([xpair, q], 2) 
        assert xpair.size()==(N, 49*49, 52+128)
        
        g = F.relu(self.g1(xpair)) # this will apply the linear transform to the last axis of xpair 
        g = F.relu(self.g2(g))
        g = F.relu(self.g3(g))
        g = F.relu(self.g4(g))
        assert g.size()==(N, 49*49, 256)
        
        f = torch.sum(g, 1)
        assert f.size()==(N, 256)
        f = F.relu(self.f1(f))
        f = F.dropout(F.relu(self.f2(f)), p=0.5, training=self.training) # typically dropout is applied after non-linearity
        f = self.f3(f) # these are the classification prediction scores!
        assert f.size()==(N, self.f3.out_features)
            
        return f 