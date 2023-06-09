from zennit import layer as zlayer
from zennit import canonizers as zcanonizers
from models.relation_network import RelationNet
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from canonizers.bn_conv_canonizer import SequentialMergeBNConv
from canonizers.bn_linear_canonizer import NamedMergeBatchNormByIndices


class CatImgTxtCanonized(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x_img_pair,qst):
        return torch.cat([x_img_pair, qst], 2)

    @staticmethod
    def backward(ctx,grad_output):
        grad1 = grad_output[:, :, :52]
        grad2 = grad_output[:, :, 52:]

        return grad1, grad2 

class CatImgCoordCanonized(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x_img,c_coord):
        return torch.cat([x_img, c_coord], -1)

    @staticmethod
    def backward(ctx,grad_output):
        grad1 = grad_output[:, :, :, :24]
        grad2 = grad_output[:, :, :, 24:]

        return grad1, grad2 

class RNCanonizer(zcanonizers.AttributeCanonizer):

    def __init__(self):
        super().__init__(self._attribute_map)
        

    @classmethod
    def _attribute_map(cls, name, module):

        if isinstance(module, RelationNet):

            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': zlayer.Sum(dim=1),
                'cat_img_text': CatImgTxtCanonized(),
                'cat_img_coord': CatImgCoordCanonized()
            }
            return attributes
        return None


    @staticmethod
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

        #### STEP 1: Address img/coord concatenation
        x = self.cat_img_coord.apply(x, self.prepare_data(self.xy_coord.repeat(N, 1, 1, 1), double, use_gpu))
        # x = torch.cat([x, self.prepare_data(self.xy_coord.repeat(N, 1, 1, 1), double, use_gpu)], 3) 
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

        #### STEP 2: Address xpair/question concatenation
        # Append question to the object pairs:
        # xpair = torch.cat([xpair, q], 2) 
        xpair = self.cat_img_text.apply(xpair,q)
        assert xpair.size()==(N, 49*49, 52+128)
        
        g = F.relu(self.g1(xpair)) # this will apply the linear transform to the last axis of xpair 
        g = F.relu(self.g2(g))
        g = F.relu(self.g3(g))
        g = F.relu(self.g4(g))
        assert g.size()==(N, 49*49, 256)
        
        # f = torch.sum(g, 1)
        f = self.canonizer_sum(g)

        assert f.size()==(N, 256)
        f = F.relu(self.f1(f))
        f = F.dropout(F.relu(self.f2(f)), p=0.5, training=self.training) # typically dropout is applied after non-linearity
        f = self.f3(f) # these are the classification prediction scores!
        assert f.size()==(N, self.f3.out_features)
            
        return f 

class RelationNetCanonizer(zcanonizers.CompositeCanonizer):
    def __init__(self):
        super().__init__((
            RNCanonizer(),
        ))

class RelationNetBNConvCanonizer(zcanonizers.CompositeCanonizer):
    def __init__(self):
        super().__init__((
            RNCanonizer(),
            SequentialMergeBNConv()
        ))

class RelationNetBNAllCanonizer(zcanonizers.CompositeCanonizer):
    def __init__(self):
        super().__init__((
            RNCanonizer(),
            SequentialMergeBNConv(),
            NamedMergeBatchNormByIndices({'image_encoder.11': ('g1', [(0, 24), (26, 50)])}),
        ))

class RelationNetBNOnlyCanonizer(zcanonizers.CompositeCanonizer):
    def __init__(self):
        super().__init__((
            SequentialMergeBNConv(),
            NamedMergeBatchNormByIndices({'image_encoder.11': ('g1', [(0, 24), (26, 50)])}),
        ))