"""
Definition of the ESIM model.
"""
# Aurelien Coet, 2018.

import torch
import torch.nn as nn

from .layers_predict_and_explain import RNNDropout, Seq2SeqEncoder, SoftmaxAttention, Decoder
from .utils import get_mask, replace_masked


class ESIM(nn.Module):
    """
    Implementation of the ESIM model presented in the paper "Enhanced LSTM for
    Natural Language Inference" by Chen et al.
    """

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
                 padding_idx=0,
                 dropout=0.5,
                 num_classes=3,
                 decoder_hidden_size=512,
                 device="cpu"):
        """
        Args:
            vocab_size: The size of the vocabulary of embeddings in the model.
            embedding_dim: The dimension of the word embeddings.
            hidden_size: The size of all the hidden layers in the network.
            embeddings: A tensor of size (vocab_size, embedding_dim) containing
                pretrained word embeddings. If None, word embeddings are
                initialised randomly. Defaults to None.
            padding_idx: The index of the padding token in the premises and
                hypotheses passed as input to the model. Defaults to 0.
            dropout: The dropout rate to use between the layers of the network.
                A dropout rate of 0 corresponds to using no dropout at all.
                Defaults to 0.5.
            num_classes: The number of classes in the output of the network.
                Defaults to 3.
            device: The name of the device on which the model is being
                executed. Defaults to 'cpu'.
        """
        super(ESIM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.decoder_hidden_size = decoder_hidden_size
        self.dropout = dropout
        self.device = device

        self._word_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx,
                                            _weight=embeddings)

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            # self._rnn_dropout = nn.Dropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        self.hidden_size,
                                        bidirectional=True)

        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))                                            
                                             
        self._decoding = Decoder(nn.LSTM,
                                 self.embedding_dim + 2*4*self.hidden_size,#4 * self.hidden_size,
                                 self.decoder_hidden_size,
                                 self.vocab_size,
                                 bidirectional=False)
        

        # Initialize all weights and biases in the model.
        self.apply(_init_esim_weights)

    def forward(self,
                premises,
                premises_lengths,
                hypotheses,
                hypotheses_lengths,
                explanations,
                explanations_lengths,
                index_to_word_dict,
                isTrain=False,
                labels=None,
                print_explanations=False):
        """
        Args:
            premises: A batch of varaible length sequences of word indices
                representing premises. The batch is assumed to be of size
                (batch, premises_length).
            premises_lengths: A 1D tensor containing the lengths of the
                premises in 'premises'.
            hypothesis: A batch of varaible length sequences of word indices
                representing hypotheses. The batch is assumed to be of size
                (batch, hypotheses_length).
            hypotheses_lengths: A 1D tensor containing the lengths of the
                hypotheses in 'hypotheses'.
            explanations: A batch of varaible length sequences of word indices
                representing explanations. The batch is assumed to be of size
                (batch, explanation_length).
            explanations_lengths: A 1D tensor containing the lengths of the
                explanations in 'premises'.
            isTrain: a boolean indicator for training time. Defaults to False.
            labels: A batch of labels - used in training time for the decoder                

        Returns:
            logits: A tensor of size (batch, num_classes) containing the
                logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing
                the probabilities of each output class in the model.
            logits_decoder: A tensor of size (batch, decoder_n_layers, vocab_size) containing
                the logits for each word in the vocabulary.
            preobabilities_decoder: A tensor of size (batch, decoder_n_layers, vocab_size) containing
                the probabilities of each word in the vocabulary.
        """
        label_to_ind = {0:4, 1:5, 2:6}
        
        premises_mask = get_mask(premises, premises_lengths).to(self.device)
        hypotheses_mask = get_mask(hypotheses, hypotheses_lengths)\
            .to(self.device)
         
        # Prepend to each explanation the gold label's embedding
        if isTrain:
            label_inds = torch.tensor([label_to_ind[x.item()] for x in labels]).view(-1,1).to(self.device)
        
            explanations = torch.cat([label_inds, explanations], dim=1)
            explanations_lengths += 1

        embedded_premises = self._word_embedding(premises)
        embedded_hypotheses = self._word_embedding(hypotheses)
        if isTrain:
            embedded_explanations = self._word_embedding(explanations)

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)
            if isTrain:
                embedded_explanations = self._rnn_dropout(embedded_explanations)
            
                    
        encoded_premises = self._encoding(embedded_premises,
                                          premises_lengths)
        encoded_hypotheses = self._encoding(embedded_hypotheses,
                                            hypotheses_lengths)
        
        attended_premises, attended_hypotheses =\
            self._attention(encoded_premises, premises_mask,
                            encoded_hypotheses, hypotheses_mask)

        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses -
                                         attended_hypotheses,
                                         encoded_hypotheses *
                                         attended_hypotheses],
                                        dim=-1)

        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_hypotheses = self._rnn_dropout(projected_hypotheses)

        v_ai = self._composition(projected_premises, premises_lengths)
        v_bj = self._composition(projected_hypotheses, hypotheses_lengths)

        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1)
                                                  .transpose(2, 1), dim=1)\
            / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)
        
        logits = self._classification(v)
        probabilities = nn.functional.softmax(logits, dim=-1)
        
        # If in training mode, append the gold label at the start of each input to the decoder
        if isTrain:
            input_dec = torch.cat([embedded_explanations,
                                   v.repeat(1,embedded_explanations.size()[1])
                                    .view(embedded_explanations.size()[0], 
                                          embedded_explanations.size()[1], 
                                          v.size()[1])], dim=-1)
            
            logits_dec, _ = self._decoding(input_dec, isTrain=True)
            probabilities_dec = nn.functional.softmax(logits_dec, dim=-1)
            
#            if print_explanations:
#                print("Explanations for batch:")
#                pred_indices = torch.max(probabilities_dec, dim=2)[1]
#                for i in range(embedded_explanations.shape[0]):
#                    expl = ""
#                    for j in range(pred_indices.shape[1]):
#                        curr_word = index_to_word_dict[pred_indices[i,j].item()]
#                        if curr_word == "_EOS_":
#                            break
#                        else:
#                            expl += " " + curr_word
#                    print("*" * 20)
#                    print("Explanation " + str(i) + ": " + expl)
                    
            
            return logits, probabilities, logits_dec, probabilities_dec
        
        # If in dev/test mode, need to feed at each timestep the output of the previous step
        else:
            # Prepend predicted label
            batch_size = embedded_premises.shape[0]   
            _, out_classes = probabilities.max(dim=1)
            out_classes_inds = torch.tensor([label_to_ind[x.item()] for x in out_classes]).view(-1,1).to(self.device)
            out_classes_embeddings = self._word_embedding(out_classes_inds)
            assert out_classes_embeddings.shape[0] == batch_size and out_classes_embeddings.shape[1] == 1 and out_classes_embeddings.shape[2] == self.embedding_dim
#            explanations = torch.cat([out_classes_inds, explanations], dim=1)
#            explanations_lengths += 1
            
            # Append v after first word (predicted label) of each explanation
            max_T_decoder = 51        
            dec_inp_t = torch.cat([out_classes_embeddings, v.view(batch_size,1,v.shape[1])], dim=-1)
            
                        
            pred_expls = [""] * batch_size
            finished = [False] * batch_size
            logits_dec = torch.zeros(batch_size, 51, self.vocab_size).to(self.device)
            
#            init_0 = torch.autograd.Variable(torch.zeros(1, batch_size, self.decoder_hidden_size)).to(self.device)
#            ht = (init_0, init_0)
            t = 0
            while t < max_T_decoder and False in finished:
                t += 1
                word_embed = torch.zeros(batch_size, 1, self.embedding_dim)
                if t == 1:
                    dec_out_t, ht = self._decoding(dec_inp_t)
                else:
                    dec_out_t, ht = self._decoding(dec_inp_t, ht)
                assert dec_out_t.shape[0] == batch_size and dec_out_t.shape[1] == 1 and dec_out_t.shape[2] == self.vocab_size
                logits_dec[:,t-1] = dec_out_t.view(batch_size, self.vocab_size)
                i_t = torch.max(dec_out_t, dim=2)[1]
                assert i_t.shape[0] == batch_size and i_t.shape[1] == 1
                
                pred_words = [index_to_word_dict[i.item()] for i in i_t[:,0]]
                assert len(pred_words) == batch_size, "pred_words " + str(len(pred_words)) + " batch_size " + str(batch_size)
                
                for i in range(batch_size):
                    if pred_words[i] == "_EOS_":
                        finished[i] = True
                    if not finished[i]:
                        pred_expls[i] += " " + pred_words[i]
                    word_embed[i,0] = self._word_embedding(i_t[i,:]).view(-1)
                    #word_embed[i,0] = self._word_embedding(out_classes_inds[i,:]).view(-1)
                word_embed = torch.autograd.Variable(word_embed.to(self.device))
                assert word_embed.shape[0] == batch_size and word_embed.shape[1] == 1 and word_embed.shape[2] == self.embedding_dim
                dec_inp_t = torch.cat([word_embed, v.view(batch_size,1,v.shape[1])], dim=-1)
                
            probabilities_dec = nn.functional.softmax(logits_dec, dim=-1)
            
            if print_explanations:
                print("Explanations for batch:")
                for i in range(batch_size):
                    premise_str = ""
                    premise_inds = premises[i,:]
                    premise_words = [index_to_word_dict[x.item()] for x in premise_inds]
                    t = 0
                    while premise_words[t] != "_EOS_" and t < premises_lengths[i]:
                        premise_str += " " + premise_words[t]
                        t += 1
                    
                    hypothesis_str = ""
                    hypothesis_inds = hypotheses[i,:]
                    hypothesis_words = [index_to_word_dict[x.item()] for x in hypothesis_inds]
                    t = 0
                    while hypothesis_words[t] != "_EOS_" and t < hypotheses_lengths[i]:
                        hypothesis_str += " " + hypothesis_words[t]
                        t += 1
                    
                    print("*" * 20) 
                    print("Premise " + str(i) + ": " + premise_str)
                    print("Hypothesis " + str(i) + ": " + hypothesis_str)
                    print("Explanation " + str(i)+ ": " + pred_expls[i])
                    
            
            return logits, probabilities, logits_dec, probabilities_dec
                        
            
            

                        

        


def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
