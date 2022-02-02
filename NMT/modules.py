import random
import torch
from torch import nn
from torch.nn import functional as F

def softmax(x, temperature=15): # use your temperature
    e_x = torch.exp(x / temperature)
    return e_x / torch.sum(e_x, dim=0)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        
        #src = [src sent len, batch size]
        # Compute an embedding from the src data and apply dropout to it
        embedded = self.dropout(self.embedding(src))
        #embedded = [src sent len, batch size, emb dim]
        
        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        
        outputs, (hidden, cell) = self.rnn(embedded)
        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        if self.bidirectional:
            hidden = hidden.reshape(self.n_layers, 2, -1, self.hid_dim) # n_layers, 2, batch size, hid dim
            hidden = hidden.transpose(1, 2).reshape(self.n_layers, -1, 2 * self.hid_dim) # n_layers, batch size, 2 * hid dim
            cell = cell.reshape(self.n_layers, 2, -1, self.hid_dim) # n_layers, 2, batch size, hid dim
            cell = cell.transpose(1, 2).reshape(self.n_layers, -1, 2 * self.hid_dim) # n_layers, batch size, 2 * hid dim
        
        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, enc_hid_dim)
        self.v = nn.Linear(enc_hid_dim, 1)
        
    def forward(self, hidden, encoder_outputs):
        
        # encoder_outputs = [src sent len, batch size, enc_hid_dim]
        # [src sent len, batch size, hid dim * enc_n directions]
        # hidden = [1, batch size, dec_hid_dim]
        # repeat hidden and concatenate it with encoder_outputs
        hidden = hidden.repeat(encoder_outputs.shape[0], 1, 1) # '''your code'''
        hidden = torch.cat((hidden, encoder_outputs),2) # '''your code'''
        # calculate energy
        energy = torch.tanh(self.attn(hidden)) # '''your code'''
        # get attention, use softmax function which is defined, can change temperature
        attn = softmax(self.v(energy)) # '''your code'''
        return attn # '''your code'''
    

class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim) # '''your code'''
        
        self.rnn = nn.GRU(emb_dim+enc_hid_dim, dec_hid_dim,num_layers=1,dropout=dropout) # use GRU
        
        self.out = nn.Linear(dec_hid_dim*3, output_dim) # '''your code'''  linear layer to get next word
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_data, hidden, encoder_outputs):
        #hidden = [n layers * n directions, batch size, hid dim]
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        
        input_data = input_data.unsqueeze(0) # because only one word, no words sequence 
        
        #input_data = [1, batch size]
        
        embedded = self.dropout(self.embedding(input_data))
        #embedded = [1, batch size, emb dim]
        
        attn = self.attention(hidden[-1],encoder_outputs)
        
        # get weighted sum of encoder_outputs
        weighted = torch.bmm(attn.permute(1,2,0),encoder_outputs.permute(1,0,2)) #'''your code'''
        # concatenate weighted sum and embedded
        weighted = weighted.permute(1,0,2) #'''your code'''
        
        output, hidden = self.rnn(torch.cat((embedded,weighted),2),hidden[-1].unsqueeze(0)) #'''your code'''
        
        # get predictions
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.out(torch.cat([output,weighted,hidden.squeeze(0)],1)) #'''your code'''
        #prediction = [batch size, output dim]
        return prediction, hidden #'''your code'''
        

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim*(encoder.bidirectional + 1) == decoder.dec_hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"  
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_states, hidden, cell = self.encoder(src)
        #first input to the decoder is the <sos> tokens
        input_data = trg[0,:]
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input_data, hidden, enc_states) #'''your code'''
            outputs[t] = output
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            #get the highest predicted token from our predictions
            top1 = output.argmax(-1) 
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input_data = trg[t] if teacher_force else top1
        
        return outputs
