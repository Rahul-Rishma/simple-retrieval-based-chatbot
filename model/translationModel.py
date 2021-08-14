import random
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

import pandas as pd
import numpy as np
import re
from tqdm.notebook import tqdm

df = pd.read_csv('model\Hindi_English_Truncated_Corpus.csv')

df.drop(columns=['source'], inplace=True)

df = df.iloc[:10000, :]


def cleanerEng(x):
    x = str(x)
    x = x.lower()
    x = re.sub(r'[^a-z0-9]+', ' ', x)
    if len(x) > 150:
        x = x[:150]
    return x


def cleanerHindi(x):
    x = str(x)
    x = re.sub(r'[-.।|,?;:<>&$₹]+', ' ', x)
    if len(x) > 150:
        x = x[:150]
    return x


df.iloc[:, 0] = df['english_sentence'].apply(func=cleanerEng)
df.iloc[:, 1] = df['hindi_sentence'].apply(func=cleanerHindi)
df.iloc[:, 0] = df['english_sentence'].apply(func=lambda x: (str(x).split()))
df.iloc[:, 1] = df['hindi_sentence'].apply(func=lambda x: (str(x).split()))


def addTokens(x, start=False):
    x.append('<END>')
    if start:
        x.insert(0, '<START>')
    return list(x)


df.iloc[:, 0] = df['english_sentence'].apply(func=addTokens, start=False)
df.iloc[:, 1] = df['hindi_sentence'].apply(func=addTokens, start=True)

data = df.values


class vocab:

    def __init__(self, data, token=True):
        self.data = data
        if token:
            self.word2idx = {'<START>': 1, '<END>': 2, '<PAD>': 0}
            self.idx2word = {1: '<START>', 2: '<END>', 0: '<PAD>'}
            self.idx = 2

        else:
            self.word2idx = {'<PAD>': 0, '<END>': 1}
            self.idx2word = {0: '<PAD>', 1: '<END>'}
            self.idx = 1

        self.x = []
        self.create()
        self.vocab_size = self.idx + 1

    def create(self):
        max_len = 0
        for sentence in self.data:
            max_len = max(max_len, len(sentence))
            for word in sentence:
                if self.word2idx.get(word) is None:
                    self.idx += 1
                    self.word2idx[word] = self.idx
                    self.idx2word[self.idx] = word

        for sentence in self.data:
            sent = []
            for word in sentence:
                sent.append(self.word2idx[word])

            for i in range(len(sentence), max_len+1):
                sent.append(0)

            self.x.append(torch.Tensor(sent))


English_vocab = vocab(data[:, 0], token=False)
Hindi_vocab = vocab(data[:, 1], token=True)


class parallelData(Dataset):

    def __init__(self):
        self.x = English_vocab.x
        self.y = Hindi_vocab.x

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)


dataset = parallelData()

a = dataset[0][0].shape[0]
b = dataset[0][1].shape[0]
for i in range(len(dataset)):
    if a != dataset[i][0].shape[0] or b != dataset[i][1].shape[0]:
        print(a, dataset[i][0].shape[0], b, dataset[i][1].shape[0])


class encoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, layers, bidirectional):
        '''
        input_size = size of vocab
        embedding_size = embedding dim
        hidden_size = hidden state size
        layer = num of layers of lstms
        '''
        super().__init__()

        self.embed = nn.Embedding(
            num_embeddings=input_size, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                            num_layers=layers, batch_first=True, bidirectional=bidirectional)
        self.bidirectional = bidirectional
        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x):
        '''
        x shape = [batch_size, sentence]
        one complete sentence represents a "sequence"
        '''
        x = self.embed(x)

        output, (hidden_state, cell_state) = self.lstm(x)

        if self.bidirectional:
            hidden = torch.cat((hidden_state[0:1], hidden_state[1:2]), dim=2)

            cell = torch.cat((cell_state[0:1], cell_state[1:2]), dim=2)
            hidden_state = self.fc_hidden(hidden)
            cell_state = self.fc_cell(cell)

        return output, hidden_state, cell_state

# _---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class decoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, layers):
        '''
        same configuration as encoder
        here input_size = size of hindi vocab
        '''
        super().__init__()

        self.embed = nn.Embedding(
            num_embeddings=input_size, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size, num_layers=layers, batch_first=True)

        self.fc = nn.Linear(in_features=hidden_size, out_features=input_size)

    def forward(self, x, hidden_state, cell_state):
        '''
        seq_len would be 1 as input is  one word not the whole sentence
        x = [batch_size] ->required-> [batch_size, 1] (1 is seq_len)
        '''

        x = x.reshape(-1, 1)

        x = self.embed(x)

        output, (hidden_state, cell_state) = self.lstm(
            x, (hidden_state, cell_state))
        output = self.fc(output)

        output = output.squeeze(dim=1)

        return output, hidden_state, cell_state


# _---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class AttnDecoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, layers):
        super().__init__()

        self.embed = nn.Embedding(
            num_embeddings=input_size, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(input_size=hidden_size*2 + embedding_size,
                            hidden_size=hidden_size, num_layers=layers, batch_first=True)

        self.fc = nn.Linear(in_features=hidden_size, out_features=input_size)

        self.energy = nn.Linear(hidden_size*3, 1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden_state, cell_state, encoder_states):
        seq_len = encoder_states.shape[1]
        batch_size = encoder_states.shape[0]
        hidden_size = encoder_states.shape[2]

        h_new = hidden_state.repeat(seq_len, 1, 1)

        h_new = h_new.permute(1, 0, 2)

        energy = self.energy(torch.cat((h_new, encoder_states), dim=2))
        att_weights = self.softmax(energy)
        att_weights = att_weights.permute(0, 2, 1)

        context = torch.bmm(att_weights, encoder_states)

        x = x.reshape(-1, 1)
        x = self.embed(x)

        input_new = torch.cat((context, x), dim=2)

        output, (hidden_state, cell_state) = self.lstm(
            input_new, (hidden_state, cell_state))
        output = self.fc(output)

        output = output.squeeze(dim=1)
        del h_new
        del context
        del input_new
        return output, hidden_state, cell_state

# _---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, target, teaching_force=0.5):
        batch_size = input.shape[0]
        seq_len = target.shape[1]
        hindi_vocab_size = Hindi_vocab.vocab_size

        output = torch.zeros(
            (seq_len, batch_size, hindi_vocab_size)).to(device)

        _, hidden, cell = self.encoder(input)
        target = target.permute(1, 0)
        x = target[0]

        for i in range(1, seq_len):
            out, hidden, cell = self.decoder(x, hidden, cell)
            output[i] = out

            decoder_guess = out.argmax(1)

            if random.random() < teaching_force:
                x = target[i]
            else:
                x = decoder_guess

        return output

# _---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class Attnseq2seq(nn.Module):
    def __init__(self, encoder, att_decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = att_decoder

    def forward(self, input, target, teaching_force=0.5):
        batch_size = input.shape[0]
        seq_len = target.shape[1]
        hindi_vocab_size = Hindi_vocab.vocab_size

        output = torch.zeros(
            (seq_len, batch_size, hindi_vocab_size)).to(device)

        encoder_states, hidden, cell = self.encoder(input)
        target = target.permute(1, 0)
        x = target[0]

        for i in range(1, seq_len):
            out, hidden, cell = self.decoder(x, hidden, cell, encoder_states)
            output[i] = out

            decoder_guess = out.argmax(1)

            if random.random() < teaching_force:
                x = target[i]
            else:
                x = decoder_guess

        return output


epochs = 120
learning_rate = 0.0006
batch_size = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_size = 256
hidden_size = 256
layers = 1
bidirection = True

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

ENC = encoder(English_vocab.vocab_size, embedding_size, hidden_size, layers, bidirection).to(device)

DE = AttnDecoder(Hindi_vocab.vocab_size, embedding_size, hidden_size, layers).to(device)

model = Attnseq2seq(ENC,DE).to(device)
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=0)

train_loss = []
for epoch in tqdm(range(epochs)):
  for id,(x,y) in (enumerate(tqdm(loader))):
    x = x.long().to(device)
    y = y.long().to(device)#[batch,seq]

    output = model(x,y,1)# [seq, batch, vocab]
    output = output[1:].reshape(-1,output.shape[2])
    y = y.permute(1,0)#[seq, batch]
    y = y[1:].reshape(-1)

    optimizer.zero_grad()
    loss = criterion(output,y)

    loss.backward()
    optimizer.step()

    # if id%20 == 0:
  print(f'[{epoch+1}/{epochs}] loss=>{loss.item()}')
  train_loss.append(loss.item())

torch.save(model.state_dict(),'model\model.pt')  

model.load_state_dict(torch.load(
    'model\model.pt', map_location=torch.device('cpu')))


def prediction(x):
    for idx in x:
        if idx == 0:
            break

    print()

    x = x.long().reshape(1, -1).to(device)
    ans = translate(x)
    res = []
    for id in ans:
        res.append(Hindi_vocab.idx2word[id])

    return res


def translate(input):
    with torch.no_grad():
        guess = []
        encoder_states, hidden, cell = model.encoder(input)

        x = torch.ones((1)).long().to(device)
        while True:
            out, hidden, cell = model.decoder(x, hidden, cell, encoder_states)
            x = out.argmax(1)
            guess.append(int(x[0].detach().cpu()))

            if x == 2:
                break

    return guess


def get_prediction(val):
    val = cleanerEng(val)
    keys = np.array([])
    val = val.split()
    for word in val:
        word = word.lower()
        for key, value in English_vocab.idx2word.items():
            if word == value:
                keys = np.append(keys, key)
    keys = np.append(keys, 1)
    while(len(keys) != 37):
        keys = np.append(keys, 0)
    target = torch.tensor(keys, dtype=torch.float32)
    listToString(prediction(target))


def listToString(words):
    words = words[0:len(words)-1]
    sentence = ' '.join(map(str, words))
    print(sentence)