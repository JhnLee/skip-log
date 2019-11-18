import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab, embedding_hidden_dim, gru_hidden_dim, num_hidden_layer, dropout_p):
        super(Encoder, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.num_hidden_layer = num_hidden_layer
        self.embedding_layer = nn.Embedding(len(vocab), embedding_hidden_dim, padding_idx=vocab.index('<PAD>'))
        self.dropout = nn.Dropout(dropout_p)
        self.gru_layer = nn.GRU(embedding_hidden_dim, gru_hidden_dim,
                                num_layers=num_hidden_layer,
                                batch_first=True,
                                dropout=dropout_p if num_hidden_layer > 1 else 0,
                                bidirectional=True)
        self.fc = nn.Linear(gru_hidden_dim * 2, gru_hidden_dim)

    def forward(self, input_sequence):

        # input_sequence : (B x L)
        # input_length : (B)
        # N -> num_layers

        batch_size, max_len = input_sequence.shape

        embedded = self.embedding_layer(input_sequence)  # (B x L x h)
        embedded = self.dropout(embedded)
        encoder_outputs, encoder_hidden = self.gru_layer(embedded)  # (B x L x 2H), (2*N x B x H)

        # Sum forward and backward hidden state   (B x L x 2H)  =>  (B x L x H)
        encoder_outputs = encoder_outputs[:, :, :self.gru_hidden_dim] + encoder_outputs[:, :, self.gru_hidden_dim:]

        # Concat forward and backward hidden states w/ activation function
        encoder_hidden = encoder_hidden.view(self.num_hidden_layer, 2, batch_size, self.gru_hidden_dim)[-1]
        encoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), 1)  # (2 x B x H)  =>  (B x 2H)
        encoder_hidden = self.fc(encoder_hidden).unsqueeze(0)  # (B x 2H)  =>  (1 x B x H)

        return encoder_outputs, encoder_hidden


class Decoder(nn.Module):
    def __init__(self, vocab, embedding_hidden_dim, gru_hidden_dim, attention_method):
        super(Decoder, self).__init__()
        self.gru_layer = nn.GRU(embedding_hidden_dim, gru_hidden_dim)
        self.attention_layer = Attention(attention_method, gru_hidden_dim * 2)
        self.concat_layer = nn.Linear(gru_hidden_dim * 2, gru_hidden_dim)
        self.fc_layer = nn.Linear(gru_hidden_dim, len(vocab))

    def forward(self, target_embedded, encoder_outputs, last_hidden, encoder_mask):

        # target_embedded : (1 x B x h)
        # encoder_outputs : (B x L x H)
        # last_hidden : (1 x B x H)

        decoder_output, last_hidden = self.gru_layer(target_embedded, last_hidden)  # (1 x B x H)

        attention_weight = self.attention_layer(decoder_output, encoder_outputs, encoder_mask)  # (B x 1 x L)
        context = attention_weight.bmm(encoder_outputs)  # (B x 1 x L) * (B x L x H) = (B x 1 x H)

        concat_input = torch.cat((decoder_output.squeeze(0), context.squeeze(1)), 1)  # (B x 2H)
        concat_output = torch.tanh(self.concat_layer(concat_input))  # (B x H)

        out = self.fc_layer(concat_output).unsqueeze(0)  # (B x V)

        return out, last_hidden


class SkipLog(nn.Module):
    def __init__(self, vocab, embedding_hidden_dim, num_hidden_layer, gru_hidden_dim, device,
                 dropout_p=0.1, attention_method='dot'):
        super(SkipLog, self).__init__()
        self.device = device
        self.vocab_size = len(vocab)
        self.encoder = Encoder(vocab, embedding_hidden_dim, gru_hidden_dim, num_hidden_layer, dropout_p)
        self.embedding_layer = nn.Embedding(self.vocab_size, embedding_hidden_dim, padding_idx=vocab.index('<PAD>'))
        self.dropout = nn.Dropout(dropout_p)
        self.prev_decoder = Decoder(vocab, embedding_hidden_dim, gru_hidden_dim, attention_method)
        self.next_decoder = Decoder(vocab, embedding_hidden_dim, gru_hidden_dim, attention_method)
        self.prev_loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.index('<PAD>'))
        self.next_loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.index('<PAD>'))

    def forward(self, encoder_mask, encoder_input, decoder_input, decoder_target):

        # encoder_input : (B x L)
        # decoder_input : (B x 2L)
        # decoder_target : (B x 2L)

        batch_size, max_len = encoder_input.shape

        encoder_outputs, hidden = self.encoder(encoder_input)  # (B x L x 2H), (B x H)

        decoder_embedded = self.embedding_layer(decoder_input.transpose(0, 1))  # (2L x B x H)
        decoder_embedded = self.dropout(decoder_embedded)

        prev_embedded = decoder_embedded[:max_len]
        next_embedded = decoder_embedded[max_len:]

        prev_hidden = hidden
        next_hidden = hidden

        outputs = torch.zeros(2 * max_len, batch_size, self.vocab_size).to(self.device)  # (2L x B x V)

        for t in range(max_len):
            prev_output, prev_hidden = self.prev_decoder(target_embedded=prev_embedded[t].unsqueeze(0),
                                                         encoder_outputs=encoder_outputs,
                                                         last_hidden=prev_hidden,
                                                         encoder_mask=encoder_mask)
            next_output, next_hidden = self.next_decoder(target_embedded=next_embedded[t].unsqueeze(0),
                                                         encoder_outputs=encoder_outputs,
                                                         last_hidden=next_hidden,
                                                         encoder_mask=encoder_mask)

            outputs[t] = prev_output.squeeze(0)
            outputs[max_len + t] = next_output.squeeze(0)

        # (2L x B x V) , (B x 2L)  =>  (B*2L x V) , (B*2L)
        prev_target = decoder_target.transpose(1, 0)[:max_len]
        next_target = decoder_target.transpose(1, 0)[max_len:]

        prev_loss = self.prev_loss_fn(outputs[:max_len].view(-1, self.vocab_size), prev_target.reshape(-1))
        next_loss = self.next_loss_fn(outputs[max_len:].view(-1, self.vocab_size), next_target.reshape(-1))

        return outputs, prev_loss + next_loss


class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        """ Implementation of various attention score function """
        super(Attention, self).__init__()
        self.method = method

        if self.method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)

        elif self.method == 'concat':  # Additive
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Linear(hidden_size, 1)

        elif self.method == 'dot':
            pass

    def forward(self, decoder_outputs, encoder_outputs, encoder_mask):

        attn_energies = self.score(decoder_outputs, encoder_outputs, encoder_mask)
        attn_weight = torch.softmax(attn_energies, 2)

        return attn_weight  # (B x 1 x L)

    def score(self, decoder_outputs, encoder_outputs, encoder_mask):
        # encoder outputs : (B x L x H)
        decoder_outputs = decoder_outputs.transpose(0, 1)  # (B x 1 x H)
        if self.method == 'dot':
            # TODO : scaled dot product attention
            energy = torch.bmm(decoder_outputs, encoder_outputs.transpose(1, 2))  # (B x 1 x L)

        elif self.method == 'general':
            energy = torch.bmm(decoder_outputs, self.attn(encoder_outputs).transpose(1, 2))  # (B x 1 x L)

        elif self.method == 'concat':
            seq_length = encoder_outputs.shape[1]
            concat = torch.cat((decoder_outputs.repeat(1, seq_length, 1), encoder_outputs), 2)
            energy = self.v(torch.tanh(self.attn(concat))).transpose(1, 2)  # (B x 1 x L)

        else:
            raise ValueError("Invalid attention method")

        # Mask to pad token
        energy = energy.masked_fill(encoder_mask.unsqueeze(1) == 0, -1e10)

        return energy

