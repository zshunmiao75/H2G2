# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

import dgl
import dgl.nn.pytorch as dglnn
from torch_geometric.nn import GCNConv



def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=16,
                initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BERTLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class BERTEmbeddings(nn.Module):
    def __init__(self, config):
        super(BERTEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.speaker_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)            

    def forward(self, input_ids, speaker_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        speaker_embeddings = self.speaker_embeddings(speaker_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings + speaker_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BERTSelfAttention(nn.Module):
    def __init__(self, config):
        super(BERTSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BERTSelfOutput(nn.Module):
    def __init__(self, config):
        super(BERTSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTAttention(nn.Module):
    def __init__(self, config):
        super(BERTAttention, self).__init__()
        self.self = BERTSelfAttention(config)
        self.output = BERTSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BERTIntermediate(nn.Module):
    def __init__(self, config):
        super(BERTIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BERTOutput(nn.Module):
    def __init__(self, config):
        super(BERTOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTLayer(nn.Module):
    def __init__(self, config):
        super(BERTLayer, self).__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BERTEncoder(nn.Module):
    def __init__(self, config):
        super(BERTEncoder, self).__init__()
        layer = BERTLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])    

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BERTPooler(nn.Module):
    def __init__(self, config):
        super(BERTPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config: BertConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(BertModel, self).__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)

    def forward(self, input_ids, speaker_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        #extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, speaker_ids, token_type_ids)
        all_encoder_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output

class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):#GCN_layer(graph_big, {"node": features})["node"]
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn
class MultiHeadAttention_a(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention_a, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn
class TurnLevelLSTM(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 lstm_dropout,
                 dropout_rate):
        super(TurnLevelLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=lstm_dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.bilstm2hiddnesize = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, inputs):
        word = inputs# 1,23,768
        if len(word[0]) > 3:
            for i in range(len(word[0])):
                if i == 0:
                    word_out = self.lstm(word.squeeze(0)[i:i+3].unsqueeze(0))
                    word_out = word_out[0].squeeze(0)
                    lstm_out = word_out[0].unsqueeze(0)
                elif i == (len(word)-1):
                    word_out = self.lstm(word.squeeze(0)[i-2:i+1].unsqueeze(0)) # 3,1,768
                    word_out = word_out[0].squeeze(0)
                    word_out = word_out[2].unsqueeze(0)
                    lstm_out = torch.cat((lstm_out,word_out),0)
                else:
                    word_out = self.lstm(word.squeeze(0)[i-1:i+2].unsqueeze(0))
                    word_out = word_out[0].squeeze(0)
                    word_out = word_out[1].unsqueeze(0)
                    lstm_out = torch.cat((lstm_out,word_out),0)
        else:
            lstm_out = self.lstm(word)[0].squeeze(0)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.bilstm2hiddnesize(lstm_out)
        return lstm_out


class MultiGraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, args, mem_dim, layers, heads):
        super(MultiGraphConvLayer, self).__init__()
        self.args = args
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = nn.Dropout(args.gcn_dropout)

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        self.weight_list = nn.ModuleList()

        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def forward(self, adj_list, gcn_inputs, src_mask):

        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[i]
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                index = i * self.layers + l

                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)

                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))

            gcn_ouputs = torch.cat(output_list, dim=2)
            gcn_ouputs = gcn_ouputs + gcn_inputs

            multi_head_list.append(gcn_ouputs)

        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)

        return adj_list,out, src_mask

class GCN_Pool_for_Single(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, args, mem_dim, layers):
        super(GCN_Pool_for_Single, self).__init__()
        self.args = args
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim
        self.gcn_drop = nn.Dropout(args.gcn_dropout)

        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, 1)

        # dcgcn block
        self.weight_list = nn.ModuleList()

        self.weight_list.append(nn.Linear(self.mem_dim, self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.linear_output = self.linear_output.cuda()

    def forward(self, adj, gcn_inputs):

        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs


        for l in range(self.layers):

            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            outputs = self.gcn_drop(gAxW)

        gcn_outputs = outputs
        out = self.linear_output(gcn_outputs)

        return out

def Top_K(score, ratio):
    #batch_size = score.size(0)
    node_sum = score.size(1)
    score = score.view(-1,node_sum)
    K = int(ratio*node_sum)+1
    Top_K_values, Top_K_indices =  score.topk(K, largest=False, sorted=False)
    return Top_K_values, Top_K_indices

class SAGPool_Multi(torch.nn.Module):
    def __init__(self,args, ratio=0.5,non_linearity=torch.tanh, heads = 3):
        super(SAGPool_Multi,self).__init__()
        #self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = GCN_Pool_for_Single(args, args.graph_hidden_size,1)
        self.linear_transform = nn.Linear(args.graph_hidden_size, args.graph_hidden_size//heads)
        self.non_linearity = non_linearity

    def forward(self, adj_list, x, src_mask):
        '''if batch is None:
            batch = edge_index.new_zeros(x.size(0))'''
        # x = x.unsqueeze(-1) if x.dim() == 1 else x
        adj_list_new = []
        src_mask_list = []
        #src_mask_source = src_mask
        x_list = []
        x_select_list = []
        for adj in adj_list:

            score = self.score_layer(adj, x)

            _, idx = Top_K(score, self.ratio)

            x_selected = self.linear_transform(x)
            x_select_list.append(x_selected)



            for i in range(src_mask.size(0)):
                for j in range(idx.size(1)):
                    src_mask[i][0][idx[i][j]] = False
            src_mask_list.append(src_mask)
            adj_list_new.append(adj)
        src_mask_out = torch.zeros_like(src_mask_list[0]).cuda()

        x = torch.cat(x_select_list, dim=2)
        for src_mask_i in src_mask_list:
            src_mask_out = src_mask_out + src_mask_i

        return adj_list_new, x, src_mask_out

def kl_div_gauss(mean_1, mean_2, std_1, std_2):
    kld_element = 0.5*(2*torch.log(std_2) - 2*torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2))/std_2.pow(2) -1)
    return kld_element

class GGG(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.num_heads = args.heads
        self.hid_dim = args.graph_hidden_size
        self.R = self.hid_dim // self.num_heads
        self.max_len_left = args.max_offset
        self.max_len_right = args.max_offset


        self.transform = nn.Linear(self.hid_dim*2, self.hid_dim)
        self.gauss = nn.Linear(self.hid_dim, self.num_heads * 2)

        self.pool_X = nn.MaxPool1d(args.max_offset*2 + 1, stride=1, padding=args.max_offset)
        self.dropout = nn.Dropout(args.input_dropout)


    def forward(self, x, adj, mask = None):
        B, T, C = x.size()#x 4,128,300   gcn_inputs,adj
        H = self.num_heads



        x_pooled = self.pool_X(x)  #最大池化
        x_new = torch.cat([x_pooled,x],dim=2)
        x_new = self.transform(x_new) #全连接
        x_new = self.dropout(x_new)



        gauss_parameters = self.gauss(x_new)  #高斯分布
        gauss_mean, gauss_std = gauss_parameters[:,:,:H], F.softplus(gauss_parameters[:,:,H:])



        kl_div = kl_div_gauss(gauss_mean.unsqueeze(1).repeat(1,T,1,1),gauss_mean.unsqueeze(2).repeat(1,1,T,1),gauss_std.unsqueeze(1).repeat(1,T,1,1),gauss_std.unsqueeze(2).repeat(1,1,T,1))
        adj_multi = kl_div

        attn_adj_list = [attn_adj.squeeze(3) for attn_adj in torch.split(adj_multi, 1, dim=3)]



        return x_new, attn_adj_list#4,128,300    4 128 128

class PoolGCN(nn.Module):
    def  __init__(self, config, args):
        super().__init__()

        self.in_dim = config.hidden_size
        self.mem_dim = args.graph_hidden_size
        self.input_W_G = nn.Linear(self.in_dim, self.mem_dim)
        self.in_drop = nn.Dropout(args.input_dropout)
        self.num_layers = args.num_graph_layers
        self.layers = nn.ModuleList()
        self.heads = args.heads
        self.sublayer_first = args.sublayer_first
        self.sublayer_second = args.sublayer_second



        # gcn layer
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(MultiGraphConvLayer(args, self.mem_dim, self.sublayer_first, self.heads))
                self.layers.append(SAGPool_Multi(args, ratio=args.pooling_ratio, heads= self.heads))
                self.layers.append(MultiGraphConvLayer(args, self.mem_dim, self.sublayer_second, self.heads))
                self.layers.append(SAGPool_Multi(args, ratio=args.pooling_ratio, heads = self.heads))
            else:
                self.layers.append(MultiGraphConvLayer(args, self.mem_dim, self.sublayer_first, self.heads))
                self.layers.append(SAGPool_Multi(args, ratio=args.pooling_ratio, heads= self.heads))
                self.layers.append(MultiGraphConvLayer(args, self.mem_dim, self.sublayer_second, self.heads))
                self.layers.append(SAGPool_Multi(args, ratio=args.pooling_ratio, heads = self.heads))
        self.agg_nodes_num = int(len(self.layers)//2 * self.mem_dim )
        self.aggregate_W = nn.Linear(self.agg_nodes_num, self.mem_dim)

        self.attn = MultiHeadAttention_a(self.heads, self.mem_dim)

        self.GGG = GGG(config, args)



    def forward(self, adj, inputs, input_id):
        #adj 16 128 128
        src_mask = (input_id != 0).unsqueeze(-2)#16,1,128
        src_mask = src_mask[:,:,:adj.size(2)]
        embs = self.in_drop(inputs)   #dropout
        gcn_inputs = embs#torch.Size([16, 128, 768])
        gcn_inputs = self.input_W_G(gcn_inputs)#nn.Linear(self.in_dim, self.mem_dim) 16 768 300

        layer_list = []



        gcn_inputs, attn_adj_list = self.GGG(gcn_inputs,adj)
        outputs = gcn_inputs# 4 128 300

        for i in range(len(self.layers)):
            if i < 4:
                attn_adj_list, outputs, src_mask = self.layers[i](attn_adj_list, outputs, src_mask)
                if i==0:
                    src_mask_input = src_mask
                if i%2 !=0:
                    layer_list.append(outputs)

            else:
                attn_tensor = self.attn(outputs, outputs, src_mask)
                attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
                attn_adj_list, outputs, src_mask = self.layers[i](attn_adj_list, outputs, src_mask)

                if i%2 !=0:
                    layer_list.append(outputs)


        aggregate_out = torch.cat(layer_list, dim=2)

        dcgcn_output = self.aggregate_W(aggregate_out) #全连接

        mask_out = src_mask.reshape([src_mask.size(0),src_mask.size(2),src_mask.size(1)])




        return dcgcn_output, mask_out, layer_list, src_mask_input

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -1e12)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)




class HAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(HAttentionLayer, self).__init__()
#        self.dropout = dropout # dropout率
        self.hidden_size = hidden_size # 输入特征维度
#        self.out_features = out_features # 输出特征维度
 #       self.alpha = alpha # LeakyReLU的负斜率


        # 定义可学习的参数
        self.W = nn.Linear(self.hidden_size, 1) # 权重矩阵W

        self.score =  nn.Linear(self.hidden_size, 1)
        # 定义LeakyReLU激活函数
        self.leakyrelu = nn.LeakyReLU(0.01)

    def forward(self, input, word_att, a_idx, b_idx, speak_num,speak_all):
        # input shape: (batch,N, in_features), N为节点数
        # adj shape: (N, N)，邻接矩阵
        # 线性变换
        #a_idx (6,1) speak_num(6,1) ,speak_all(6,4,512)
#        batch_num = input.shape[0]
        x_output = list()
        for input_x in range(len(input)):
            h = input[input_x]#(512,768)
            N = h.size()[0] # 获得节点数
            e = word_att[input_x][0]
        # 计算注意力分数
#            e =  Hgcn_e (h,N,self.hidden_size,self.a)torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.hidden_size)
        # a_input shape: (N, N, 2 * out_features)
#            e = self.leakyrelu(e)
        # e shape: (N, N)
        # softmax归一化
#        zero_vec = -9e15 * torch.ones_like(e)  # 一个很小的向量，用于填充邻接矩阵中的0元素
#        attention = torch.where(adj > 0, e, zero_vec)  # 将邻接矩阵中的0元素替换为很小的向量
#            attention = F.softmax(e, dim=1)  # 按行进行softmax归一化，得到注意力系数
            attention = e  # 按行进行softmax归一化，得到注意力系数

            a_indices = torch.zeros(N).cuda()
            for a_num in a_idx[input_x]:
                if int(a_num.item()) != 0:
                    attention_a = attention[int(a_num.item())]
                    _, a_index = torch.topk(attention_a, 20)
                    a_indices[a_index] = 1
            b_indices = torch.zeros(N).cuda()
            for b_num in b_idx[input_x]:
                if int(b_num.item()) != 0:
                    attention_b = attention[int(b_num.item())]
                    _, b_index = torch.topk(attention_b, 20)
                    b_indices[b_index] = 1
#                   b_indices.index_fill(1, b_index, 1)



            Hyper = torch.stack((a_indices,b_indices),0)
            for num in range(int(speak_num[input_x][0])):
                if num < 3:
                    Hyper = torch.cat((Hyper,speak_all[input_x][num].unsqueeze(0)),dim = 0)#(4,512)

            HyperT = Hyper.t()
            Dv = HyperT.sum(dim=1)#(512,）
            Ev = torch.ones(N).cuda()
            Dv = torch.add(torch.diag(Dv),torch.diag(Ev))
            Dv = torch.inverse(Dv)
            Dv = torch.sqrt(Dv)

            De = Hyper.sum(dim=1)
            De = torch.diag(De)
            De = torch.inverse(De)
            De = torch.sqrt(De)

            attention = F.dropout(attention)  # 对注意力系数进行dropout
            Hyper = torch.matmul(Hyper, attention)

            Hyper_att = F.softmax(Hyper)#(4,512)

            Hyper_prime = torch.matmul(Hyper_att,h)#(4,768)

            W_Hyper = F.softmax(self.W(Hyper_prime),dim = 0) #(4,1)

            matrix = W_Hyper.view(len(Hyper))#4

            W_Hyper= torch.diag(matrix)

            A = torch.mm(Dv,HyperT)#(512,4)
            A = torch.mm(A,W_Hyper)#(512,4)
            A = torch.mm(A,De)#(512,4)
            A = torch.mm(A,Hyper)#(512,512)
            A = torch.mm(A,Dv)#(512,512)

            X = torch.matmul(A, h)  # (512,768)
            X_score = self.score(X)  # (512,1)
#            X_score = F.softmax(X_score, dim=0)  # (512,1)
            #           k_num = 0.1*N
            topk_values, topk_indices = torch.topk(X_score.view(-1), k=2)
            for num,top_id in enumerate(topk_indices):
                if num == 0:
                    X_top =  X[top_id]
                else:
                    X_top = torch.cat((X_top,X[top_id]),dim = -1)
            #            X_fin = torch.mean(X_top, dim = 0)
            if input_x == 0:
                x_output = X_top.unsqueeze(0)  # (6,1536)
            else:
                x_output = torch.cat((x_output,X_top.unsqueeze(0)),dim = 0)
 #       if self.concat:
            # 如果拼接，返回激活后的输出
#            return F.elu(h_prime)
#        else:
            # 如果不拼接，返回原始输出
        return x_output


class HG2G_BERT(nn.Module):
    def __init__(self, config, num_labels, args, gcn_layers=2, activation='relu', gcn_dropout=0.6):
        super(HG2G_BERT, self).__init__()
        self.args = args
        self.bert = BertModel(config)

        self.conv1 = GCNConv(768, 768)

        self.gcn = PoolGCN(config, args)  # multi-view GCN
        self.Hgcn = HAttentionLayer(768)  # hypergraph
        self.gcn_dim = config.hidden_size
        self.gcn_layers = gcn_layers
        self.num_labels = num_labels

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            assert 1 == 2, "you should provide activation function."

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
#        self.classifier = nn.Linear(config.hidden_size*2, self.num_labels)#768*3*3
        self.classifier = nn.Linear((config.hidden_size*2*(self.gcn_layers+1)+3*config.hidden_size), self.num_labels)#768*3*3
#        self.lyer = nn.Linear(config.hidden_size*2, config.hidden_size)#768*3*3

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.turnAttention = MultiHeadAttention(config.num_attention_heads, config.hidden_size, self.attention_head_size, self.attention_head_size, config.attention_probs_dropout_prob)
        self.wordAttention = MultiHeadAttention(1, config.hidden_size, 768, 768, config.attention_probs_dropout_prob)

        rel_name_lists = ['speaker', 'dialog', 'entity','multi']
        self.GCN_layers = nn.ModuleList([RelGraphConvLayer(self.gcn_dim, self.gcn_dim, rel_name_lists,
                                                           num_bases=len(rel_name_lists), activation=self.activation,
                                                           self_loop=True, dropout=gcn_dropout)
                                          for i in range(self.gcn_layers)])
        #GCN_layer(graph_big, {"node": features})["node"]
        self.LSTM_layers = nn.ModuleList([TurnLevelLSTM(config.hidden_size, 2, 0.2, 0.4) for i in range(self.gcn_layers)])
        
    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                speaker_ids,
                graphs,
                mention_id,
                a_idx,
                b_idx,
                speak_num,
                speak_all,
                labels=None, turn_mask=None):
        slen = input_ids.size(1)
        adj = torch.ones(input_ids.size(0), slen, slen).cuda()
        sequence_outputs, pooled_outputs = self.bert(input_ids, speaker_ids, token_type_ids, attention_mask)
        #torch.Size([6, 512, 768])
        _,word_att = self.wordAttention(sequence_outputs, sequence_outputs, sequence_outputs)
        Hgcn_out = self.Hgcn(sequence_outputs,word_att,a_idx, b_idx, speak_num,speak_all)

        h, pool_mask,layer_list, src_mask_input = self.gcn(adj, sequence_outputs,input_ids)
        h_out = pool(h, pool_mask, type="avg")
#        output_multi = self.dropout(torch.cat([pooled_outputs,h_out],dim=1))#  6,768*2
        output_multi = self.dropout(h_out)#  6,768*2
#        new_output_multi = self.lyer(output_multi)
        features = None
        sequence_outputs, _ = self.turnAttention(sequence_outputs, sequence_outputs, sequence_outputs, turn_mask)#上下句关联
        # torch.Size([6, 512, 768])
        num_batch_turn = []
        #文本 num = len(attention_mask[i].sum())-1   feature[len(attention_mask[i].sum())-4) len(attention_mask[i].sum())
        for i in range(len(graphs)): 
            sequence_output = sequence_outputs[i] # 512 768
            mention_num = torch.max(mention_id[i])  #总轮次
            num_batch_turn.append(mention_num+2)# 总轮次+1
            mention_index = get_cuda(
                (torch.arange(mention_num) + 1).unsqueeze(1).expand(-1, slen))  #1111111 2222222 33333333 4444444444
            mentions = mention_id[i].unsqueeze(0).expand(mention_num, -1)   # (512  1,512 28,512
            select_metrix = (mention_index == mentions).float()    #把每一句单独提出来 【0 1 1 1 2 2 2】 【0 0.1 0.1 0。1 00000】
            word_total_numbers = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, slen)    #每一句几个词   【512】
            select_metrix = torch.where(word_total_numbers > 0, select_metrix / word_total_numbers, select_metrix)  #

            x = torch.mm(select_metrix, sequence_output)  #([9, 512]) ([512, 768]) 每一句特征向量
            x = torch.cat((pooled_outputs[i].unsqueeze(0), x), dim=0) #(10,768) cat 文档
            x = torch.cat((x, output_multi[i].unsqueeze(0)), dim=0) #(10,768) cat 文档
            if features is None:
                features = x
            else:
                features = torch.cat((features, x), dim=0)  #6篇句子特征

        graph_big = dgl.batch(graphs)#打包成大图
        output_features = [features]#(107,768)

        for layer_num, GCN_layer in enumerate(self.GCN_layers):#两层
            start = 0
            new_features = []
            for idx in num_batch_turn:
                new_features.append(features[start])
                lstm_out = self.LSTM_layers[layer_num](features[start+1:start+idx-3].unsqueeze(0))#idx词的数量 从1 到文本词 23 768
                new_features += lstm_out
                new_features.append(features[start+idx-3])
                new_features.append(features[start+idx-2])
                new_features.append(features[start+idx-1])
                start += idx
            features = torch.stack(new_features)
            features = GCN_layer(graph_big, {"node": features})["node"]  
            output_features.append(features)  #3层     output feature      len 3   107 768

        graphs = dgl.unbatch(graph_big)

        graph_output = list() 

        fea_idx = 0
        for i in range(len(graphs)):
            node_num = graphs[i].number_of_nodes('node')#文本句子+实体1+实体2
            intergrated_output = None
            for j in range(self.gcn_layers + 1):
                if intergrated_output == None:
                    intergrated_output = output_features[j][fea_idx+ node_num - 3]#+768
                else:
                    intergrated_output = torch.cat((intergrated_output, output_features[j][fea_idx + node_num - 3]), dim=-1)
#                intergrated_output = torch.cat((intergrated_output, output_features[j][fea_idx + node_num - 3]),dim=-1)  # +768  同一纬度
                intergrated_output = torch.cat((intergrated_output, output_features[j][fea_idx + node_num - 2]),dim=-1)
#            intergrated_output = torch.cat((intergrated_output, output_features[j][fea_idx + node_num - 1]),dim=-1)
            fea_idx += node_num
            graph_output.append(intergrated_output)#list append
        graph_output = torch.stack(graph_output)# tensor stack ([6,6912])
        cat_output = torch.cat((output_multi,graph_output),dim = 1)
        cat_output = torch.cat((cat_output,Hgcn_out),dim = 1)


        pooled_output = self.dropout(cat_output)
#        pooled_output = self.dropout(graph_output)
        logits = self.classifier(pooled_output)#(6,36)
        logits = logits.view(-1, self.num_labels)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()#sigmoid +bceloss
            labels = labels.view(-1, self.num_labels)
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
