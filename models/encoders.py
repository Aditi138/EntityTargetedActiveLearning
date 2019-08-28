__author__ = 'chuntingzhou'
from utils.util import *

''' Designing idea: the encoder should be agnostic to the input, it can be either
    arbitrary spans, characters, or words, or even raw feature. However, user has to specify
    whether to have the lookup table for any input.

    There are also two ways to feed in multiple input features:
    (a) First concatenate all features for each position, and then use them as features for one encoder, e.g. bilstm
    (b) Use multiple encoders for multiple features then combine outputs from multiple encoders, either concat them
        or feed them to another encoder.'''


class Encoder():
    def __init__(self):
        pass

    def encode(self):
        raise NotImplementedError

# class concat_input_encoder(encoder):
#     def __init__(self, model, lookups, lookup_table_dims):
#         # length of elements in lookup_table_dims == number of elements in lookups which are true
#         self.num_inputs = len(lookups)
#         self.lookups = lookups
#         self.lookup_params = []
#         for i, lookup in enumerate(lookups):
#             if lookup == 1:
#                 # add loop up parameters
#                 self.lookup_params.append(model.add_lookup_parameters((lookup_table_dims[i][0], lookup_table_dims[i][1])))
#             elif lookup == 2:
#                 # add normal transformation parameters
#                 # dims: discrete_feature_num, continuous_emb_dim
#                 # the input should concatenate all the discrete features together first
#                 self.lookup_params.append(model.add_parameters((lookup_table_dims[i][0], lookup_table_dims[i][1])))
#             else:
#                 self.lookup_params.append(0)
#
#     def prepare_inputs(self, inputs):
#         # inputs: (a)
#         input_features = []
#         for i, lookup in enumerate(self.lookups):
#             if lookup == 1:


class Lookup_Encoder(Encoder):
    def __init__(self, model, args, vocab_size, emb_size, padding_token=None, pretrain_embedding=None, isFeatureEmb=False):
        Encoder.__init__(self)
        self.padding_token = padding_token
        self.map_pretrain = args.map_pretrain
        self.pretrain_fix = args.pretrain_fix
        self.isFeatureEmb = isFeatureEmb
        if args.map_pretrain:
            self.W_map = model.add_parameters((args.map_dim, emb_size))
            self.b_map = model.add_parameters(args.map_dim)
            self.b_map.zero()
        if pretrain_embedding is not None:
            self.lookup_table = model.lookup_parameters_from_numpy(pretrain_embedding)
        else:
            self.lookup_table = model.add_lookup_parameters((vocab_size, emb_size))

    def encode(self, input_seqs):
        transpose_inputs, _ = transpose_input(input_seqs, self.padding_token)
        embs = [dy.lookup_batch(self.lookup_table, wids) for wids in transpose_inputs]
        if self.pretrain_fix and not self.isFeatureEmb:
            embs = [dy.nobackprop(emb) for emb in embs]
        # TODO: initialize <unk> with ones vector, initialize W_map with identity matrix
        if self.map_pretrain and not self.isFeatureEmb:
            if not self.pretrain_fix:
                embs = [dy.nobackprop(emb) for emb in embs]
            W_map = dy.parameter(self.W_map)
            b_map = dy.parameter(self.b_map)
            embs = [dy.affine_transform([b_map, W_map, emb]) for emb in embs]
        return embs


class Discrete_Feature_Encoder(Encoder):
    def __init__(self, model, num_feats, to_dim):
        Encoder.__init__(self)
        self.num_feats = num_feats
        self.to_dim = to_dim
        self.W_feat_emb = model.add_parameters((to_dim, num_feats))

    def encode(self, input_feats):
        batch_size = len(input_feats)
        # after transpose: input_feats: [(num_feats, batch_size)]
        input_feats = transpose_discrete_features(input_feats)
        W_feat_emb = dy.parameter(self.W_feat_emb)
        output_emb = []
        for wif in input_feats:
            extend_wif = dy.transpose(dy.concatenate_cols([wif for _ in range(self.to_dim)]))
            feature_emb = dy.cmult(extend_wif, W_feat_emb)
            output_emb.append(dy.reshape(feature_emb, (self.to_dim * self.num_feats, ), batch_size=batch_size))
        return output_emb


class CNN_Encoder(Encoder):
    def __init__(self, model, emb_size, win_size=3, filter_size=64, dropout=0.5, vocab_size=0, padding_token=0, lookup_emb=None):
        Encoder.__init__(self)
        self.vocab_size = vocab_size # if 0, no lookup tables
        self.win_size = win_size
        self.filter_size = filter_size
        self.emb_size = emb_size
        self.dropout_rate = dropout
        self.paddding_token = padding_token
        if vocab_size != 0:
            print("In CNN encoder: creating lookup embedding!")
            self.lookup_emb = model.add_lookup_parameters((vocab_size, 1, 1, emb_size))
        else:
            assert lookup_emb is not None
            print("In CNN encoder: reusing lookup embedding!")
            self.lookup_emb = lookup_emb

        self.W_cnn = model.add_parameters((1, win_size, emb_size, filter_size))
        self.b_cnn = model.add_parameters((filter_size))
        self.b_cnn.zero()

    def _cnn_emb(self, input_embs, training):
        # input_embs: (h, time_step, dim, batch_size), h=1
        if self.dropout_rate > 0 and training:
            input_embs = dy.dropout(input_embs, self.dropout_rate)
        W_cnn = dy.parameter(self.W_cnn)
        b_cnn = dy.parameter(self.b_cnn)

        cnn_encs = dy.conv2d_bias(input_embs, W_cnn, b_cnn, stride=(1, 1), is_valid=False)
        tanh_cnn_encs = dy.tanh(cnn_encs)
        max_pool_out = dy.reshape(dy.max_dim(tanh_cnn_encs, d=1), (self.filter_size,))
        # rec_pool_out = dy.rectify(max_pool_out)
        return max_pool_out

    def encode(self, input_seqs, training=True, char=True):
        batch_size = len(input_seqs)
        sents_embs = []
        if char:
            # we don't batch at first, we batch after cnn
            for sent in input_seqs:
                sent_emb = []
                for w in sent:
                    if len(w) < self.win_size:
                        w += [self.paddding_token] * (self.win_size - len(w))
                    input_embs = dy.concatenate([dy.lookup(self.lookup_emb, c) for c in w], d=1)
                    w_emb = self._cnn_emb(input_embs, training)  # (filter_size, 1)
                    sent_emb.append(w_emb)
                sents_embs.append(sent_emb)
            sents_embs, sents_mask = transpose_and_batch_embs(sents_embs, self.filter_size) # [(filter_size, batch_size)]
        else:
            for sent in input_seqs:
                if self.vocab_size != 0:
                    if len(sent) < self.win_size:
                        sent += [0] * (self.win_size - len(sent))
                    input_embs = dy.concatenate([dy.lookup(self.lookup_emb, w) for w in sent], d=1)
                else:
                    # input_seqs: [(emb_size, batch_size)]
                    if len(sent) < self.win_size:
                        sent += [dy.zeros(self.emb_size)] * (self.win_size - len(sent))
                    input_embs = dy.transpose(dy.concatenate_cols(sent)) # (time_step, emb_size, bs)
                    input_embs = dy.reshape(input_embs, (1, len(sent), self.emb_size), )

                sent_emb = self._cnn_emb(input_embs, training)  # (filter_size, 1)
                sents_embs.append(sent_emb)
            sents_embs = dy.reshape(dy.concatenate(sents_embs, d=1), (self.filter_size,), batch_size =batch_size) # (filter_size, batch_size)

        return sents_embs


class BiRNN_Encoder(Encoder):
    def __init__(self,
                 model,
                 input_dim,
                 hidden_dim,
                 emb_dropout_rate=0.3,
                 output_dropout_rate=0.5,
                 padding_token=None,
                 vocab_size=0,
                 emb_size=0,
                 layer=1,
                 rnn="lstm",
                 vocab_emb=None):
        Encoder.__init__(self)
        # self.birnn = dy.BiRNNBuilder(layer, input_dim, hidden_dim, model, dy.LSTMBuilder if rnn == "lstm" else dy.GRUBuilder)
        self.fwd_RNN = dy.LSTMBuilder(layer, input_dim, hidden_dim, model) if rnn == "lstm" else dy.GRUBuilder(layer, input_dim, hidden_dim, model)
        self.bwd_RNN = dy.LSTMBuilder(layer, input_dim, hidden_dim, model) if rnn == "lstm" else dy.GRUBuilder(layer, input_dim, hidden_dim, model)

        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.padding_token = padding_token
        self.drop_out_rate = output_dropout_rate
        self.emb_drop_rate = emb_dropout_rate
        self.hidden_dim = hidden_dim
        if vocab_size > 0:
            print("In BiRNN, creating lookup table!")
            self.vocab_emb = model.add_lookup_parameters((vocab_size, emb_size))
        else:
            if vocab_emb is not None:
                # assert vocab_emb is not None
                self.vocab_emb = vocab_emb
            else:
                self.vocab_emb = None

    def encode(self, input_seqs, training=True, char=False):
        if char:
            return self.encode_word(input_seqs, training=training)
        else:
            return self.encode_seq(input_seqs, training=training)

    def encode_seq(self, input_seqs, training=True, char=False):
        if self.vocab_emb is not None:
            # input_seqs = [[w1, w2],[]]
            transpose_inputs, _ = transpose_input(input_seqs, self.padding_token)
            if self.vocab_size != 0:
                w_embs = [dy.dropout(dy.lookup_batch(self.vocab_emb, wids),
                                     self.emb_drop_rate) if self.emb_drop_rate > 0. and training
                          else dy.lookup_batch(self.vocab_emb, wids)
                          for wids in transpose_inputs]
            else:
                # print "In our case, use parameters shared by CNN char encoder, need conversion!"
                vocab_emb = dy.parameter(self.vocab_emb)
                vocab_size = vocab_emb.dim()[0][-1]
                # print "In BiRNN Char vocab size: ", vocab_size
                vocab_emb = dy.reshape(vocab_emb, (self.input_dim, vocab_size))  # expression, not lookup_parameters

                # for wids in transpose_inputs:
                #     print wids
                #     print vocab_emb.dim()
                #     a = dy.pick_batch(vocab_emb, wids, dim=1)
                #     print a.value()
                # Special case handler: use pick_batch
                w_embs = [dy.dropout(dy.pick_batch(vocab_emb, wids, dim=1),
                                     self.emb_drop_rate) if self.emb_drop_rate > 0. and training
                          else dy.pick_batch(vocab_emb, wids, dim=1)
                          for wids in transpose_inputs]
                # print "In BiRNN char: ", w_embs[0].dim()
        else:
            w_embs = [dy.dropout(emb, self.emb_drop_rate) if self.emb_drop_rate > 0. and training else emb for emb in input_seqs]
        # if vocab_size = 0: input_seqs = [(input_dim, batch_size)]

        w_embs_r = w_embs[::-1]
        # birnn_outputs = [dy.dropout(emb, self.drop_out_rate) if self.drop_out_rate > 0. else emb for emb in self.birnn.transduce(w_embs)]
        fwd_vectors = self.fwd_RNN.initial_state().transduce(w_embs)
        bwd_vectors = self.bwd_RNN.initial_state().transduce(w_embs_r)[::-1]

        if char:
            return dy.concatenate([fwd_vectors[-1], bwd_vectors[0]])

        birnn_outputs = [dy.dropout(dy.concatenate([fwd_v, bwd_v]), self.drop_out_rate) if self.drop_out_rate > 0.0 and training
                         else dy.concatenate([fwd_v, bwd_v])
                         for (fwd_v, bwd_v) in zip(fwd_vectors, bwd_vectors)]
        return birnn_outputs

    def encode_word(self, input_seqs, training=True):
        # embedding dropout rate is 0.0, because we dropout at the later stage of RNN
        sents_embs = []

        for sent in input_seqs:
            sent_emb = []
            for w in sent:
                w_emb = self.encode_seq([w], training=training, char=True)
                sent_emb.append(w_emb)
            sents_embs.append(sent_emb)
        sents_embs, sents_mask = transpose_and_batch_embs(sents_embs, self.hidden_dim*2)  # [(hidden_dim*2, batch_size)]
        return sents_embs