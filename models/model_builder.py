__author__ = 'chuntingzhou and aditichaudhary'
from encoders import *
from decoders import *
from collections import defaultdict
from copy import deepcopy

#np.set_printoptions(threshold='nan')


class CRF_Model(object):
    def __init__(self, args, data_loader, lm_data_loader=None):
        self.save_to = args.save_to_path
        self.load_from = args.load_from_path
        tag_to_id = data_loader.tag_to_id
        self.constraints = None
        # print self.constraints

        #partial CRF
        self.use_partial = args.use_partial
        self.tag_to_id = tag_to_id
        self.B_UNK = data_loader.B_UNK
        self.I_UNK = data_loader.I_UNK

        #active learning for partial annotations
        self.entropy_spans = defaultdict(lambda: 0)
        self.full_sentences = {}
        self.use_CFB = args.use_CFB
        self.addbias = args.addbias
        self.B_tags = []
        self.I_tags = []
        self.O_tags = []
        B_tags = []
        I_tags = []
        for tag in tag_to_id:
            if "B-" in tag:
                B_tags.append(tag)
            elif "I-" in tag:
                I_tags.append(tag)
            elif tag == "O":
                self.O_tags.append(tag_to_id[tag])
        B_tags = sorted(B_tags)
        I_tags = sorted(I_tags)
        self.B_tags = [tag_to_id[tag] for tag in B_tags]
        self.I_tags = [tag_to_id[tag] for tag in I_tags]

    def forward(self, sents, char_sents, feats, bc_feats, training=True):
        raise NotImplementedError

    def save(self):
        if self.save_to is not None:
            self.model.save(self.save_to)
        else:
            print('Save to path not provided!')

    def load(self, path=None):
        if path is None:
            path = self.load_from
        if self.load_from is not None or path is not None:
            print('Load model parameters from %s!' % path)
            self.model.populate(path)
        else:
            print('Load from path not provided!')

    def cal_loss(self, sents, char_sents, ner_tags, feats, bc_feats, known_tags, lm_batch=None, training=True):
        birnn_outputs = self.forward(sents, char_sents, feats, bc_feats, training=training)
        crf_loss = self.crf_decoder.decode_loss(birnn_outputs, ner_tags,self.use_partial, known_tags, self.tag_to_id, self.B_UNK, self.I_UNK)
        return crf_loss#, sum_s, sent_s

    def eval(self, sents, char_sents, feats, bc_feats, training=False,type="eval"):
        birnn_outputs = self.forward(sents, char_sents, feats, bc_feats, training=training)
        best_score, best_path, tag_scores = self.crf_decoder.decoding(birnn_outputs, self.O_tags, addbias=self.addbias)
        best_path_copy = deepcopy(best_path)
        if type == "test":
            alpha_value, alphas = self.crf_decoder.forward_alg(tag_scores)
            beta_value, betas = self.crf_decoder.backward_one_sequence(tag_scores)
            # print("Alpha:{0} Beta:{1}".format(alpha_value.value(), beta_value.value()))
            sent = sents[0]
            gammas = []
            sum = []
            for i in range(len(sent)):
                gammas.append(alphas[i] + betas[i] - alpha_value)

            if self.use_CFB:
                self.crf_decoder.get_uncertain_subsequences_CFB(sent, tag_scores, alphas, betas, alpha_value, gammas,
                                                                best_path_copy, self.tag_to_id
                                                                , self.B_UNK, self.I_UNK)

            else:
                self.crf_decoder.get_uncertain_subsequences(sent, tag_scores, alphas, betas, alpha_value, gammas,
                                                       best_path_copy
                                                       , self.B_tags, self.I_tags, self.O_tags)


            return best_score - alpha_value, best_path
        else:
            return  best_score, best_path

    def eval_scores(self, sents, char_sents, feats, bc_feats, training=False):
        birnn_outputs = self.forward(sents, char_sents, feats, bc_feats, training=training)
        tag_scores, transit_score = self.crf_decoder.get_crf_scores(birnn_outputs)
        return tag_scores, transit_score


class vanilla_NER_CRF_model(CRF_Model):
    ''' Implement End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF. '''
    def __init__(self, args, data_loader, lm_data_loader=None):
        # super(vanilla_NER_CRF_model, self).__init__(args, data_loader)
        self.model = dy.Model()
        self.args = args
        super(vanilla_NER_CRF_model, self).__init__(args, data_loader)

        self.res_discrete = args.res_discrete_feature

        ner_tag_size = data_loader.ner_vocab_size
        char_vocab_size = data_loader.char_vocab_size
        word_vocab_size = data_loader.word_vocab_size
        word_padding_token = data_loader.word_padding_token

        char_emb_dim = args.char_emb_dim
        word_emb_dim = args.word_emb_dim
        tag_emb_dim = args.tag_emb_dim
        birnn_input_dim = args.cnn_filter_size + args.word_emb_dim
        hidden_dim = args.hidden_dim
        src_ctx_dim = args.hidden_dim * 2

        cnn_filter_size = args.cnn_filter_size
        cnn_win_size = args.cnn_win_size
        output_dropout_rate = args.output_dropout_rate
        emb_dropout_rate = args.emb_dropout_rate

        if args.use_discrete_features:
            self.num_feats = data_loader.num_feats
            self.feature_encoder = Discrete_Feature_Encoder(self.model, self.num_feats, args.feature_dim)
            if self.res_discrete:
                src_ctx_dim += args.feature_dim * self.num_feats
            else:
                birnn_input_dim += args.feature_dim * self.num_feats

        if args.use_brown_cluster:
            bc_num = args.brown_cluster_num
            bc_dim = args.brown_cluster_dim
            # for each batch, the length of input seqs are the same, so we don't have bother with padding
            self.bc_encoder = Lookup_Encoder(self.model, args, bc_num, bc_dim, word_padding_token, isFeatureEmb=True)

            if self.res_discrete:
                src_ctx_dim += bc_dim
            else:
                birnn_input_dim += bc_dim

        self.char_cnn_encoder = CNN_Encoder(self.model, char_emb_dim, cnn_win_size, cnn_filter_size,
                                            0.0, char_vocab_size, data_loader.char_padding_token)
        if args.pretrain_emb_path is None:
            self.word_lookup = Lookup_Encoder(self.model, args, word_vocab_size, word_emb_dim, word_padding_token)
        else:
            print("In NER CRF: Using pretrained word embedding!")
            self.word_lookup = Lookup_Encoder(self.model, args, word_vocab_size, word_emb_dim, word_padding_token, data_loader.pretrain_word_emb)
        self.birnn_encoder = BiRNN_Encoder(self.model, birnn_input_dim, hidden_dim, emb_dropout_rate=emb_dropout_rate,
                                           output_dropout_rate=output_dropout_rate)

        self.crf_decoder = chain_CRF_decoder(args, self.model, src_ctx_dim, tag_emb_dim, ner_tag_size, constraints=self.constraints)

    def forward(self, sents, char_sents, feats, bc_feats, training=True):
        char_embs = self.char_cnn_encoder.encode(char_sents, training=training)
        word_embs = self.word_lookup.encode(sents)

        if self.args.use_discrete_features:
            feat_embs = self.feature_encoder.encode(feats)

        if self.args.use_brown_cluster:
            bc_feat_embs = self.bc_encoder.encode(bc_feats)

        if self.args.use_discrete_features and self.args.use_brown_cluster:
            concat_inputs = [dy.concatenate([c, w, f, b]) for c, w, f, b in
                             zip(char_embs, word_embs, feat_embs, bc_feat_embs)]
        elif self.args.use_brown_cluster and not self.args.use_discrete_features:
            concat_inputs = [dy.concatenate([c, w, f]) for c, w, f in
                             zip(char_embs, word_embs, bc_feat_embs)]
        elif self.args.use_discrete_features and not self.args.use_brown_cluster:
            concat_inputs = [dy.concatenate([c, w, f]) for c, w, f in
                             zip(char_embs, word_embs, feat_embs)]
        else:
            concat_inputs = [dy.concatenate([c, w]) for c, w in zip(char_embs, word_embs)]

        birnn_outputs = self.birnn_encoder.encode(concat_inputs, training=training)
        return birnn_outputs
