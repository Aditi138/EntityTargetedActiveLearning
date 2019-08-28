__author__ = 'chuntingzhou and aditichaudhary'
from utils.util import *
import numpy as np
from collections import defaultdict
from scipy.special import logsumexp


class Decoder():
    def __init__(self, tag_size):
        # type: () -> object
        pass

    def decode_loss(self):
        raise NotImplementedError

    def decoding(self):
        raise NotImplementedError


def constrained_transition_init(transition_matrix, contraints):
    '''
    :param transition_matrix: numpy array, (from, to)
    :param contraints: [[from_indexes], [to_indexes]]
    :return: newly initialized transition matrix
    '''
    for cons in contraints:
        transition_matrix[cons[0], cons[1]] = -1000.0
    return transition_matrix

class chain_CRF_decoder(Decoder):
    ''' For NER and POS Tagging. '''

    def __init__(self, args, model, src_output_dim, tag_emb_dim, tag_size, constraints=None):
        Decoder.__init__(self, tag_size)
        self.model = model
        self.start_id = tag_size
        self.end_id = tag_size + 1
        self.tag_size = tag_size + 2
        tag_size = tag_size + 2
        self.args = args

        # optional: transform the hidden space of src encodings into the tag embedding space
        self.W_src2tag_readout = model.add_parameters((tag_emb_dim, src_output_dim))
        self.b_src2tag_readout = model.add_parameters((tag_emb_dim))
        self.b_src2tag_readout.zero()

        self.W_scores_readout2tag = model.add_parameters((tag_size, tag_emb_dim))
        self.b_scores_readout2tag = model.add_parameters((tag_size))
        self.b_scores_readout2tag.zero()

        # (to, from), trans[i] is the transition score to i
        init_transition_matrix = np.random.randn(tag_size, tag_size) # from, to
        init_transition_matrix[:, self.end_id] = -1000.0
        init_transition_matrix[self.start_id, :] = -1000.0
        if False and constraints is not None:
            init_transition_matrix = constrained_transition_init(init_transition_matrix, constraints)
        # print init_transition_matrix
        #self.transition_matrix = model.add_lookup_parameters((tag_size, tag_size),
        #                                                    init=dy.NumpyInitializer(init_transition_matrix))
	self.transition_matrix = model.lookup_parameters_from_numpy(init_transition_matrix) # (to, from)

        self.ngram = args.ngram

        self.entropy_threshold = args.entropy_threshold
        if args.entropy_threshold is not None and args.use_CFB:
            self.entropy_threshold = args.entropy_threshold * -1

        self.prob_threshold = np.NINF
        self.entropy_spans = defaultdict(lambda: 0)
        self.most_uncertain_entropy_spans = {}
        self.entropy_spans_number = defaultdict(lambda: 0)
        self.full_sentences = defaultdict(list)
        self.avg_spans_in_sent_entropy = defaultdict(list)
        self.SPAN_wise = args.SPAN_wise

    def forward_alg(self, tag_scores):
        ''' Forward DP for CRF.
        tag_scores (list of batched dy.Tensor): (tag_size, batchsize)
        '''
        # Be aware: if a is lookup_parameter with 2 dimension, then a[i] returns one row;
        # if b = dy.parameter(a), then b[i] returns one column; which means dy.parameter(a) already transpose a
        # transpose_transition_score = self.transition_matrix
        transpose_transition_score = dy.parameter(self.transition_matrix) # (from, to)

        # alpha(t', s) = the score of sequence from t=0 to t=t' in log space
        # np_init_alphas = -100.0 * np.ones((self.tag_size, batch_size))
        # np_init_alphas[self.start_id, :] = 0.0
        # alpha_tm1 = dy.inputTensor(np_init_alphas, batched=True)
	alphas = []

        alpha_tm1 = transpose_transition_score[self.start_id] + tag_scores[0]
        # self.transition_matrix[i]: from i, column
        # transpose_score[i]: to i, row
        # transpose_score: to, from
	alphas.append(alpha_tm1)

        for tag_score in tag_scores[1:]:
            # extend for each transit <to>
            alpha_tm1 = dy.concatenate_cols([alpha_tm1] * self.tag_size)  # (from, to, batch_size)
            # each column i of tag_score will be the repeated emission score to tag i
            tag_score = dy.transpose(dy.concatenate_cols([tag_score] * self.tag_size))
            alpha_t = alpha_tm1 + transpose_transition_score + tag_score
            alpha_tm1 = log_sum_exp_dim_0(alpha_t)  # (tag_size, batch_size)
	    alphas.append(alpha_tm1)

        terminal_alpha = log_sum_exp_dim_0(alpha_tm1 + self.transition_matrix[self.end_id])  # (1, batch_size)
        return terminal_alpha,alphas

    def score_one_sequence(self, tag_scores, tags, batch_size):
        ''' tags: list of tag ids at each time step '''
        # print tags, batch_size
        # print batch_size
        # print "scoring one sentence"
        tags = [[self.start_id] * batch_size] + tags  # len(tag_scores) = len(tags) - 1
        score = dy.inputTensor(np.zeros(batch_size), batched=True)
        # tag_scores = dy.concatenate_cols(tag_scores) # tot_tags, sent_len, batch_size
        # print "tag dim: ", tag_scores.dim()
        for i in range(len(tags) - 1):
            score += dy.pick_batch(dy.lookup_batch(self.transition_matrix, tags[i + 1]), tags[i]) \
                    + dy.pick_batch(tag_scores[i], tags[i + 1])
        score += dy.pick_batch(dy.lookup_batch(self.transition_matrix, [self.end_id]*batch_size), tags[-1])
        return score

    def backward_one_sequence(self, tag_scores):
        ''' Backward DP for CRF.
        tag_scores (list of batched dy.Tensor): (tag_size, batchsize)
        '''
        # Be aware: if a is lookup_parameter with 2 dimension, then a[i] returns one row;
        # if b = dy.parameter(a), then b[i] returns one column; which means dy.parameter(a) already transpose a
        transpose_transition_score = dy.parameter(self.transition_matrix)
        # transpose_transition_score = dy.parameter(self.transition_matrix)

        # alpha(t', s) = the score of sequence from t=0 to t=t' in log space
        # np_init_alphas = -100.0 * np.ones((self.tag_size, batch_size))
        # np_init_alphas[self.start_id, :] = 0.0
        # alpha_tm1 = dy.inputTensor(np_init_alphas, batched=True)
        betas = []
        # beta_tp1 = self.transition_matrix[self.end_id] + tag_scores[-1]
        # beta_tp1 = dy.inputTensor(np.zeros(self.tag_size))
        beta_tp1 = self.transition_matrix[self.end_id]
        betas.append(beta_tp1)
        # self.transition_matrix[i]: from i, column
        # transpose_score[i]: to i, row
        # transpose_score: to, from
        seq_len = len(tag_scores)
        tag_scores.reverse()
        for tag_score in tag_scores[0:seq_len - 1]:
            # extend for each transit <to>
            beta_tp1 = dy.concatenate_cols([beta_tp1] * self.tag_size)  # (to, from, batch_size)
            # each column i of tag_score will be the repeated emission score to tag i
            tag_score = dy.concatenate_cols([tag_score] * self.tag_size)  # (to, from)
            beta_t = beta_tp1 + dy.transpose(transpose_transition_score) + tag_score  # (to, from)
            beta_tp1 = log_sum_exp_dim_0(beta_t)  # (tag_size, batch_size)
            betas.append(beta_tp1)

        # betas.append(beta_tp1 + transpose_transition_score[self.start_id] + tag_scores[-1])
        terminal_beta = log_sum_exp_dim_0(
            beta_tp1 + transpose_transition_score[self.start_id] + tag_scores[-1])  # (1, batch_size)
        betas.reverse()
        return terminal_beta, betas

    def get_uncertain_subsequences(self, sents, tag_scores, alphas, betas, Z, gammas,
                                   best_path, B_tags, I_tags, O_tags):
        # predicted_path = deepcopy(best_path)
        # transition_B_O = np.array(dy.pick_batch(dy.lookup_batch(self.transition_matrix, O_tags), B_tags).value())
        # transition_I_O = np.array(dy.pick_batch(dy.lookup_batch(self.transition_matrix, O_tags), I_tags).value())
        # transition_B_I = np.array(dy.pick_batch(dy.lookup_batch(self.transition_matrix, I_tags), B_tags).value())
        # transition_I_I = np.array(dy.pick_batch(dy.lookup_batch(self.transition_matrix, I_tags), I_tags).value())


        first = True
        Z = Z.value()
        for i in range(len(sents)):
            # log_p_alpha = np.array(alphas[i].value())[B_tags]
            # transition = transition_B_O
            log_pin = logsumexp(np.array(gammas[i].value())[B_tags]) #Prob (y=start_entity|x)= log_sum{tags}#  (e^log(P=B-tag|x))

            for j in range(i + 1, len(sents)):
                if (j - i) > self.ngram:
                    break

                log_p_out = np.array(gammas[j].value())[O_tags]
                log_p = log_pin + log_p_out
                p = np.exp(log_p)
                if p > 1.0:
                    #print(p, log_p)
                    H= 0.0
                else:
                    H = -(p * log_p) - ((1-p)* np.log(1-p))

                if H > self.entropy_threshold:
                    # best_path = deepcopy(predicted_path)
                    span = " ".join([str(x) for x in sents[i:j]])
                    sent = " ".join([str(x) for x in sents])
                    self.entropy_spans[span] += H
                    self.entropy_spans_number[span] += 1

                    if self.SPAN_wise:
                        if span in self.most_uncertain_entropy_spans:
                            (existing_entropy, _,_,_,_) = self.most_uncertain_entropy_spans[span]
                            #if H > existing_entropy:
                            #    self.most_uncertain_entropy_spans[span] = (H, sent, i,j,best_path)
                            self.most_uncertain_entropy_spans[span] =  (self.entropy_spans[span], sent, i, j, best_path)
                        else:
                            self.most_uncertain_entropy_spans[span] = (H, sent, i,j,best_path)

                    # for k in range(i,j+1):
                    #     best_path[k] = -10
                    self.full_sentences[sent].append((i, j, best_path, self.entropy_spans[span]))
                    self.avg_spans_in_sent_entropy[sent].append(span)
                    # self.full_sentences[span] = (sents,best_path,predicted_path, self.entropy_spans[span])

                log_pin += logsumexp(np.array(gammas[j].value())[I_tags])
                if log_pin < np.log(1e-4):
                    break


    def get_uncertain_subsequences_CFB(self, sents, tag_scores, alphas, betas, Z, gammas,
                                       best_path, tag_to_id,B_UNK, I_UNK):
        first = True
        Z = Z.value()
        entropy_spans_number = defaultdict(lambda :0)
        for i in range(len(sents)):
            known_tags =np.array([[0]] * len(sents)).reshape((len(sents),1,1))
            known_tags[i][0][0] = 1
            tags = np.array([B_UNK] * len(sents)).reshape((len(sents),1))
            tags[i][0] = best_path[i]
            for j in range(i + 1, len(sents)):
                if (j - i + 1) > self.ngram:
                    break
                tags[j][0] = best_path[j]
                known_tags[j][0][0] = 1
                Z_span = self.score_one_sequence_partial(tag_scores, tags, 1, known_tags, tag_to_id, B_UNK, I_UNK)
                confidence = Z_span.value() - Z

                if confidence < self.entropy_threshold:
                    # best_path = deepcopy(predicted_path)
                    span = " ".join([str(x) for x in sents[i:j + 1]])
                    sent = " ".join([str(x) for x in sents])

                    if self.SPAN_wise:
                        if span in self.most_uncertain_entropy_spans:
                            (existing_threshold, _,_,_,_) = self.most_uncertain_entropy_spans[span]
                            if confidence < existing_threshold:
                                self.most_uncertain_entropy_spans[span] = (confidence, sent, i, j,best_path)
			else:
			    self.most_uncertain_entropy_spans[span] = (confidence, sent, i, j,best_path)

                    self.entropy_spans[span] += confidence
                    self.entropy_spans_number[span] += 1
                    #self.full_sentences[sent].append((i, j, best_path, self.entropy_spans[span]))
                    self.full_sentences[sent].append((i, j, best_path, span))
                    #self.avg_spans_in_sent_entropy[sent].append(self.entropy_spans[span])
                    self.avg_spans_in_sent_entropy[sent].append(span)


    def decode_loss(self, src_encodings, tgt_tags, use_partial, known_tags, tag_to_id, B_UNK, I_UNK):
        # This is the batched version which requires bucketed batch input with the same length.
        '''
        The length of src_encodings and tgt_tags are time_steps.
        src_encodings: list of dynet.Tensor (src_output_dim, batch_size)
        tgt_tags: list of tag ids [(1, batch_size)]
        return: average of negative log likelihood
        '''
        # TODO: transpose tgt tags first
        batch_size = len(tgt_tags)
        tgt_tags, tgt_mask = transpose_input(tgt_tags, 0)
        known_tags, _ = transpose_input(known_tags, 0)

        W_src2tag_readout = dy.parameter(self.W_src2tag_readout)
        b_src2tag_readout = dy.parameter(self.b_src2tag_readout)
        W_score_tag = dy.parameter(self.W_scores_readout2tag)
        b_score_tag = dy.parameter(self.b_scores_readout2tag)

        tag_embs = [dy.tanh(dy.affine_transform([b_src2tag_readout, W_src2tag_readout, src_encoding])) for src_encoding
                    in src_encodings]

        tag_scores = [dy.affine_transform([b_score_tag, W_score_tag, tag_emb]) for tag_emb in tag_embs]

        # scores over all paths, all scores are in log-space
        forward_scores,_ = self.forward_alg(tag_scores)

        if use_partial:
            gold_score = self.score_one_sequence_partial(tag_scores, tgt_tags, batch_size, known_tags, tag_to_id, B_UNK,
                                                         I_UNK)
        else:
            gold_score = self.score_one_sequence(tag_scores, tgt_tags, batch_size)

        # negative log likelihood
        loss = dy.sum_batches(forward_scores - gold_score) / batch_size
        return loss #, dy.sum_batches(forward_scores)/batch_size, dy.sum_batches(gold_score) / batch_size

    def makeMask(self, batch_size, known_tags, tag_to_id, tags, index, B_UNK, I_UNK):
        mask_w_0 = np.array([[-1000] * self.tag_size])
        mask_w_0 = np.transpose(mask_w_0)
        mask_w_0_all_s = np.reshape(np.array([mask_w_0] * batch_size), (self.tag_size, batch_size))

        mask_idx = []
        tag_vals = []
        for idx, w0_si in enumerate(known_tags[index]):
            if w0_si[0] == 1:
                mask_idx.append(idx)
                tag_vals.append(tags[index][idx])
            else:
                if tags[index][idx] == B_UNK:
                    if self.args.misc:
                        possible_labels = ["B-LOC", "B-PER", "B-ORG", "B-MISC", "O","I-LOC", "I-PER", "I-ORG", "I-MISC"]
                    else:
                        possible_labels = ["B-LOC", "B-PER", "B-ORG", "B-GPE", "O","I-LOC", "I-PER", "I-ORG", "I-GPE"]
                    for pl in possible_labels:
                        mask_idx.append(idx)
                        tag_vals.append(tag_to_id[pl])
        mask_w_0_all_s[tag_vals, mask_idx] = 0
        return mask_w_0_all_s

    def score_one_sequence_partial(self, tag_scores, tags, batch_size, known_tags, tag_to_id, B_UNK, I_UNK):
        transpose_transition_score = dy.parameter(self.transition_matrix)

        alpha_tm1 = transpose_transition_score[self.start_id] + tag_scores[0]

        mask_w_0_all_s = self.makeMask(batch_size, known_tags, tag_to_id, tags, 0, B_UNK, I_UNK)
        i = 1
        alpha_tm1 = alpha_tm1 + dy.inputTensor(mask_w_0_all_s, batched=True)
        for tag_score in tag_scores[1:]:
            alpha_tm1 = dy.concatenate_cols([alpha_tm1] * self.tag_size)  # (from, to, batch_size)
            tag_score = dy.transpose(dy.concatenate_cols([tag_score] * self.tag_size))
            alpha_t = alpha_tm1 + transpose_transition_score + tag_score
            alpha_tm1 = log_sum_exp_dim_0(alpha_t)  # (tag_size, batch_size)
            mask_w_i_all_s = self.makeMask(batch_size, known_tags, tag_to_id, tags, i, B_UNK, I_UNK)
            alpha_tm1 = alpha_tm1 + dy.inputTensor(mask_w_i_all_s, batched=True)
            i = i + 1

        terminal_alpha = log_sum_exp_dim_0(alpha_tm1 + self.transition_matrix[self.end_id])  # (1, batch_size)
        return terminal_alpha


    def get_crf_scores(self, src_encodings):
        W_src2tag_readout = dy.parameter(self.W_src2tag_readout)
        b_src2tag_readout = dy.parameter(self.b_src2tag_readout)
        W_score_tag = dy.parameter(self.W_scores_readout2tag)
        b_score_tag = dy.parameter(self.b_scores_readout2tag)

        tag_embs = [dy.tanh(dy.affine_transform([b_src2tag_readout, W_src2tag_readout, src_encoding]))
                    for src_encoding in src_encodings]
        tag_scores = [dy.affine_transform([b_score_tag, W_score_tag, tag_emb]) for tag_emb in tag_embs]

        transpose_transition_score = dy.parameter(self.transition_matrix)  # (from, to)

        return transpose_transition_score.npvalue(), [ts.npvalue() for ts in tag_scores]

    def decoding(self, src_encodings,OTag, addbias=False):
        ''' Viterbi decoding for a single sequence. '''
        W_src2tag_readout = dy.parameter(self.W_src2tag_readout)
        b_src2tag_readout = dy.parameter(self.b_src2tag_readout)
        W_score_tag = dy.parameter(self.W_scores_readout2tag)
        #b_score_tag = dy.parameter(self.b_scores_readout2tag)

        tag_embs = [dy.tanh(dy.affine_transform([b_src2tag_readout, W_src2tag_readout, src_encoding]))
                    for src_encoding in src_encodings]
        if addbias:
            b_score_tag = np.zeros(self.tag_size)
            b_score_tag[OTag] = 0.5
            b_score_tag = dy.inputTensor(b_score_tag)


        tag_scores = [dy.affine_transform([b_score_tag, W_score_tag, tag_emb]) for tag_emb in tag_embs]

        back_trace_tags = []
        np_init_alpha = np.ones(self.tag_size) * -2000.0
        np_init_alpha[self.start_id] = 0.0
        max_tm1 = dy.inputTensor(np_init_alpha)
        transpose_transition_score = dy.parameter(self.transition_matrix)  # (from, to)

        for i, tag_score in enumerate(tag_scores):
            max_tm1 = dy.concatenate_cols([max_tm1] * self.tag_size)
            max_t = max_tm1 + transpose_transition_score
            if i != 0:
                eval_score = max_t.npvalue()[:-2, :]
            else:
                eval_score = max_t.npvalue()
            best_tag = np.argmax(eval_score, axis=0)
            back_trace_tags.append(best_tag)
            max_tm1 = dy.inputTensor(eval_score[best_tag, range(self.tag_size)]) + tag_score

        terminal_max_T = max_tm1 + self.transition_matrix[self.end_id]
        eval_terminal = terminal_max_T.npvalue()[:-2]
        best_tag = np.argmax(eval_terminal, axis=0)
        best_path_score = eval_terminal[best_tag]

        best_path = [best_tag]
        for btpoint in reversed(back_trace_tags):
            best_tag = btpoint[best_tag]
            best_path.append(best_tag)
        start = best_path.pop()
        assert start == self.start_id
        best_path.reverse()
        return best_path_score, best_path, tag_scores

    def cal_accuracy(self, pred_path, true_path):
        return np.sum(np.equal(pred_path, true_path).astype(np.float32)) / len(pred_path)


def ensemble_viterbi_decoding(l_tag_scores, l_transit_score, tag_size):
    back_trace_tags = []
    tag_size = tag_size + 2
    start_id = tag_size - 2
    end_id = tag_size - 1
    max_tm1 = np.ones(tag_size) * -2000.0
    max_tm1[start_id] = 0.0

    tag_scores = []
    for i in range(len(l_tag_scores[0])):
        tag_scores.append(sum([ts[i] for ts in l_tag_scores]) / len(l_tag_scores))
    transpose_transition_score = sum(l_transit_score) / len(l_transit_score)  # (from, to)

    for i, tag_score in enumerate(tag_scores):
        max_tm1 = np.tile(np.expand_dims(max_tm1, axis=1), (1, tag_size))
        max_t = max_tm1 + transpose_transition_score
        if i != 0:
            eval_score = max_t[:-2, :]
        else:
            eval_score = max_t
        best_tag = np.argmax(eval_score, axis=0)
        back_trace_tags.append(best_tag)
        max_tm1 = eval_score[best_tag, range(tag_size)] + tag_score

    terminal_max_T = max_tm1 + transpose_transition_score[:, end_id]
    eval_terminal = terminal_max_T[:-2]
    best_tag = np.argmax(eval_terminal, axis=0)
    best_path_score = eval_terminal[best_tag]

    best_path = [best_tag]
    for btpoint in reversed(back_trace_tags):
        best_tag = btpoint[best_tag]
        best_path.append(best_tag)
    start = best_path.pop()
    assert start == start_id
    best_path.reverse()
    return best_path_score, best_path


class classifier(Decoder):
    def __init__(self, model, input_dim, tag_size):
        self.W_softmax = model.add_parameters((tag_size, input_dim))
        self.b_softmax = model.add_parameters((tag_size))

    def decode_loss(self, src_encoding, tgt_tags):
        batch_size = len(tgt_tags)
        tgt_tags, tgt_mask = transpose_input(tgt_tags, 0)

        assert len(src_encoding) == len(tgt_tags)

        W_softmax = dy.parameter(self.W_softmax)
        b_softmax = dy.parameter(self.b_softmax)

        predictions = [dy.affine_transform([b_softmax, W_softmax, src_emb]) for src_emb in src_encoding]

        losses = [dy.pickneglogsoftmax_batch(pred, tgt) for pred, tgt in zip(predictions, tgt_tags)]

        loss = dy.sum_batches(dy.esum(losses)) / (batch_size * len(src_encoding))

        return loss

    def decoding(self, src_encoding):
        W_softmax = dy.parameter(self.W_softmax)
        b_softmax = dy.parameter(self.b_softmax)
        predictions = [dy.affine_transform([b_softmax, W_softmax, src_emb]) for src_emb in src_encoding]

        predictions = [np.argmax(pred.npvalue()) for pred in predictions]

        return None, predictions
