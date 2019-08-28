__author__ = 'chuntingzhou and aditichaudhary'
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def evaluate(data_loader, path, model, model_name,type="dev"):
    sents, char_sents, tgt_tags, discrete_features, bc_feats,_ = data_loader.get_data_set(path, args.lang, source="dev")

    prefix = model_name + "_" + str(uid)
    # tot_acc = 0.0
    predictions = []
    gold_standards = []
    sentences = []
    i = 0
    sentence_gold = {}

    score_sent = {}
    for sent, char_sent, tgt_tag, discrete_feature, bc_feat in zip(sents, char_sents, tgt_tags, discrete_features, bc_feats):
        dy.renew_cg()
        sent, char_sent, discrete_feature, bc_feat = [sent], [char_sent], [discrete_feature], [bc_feat]
        best_score, best_path = model.eval(sent, char_sent, discrete_feature, bc_feat, training=False,type=type)

        assert len(best_path) == len(tgt_tag)
        # acc = model.crf_decoder.cal_accuracy(best_path, tgt_tag)
        # tot_acc += acc
        predictions.append(best_path)
        gold_standards.append(tgt_tag)

        sentences.append(sent)
        sent_key = " ".join([str(x) for x in sent[0]])
        sentence_gold[sent_key] = tgt_tag
        score_sent[sent_key] = best_score

        i += 1
        if i % 1000 == 0:
            print("Testing processed %d lines " % i)

    pred_output_fname = "%s/%s_pred_output.txt" % (args.eval_folder,prefix)
    eval_output_fname = "%s_eval_score.txt" % (prefix)
    with open(pred_output_fname, "w") as fout:
        for sent, pred, gold in zip(sentences, predictions, gold_standards):
            for s, p, g in zip(sent[0], pred, gold):
                fout.write(data_loader.id_to_word[int(s)] + " " + data_loader.id_to_tag[g] + " " + data_loader.id_to_tag[p] + "\n")
            fout.write("\n")

    os.system("%s/conlleval.v2 < %s > %s" % (args.eval_folder,pred_output_fname, eval_output_fname))

    with open(eval_output_fname, "r") as fin:
        lid = 0
        for line in fin:
            if lid == 1:
                fields = line.split(";")
                acc = float(fields[0].split(":")[1].strip()[:-1])
                precision = float(fields[1].split(":")[1].strip()[:-1])
                recall = float(fields[2].split(":")[1].strip()[:-1])
                f1 = float(fields[3].split(":")[1].strip())
            lid += 1

    output = open(eval_output_fname, "r").read().strip()
    print(output)
    if type == "dev":
        os.system("rm %s" % (eval_output_fname,))
        os.system("rm %s" % (pred_output_fname,))   


    return acc, precision, recall, f1,sentence_gold, score_sent


def replace_singletons(data_loader, sents, replace_rate):
    new_batch_sents = []
    for sent in sents:
        new_sent = []
        for word in sent:
            if word in data_loader.singleton_words:
                new_sent.append(word if np.random.uniform(0., 1.) > replace_rate else data_loader.word_to_id["<unk>"])
            else:
                new_sent.append(word)
        new_batch_sents.append(new_sent)
    return new_batch_sents


def main(args):
    prefix = args.model_name + "_" + str(uid)
    print("PREFIX: %s" % prefix)
    final_darpa_output_fname = "%s/%s_output.conll" % (args.eval_folder,prefix)
    best_output_fname = "%s/best_%s_output.conll" % (args.eval_folder,prefix)
    ner_data_loader = NER_DataLoader(args)
    print(ner_data_loader.id_to_tag)

    #Loading training data from CoNLL format
    if not args.data_aug:
        sents, char_sents, tgt_tags, discrete_features, bc_features,known_tags = ner_data_loader.get_data_set(args.train_path, args.lang)
    else:
        sents_tgt, char_sents_tgt, tags_tgt, dfs_tgt, bc_feats_tgt,known_tags_tgt = ner_data_loader.get_data_set(args.tgt_lang_train_path, args.lang)
        sents_aug, char_sents_aug, tags_aug, dfs_aug, bc_feats_aug, known_tags_aug= ner_data_loader.get_data_set(args.aug_lang_train_path, args.aug_lang)
        sents, char_sents, tgt_tags, discrete_features, bc_features,known_tags = sents_tgt+sents_aug, char_sents_tgt+char_sents_aug, tags_tgt+tags_aug, dfs_tgt+dfs_aug, bc_feats_tgt+bc_feats_aug,known_tags_tgt+known_tags_aug


    print("Data set size (train): %d" % len(sents))
    print("Number of discrete features: ", ner_data_loader.num_feats)
    epoch = bad_counter = updates = tot_example = cum_loss = 0
    patience = args.patience

    display_freq = 100
    valid_freq = args.valid_freq
    batch_size = args.batch_size


    print("Using Char CNN model!")
    model = vanilla_NER_CRF_model(args, ner_data_loader)
    inital_lr = args.init_lr

    if args.fineTune:
        print("Loading pre-trained model!")
        model.load()

        if len(sents) < 100:
            inital_lr = 0.0001
        else:
            inital_lr = args.init_lr #+ inital_lr * len(sents) / 1500.0


    trainer = dy.MomentumSGDTrainer(model.model, inital_lr, 0.9)

    def _check_batch_token(batch, id_to_vocab):
        for line in batch:
            print([id_to_vocab[i] for i in line])

    def _check_batch_char(batch, id_to_vocab):
        for line in batch:
            print([u" ".join([id_to_vocab[c] for c in w]) for w in line])

    lr_decay = args.decay_rate

    # decay_patience = 3
    # decay_num = 0
    valid_history = []
    best_results = [0.0, 0.0, 0.0, 0.0]
    while epoch <= args.tot_epochs:
        batches = make_bucket_batches(
                zip(sents, char_sents, tgt_tags, discrete_features, bc_features, known_tags), batch_size)

        for b_sents, b_char_sents, b_ner_tags, b_feats, b_bc_feats, b_known_tags in batches:
            dy.renew_cg()

            if args.replace_unk_rate > 0.0:
                b_sents = replace_singletons(ner_data_loader, b_sents, args.replace_unk_rate)
            # _check_batch_token(b_sents, ner_data_loader.id_to_word)
            # _check_batch_token(b_ner_tags, ner_data_loader.id_to_tag)
            # _check_batch_char(b_char_sents, ner_data_loader.id_to_char)

            loss = model.cal_loss(b_sents, b_char_sents, b_ner_tags, b_feats, b_bc_feats, b_known_tags, training=True)
            loss_val = loss.value()
            cum_loss += loss_val * len(b_sents)
            tot_example += len(b_sents)

            updates += 1
            loss.backward()
            trainer.update()

            if updates % display_freq == 0:
                print("Epoch = %d, Updates = %d, CRF Loss=%f, Accumulative Loss=%f." % (epoch, updates, loss_val, cum_loss*1.0/tot_example))
            if updates % valid_freq == 0:
                acc, precision, recall, f1,_,_ = evaluate(ner_data_loader, args.dev_path, model, args.model_name)

                if len(valid_history) == 0 or f1 > max(valid_history):
                    bad_counter = 0
                    best_results = [acc, precision, recall, f1]
                    if updates > 0:
                        print("Saving the best model so far.......")
                        model.save()
                else:
                    bad_counter += 1
                    if args.lr_decay and bad_counter >= 3 and os.path.exists(args.save_to_path):
                        bad_counter = 0
                        model.load()
                        lr = inital_lr / (1 + epoch * lr_decay)
                        print("Epoch = %d, Learning Rate = %f." % (epoch, lr))
                        trainer = dy.MomentumSGDTrainer(model.model, lr)

                if bad_counter > patience:
                    print("Early stop!")
                    print("Best on validation: acc=%f, prec=%f, recall=%f, f1=%f" % tuple(best_results))

                    acc, precision, recall, f1,sentence_gold, score_sent = evaluate(ner_data_loader, args.test_path, model, args.model_name,"test")
                    if args.SPAN_wise:
                        createAnnotationOutput_SPAN_wise(args, model, ner_data_loader, sentence_gold, score_sent)

                    exit(0)
                valid_history.append(f1)
        epoch += 1



    _,_,_,_,sentence_gold, score_sent = evaluate(ner_data_loader, args.test_path, model, args.model_name,"test")
    if args.SPAN_wise:
        createAnnotationOutput_SPAN_wise(args, model, ner_data_loader, sentence_gold, score_sent)
    print("All Epochs done.")

def createAnnotationOutput_SPAN_wise(args, model, data_loader, sentence_gold, score_sent):
    # normalize all the entropy_spans ONLY DONE for the CFB


    reverse = True #For ETAL we look at the highest entropy ones, hence sorting is reversed
    if args.use_CFB: #For CFEAL we look at the least confident, hence sorting is not reversed
        reverse = False


    # Order the sentences by entropy of the spans
    fout=  codecs.open(args.to_annotate, "w", encoding='utf-8')

    sorted_spans = sorted(model.crf_decoder.most_uncertain_entropy_spans,  key=lambda k:model.crf_decoder.most_uncertain_entropy_spans[k],reverse=reverse)
    print("Total unique spans: {0}".format(len(sorted_spans)))
    count_span = args.k
    count_tokens = args.k
	
    #DEBUG Print Span Entropy in the sorted order of selected spans
    fdebug = codecs.open("./" + args.model_name + "_span_entropy_debug.txt", "w", encoding='utf-8')

    for sorted_span in sorted_spans:

        span_words= []
        if count_tokens <=0:
            break
        (span_entropy,sentence_key, start, end,best_path) = model.crf_decoder.most_uncertain_entropy_spans[sorted_span]
        gold_path = sentence_gold[sentence_key]
        sent = sentence_key.split()

        for t in sorted_span.split():
            span_words.append(data_loader.id_to_word[int(t)])
        fdebug.write(" ".join(span_words) + " " + str(span_entropy) + "\n")

        first = True
        path = deepcopy(best_path)
        for i in range(start, end):
            if first:
                path[i] = -10 #Id for B-UNK
                first = False
            else:
                path[i] = -11 #Id for I-UNK

        idx = 0
        for token, tag in zip(sent, path):

            if tag == -10:
                tag_label = "B-UNK"
                count_span -= 1
                count_tokens -= 1
            elif tag == -11:
                tag_label = "I-UNK"
                count_tokens -= 1
            else:
                tag_label = data_loader.id_to_tag[tag]

            gold_tag_label = data_loader.id_to_tag[gold_path[idx]]
            idx += 1
            fout.write(data_loader.id_to_word[int(token)] + "\t" + tag_label + "\t" + gold_tag_label + "\n")

        fout.write("\n")

    print("Total unique spans for exercise: {0}".format(args.k))

    #SAL: Select most uncertain sequence
    basename = os.path.basename(args.to_annotate).replace(".conll", "")
    LC_output_file = os.path.dirname(args.to_annotate) + "/" + basename + "_LC.conll"
    count_tokens = args.k
    with codecs.open(LC_output_file, "w", encoding='utf-8') as fout:
        idx = 0
        for sentence_key in sorted(score_sent.keys(), reverse=False):
            if count_tokens<=0:
                break
            sent = sentence_key.split()
            gold_path = sentence_gold[sentence_key]
            token_count = 0
            for token in sent:
                count_tokens -= 1
                gold_tag_label = data_loader.id_to_tag[gold_path[token_count]]
                token_count += 1
                fout.write(data_loader.id_to_word[int(token)] + "\t" + "UNK " + gold_tag_label + "\n")
            fout.write("\n")
            idx += 1


def test_single_model(args):
    ner_data_loader = NER_DataLoader(args)
    # ugly: get discrete number features
    _, _, _, _, _,_ = ner_data_loader.get_data_set(args.train_path, args.lang)

    print("Using Char CNN model!")
    model = vanilla_NER_CRF_model(args, ner_data_loader)
    model.load()

    _,_,_,_,sentence_gold, score_sent = evaluate(ner_data_loader, args.test_path, model, args.model_name,"test")
    if args.SPAN_wise:
        createAnnotationOutput_SPAN_wise(args, model, ner_data_loader, sentence_gold, score_sent)




from args import init_config

args = init_config()
from models.model_builder import *
import os
import uuid
from dataloaders.data_loader import *
uid = uuid.uuid4().get_hex()[:6]

if __name__ == "__main__":
    # args = init_config()
    if args.mode == "train":
        if args.load_from_path is not None:
            args.load_from_path = args.load_from_path
        else:
            args.load_from_path = args.save_to_path
        main(args)

    elif args.mode == "test_1":
        test_single_model(args)

    else:
        raise NotImplementedError
