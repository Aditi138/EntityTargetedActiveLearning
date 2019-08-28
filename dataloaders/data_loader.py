__author__ = 'chuntingzhou and aditichaudhary'
import os
from utils.util import *
from utils.features import *
#from utils.segnerfts import orm_morph as ormnorm

#tagset = ['B-LOC','B-PER','B-MISC', 'B-ORG','I-LOC','I-PER','I-MISC', 'I-ORG','O']
tagset = ['B-LOC','B-PER','B-GPE', 'B-ORG','I-LOC','I-PER','I-GPE', 'I-ORG','O']

class NER_DataLoader():
    def __init__(self, args, special_normal=False):
        # This is data loader as well as feature extractor!!

        self.args = args
        if args.train_ensemble:
            self.train_path = args.full_data_path
        else:
            self.train_path = args.train_path
        self.test_path = args.test_path
        self.dev_path = args.dev_path
        self.args = args

        self.tag_vocab_path = self.train_path + ".tag_vocab"
        self.word_vocab_path = self.train_path + ".word_vocab"
        self.char_vocab_path = self.train_path + ".char_vocab"

        self.pretrained_embedding_path = args.pretrain_emb_path
        self.use_discrete_feature = args.use_discrete_features
        self.use_brown_cluster = args.use_brown_cluster
        self.orm_norm = args.oromo_normalize
        self.orm_lower = args.train_lowercase_oromo

        self.train_senttypes = self.dev_senttypes = self.test_senttypes = None

        if self.use_brown_cluster:
            self.brown_cluster_dicts = get_brown_cluster(args.brown_cluster_path)
            self.brown_cluster_dicts['<unk>'] = len(self.brown_cluster_dicts)
            args.brown_cluster_num = len(self.brown_cluster_dicts)
        else:
            self.brown_cluster_dicts = None

        print("Generating vocabs from training file ....")
        paths_to_read = [self.train_path, self.dev_path, self.test_path]

        if args.fixedVocab: #Make vaocabulary from the args.aug_lang_train_path
            _, self.word_to_id, self.char_to_id = self.read_files([self.args.aug_lang_train_path])
            self.tag_to_id  = {}
            # self.word_to_id = {}
            # self.char_to_id = {}
            for tag in tagset:
                if args.misc:
                    tag = tag.replace("GPE", "MISC")
                self.tag_to_id[tag] = len(self.tag_to_id)
        else:
            self.tag_to_id, self.word_to_id, self.char_to_id = self.read_files(paths_to_read)


        self.word_padding_token = 0
        self.char_padding_token = 0

        if self.pretrained_embedding_path is not None:
            self.pretrain_word_emb, self.word_to_id, self.char_to_id = get_pretrained_emb(self.args.fixedVocab, self.pretrained_embedding_path,
                                                                         self.word_to_id, self.char_to_id, args.word_emb_dim)

        # for char vocab and word vocab, we reserve id 0 for the eos padding, and len(vocab)-1 for the <unk>
        self.id_to_tag = {v: k for k, v in self.tag_to_id.iteritems()}
        self.id_to_word = {v: k for k, v in self.word_to_id.iteritems()}
        self.id_to_char = {v: k for k, v in self.char_to_id.iteritems()}

        self.ner_vocab_size = len(self.id_to_tag)
        self.word_vocab_size = len(self.id_to_word)
        self.char_vocab_size = len(self.id_to_char)

        self.cap_ratio_dict = None

        #Partial CRF
        self.B_UNK = self.ner_vocab_size + 1
        self.I_UNK = self.ner_vocab_size + 2

        print("Size of vocab after: %d" % len(self.word_to_id))
        print("NER tag num=%d, Word vocab size=%d, Char Vocab size=%d" % (self.ner_vocab_size, self.word_vocab_size, self.char_vocab_size))


    @staticmethod
    def exists(path):
        return os.path.exists(path)

    def read_one_line(self, line, tag_set, word_dict, char_set):
        for w in line:
            fields = w.split()
            if len(fields) !=2:
                print("ERROR! Incorrect number of fields in the file, required two.")
                print(fields)
                exit(0)
            word = fields[0]
            ner_tag = fields[-1]

            for c in word:
                char_set.add(c)
            if "UNK" not in ner_tag:
                if self.args.misc:
                    ner_tag = ner_tag.replace("GPE","MISC")
                tag_set.add(ner_tag)
            word_dict[word] += 1

    def get_vocab_from_set(self, a_set, shift=0):
        vocab = {}
        for i, elem in enumerate(a_set):
            vocab[elem] = i + shift

        return vocab

    def get_vocab_from_dict(self, a_dict, shift=0, remove_singleton=False):
        vocab = {}
        i = 0
        self.singleton_words = set()

        #Sort the defaultdict
        sortedDict = sorted(a_dict.iteritems(), key=lambda (k, v): v, reverse=True)
        for (k,v) in sortedDict:

        #for k, v in a_dict.iteritems():
            if v == 1:
                self.singleton_words.add(i + shift)
            if remove_singleton:
                if v > 1:
                    # print k, v
                    vocab[k] = i + shift
                    i += 1
            else:
                vocab[k] = i + shift
                i += 1
        print("Singleton words number: %d" % len(self.singleton_words))
        return vocab

    def read_files(self, paths):
        # word_list = []
        # char_list = []
        # tag_list = []
        word_dict = defaultdict(lambda: 0)
        char_set = set()
        tag_set = set()

        def _read_a_file(path):
            with codecs.open(path, "r", "utf-8") as fin:
                to_read_line = []
                for line in fin:
                    if line.strip() == "":
                        self.read_one_line(to_read_line, tag_set, word_dict, char_set)
                        to_read_line = []
                    else:
                        to_read_line.append(line.strip())
                self.read_one_line(to_read_line, tag_set, word_dict, char_set)

        for path in paths:
            _read_a_file(path)

        tag_vocab = self.get_vocab_from_set(tag_set)
        word_vocab = self.get_vocab_from_dict(word_dict, 1, self.args.remove_singleton)
        char_vocab = self.get_vocab_from_set(char_set, 1)

        return tag_vocab, word_vocab, char_vocab

    def get_data_set(self, path, lang, source="train"):
        sents = []
        char_sents = []
        tgt_tags = []
        discrete_features = []
        bc_features = []
        known_tags = []

        if source == "train":
            sent_types = self.train_senttypes
        else:
            sent_types = self.dev_senttypes

        def add_sent(one_sent, type):
            temp_sent = []
            temp_ner = []
            temp_char = []
            temp_bc = []
            sent = []
            temp_known_tag = []
            for w in one_sent:
                fields = w.split()
                if len(fields)!=2:
                    fields = w.split("\t")
                assert len(fields)==2
                word = fields[0]
                sent.append(word)
                ner_tag = fields[-1]
                if self.use_brown_cluster:
                    temp_bc.append(self.brown_cluster_dicts[word] if word in self.brown_cluster_dicts else self.brown_cluster_dicts["<unk>"])

                if self.args.fixedVocab:
                    if word in self.word_to_id:
                        temp_sent.append(self.word_to_id[word])
                    elif word.lower() in self.word_to_id:
                        temp_sent.append(self.word_to_id[word.lower()])
                    else:
                        temp_sent.append(self.word_to_id["<unk>"])
                else:
                    temp_sent.append(self.word_to_id[word] if word in self.word_to_id else self.word_to_id["<unk>"])

                if "B-UNK" in ner_tag:
                    temp_ner.append(self.B_UNK)
                elif "I-UNK" in ner_tag:
                    temp_ner.append(self.I_UNK)
                else:
                    if self.args.misc:
                        ner_tag = ner_tag.replace("GPE","MISC")
                    temp_ner.append(self.tag_to_id[ner_tag])

                if "UNK" in ner_tag:
                    temp_known_tag.append([0])
                else:
                    temp_known_tag.append([1])

                temp_char.append([self.char_to_id[c] if c in self.char_to_id else self.char_to_id["<unk>"] for c in word])

            sents.append(temp_sent)
            char_sents.append(temp_char)
            tgt_tags.append(temp_ner)
            bc_features.append(temp_bc)
            known_tags.append(temp_known_tag)
            if not self.args.isLr:
                discrete_features.append([])
            else:
                discrete_features.append(get_feature_sent(lang, sent, self.args, self.cap_ratio_dict, type=type))

            # print len(discrete_features[-1])

        with codecs.open(path, "r", "utf-8") as fin:
            i = 0
            one_sent = []
            for line in fin:
                if line.strip() == "":
                    if len(one_sent) > 0:
                        add_sent(one_sent, sent_types[i] if sent_types is not None else None)
                        i += 1
                        if i % 1000 == 0:
                            print("Processed %d training data." % (i,))
                    one_sent = []
                else:
                    one_sent.append(line.strip())

            if len(one_sent) > 0:
                add_sent(one_sent, sent_types[i] if sent_types is not None else None)
                i += 1

        if sent_types is not None:
            assert i == len(sent_types), "Not match between number of sentences and sentence types!"

        if self.use_discrete_feature:
            self.num_feats = len(discrete_features[0][0])
        else:
            self.num_feats = 0
        return sents, char_sents, tgt_tags, discrete_features, bc_features, known_tags
