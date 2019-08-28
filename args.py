def init_config():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynet-mem", default=1000, type=int)
    parser.add_argument("--dynet-seed", default=5783287, type=int)
    parser.add_argument("--dynet-gpu")

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--eval_folder", type=str, default="../eval")
    parser.add_argument("--lang", default="english", help="the target language")
    parser.add_argument("--train_ensemble", default=False, action="store_true")
    parser.add_argument("--full_data_path", type=str, default=None, help="when train_ensemble is true, this one is the full data path from which to load vocabulary.")
    parser.add_argument("--train_path", default="../datasets/english/eng.train.bio.conll", type=str)
    # parser.add_argument("--train_path", default="../datasets/english/debug_train.bio", type=str)
    parser.add_argument("--monolingual_data_path", default=None, type=str)
    parser.add_argument("--dev_path", default="../datasets/english/eng.dev.bio.conll", type=str)
    parser.add_argument("--test_path", default="../datasets/english/eng.test.bio.conll", type=str)
    parser.add_argument("--new_test_path", default="../datasets/english/eng.test.bio.conll", type=str)
    parser.add_argument("--new_test_conll", default="../datasets/english/eng.test.bio.conll", type=str)
    parser.add_argument("--save_to_path", default="../saved_models/")
    parser.add_argument("--load_from_path", default=None)
    parser.add_argument("--train_filename_path", default=None, type=str)
    parser.add_argument("--dev_filename_path", default=None, type=str)
    parser.add_argument("--test_filename_path", default=None, type=str)


    parser.add_argument("--model_arc", default="char_cnn", choices=["char_cnn", "char_birnn", "char_birnn_cnn", "sep", "sep_cnn_only"], type=str)
    parser.add_argument("--tag_emb_dim", default=50, type=int)
    parser.add_argument("--pos_emb_dim", default=50, type=int)
    parser.add_argument("--char_emb_dim", default=30, type=int)
    parser.add_argument("--word_emb_dim", default=100, type=int)
    parser.add_argument("--cnn_filter_size", default=30, type=int)
    parser.add_argument("--cnn_win_size", default=3, type=int)
    parser.add_argument("--rnn_type", default="lstm", choices=['lstm', 'gru'], type=str)
    parser.add_argument("--hidden_dim", default=200, type=int, help="token level rnn hidden dim")
    parser.add_argument("--char_hidden_dim", default=25, type=int, help="char level rnn hidden dim")
    parser.add_argument("--layer", default=1, type=int)

    parser.add_argument("--replace_unk_rate", default=0.0, type=float, help="uses when not all words in the test data is covered by the pretrained embedding")
    parser.add_argument("--remove_singleton", default=False, action="store_true")
    parser.add_argument("--map_pretrain", default=False, action="store_true")
    parser.add_argument("--map_dim", default=100, type=int)
    parser.add_argument("--pretrain_fix", default=False, action="store_true")

    parser.add_argument("--output_dropout_rate", default=0.5, type=float, help="dropout applied to the output of birnn before crf")
    parser.add_argument("--emb_dropout_rate", default=0.3, type=float, help="dropout applied to the input of token-level birnn")
    parser.add_argument("--valid_freq", default=500, type=int)
    parser.add_argument("--tot_epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--init_lr", default=0.015, type=float)
    parser.add_argument("--lr_decay", default=False, action="store_true")
    parser.add_argument("--decay_rate", default=0.05, action="store", type=float)
    parser.add_argument("--patience", default=3, type=int)

    parser.add_argument("--tagging_scheme", default="bio", choices=["bio", "bioes"], type=str)

    parser.add_argument("--data_aug", default=False, action="store_true", help="If use data_aug, the train_path should be the combined training file")
    parser.add_argument("--aug_lang", default="english", help="the language to augment the dataset")
    parser.add_argument("--aug_lang_train_path", default=None, type=str)
    parser.add_argument("--tgt_lang_train_path", default="../datasets/english/eng.train.bio.conll", type=str)

    parser.add_argument("--pretrain_emb_path", type=str, default=None)
    parser.add_argument("--res_discrete_feature", default=False, action="store_true", help="residual use of discrete features")

    parser.add_argument("--feature_birnn_hidden_dim", default=50, type=int, action="store")

    parser.add_argument("--use_discrete_features", default=False, action="store_true", help="David's indicator features")
    parser.add_argument("--split_hashtag", default=False, action="store_true", help="indicator of preceding hashtags")
    parser.add_argument("--cap", default=False, action="store_true", help="capitalization feature")
    parser.add_argument("--feature_dim", type=int, default=10, help="dimension of discrete features")

    parser.add_argument("--use_brown_cluster", default=False, action="store_true")
    parser.add_argument("--brown_cluster_path", action="store", type=str, help="path to the brown cluster features")
    parser.add_argument("--brown_cluster_num", default=0, type=int, action="store")
    parser.add_argument("--brown_cluster_dim", default=30, type=int, action="store")

    # Use trained model to test
    parser.add_argument("--mode", default="train", type=str, choices=["train", "test_1"],
                        help="test_1: use one model")

    # Partial CRF
    parser.add_argument("--use_partial", default=False, action="store_true")

    # Active Learning
    parser.add_argument("--ngram", default=2, type=int)
    parser.add_argument("--to_annotate", type=str,default="./annotate.txt")
    parser.add_argument("--entropy_threshold", type=float, default=None)
    parser.add_argument("--use_CFB", default=False, action="store_true")
    parser.add_argument("--SPAN_wise", default=False, action="store_true", help="get span wise scores, even if there are duplicates.")
    parser.add_argument("--k", default=200, type=int, help="fixed number of spans to annotate")
    parser.add_argument("--debug", type=str)
    # Format of test output
    parser.add_argument("--test_conll", default=False, action="store_true")
    parser.add_argument("--fixedVocab", default=False, action="store_true", help="for loading pre-trained model")
    parser.add_argument("--fineTune", default=False, action="store_true", help="for loading pre-trained model")
    parser.add_argument("--run",default=0, type=int)
    parser.add_argument("--misc",default=False, action="store_true")
    parser.add_argument("--addbias", default=False, action="store_true")
    args = parser.parse_args()


    if args.train_ensemble:
        # model_name = ens_1_ + original
        # set dynet seed manually
        ens_no = int(args.model_name.split("_")[1])
        # dyparams = dy.DynetParams()
        # dyparams.set_random_seed(ens_no + 5783287)
        # dyparams.init()

        import dynet_config
        dynet_config.set(random_seed=ens_no + 5783290)
        # if args.cuda:
        #     dynet_config.set_gpu()

        # args.train_path = args.train_path.split(".")[0] + "_" + str(ens_no) + ".conll"

    if args.full_data_path is None:
        args.full_data_path = args.train_path
    args.save_to_path = args.save_to_path + args.model_name + ".model"
    print(args)
    return args
