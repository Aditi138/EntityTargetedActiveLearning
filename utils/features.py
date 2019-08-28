import codecs
import numpy as np
import pdb


def get_feature_sent(lang, sent, args, cap_ratio_dict, type=None):
    dsf = []
    individual_feats = []

    if args.cap and not args.use_discrete_features:
        cap_feat = [w[0].isupper() for w in sent]
        individual_feats.append(cap_feat)

    if args.cap_ratio_path is not None:
        cap_feats = []
        for w in sent:
            # feat = np.zeros(4,)
            feat = [0, 0, 0, 0]
            if w in cap_ratio_dict:
                feat[cap_ratio_dict[w]] = 1
            cap_feats.append(feat)
        individual_feats.append(cap_feats)

    # individual_feats = zip(*individual_feats) # [(), ()]
    if len(dsf) > 0 and len(individual_feats) > 0:
        # individual_feats = [list(i) for i in individual_feats]
        dsf = [list(i) for i in dsf]
        # for i, d in zip(individual_feats, dsf):
        #    print i, d
        #    print len(i), len(d)
        new_feat = [list(tuple(i + d)) for i, d in zip(individual_feats[0], dsf)]
        # pdb.set_trace()
        return new_feat
    elif len(individual_feats) > 0:
        return individual_feats
    elif len(dsf) > 0:
        return dsf
    else:
        return []


def get_brown_cluster(path):
    bc_dict = dict()
    linear_map = dict()
    with codecs.open(path, "r", "utf-8") as fin:
        for line in fin:
            fields = line.strip().split('\t')
            if len(fields) == 3:
                word = fields[1]
                binary_string = fields[0]
                bid = int(binary_string, 2)
                if bid not in linear_map:
                    linear_map[bid] = len(linear_map)
                bc_dict[word] = linear_map[bid]
    return bc_dict
