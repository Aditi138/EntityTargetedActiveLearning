import codecs, argparse


def pickKTokens(args):
    with codecs.open(args.input, "r", encoding='utf-8') as fin, codecs.open(args.output, "w", encoding='utf-8') as fout:
        count = args.k
        one_sent = []
        for line in fin:
            if line == "" or line == "\n":
                for s in one_sent:
                    fout.write(s + "\n")
                fout.write('\n')
                one_sent = []
                if count <=0:
                    break

            else:
                tokens = line.strip().split("\t")
                tag  = tokens[1]
                token = tokens[0]
                if "UNK" in tag:
                    count -= 1

                one_sent.append(line.strip())


        if len(one_sent) > 0:
            for s in one_sent:
                fout.write(s + "\n")
            fout.write('\n')

def pickKTokensRev(args):
    with codecs.open(args.input, "r", encoding='utf-8') as fin, codecs.open(args.output, "w", encoding='utf-8') as fout:
        count = args.k
        one_sent = []
        for line in fin:
            if line == "" or line == "\n":
                for s in one_sent:
                    fout.write(s + "\n")
                fout.write('\n')
                one_sent = []
                if count <=0:
                    break

            else:
                tokens = line.strip().split("\t")
                tag  = tokens[1]
                token = tokens[0]
                count -= 1

                one_sent.append(line.strip())


        if len(one_sent) > 0:
            for s in one_sent:
                fout.write(s + "\n")
            fout.write('\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--k", type=int)
    parser.add_argument("--output",type=str)
    args = parser.parse_args()

    #pickKTokens(args)
    pickKTokensRev(args)
