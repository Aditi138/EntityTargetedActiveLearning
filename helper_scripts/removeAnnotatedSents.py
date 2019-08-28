import argparse
import codecs

def selectUnAnnotated(args):
    annotated_sents = set()
    with codecs.open(args.annotated, "r",encoding='utf-8') as fin:
        sent = []
        count = 0
        for line in fin:
            line = line.strip()
            if line == "" or line == "\n":
                annotated_sents.add(" ".join(sent))
                count +=1 
                sent =[]
            else:
                tokens = line.split("\t")
                sent.append(tokens[0])

    print(count, len(annotated_sents))
    fout = codecs.open("./annotated_sents.txt","w", encoding='utf-8') 
    for sent in annotated_sents:
        fout.write(sent + "\n")

    ffull = codecs.open("./orig_sents.txt","w", encoding='utf-8') 
    with codecs.open(args.input, "r", encoding='utf-8') as fin,  codecs.open(args.output, "w", encoding='utf-8') as fout:
        sent = []
        tokens = []
        for line in fin:
            line = line.strip()
            if line == "" or line == "\n":
                sentence = " ".join(tokens)
                ffull.write(sentence + "\n")
                tokens = []
                if sentence not in annotated_sents:
#q                    print(sentence)
                    for l in sent:
                        fout.write(l + "\n")
                    fout.write("\n")
                sent = []
            else:
                sent.append(line)
                tokens.append(line.split("\t")[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",type=str)
    parser.add_argument("--annotated", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    print(args)
    selectUnAnnotated(args)
