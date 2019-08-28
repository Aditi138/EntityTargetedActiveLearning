import codecs
import argparse
from copy import deepcopy






def annotate(input, output):
    gold_lines = []

    with codecs.open(input, "r", encoding='utf-8') as fin,codecs.open(output, "w", encoding='utf-8') as fout:
        actual_line = []
        actual_one_line = []

        crf_line= []
        crf_one_line = []

        gold_one_line = []
        prev = ""
        for line in fin:
            if line == "" or line == "\n":
                #fout.write("\n")
                actual_line.append(actual_one_line)
                actual_one_line  = []

                crf_line.append(crf_one_line)
                crf_one_line = []

                gold_lines.append(gold_one_line)
                gold_one_line = []

		prev = ""
            else:
                tokens  = line.strip().split()
                gold_one_line.append(tokens[-1])

                if "UNK" in tokens[1]:   #Find the true start of the entity
                    #fout.write(tokens[0] + "\t" + tokens[-1] + '\n')
                    actual_one_line.append(tokens[0] + "\t" + tokens[-1])
                    prev = tokens[-1]


                else:
                    #fout.write(tokens[0] + "\t" + tokens[1] + '\n')
                   # actual_one_line.append(tokens[0] + "\t" + tokens[1])
                    if prev != "" and tokens[-1].startswith("I-"):
                        BIO_tag = tokens[-1]
                        prev  =tokens[-1]
                    else:
                        if args.needUNK:
                            BIO_tag = "B-UNK"
                        else:
                            #BIO_tag = "O"
                             BIO_tag = tokens[1]
                        prev = ""
                    actual_one_line.append(tokens[0] + "\t" + BIO_tag)


        index = 0
        lines = []
        one_line = []
        for line in actual_line:
            prev = ""
            for token_tag in line:
                current_tag = token_tag.split("\t")[-1]
                token = token_tag.split("\t")[0]


                if prev != "":
                    if prev == "O" and "I-" in current_tag:
                        #print("Check index :{0} for inconsistency {1}".format(index, token))
                        token_tag = token + "\t" + "B-" + current_tag.split("-")[-1]

                    if (prev == "B-PER" or prev == "I-PER") and current_tag in ['I-LOC, I-ORG, I-GPE']:
                        #print("Check index :{0} for inconsistency {1}".format(index, token))
                        token_tag = token + "\t" + "I-PER"

                    if (prev == "B-GPE" or prev == "I-GPE") and current_tag in ['I-LOC, I-ORG, I-PER']:
                        #print("Check index :{0} for inconsistency".format(index,token))
                        token_tag = token + "\t" + "I-GPE"

                    if (prev == "B-LOC" or prev == "I-LOC") and current_tag in ['I-PER, I-ORG, I-GPE']:
                        #print("Check index :{0} for inconsistency {1}".format(index,token))
                        token_tag = token + "\t" + "I-LOC"

                    if (prev == "B-ORG" or prev == "I-ORG") and current_tag in ['I-LOC, I-PER, I-GPE']:
                        #print("Check index :{0} for inconsistency {1}".format(index,token))
                        token_tag = token + "\t" + "I-ORG"



                prev = current_tag

                index +=1
                one_line.append(token_tag)
                #fout.write(token_tag + "\n")
            index += 1
            lines.append(one_line)
            one_line =[]
            #fout.write("\n")
        print(len(lines))
        for line_num, line in enumerate(lines):
            prev = ""
            for token_num, token_tag in enumerate(line):
                token = token_tag.split("\t")[0]
                tag = token_tag.split("\t")[-1]
                if prev != "":
                    if prev in ["B-UNK","O"] and tag in ["I-LOC", "I-GPE", "I-LOC", "I-MISC","I-PER","I-ORG"]:
                        gold_one_line = gold_lines[line_num]
                        gold_cur_tag = gold_one_line[token_num]
                        temp_num = deepcopy(token_num)
                        while not gold_cur_tag.startswith("B-"):
                            temp_num -=1
                            gold_cur_tag =  gold_one_line[temp_num]
                            line[temp_num] = line[temp_num].split("\t")[0] + "\t" + gold_cur_tag
                prev = tag

            for token_tag in line:
                fout.write(token_tag + '\n')
            fout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Active learning output")
    parser.add_argument("--output", type=str, default=None, help ="Simulated NI with gold annotations in place of UNK")
    parser.add_argument("--needUNK", default=False, action="store_true", help="Simulated NI with gold annotations in place of UNK")
    args = parser.parse_args()

    annotate(args.input, args.output)
