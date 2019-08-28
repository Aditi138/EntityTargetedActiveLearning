import codecs
import argparse


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("--files", help="File 1",
                        default=None,nargs='+')

#arg_parser.add_argument("--file2", help="File 2",
#                        default=None,
#                        type=str)

arg_parser.add_argument("--output", help="Output File",
                        default=None,
                        type=str)

args = arg_parser.parse_args()
print("Args used for this run:")
print(args)


files = args.files
fout = codecs.open(args.output, "w", encoding='utf-8')

for i in files:
    with codecs.open(i,"r", encoding='utf-8') as fin:
        for line in fin:
            fout.write(line)
        print "Done reading file: " + str(i)
        fout.write("\n")
