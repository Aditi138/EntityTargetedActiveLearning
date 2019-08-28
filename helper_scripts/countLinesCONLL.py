import codecs
import argparse

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("--input", help="Folder with the raw text data",
                        default=None,
                        type=str)

args = arg_parser.parse_args()
print("Args used for this run:")
print(args)

with codecs.open(args.input,"r",encoding='utf-8') as fin:
	index = 0
	one_line = []	
	for line in fin:
		if line == "" or line == "\n":
			if len(one_line) > 0:
				index +=1
			one_line = []
		else:
			line = line.strip()
			one_line.append(line)

if len(one_line)>0:
	index = index + 1
print index
		
