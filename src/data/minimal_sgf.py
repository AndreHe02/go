import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
args = parser.parse_args()
params = vars(args)

with open(params["file"]) as f:
    output = ""
    text = f.read().replace('\n', '')
    text = text.split(";")
    for val in text:
        if val[0:2] == "W[" or val[0:2] == "B[":
            output += (";" + val.split("]")[0] + "]")
    print(output)
