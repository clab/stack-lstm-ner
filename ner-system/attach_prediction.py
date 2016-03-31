import sys
import argparse

parser = argparse.ArgumentParser(description='Give Me Conll data.')
parser.add_argument('-p', type=str, help='prediction file')
parser.add_argument('-t', type=str, help='test file')
parser.add_argument('-o', type=str, help='output destination')
args = parser.parse_args()

f = open(args.o,'w')

for test_line, pred_line in zip(open(args.t), open(args.p)):
    test_line = test_line.strip().split()
    pred_line = pred_line.strip().split()
    
    
    if len(test_line) > 0:
        #assert test_line[0] == pred_line[0], "your prediction is not aligned to test file"
        test_line.append(pred_line[-1])
        f.write('{}\n'.format(" ".join(test_line)))
    else:
        f.write("\n")



