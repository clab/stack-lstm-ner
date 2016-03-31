import sys
import argparse

parser = argparse.ArgumentParser(description='Give Me Conll data.')
parser.add_argument('-f', type=str, help='conll file')
args = parser.parse_args()

for line in open(args.f):
    sentence, actions = map(lambda x: x.strip().split(), line.strip().split('|||'))
    stack_state, buffer_state = [], sentence
    #print sentence, actions

    sys.stdout.write('\n')
    for action in actions:
        sys.stdout.write('{}{}\n'.format(str(stack_state), str(buffer_state)))
        if action[0] == 'O':
            assert len(stack_state) == 0
            buffer_state = buffer_state[1:]
        elif action[0] == 'S':
            stack_state.append(buffer_state[0])
            buffer_state = buffer_state[1:]
        elif action[0] == 'R':
            stack_state = []
        sys.stdout.write('{}\n'.format(action))
    assert len(stack_state) == 0 and len(buffer_state) == 0
    sys.stdout.write('{}{}\n'.format(str(stack_state), str(buffer_state)))

