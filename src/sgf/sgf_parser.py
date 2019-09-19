import sys
sys.path.append('.')
from collections import namedtuple
import re
from sgf.str_util import find_c_after, find_all_occ
import warnings

annotation_dir = '../annotations/'
char_map = 'abcdefghijklmnopqrs'
char_map = {c: idx for idx, c in enumerate(char_map)}
removed = {'=\\'}
debug = True
type_of_interest = {'C', 'W', 'B', 'AW', 'AB'}
AW_count, AB_count = 0, 0

class SGFParseError(Exception):
    pass

Bracket = namedtuple('Bracket', ['type', 'content'])
Move = namedtuple('Move', ['player', 'r', 'c'])

class Branch:

    def __init__(self, start, parent):
        self.parent = parent
        self.start, self.children = start, []
        self.non_branch_s = None
        self.cur_pointer = self.start
        self.end = None
        self.segments = []
        self.completed = False

    def accept_bracket(self, bracket):
        bracket_start, bracket_end = bracket
        if self.cur_pointer != bracket_start:
            self.segments.append(('text', (self.cur_pointer, bracket_start)))
        self.segments.append(('bracket' , (bracket_start, bracket_end + 1)))
        self.cur_pointer = bracket_end + 1

    def retrieve_str(self, str_start, str_end):
        return self.non_branch_s[str_start - self.start: str_end - self.start]


    def complete_segs(self):
        self.completed_brackets = []
        idx = 0
        type_info = ''
        while idx < len(self.segments):
            if self.segments[idx][0] == 'text':
                if self.segments[idx + 1][0] != 'bracket':
                    raise SGFParseError('two consecutive text blocks without bracket content.')
                type_info = self.retrieve_str(*self.segments[idx][1]).strip().replace(';', '')
            else:
                content = self.retrieve_str(*self.segments[idx][1])[1:-1]
                if type_info in type_of_interest:
                    self.completed_brackets.append(Bracket(type=type_info,
                                                           content=content))
                if type_info not in ('AB', 'AW'):
                    type_info = ''
            idx += 1
        self.completed = True




class SGF:

    def __init__(self, root, meta_info):
        self.root, self.meta_info = root, meta_info


class SGFNode:

    def __init__(self, parent, move):
        self.parent, self.move = parent, move
        self.comments = []
        self.children = []
        self.board = None

    def get_board(self):
        if self.board is not None:
            return self.board
        elif self.parent is None:
            pass



open_branch = re.compile(r'(\()')

def parse_paren(s, brackets):
    bracket_region = set()
    for start, end in brackets:
        for i in range(start, end + 1):
            bracket_region.add(i)
    open_parens, close_parens = [[i for i in find_all_occ(s, c) if i not in bracket_region] for c in ('(', ')')]
    all_parens = [(idx, '(') for idx in open_parens] + [(idx, ')') for idx in close_parens]
    all_parens = sorted(all_parens)
    result, stack = [], []
    for paren in all_parens:
        if paren[1] == '(':
            new_branch = Branch(start=paren[0] + 1, parent=stack[-1] if len(stack) > 0 else None)
            if len(stack) > 0:
                last_branch = stack[-1]
                if last_branch.non_branch_s is None:
                    last_branch.non_branch_s = s[last_branch.start:paren[0]]
                last_branch.children.append(new_branch)
            stack.append(new_branch)
            result.append(new_branch)
        elif len(stack) == 0:
            warnings.warn('parens unambiguous but still well-formed')
        else:
            last_branch = stack[-1]
            if last_branch.non_branch_s is None:
                last_branch.non_branch_s = s[last_branch.start:paren[0]]
            del stack[-1]
    if len(stack) != 0:
        raise SGFParseError('parens are not well-formed, %d open parens remained.' % len(stack))
    for branch in result:
        branch.end = branch.start + len(branch.non_branch_s)
    return result


def parse_bracket(s):
    cur, results = 0, []
    while cur < len(s):
        left = find_c_after('[', s, cur)
        if left == -1:
            break
        right = find_c_after(']', s, left)
        if right == -1:
            raise SGFParseError('bracket is not well-formed.')
        results.append((left, right))
        cur = right + 1
    return results

def branch2sgfnode(branch, parent_node=None):
    start = None
    for bracket in branch.completed_brackets:
        if bracket.type in ('W', 'B', 'AW', 'AB'):
            player = 0 if bracket.type[-1] == 'B' else 1
            if bracket.content =='tt':
                continue
            if bracket.content == '':
                continue
            r, c = [char_map[coord] for coord in bracket.content.strip()]
            new_sgfnode = SGFNode(parent=parent_node, move=Move(player=player, r=r, c=c))
            if start is None:
                start = new_sgfnode
            if parent_node is not None:
                parent_node.children.append(new_sgfnode)
            parent_node = new_sgfnode
        elif bracket.type == 'C':
            comment = bracket.content
            if parent_node is not None:
                parent_node.comments.append(comment)
    for child_branch in branch.children:
        branch2sgfnode(child_branch, parent_node)
    return start

def print_moves(node):
    cur = node

    while True:
        print(cur.move)
        if len(cur.children) != 0:
            cur = cur.children[0]
        else:
            break
    return cur


def parser_sgf(f_name):
    with open(f_name, 'r') as in_file:
        s = in_file.read()
        for x in removed:
            s = s.replace(x, ' ')
        brackets = parse_bracket(s)
        branches = parse_paren(s, brackets)
        root = branches[0]
        cur_branch_idx, cur_bracket_idx = 0, 0
        while cur_bracket_idx < len(brackets):
            bracket = brackets[cur_bracket_idx]
            branch = branches[cur_branch_idx]
            if bracket[0] > branch.end:
                branch.complete_segs()
                cur_branch_idx += 1
            else:
                branch.accept_bracket(bracket)
                cur_bracket_idx += 1
        if not branches[-1].completed:
            branches[-1].complete_segs()
        sgf_root = branch2sgfnode(root)
        if debug:
            print_moves(sgf_root)
    print('parsing %s successful.' % f_name)


if __name__ == '__main__':
    import os
    f_names = os.listdir(annotation_dir)
    for f_name in f_names:
        print(f_name)
        try:
            parser_sgf(annotation_dir + f_name)
        except UnicodeDecodeError:
            pass
        except SGFParseError as e:
            print(f_name)
            raise e
        break