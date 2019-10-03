from sgf.sgf_parser import parser_sgf, SGFParseError, board_rep
import pickle as pkl
import os
annotation_dir = '../annotations/'
b2cdir = 'b2c/'
if not os.path.exists(b2cdir):
    os.mkdir(b2cdir)

def create_sgf(node):
    s = 'SZ[%d];' % node.board_size
    for move in node.history:
        orig_bracket = move.orig_bracket
        s += '%s[%s];' % (orig_bracket.type, orig_bracket.content)
    return s

def saveb2c(f_name, node):
    comment_id = 0
    for state in node.all_children():
        comments = [content for tag, content in state.comments if tag == 'C']
        if len(comments) != 0:
            all_comments = '\n'.join(comments)
            sgf = create_sgf(state)
            saved_content = {
                'f_name': f_name,
                'comment_index': comment_id,
                'comments': all_comments,
                'board_state': state.board,
                'board_viz': board_rep(state.board),
                'sgf': sgf
            }
            pkl.dump(saved_content, open('%s%s-%d.pkl' % (b2cdir, f_name, comment_id), 'wb'))
            comment_id += 1

if __name__ == '__main__':
    import os
    f_names = os.listdir(annotation_dir)
    error_set = set()
    for f_name in f_names:
        try:
            print('parsing %s' % f_name)
            node = parser_sgf(annotation_dir + f_name)
            saveb2c(f_name, node)
        except UnicodeDecodeError:
            print('file encoding error')
            pass
        except SGFParseError as e:
            print('Parser error.')
            error_set.add(f_name)
    print('%d files cannot be parsed.' % len(error_set))
    print(error_set)