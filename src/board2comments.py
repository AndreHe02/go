from sgf.sgf_parser import parser_sgf, SGFParseError, board_rep
import pickle as pkl
import os
#annotation_dir = '../data/annotations/'
#b2cdir = 'b2c/'
#if not os.path.exists(b2cdir):
#    os.mkdir(b2cdir)

def create_sgf(node):
    s = 'SZ[%d];' % node.board_size
    for move in node.history:
        orig_bracket = move.orig_bracket
        s += '%s[%s];' % (orig_bracket.type, orig_bracket.content)
    return s

def saveb2c(f_name, node):
    comment_id = 0
    sgf_content = []
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
            sgf_content.append(saved_content)
            #pkl.dump(saved_content, open('%s%s-%d.pkl' % (b2cdir, f_name, comment_id), 'wb'))
            comment_id += 1
    return sgf_content

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', default='../data/annotations/')
    parser.add_argument('-o', '--output', default='b2c/')
    args = parser.parse_args()

    annotation_dir = args.dir
    b2cdir = args.output

    if not os.path.exists(b2cdir):
        os.mkdir(b2cdir)

    import os
    f_names = os.listdir(annotation_dir)
    error_set = set()
    content = []
    for f_name in f_names:
        try:
            print('parsing %s' % f_name)
            node = parser_sgf(annotation_dir + f_name)
            sgf_content = saveb2c(f_name, node)
            content.extend(sgf_content)
        except UnicodeDecodeError:
            print('file encoding error')
            pass
        except SGFParseError as e:
            print('Parser error.')
            error_set.add(f_name)
        except KeyError as e:
            print('malformed file')
            error_set.add(f_name)
    pkl.dump(content, open(os.path.join(b2cdir, 'annotations.pkl'), 'wb'))
    print('%d files cannot be parsed.' % len(error_set))
    print(error_set)