def find_c_after(c, s, idx):
    cur = idx - 1
    even = True
    while cur >= 0:
        if s[cur] == '\\':
            cur -= 1
            even = not even
        else:
            break

    for k in range(idx, len(s)):
        if s[k] == c:
            if even:
                return k
            even = True
        else:
            if s[k] == '\\':
                even = not even
            else:
                even = True
    return -1

def find_all_occ(s, c):
    result = []
    cur_idx = 0
    while cur_idx < len(s):
        i = find_c_after(c, s, cur_idx)
        if i == -1:
            break
        result.append(i)
        cur_idx = i + 1
    return result