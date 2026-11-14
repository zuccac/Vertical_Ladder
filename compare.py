"""compare.py -- macro-by-macro comparison; reads \\newcommand and \\renewcommand (later definitions win);
thousands separators {,} and , and a sign on zero (+0, -0) are treated as the same value."""
import re, sys
def norm(v):
    v = v.replace('{,}', ',')
    return v[1:] if re.fullmatch(r'[+-]0(\.0+)?', v) else v


def load(*fs):
    out = {}
    for f in fs:
        t = open(f).read().replace('{,}', '\x00')
        for k, v in re.findall(r'\\(?:re)?newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', t):
            out[k] = norm(v.replace('\x00', '{,}'))
    return out
i = sys.argv.index('--')
old, new = load(*sys.argv[1:i]), load(*sys.argv[i + 1:])
same = [k for k in old if k in new and old[k] == new[k]]
diff = [k for k in old if k in new and old[k] != new[k]]
miss = [k for k in old if k not in new]
extra = [k for k in new if k not in old]
print(f'published {len(old)} | rebuilt {len(new)} | identical {len(same)} | different {len(diff)} | missing {len(miss)} | extra {len(extra)}')
for k in diff: print(f'  DIFF {k}: published {old[k]}  rebuilt {new[k]}')
if miss: print('  MISSING:', ' '.join(miss))
if extra: print('  EXTRA:', ' '.join(extra))
