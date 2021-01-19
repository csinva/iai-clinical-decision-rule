import os
import re
from os.path import join as oj

NOTEBOOKS_DIR = '../notebooks'
OUTPUT_FILE = 'readme.md'
ss = 'This is an overview of the markdown contents of all the notebooks / scripts in this directory.\n\n'


# add pcs.md
# pcs = open('pcs.md', 'r').read()
# ss += pcs + '\n\n'

for fname in sorted(os.listdir(NOTEBOOKS_DIR)):
    if fname.endswith('.md') and not fname == OUTPUT_FILE:
        with open(oj(NOTEBOOKS_DIR, fname), 'r') as f:
            s = f.read()
        s = re.sub("\```([^`]+)\```", '', s) # remove all code blocks
        s = re.sub("\---([^`]+)\---", '', s) # remove header
        s = s.replace('#### ', '##### ') # make all headers one header lower
        s = s.replace('### ', '#### ') # make all headers one header lower
        s = s.replace('## ', '### ') # make all headers one header lower
        s = s.replace('# ', '## ') # make all headers one header lower
        s = re.sub("((\r?\n|\r)\d*)+(\r?\n|\r)", '\n\n', s) # remove double newlines
        s = f'# {fname[:-3]}\n' + s
        ss += s
        os.remove(oj(NOTEBOOKS_DIR, fname))
        
                
with open(oj(NOTEBOOKS_DIR, OUTPUT_FILE), 'w') as f:
    f.write(ss)