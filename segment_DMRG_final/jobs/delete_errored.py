import glob, os, sys
folder = sys.argv[1] if len(sys.argv) > 1 else '.'
for f in glob.glob(folder + '/python3-*.out'):
    if 'ValueError: no matching charge sectors' in open(f, errors='replace').read():
        os.remove(f)