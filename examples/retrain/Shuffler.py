import sys
import os
import random

#
cwd = "D:/handy/work/project/kwin/kwin4/examples/retrain/test/5" # sys.argv[1]
dir = cwd + '/'
print(cwd)

if True:
    src_names = os.listdir(cwd)
    for filename in src_names:
        os.rename(dir + filename, dir + "_" + filename)

    #
    src_names = os.listdir(cwd)
    random.shuffle(src_names)

    i = 0
    for filename in src_names:
        if i % 100 == 0:
            print("%dth work has been completed" %(i))

        _, file_extension = os.path.splitext(filename)
        new_name = dir + str(i) + file_extension
        os.rename(dir + filename, new_name)
        i += 1

    print("All works are done")
