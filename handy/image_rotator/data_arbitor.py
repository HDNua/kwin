from kwin import *
import dataset
from PIL import Image


#
def arbitor(target_parent, target_dir):
    """

    :param target_parent:
    :param target_dir:
    :return:
    """
    #
    src_names = os.listdir(target_dir)
    i = 1
    for filename in src_names:
        if i % 100 == 0:
            print("%dth work has been completed" % (i))

        target_angle = "000"
        if i % 4 == 1:
            target_angle = "090"
        elif i % 4 == 2:
            target_angle = "180"
        elif i % 4 == 3:
            target_angle = "270"

        _, file_extension = os.path.splitext(filename)
        src_name = target_dir + filename
        dst_name = "%s/%s/%s" % (target_parent, target_angle, filename)

        if False:
            os.rename(src_name, dst_name)
        else:
            img = Image.open(src_name)
            img_out = img.rotate(-int(target_angle), expand=True)
            # img_out = img_out.resize((72, 40))
            img_out.save(dst_name)

        i += 1

    print("All works of [%s] are done" % (target_dir))


#
cwd = dataset.train_path()  # resource_path("wr") + '/' # sys.argv[1]
print('current working directory is [%s]' %(cwd))

#
src_names = os.listdir(cwd)
if True:
    for filename in src_names:
        if filename == "000":
            continue
        elif filename == "090":
            continue
        elif filename == "180":
            continue
        elif filename == "270":
            continue

        target_dir = cwd + '/' + filename + '/'
        arbitor(cwd, target_dir)

#
print("All works are done")
