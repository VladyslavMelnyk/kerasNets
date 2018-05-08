from utils import *

root_folder = parent_directory(__file__, level=1)

data_folder = os.path.join("/home/ubuntu/", "keras-retinanet", "data", "photos_105_labeled")

imagesets_folder = os.path.join(data_folder, "ImageSets", "Main")
image_folder = os.path.join(data_folder, "JPEGImages")
annotation_folder = os.path.join(data_folder, "Annotations")

train = read_sets_file(imagesets_folder, "trainval")
validation = read_sets_file(imagesets_folder, "test")

classes = ['sku']