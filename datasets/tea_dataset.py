from utils import *

root_folder = parent_directory(__file__, level=1)

train_folder = os.path.join("/data", "rfcn-object-detection", "data", "tea_cartons1", "train")
test_folder = os.path.join("/data", "rfcn-object-detection", "data", "tea_cartons1", "test")

tr_imagesets_folder = os.path.join(train_folder, "ImageSets")
tr_image_folder = os.path.join(train_folder, "JPEGImages")
tr_annotation_folder = os.path.join(train_folder, "Annotations")

ts_imagesets_folder = os.path.join(test_folder, "ImageSets")
ts_image_folder = os.path.join(test_folder, "JPEGImages")
ts_annotation_folder = os.path.join(test_folder, "Annotations")

box_annotation_folder = os.path.join(train_folder, "Annotations_box")
box_annotation_folder_ts = os.path.join(test_folder, "Annotations_box")

train = read_sets_file(tr_imagesets_folder, "train")
validation = read_sets_file(ts_imagesets_folder, "test")

classes = ['tea_carton1', 'tea_carton2', 'tea_carton3', 'tea_carton4', 'tea_carton5', 'tea_carton6', 'tea_carton7', 'tea_carton8', 'tea_carton9', 'tea_carton10']
classes_box = ['box']