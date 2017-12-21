from utils import *

root_folder = parent_directory(__file__, level=1)

# dataset_folder = os.path.join(root_folder, "deformable_rfcn", "Deformable-ConvNets", "data", "skoda", "train")

dataset_folder = os.path.join("/data", "rfcn-object-detection", "data", "tea_cartons1")

# dataset_folder = os.path.join(root_folder, "datasets", "skoda", "processed")

imagesets_folder = os.path.join(dataset_folder, "ImageSets")

image_folder = os.path.join(dataset_folder, "JPEGImages")

annotation_folder = os.path.join(dataset_folder, "Annotations")


classes = ['tea_carton1', 'tea_carton2', 'tea_carton3', 'tea_carton4', 'tea_carton5', 'tea_carton6', 'tea_carton7', 'tea_carton8', 'tea_carton9', 'tea_carton10']
