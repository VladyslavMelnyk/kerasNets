from datasets.tea_dataset import *
import io
import json
from pycocotools import mask

classes = classes_box
annotFolder = box_annotation_folder_ts
imgFolder = ts_image_folder
imgList = validation_box
json_path=os.path.abspath(os.path.join(annotFolder, "..", "tea_coco_test.json"))


def convert_to_COCO(img_folder, annot_folder, img_list, json_path=json_path):
    get_annot_data.counter = 0
    data = {"images":[],"type":"instances", "annotations":[], "categories":[]}
    imdata = data["images"]
    annot_data = data["annotations"]
    cat_data = data["categories"]
    print("Writing JSON metadata...")
    for i, file in enumerate(img_list):
        print "Processing file {}".format(str(file))
        imdata.extend(get_img_data(img_folder, file, i))
        annot_data.extend(get_annot_data(annot_folder, file, i))
    cat_data.extend(get_cat_data(classes))
    print("Finished proccesing all {} files!".format(i))
    data = {"images":imdata, "type":"instances", "annotations":annot_data, "categories":cat_data}
    json_dump(data, json_path)



def get_img_data(img_folder, file, i):
    img = os.path.join(img_folder, file + ".jpg")
    img = cv2.imread(img, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    height, width, channels = img.shape
    return [{"file_name":str(file+".jpg"), "height":height, "width":width, "id":i}]

def get_cat_data(classes):
    categories = []
    for i in range(1, len(classes)+1):
        categories.append({"supercategory":"none","id":i,"name":"{}".format(classes[i-1])})
    return categories


def get_annot_data(annot_folder, file, i):
    annotations = []
    annot_file = os.path.join(annot_folder, file + ".xml")
    tree = ET.parse(annot_file)
    for obj in tree.findall('object'):
        if obj.find('deleted') == True and int(obj.find('deleted').text) == 1:
            continue
        else:
            get_annot_data.counter += 1
            bbox = obj.find('bndbox')
            cat_id = classes.index(obj.find('name').text.lower().strip()) + 1
            if bbox.find('xmin').text.isdigit() == True and bbox.find('ymin').text.isdigit() == True and \
                bbox.find('xmax').text.isdigit() == True and bbox.find('ymax').text.isdigit() == True:
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)
            box = [x1, y1, x2-x1, y2-y1]
            labelMask = np.zeros((box[3],box[2]))
            labelMask[:, :] = box[3] * box[2]
            labelMask = np.expand_dims(labelMask, axis=2)
            labelMask = labelMask.astype('uint8')
            labelMask = np.asfortranarray(labelMask)
            Rs = mask.encode(labelMask)

        annotations.append({"segmentation": [[box[0], box[1], box[0], box[1] + box[3], box[0] + box[2], box[1] + box[3], box[0] + box[2], box[1]]],
         "area": float(mask.area(Rs)), "iscrowd": 0, "image_id": i,
         "bbox": box, "category_id": cat_id, "id": get_annot_data.counter, "ignore": 0})
    return annotations


def json_dump(data, json_file):
    with io.open(json_file, 'w', encoding='utf8') as json_file:
        indent = 0
        separators = (',', ':')
        ensure_ascii = False
        str_ = json.dumps(data, indent=indent, sort_keys=True, separators=separators, ensure_ascii=ensure_ascii)
        # str_ = str_[1:-2] + ',\n'  # Remove brackets and add comma
        json_file.write(unicode(str_))

if  __name__ =='__main__':convert_to_COCO(imgFolder, annotFolder, imgList)