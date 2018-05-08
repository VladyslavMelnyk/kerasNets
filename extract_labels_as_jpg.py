import matplotlib.pyplot as plt
from datasets.sku_dataset import *

annot_folder = annotation_folder
img_folder = image_folder
imglist = train

for cls in classes:
    os.mkdir(os.path.join(data_folder, cls)

for i, file in enumerate(imglist):
       print "Processing file {}".format(str(file))
        img = os.path.join(img_folder, file + ".jpg")
        img = cv2.imread(img, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        annot_file = os.path.join(annot_folder, file + ".xml")
        tree = ET.parse(annot_file)
        for j, obj in enumerate(tree.findall('object')):
            if obj.find('deleted') == True and int(obj.find('deleted').text) == 1:
                continue
            else:
                bbox = obj.find('bndbox')
                cls = classes[classes.index(obj.find('name').text.lower().strip())]
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)
                obj_img = img[y1:y2, x1:x2]
                cv2.imwrite(os.path.join(data_folder, cls, "{}{}.jpg".format(file, j)), obj_img)
                #plt.imshow(obj_img,cmap='gray')
                #plt.show()
print "Finished proccesing all {} files!".format(i) 
