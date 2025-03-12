import os
from os.path import join
import imageio

mani_data_dir = './data_dir/'
image_names = []
image_class = []
ddir = os.path.join(mani_data_dir, 'CASIA/CASIA1')
with open(join(ddir, 'fake.txt')) as f:
    contents = f.readlines()
    for content in contents:
        image_names.append(os.path.join(ddir, 'fake', content.strip()))

ddir = os.path.join(mani_data_dir, 'CASIA/CASIA2')
with open(join(ddir, 'fake.txt')) as f:
    contents = f.readlines()
    for content in contents:
        image_names.append(os.path.join(ddir, 'fake', content.strip()))

# with open(join(ddir, 'au_list.txt')) as f:
#     contents = f.readlines()
#     for content in contents:
#         image_names.append(os.path.join(ddir, 'Au', content.strip()))


for image_name in image_names: 
    image = imageio.imread(image_name)  
    im_hight,im_width  = image.shape[:2]
    # mask
    if '.jpg' in image_name:
        mask_name = image_name.replace('fake', 'mask').replace('.jpg', '_gt.png')
    else:
        mask_name = image_name.replace('fake', 'mask').replace('.tif', '_gt.png')
    mask = imageio.imread(mask_name)
    ma_height, ma_width = mask.shape[:2]

    if im_width != ma_width or im_hight != ma_height:
        
        print('the sizes of image and mask are different: {}'.format(image_name))