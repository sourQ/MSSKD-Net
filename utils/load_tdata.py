import numpy as np
import torch.utils.data as data
from os.path import join,basename
from PIL import Image
import random
from random import randrange
import torch
import imageio


def generate_4masks(mask):
    mask_pil = Image.fromarray(mask)

    (width2, height2) = (mask_pil.width // 2, mask_pil.height // 2)
    (width3, height3) = (mask_pil.width // 4, mask_pil.height // 4)
    (width4, height4) = (mask_pil.width // 8, mask_pil.height // 8)

    mask2 = mask_pil.resize((width2, height2))
    mask3 = mask_pil.resize((width3, height3))
    mask4 = mask_pil.resize((width4, height4))

    mask = mask.astype(np.float32) / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    mask2 = np.asarray(mask2).astype(np.float32) / 255
    mask2[mask2 > 0.5] = 1
    mask2[mask2 <= 0.5] = 0

    mask3 = np.asarray(mask3).astype(np.float32) / 255
    mask3[mask3 > 0.5] = 1
    mask3[mask3 <= 0.5] = 0

    mask4 = np.asarray(mask4).astype(np.float32) / 255
    mask4[mask4 > 0.5] = 1
    mask4[mask4 <= 0.5] = 0

    mask = torch.from_numpy(mask)
    mask2 = torch.from_numpy(mask2)
    mask3 = torch.from_numpy(mask3)
    mask4 = torch.from_numpy(mask4)

    return mask, mask2, mask3, mask4


def data_aug(img, data_aug_ind):
    img = Image.fromarray(img)
    if data_aug_ind == 0:
        return np.asarray(img)
    elif data_aug_ind == 1:
        return np.asarray(img.rotate(90, expand=True))
    elif data_aug_ind == 2:
        return np.asarray(img.rotate(180, expand=True))
    elif data_aug_ind == 3:
        return np.asarray(img.rotate(270, expand=True))
    elif data_aug_ind == 4:
        return np.asarray(img.transpose(Image.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 5:
        return np.asarray(img.rotate(90, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 6:
        return np.asarray(img.rotate(180, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 7:
        return np.asarray(img.rotate(270, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
    else:
        raise Exception('Data augmentation index is not applicable.')


class TrainData(data.Dataset):
    def __init__(self, args):
        super(TrainData, self).__init__()
        path, crop_size, train_num, train_ratio, val_num = args['path'], args['crop_size'], args['train_num'], args['train_ratio'], args['val_num']

        # authentic
        authentic_names = []
        authentic_path = join(path, 'authentic')

        with open(join(authentic_path, 'authentic.txt')) as f:
            contents = f.readlines()
            for content in contents[val_num:]:
                authentic_names.append(join(authentic_path, content.strip()))

        # splice
        splice_names = []
        splice_path = join(path, 'splice')

        with open(join(splice_path, 'fake.txt')) as f:
            contents = f.readlines()
            for content in contents[val_num:]:
                splice_names.append(join(splice_path, content.strip()))

        splice_randmask = []
        splice_randmask_path = join(path, 'splice_randmask')

        with open(join(splice_randmask_path, 'fake.txt')) as f:
            contents = f.readlines()
            for content in contents:
                splice_randmask.append(join(splice_randmask_path, content.strip()))


        splice_names = splice_names + splice_randmask

        # copymove
        copymove_names = []
        copymove_path = join(path, 'copymove')

        with open(join(copymove_path, 'fake.txt')) as f:
            contents = f.readlines()
            for content in contents[val_num:]:
                copymove_names.append(join(copymove_path, content.strip()))

        # inpainting
        inpainting_names = []
        inpainting_path = join(path, 'inpainting')

        with open(join(inpainting_path, 'fake.txt')) as f:
            contents = f.readlines()
            for content in contents[val_num:]:
                inpainting_names.append(join(inpainting_path, content.strip()))

        self.image_names = [authentic_names, splice_names, copymove_names, inpainting_names]
        self.train_num = train_num
        self.train_ratio = train_ratio
        self.crop_size = crop_size

    def rgba2rgb(self, rgba, background=(255, 255, 255)):
        row, col, ch = rgba.shape

        rgb = np.zeros((row, col, 3), dtype='float32')
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

        a = np.asarray(a, dtype='float32') / 255.0

        R, G, B = background

        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B

        return np.asarray(rgb, dtype='uint8')

    def get_item(self, index):

        crop_width, crop_height = self.crop_size
        train_num = self.train_num
        train_ratio = self.train_ratio

        # get 4 class
        if index < train_num * train_ratio[0]:
            cls = 0
        elif train_num * train_ratio[0] <= index < train_num * (train_ratio[0] + train_ratio[1]):
            cls = 1
        elif train_num * (train_ratio[0] + train_ratio[1]) <= index < train_num * (
                train_ratio[0] + train_ratio[1] + train_ratio[2]):
            cls = 2
        else:
            cls = 3

        # get images in that class
        one_cls_names = self.image_names[cls]

        index = randrange(0, len(one_cls_names))

        # read the chosen image
        image_name = one_cls_names[index]
        image = imageio.imread(image_name)

        im_height, im_width, im_channel = image.shape

        if im_channel != 3:
            print(image_name)
            raise Exception('Image channel is not 3.')

        # authentic
        if cls == 0:
            if image.shape[-1] == 4:
                image = self.rgba2rgb(image)

            if im_height != crop_height or im_width != crop_width:
                # resize image
                image = Image.fromarray(image.astype(np.uint8))
                image = image.resize((crop_height, crop_width), resample=Image.BICUBIC)
                image = np.asarray(image)

            mask = np.zeros((crop_height, crop_width)).astype(np.uint8)

        # splice
        elif cls == 1:
            if '.jpg' in image_name:
                mask_name = image_name.replace('fake', 'mask').replace('.jpg', '.png')
            else:
                mask_name = image_name.replace('fake', 'mask').replace('.tif', '.png')

            mask = imageio.imread(mask_name)
            ma_height, ma_width = mask.shape[:2]

            if im_width != ma_width or im_height != ma_height:
                raise Exception('the sizes of image and mask are different: {}'.format(image_name))

            if im_height != crop_height or im_width != crop_width:
                # resize image
                image = Image.fromarray(image)
                image = image.resize((crop_height, crop_width), resample=Image.BICUBIC)
                image = np.asarray(image)
                # resize mask
                mask = Image.fromarray(mask)
                mask = mask.resize((crop_height, crop_width), resample=Image.BICUBIC)
                mask = np.asarray(mask)

        # copymove
        elif cls == 2:
            mask = imageio.imread(image_name.replace('fake', 'mask'))
            ma_height, ma_width = mask.shape[:2]

            if im_width != ma_width or im_height != ma_height:
                raise Exception('the sizes of image and mask are different: {}'.format(image_name))

            if im_height != crop_height or im_width != crop_width:
                # resize image
                image = Image.fromarray(image)
                image = image.resize((crop_height, crop_width), resample=Image.BICUBIC)
                image = np.asarray(image)
                # resize mask
                mask = Image.fromarray(mask)
                mask = mask.resize((crop_height, crop_width), resample=Image.BICUBIC)
                mask = np.asarray(mask)

        # inpainting
        elif cls == 3:
            mask = imageio.imread(image_name.replace('fake', 'mask').replace('.jpg', '.png'))
            ma_height, ma_width = mask.shape[:2]

            if im_width != ma_width or im_height != ma_height:
                raise Exception('the sizes of image and mask are different: {}'.format(image_name))

            if im_height != crop_height or im_width != crop_width:
                # resize image
                image = Image.fromarray(image)
                image = image.resize((crop_height, crop_width), resample=Image.BICUBIC)
                image = np.asarray(image)
                # resize mask
                mask = Image.fromarray(mask)
                mask = mask.resize((crop_height, crop_width), resample=Image.BICUBIC)
                mask = np.asarray(mask)

        else:
            raise Exception('class is not defined!')

        # image
        aug_index = randrange(0, 8)
        image = data_aug(image, aug_index)
        image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)

        # mask
        mask = data_aug(mask, aug_index)
        mask, mask2, mask3, mask4 = generate_4masks(mask)

        return image, [mask, mask2, mask3, mask4], cls

    def __getitem__(self, index):
        res = self.get_item(index)
        return res

    def __len__(self):
        return self.train_num


class ValData(data.Dataset):
    def __init__(self, args):
        super(ValData, self).__init__()

        path, val_num = args['path'], args['val_num']

        # authentic
        authentic_names = []
        authentic_path = join(path, 'authentic')

        with open(join(authentic_path, 'authentic.txt')) as f:
            contents = f.readlines()
            for content in contents[:val_num]:
                authentic_names.append(join(authentic_path, content.strip()))

        authentic_cls = [0] * val_num

        # splice
        splice_names = []
        splice_path = join(path, 'splice')

        with open(join(splice_path, 'fake.txt')) as f:
            contents = f.readlines()
            for content in contents[:val_num]:
                splice_names.append(join(splice_path, content.strip()))

        splice_cls = [1] * val_num

        # copymove
        copymove_names = []
        copymove_path = join(path, 'copymove')

        with open(join(copymove_path, 'fake.txt')) as f:
            contents = f.readlines()
            for content in contents[:val_num]:
                copymove_names.append(join(copymove_path, content.strip()))

        copymove_cls = [2] * val_num

        # inpainting
        inpainting_names = []
        inpainting_path = join(path, 'inpainting')

        with open(join(inpainting_path, 'fake.txt')) as f:
            contents = f.readlines()
            for content in contents[:val_num]:
                inpainting_names.append(join(inpainting_path, content.strip()))

        inpainting_cls = [3] * val_num

        self.image_names = authentic_names + splice_names + copymove_names + inpainting_names
        self.image_class = authentic_cls + splice_cls + copymove_cls + inpainting_cls

    def get_item(self, index):
        image_name = self.image_names[index]
        cls = self.image_class[index]

        image = imageio.imread(image_name)

        im_height, im_width, im_channel = image.shape

        if im_channel != 3:
            print(image_name)
            raise Exception('Image channel is not 3.')

        # image
        image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)

        # authentic
        if cls == 0:
            # mask
            mask = np.zeros((im_height, im_width))
            mask = torch.from_numpy(mask.astype(np.float32))

        # splice
        elif cls == 1:
            # mask
            mask = imageio.imread(image_name.replace('fake', 'mask').replace('.jpg', '.png'))
            mask = torch.from_numpy(mask.astype(np.float32) / 255)

        # copymove
        elif cls == 2:
            # mask
            mask = imageio.imread(image_name.replace('fake', 'mask'))
            mask = torch.from_numpy(mask.astype(np.float32) / 255)

        # inpainting
        elif cls == 3:
            # mask
            mask = imageio.imread(image_name.replace('fake', 'mask').replace('.jpg', '.png'))
            mask = torch.from_numpy(mask.astype(np.float32) / 255)

        else:
            raise Exception('class is not defined!')

        return image, mask, cls, image_name

    def __getitem__(self, index):
        res = self.get_item(index)

        return res

    def __len__(self):
        return len(self.image_names)


class Finetune_TrainData(data.Dataset):
    def __init__(self, args):
        super(Finetune_TrainData, self).__init__()
        path, crop_size, train_num, train_ratio, val_num = './data_dir', args['crop_size'], args['train_num'], args['train_ratio'], args['val_num']

        # columbia
        # columbia_names = []
        # columbia_path = join(path, 'columbia/')
        # trian_len = 100
        # trian_len_casia = 100
        # with open(join(columbia_path, 'fake.txt')) as f:
        #     contents = f.readlines()
        #     trian_len = int(len(contents)*0.75)
        #     for content in contents[0:trian_len]:
        #         columbia_names.append(join(columbia_path, 'fake',content.strip()))

        # coverage
        # coverage_names = []
        # coverage_path = join(path, 'Coverage')

        # with open(join(coverage_path, 'fake.txt')) as f:
        #     contents = f.readlines()
        #     trian_len_coverage = int(len(contents)*0.75)
        #     for content in contents[0:trian_len_coverage]:
        #         coverage_names.append(join(coverage_path,'fake', content.strip()))

        # # # casia
        casia_names = []
        trian_len_casia = 0

        # ddir = join(path, 'CASIA/CASIA1')
        # with open(join(ddir, 'fake.txt')) as f:
        #     contents = f.readlines()
        #     trian_len_casia = int(len(contents)*0.75)
        #     for content in contents[0:trian_len_casia]:
        #         casia_names.append(join(ddir, 'fake', content.strip()))

        ddir = join(path, 'CASIA/CASIA2')
        with open(join(ddir, 'fake.txt')) as f:
            contents = f.readlines()
            trian_len_casia = trian_len_casia + int(len(contents)*0.75)
            for content in contents:
                casia_names.append(join(ddir, 'fake', content.strip()))
        
        with open(join(ddir, 'au_list.txt')) as f:
            contents = f.readlines()
            trian_len_casia = trian_len_casia + int(len(contents)*0.75)
            for content in contents:
                casia_names.append(join(ddir, 'Au', content.strip()))

        # nist16
        # nist_names = []
        # nist_mask = []
        # trian_len_nist=100
        # ddir = join(path, 'NIST16')
        # file_name = 'alllist.txt'
        # with open(join(ddir, file_name)) as f:
        #     contents = f.readlines()
        #     trian_len_nist = int(len(contents)*0.75)
        #     for content in contents[0:trian_len_nist]:
        #         image_name, mask_name = content.split(' ')
        #         nist_names.append(join(ddir,'fake', image_name))
        #         nist_mask.append(join(ddir, 'mask',mask_name.strip()))

        self.image_names = [casia_names]#修改数据
        # self.mask_names = [nist_mask]#nist16数据库独有的
        self.train_num = trian_len_casia#修改每一个数据及集的长度
        self.train_ratio = train_ratio
        self.crop_size = crop_size

    def rgba2rgb(self, rgba, background=(255, 255, 255)):
        row, col, ch = rgba.shape

        rgb = np.zeros((row, col, 3), dtype='float32')
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

        a = np.asarray(a, dtype='float32') / 255.0

        R, G, B = background

        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B

        return np.asarray(rgb, dtype='uint8')

    def get_item(self, index):

        crop_width, crop_height = self.crop_size
        train_num = self.train_num
        train_ratio = self.train_ratio

        # get 4 class
        # if index < train_num * train_ratio[0]:
        #     cls = 0
        # elif train_num * train_ratio[0] <= index < train_num * (train_ratio[0] + train_ratio[1]):
        #     cls = 1
        # elif train_num * (train_ratio[0] + train_ratio[1]) <= index < train_num * (
        #         train_ratio[0] + train_ratio[1] + train_ratio[2]):
        #     cls = 2
        # else:
        #     cls = 3

        # get images in that class
        cls = 1
        cls_fl = 3
        one_cls_names = self.image_names[0]

        index = randrange(0, len(one_cls_names))

        # read the chosen image
        image_name = one_cls_names[index]
        image = imageio.imread(image_name)

        im_height, im_width, im_channel = image.shape

        if im_channel != 3:
            print(image_name)
            raise Exception('Image channel is not 3.')

        # authentic
        if cls_fl == 0:
            if image.shape[-1] == 4:
                image = self.rgba2rgb(image)

            if im_height != crop_height or im_width != crop_width:
                # resize image
                image = Image.fromarray(image.astype(np.uint8))
                image = image.resize((crop_height, crop_width), resample=Image.BICUBIC)
                image = np.asarray(image)

            mask = np.zeros((crop_height, crop_width)).astype(np.uint8)

        # columbia
        elif cls_fl == 1:
            mask_name = image_name.replace('fake','mask').replace('.tif', '_gt.png')

            mask = imageio.imread(mask_name)
            ma_height, ma_width = mask.shape[:2]

            if im_width != ma_width or im_height != ma_height:
                raise Exception('the sizes of image and mask are different: {}'.format(image_name))

            if im_height != crop_height or im_width != crop_width:
                # resize image
                image = Image.fromarray(image)
                image = image.resize((crop_height, crop_width), resample=Image.BICUBIC)
                image = np.asarray(image)
                # resize mask
                mask = Image.fromarray(mask)
                mask = mask.convert('L')
                mask = mask.resize((crop_height, crop_width), resample=Image.BICUBIC)
                mask = np.asarray(mask)

        # coverage
        elif cls_fl == 2:

            if 't' not in basename(image_name).split(".")[0]:
                cls = 0
                # mask = np.zeros((crop_height, crop_width))
                # mask = torch.from_numpy(mask.astype(np.float32))
                # mask = np.asarray(mask)
                mask = np.zeros((crop_height, crop_width)).astype(np.uint8)
            else:
                mask_name = image_name.replace('fake', 'mask').replace('t.tif', 'forged.tif')
                mask = imageio.imread(mask_name)
                ma_height, ma_width = mask.shape[:2]

                if im_width != ma_width or im_height != ma_height:
                    raise Exception('the sizes of image and mask are different: {}'.format(image_name))

            if im_height != crop_height or im_width != crop_width:
                # resize image
                image = Image.fromarray(image)
                image = image.resize((crop_height, crop_width), resample=Image.BICUBIC)
                image = np.asarray(image)
                # resize mask
                
                mask = Image.fromarray(mask)
                mask = mask.convert('L')
                mask = mask.resize((crop_height, crop_width), resample=Image.BICUBIC)
                mask = np.asarray(mask)

        # casia
        elif cls_fl == 3:
            if 'Au' in basename(image_name).split(".")[0]:
                cls = 0
                # mask = np.zeros((crop_height, crop_width))
                # mask = torch.from_numpy(mask.astype(np.float32))
                # mask = np.asarray(mask)
                mask = np.zeros((crop_height, crop_width)).astype(np.uint8)
            else:
                if '.jpg' in image_name:
                    mask_name = image_name.replace('fake', 'mask').replace('.jpg', '_gt.png')
                else:
                    mask_name = image_name.replace('fake', 'mask').replace('.tif', '_gt.png')

                mask = imageio.imread(mask_name)

                ma_height, ma_width = mask.shape[:2]

                if im_width != ma_width or im_height != ma_height:
                    raise Exception('the sizes of image and mask are different: {}'.format(image_name))

            if im_height != crop_height or im_width != crop_width:
                # resize image
                image = Image.fromarray(image)
                image = image.resize((crop_height, crop_width), resample=Image.BICUBIC)
                image = np.asarray(image)
                # resize mask
                mask = Image.fromarray(mask)
                mask = mask.convert('L')
                mask = mask.resize((crop_height, crop_width), resample=Image.BICUBIC)
                mask = np.asarray(mask)
        # casia
        elif cls_fl == 4:
            # image_name = self.image_names[0][index]
            mask_name = self.mask_names[0][index]
            # print(image_name)
            # print(mask_name)
            if 'splice' in mask_name:
                cls = 1
            elif 'remove' in mask_name:
                cls = 2
            elif 'mani' in mask_name:
                cls = 3
            else:
                cls = 0
           
            mask = imageio.imread(mask_name)

            ma_height, ma_width = mask.shape[:2]

            if im_width != ma_width or im_height != ma_height:
                raise Exception('the sizes of image and mask are different: {}'.format(image_name))

            if im_height != crop_height or im_width != crop_width:
                # resize image
                if image.shape[-1] == 4:
                    image = self.rgba2rgb(image)
                image = Image.fromarray(image)
                image = image.resize((crop_height, crop_width), resample=Image.BICUBIC)
                image = np.asarray(image)
                # image = torch.from_numpy(image.astype(np.float32) / 255)
                # image = image.permute(2, 0, 1)
                # image = np.asarray(image)
                # resize mask
                mask = Image.fromarray(mask)
                mask = mask.convert('L')
                mask = mask.resize((crop_height, crop_width), resample=Image.BICUBIC)                
                mask = np.asarray(mask)
                # mask
                   


        else:
            raise Exception('class is not defined!')

        # image
        aug_index = randrange(0, 8)
        image = data_aug(image, aug_index)
        image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)

        # mask
        mask = data_aug(mask, aug_index)
        mask, mask2, mask3, mask4 = generate_4masks(mask)
        if cls_fl == 4:
            mask = torch.abs(mask - 1)
            mask2 = torch.abs(mask2 - 1)
            mask3 = torch.abs(mask3 - 1)
            mask4 = torch.abs(mask4 - 1)


        return image, [mask, mask2, mask3, mask4], cls

    def __getitem__(self, index):
        res = self.get_item(index)
        return res

    def __len__(self):
        return self.train_num
