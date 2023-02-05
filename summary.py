import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Grayscale
from torchvision import transforms
from skimage import io, transform
import os


class Rescale(object):

    '''
     THE output_size is the desired shape of the image
     Args:
         output_size(int, tuple): if int the smaller image's edge is the output_size
                                  if tuple each element represent each image edge
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h // w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w // h
        else:
            new_h, new_w = self.output_size

        image = transform.resize(image, (new_h, new_w))

        # ======landmarks the axis are reversed =========
        landmarks = landmarks * [new_w / w, new_h / h]
        return {'image': image, 'landmarks': landmarks}


class Crop(object):
    '''
    crop the image to the desired size
    Args:
        output_size(int, tuple):  the desired size of the output
    '''
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]

        new_h, new_w = self.output_size
        start_h = np.random.randint(0, h - new_h)
        start_w = np.random.randint(0, w - new_w)
        image = image[start_h: start_h + new_h,
                      start_w: start_w + new_w]
        # landmarks
        landmarks = landmarks - [start_w, start_h]
        return {'image': image,
                'landmarks': landmarks}


class ToTorch(object):
    '''
    transform the numpy array into a torch tensor
    so we need to swap the color's position to the first dimension instead of the last dimension
    '''
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        new_img_transition = image.transpose((2, 0, 1))
        image = torch.from_numpy(new_img_transition)
        landmarks = torch.from_numpy(landmarks)
        return {'image': image,
                'landmarks': landmarks}


class FaceLandMarks(Dataset):
    def __init__(self, csv_file, root_dir, transformer=None, grayscale=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transformer = transformer
        self.grayscale = grayscale

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        image_name = self.landmarks_frame.iloc[idx, 0]
        image = io.imread(os.path.join(self.root_dir, image_name))
        img_landmarks = self.landmarks_frame.iloc[idx, 1:].values
        landmarks = img_landmarks.astype('float').reshape(-1, 2)

        sample = {'image': image,
                  'landmarks': landmarks}

        if self.transformer:
            sample = self.transformer(sample)

        return sample
    @staticmethod
    def img_landmarks(image, landmarks):
        plt.imshow(image)
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, c='r', marker='.')
        plt.pause(0.01)


class BlackWhite(Grayscale):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = image[:, :, 2]
        sample = {'image': image,
                'landmarks': landmarks}
        return sample


rescale = Rescale(220)
crop = Crop(140)
to_tensor = ToTorch()
composed = Compose([Rescale(256), Crop(214)])

transformed_data = FaceLandMarks('faces/face_landmarks.csv', 'faces/', transformer=Compose([Rescale(250),
                                                                                            transforms.CenterCrop(224),
                                                                                            ToTorch()]))

for i in range(len(transformed_data)):
    sample = transformed_data[i]
    print(i, sample['image'].shape, sample['landmarks'].shape, sep='\t')
    if i == 3:
        break
# sample = face[65]
# for i, t in enumerate([rescale, crop, composed]):
#
#     sample_ = t(sample)
#     face.img_landmarks(**sample_)
    # ax = plt.subplot(1, 3, i + 1)
    # plt.tight_layout()
    # ax.set_title(type(t).__name__)

# plt.show()

dataloader = DataLoader(transformed_data, batch_size=4, shuffle=False, num_workers=4)

# def show_landmarks_batch(sample_batched):
#     """Show image with landmarks for a batch of samples."""
#     images_batch, landmarks_batch = \
#             sample_batched['image'], sample_batched['landmarks']
#
#     batch_size = len(images_batch)
#     im_size = images_batch.size(2)


# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['landmarks'].size())

print(next(iter(dataloader)))








