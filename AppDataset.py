from torch.utils.data import Dataset
import pandas as pd
import os
from  skimage import io, transform
import matplotlib.pyplot as plt

class FaceLandMarks(Dataset):
    def __init__(self, csv_file, root_dir, transform= None):
        super().__init__()
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

        self.transform = transform

    def __len__(self):
        '''

        :return:  The size of the dataset (lenght)
        '''
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        # returns a rank 1 tensor
        image = io.imread(img_name)
        img_landmarks = self.landmarks_frame.iloc[idx, 1:].values
        landmarks = img_landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}
        # transform
        if self.transform:
            self.transform(sample)
        return sample


face = FaceLandMarks('faces/face_landmarks.csv', 'faces')
plt.figure()
for i in range(len(face)):
    sample = face[i]
    image = sample['image']
    print(image.shape)
    print(image.shape[:2])
    input()
    plt.imshow(sample['image'])
    plt.scatter(sample['landmarks'][:, 0], sample['landmarks'][:, 1], 10, 'r', '.')
    plt.pause(0.01)
    plt.show()





