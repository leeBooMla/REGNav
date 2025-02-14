import os
import os.path as osp
import tqdm
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
# from samplers import CategoriesSampler
from torch.utils.data import DataLoader
class mp3d_image_dataset(Dataset):
    # def __init__(self, set_name, args=None, return_path=False):
    def __init__(self, set_name, root_path):
        self.root = os.path.join(root_path, set_name+"_data_new")
        self.data = os.listdir(self.root)
        self.label_str = []
        self.label = []
        for image in self.data:
            label_list = image.split("_")[0:3]
            label = "_".join(label_list)
            self.label_str.append(label)
        
        self.annotated_label = json.load(open("/data4/wky/code/renet/datasets/images_semantic/label_list.json", "r"))
        self.num_class = len(self.annotated_label)
        for label in self.label_str:
            label_index = self.annotated_label.index(label)
            self.label.append(label_index)
        if set_name == 'train':

            image_size = 84

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:

            image_size = 84
            resize_size = 92

            self.transform = transforms.Compose([
                transforms.Resize([resize_size, resize_size]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                    np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index], self.label[index]
        image = self.transform(Image.open(os.path.join(self.root, image_path)).convert('RGB'))
        return image, label


if __name__ == "__main__":
    mp3d_dataset = mp3d_image_dataset("train", "/data4/wky/code/renet/datasets/images_semantic")
    train_sampler = CategoriesSampler(mp3d_dataset.label, len(mp3d_dataset.data) // 64, 10, 10)
    train_loader = DataLoader(dataset=mp3d_dataset, batch_sampler=train_sampler, num_workers=1, pin_memory=True)

    trainset_aux = mp3d_image_dataset('train',"/data4/wky/code/renet/datasets/images_semantic")
    train_loader_aux = DataLoader(dataset=trainset_aux, batch_size=64, shuffle=True, num_workers=1, pin_memory=True)
    train_loaders = {'train_loader': train_loader, 'train_loader_aux': train_loader_aux}
    tqdm_gen = tqdm.tqdm(train_loader)
    for i, ((data, train_labels), (data_aux, train_labels_aux)) in enumerate(zip(tqdm_gen, train_loader_aux), 1):
        data, train_labels = data.cuda(), train_labels.cuda()
        data_aux, train_labels_aux = data_aux.cuda(), train_labels_aux.cuda()

    # num_class = mp3d_dataset.num_class
    # num_samples = len(mp3d_dataset)
    # print(f"number of class is {num_class}")
    # print(f"num_samples is {num_samples}")