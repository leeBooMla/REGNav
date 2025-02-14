import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            random_classes = torch.randperm(len(self.m_ind))
            classes = random_classes[:self.n_cls]  # random sample num_class indices, e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indices of this class
                i = 1
                while l.shape[0] == 0:
                    c = random_classes[self.n_cls+i]
                    l = self.m_ind[c]
                    i+=1
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                if pos.shape[0] < self.n_per:
                    repeated_pos = pos.repeat(500)
                    pos = repeated_pos[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch

