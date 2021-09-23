import torchvision.transforms as transforms
from PIL import Image
import os.path
import numpy as np


class zsl_NShot:

    def __init__(self, trainData, testData, batchsz, n_way, k_shot, k_query):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """
        self.x_train, self.x_test = trainData, testData
        # self.normalization()
        self.batchsz = batchsz  # number of task
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        # print("DB: train", self.x_train[0].shape, "test", self.x_test[0].shape)

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"])}  # current epoch data cached
        # "test": self.load_data_cache(self.datasets["test"])}

    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        # print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

    # print("after norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)

    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [N,2048]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        unq_labels = np.unique(data_pack[1])
        labels = np.array(data_pack[1])

        Data = data_pack[0]

        x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
        for i in range(self.batchsz):  # one batch means one set

            x_spt, y_spt, x_qry, y_qry = [], [], [], []
            #             print('unq labels shape: '+str(unq_labels.shape)+'  ' +str(i))
            selected_cls = np.random.choice(unq_labels, 2*self.n_way, False)
            #             print(unq_labels)
            #                 print('selected: '+str(selected_cls))

            for j, cur_class in enumerate(selected_cls[:self.n_way]):
                indx = np.where(labels == cur_class)[0]
                selected_img = np.random.choice(indx, self.k_shot, False)
                x_spt.append(Data[selected_img])
                y_spt.append(labels[selected_img])
                # print('shape: '+str(np.array(x_spt).shape))

            for j, cur_class in enumerate(selected_cls[self.n_way:]):
                indx = np.where(labels == cur_class)[0]
                selected_img = np.random.choice(indx, self.k_query, False)
                x_qry.append(Data[selected_img])
                y_qry.append(labels[selected_img])
                # print('shape: '+str(np.array(x_spt).shape))

            # shuffle inside a batch
            perm = np.random.permutation(self.n_way * self.k_shot)
            x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 2048)[perm]
            y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
            perm = np.random.permutation(self.n_way * self.k_query)
            x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 2048)[perm]
            y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

            # append [N,2048] => [b, N,2048]
            x_spts.append(x_spt)
            y_spts.append(y_spt)
            x_qrys.append(x_qry)
            y_qrys.append(y_qry)

        # [b, N,2048]
        x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, 2048)
        y_spts = np.array(y_spts).astype(np.int).reshape(self.batchsz, setsz)
        # [b, N,2048]
        x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, 2048)
        y_qrys = np.array(y_qrys).astype(np.int).reshape(self.batchsz, querysz)

        data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch