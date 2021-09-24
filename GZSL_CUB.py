import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import scipy.io as sio
import math
import argparse
import random
import os
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from gzslcub2 import zsl_NShot
from sklearn import svm
torch.cuda.set_device(2)
from    torch import optim

# step 1: init dataset
print("init dataset")

dataroot = '/data'
dataset = 'CUB1_data'
image_embedding = 'res101'
class_embedding = 'att_splits'

matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
feature = matcontent['features'].T
label = matcontent['labels'].astype(int).squeeze() - 1
matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + ".mat")
# numpy array index starts from 0, matlab starts from 1
trainval_loc = matcontent['trainval_loc'].squeeze() - 1
test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

attribute = matcontent['att'].T

x = feature[trainval_loc] # train_features
train_label = label[trainval_loc].astype(int)  # train_label
att = attribute[train_label] # train attributes

x_test = feature[test_unseen_loc]  # test_feature
test_label_unseen = label[test_unseen_loc].astype(int)

unseenlabel=list(set(test_label_unseen))# test_label
unseenlabel=np.array(unseenlabel)

x_test_seen = feature[test_seen_loc]  #test_seen_feature
test_label_seen = label[test_seen_loc].astype(int) # test_seen_label

seenlabel=list(set(test_label_seen))# test_label
seenlabel=np.array(seenlabel)

test_id = np.unique(test_label_unseen)   # test_id
att_pro = attribute[test_id]

import numpy as np
path=dataroot
file='/cub_attributes_reed.npy'
attribute=15*np.load(file) # it scale the attribute in the some higher range since such
#that its attribute is in the AWA dataset attribute range since our model is optimized for the AWA and
# we are using the same parameter for all dataset. it scale the attribute in range of 100


# train set
#train_features=torch.from_numpy(x)
train_features=x
print('train_features.shape: ' + str(train_features.shape))

# a=np.concatenate((x_test,x_test_seen),axis=0)
# b=np.vstack((x_test,x_test_seen))

train_label=np.array(torch.from_numpy(train_label).unsqueeze(1))
#train_label=torch.from_numpy(train_label).unsqueeze(1)
print('train_label.shape:  '+str(train_label.shape))

# attributes
all_attributes=np.array(attribute)
print('all_attributes.shape:  '+str(all_attributes.shape))

attributes = torch.from_numpy(attribute)
# test set

# test_features=torch.from_numpy(x_test)
test_features=x_test
print('test_features.shape:  '+ str(test_features.shape))


#test_label=np.array(torch.from_numpy(test_label).unsqueeze(1))
#print('test_label.shape:  ' +str(test_label.shape))

test_label_seen=np.array(torch.from_numpy(test_label_seen).unsqueeze(1))
print('test_label_seen.shape:  ' +str(test_label_seen.shape))

testclasses_id = np.array(test_id)
print('testclasses_id.shape:  ' +str(testclasses_id.shape))

test_attributes = torch.from_numpy(att_pro).float()
print('test_attributes.shape:  ' +str(test_attributes.shape))


#test_feature_gzsl=np.vstack((test_features,x_test_seen))
#test_label_gzsl=np.vstack((test_label_unseen,test_label_seen))

test_seen_features = torch.from_numpy(x_test_seen)
print('test_seen_features.shape:  ' +str(test_seen_features.shape))

test_seen_label = torch.from_numpy(test_label_seen)

train_data = [train_features,train_label]
test_data = [test_features,test_label_unseen]

unq_train_labels=np.unique(train_label)
unq_test_labels=np.unique(test_label_unseen)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean=0, std=0.002)
        m.bias.data.fill_(0.002)


class RelationNetwork(nn.Module):
    """Graph Construction Module"""

    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1))

        self.fc3 = nn.Linear(2 * 2, 8)
        self.fc4 = nn.Linear(8, 1)

        self.m0 = nn.MaxPool2d(2)  # max-pool without padding
        self.m1 = nn.MaxPool2d(2, padding=1)  # max-pool with padding

    def forward(self, x, rn):
        x = x.view(160, 32, 4, 4)

        out = self.layer1(x)
        out = self.layer2(out)
        # flatten
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)  # no relu

        out = out.view(out.size(0), -1)  # bs*1

        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.modelG = nn.Sequential(
            nn.Linear(args.attri_dim + args.noise_dim, 2048),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048, 0.8),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048, 0.8),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, args.input_shape)
        )

    def forward(self, noise, attri, weightsM=None):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((attri, noise), -1)
        if weightsM is None:
            img = self.modelG(gen_input)
        else:
            i = 0
            weights = weightsM[0]
            for m in self.modelG.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data = weights[i]
                    m.bias.data = weights[i + 1]
                    i = i + 2
                if isinstance(m, nn.BatchNorm1d):
                    m.weight.data = weights[i]
                    m.bias.data = weights[i + 1]
                    i = i + 2
            img = self.modelG(gen_input)

        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.modelD = nn.Sequential(
            nn.Linear(args.input_shape + args.attri_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    #         self.modelD.apply(init_weights)

    def forward(self, img, attri, weightsM=None):
        d_in = torch.cat((img.view(img.size(0), -1), attri), -1)
        if weightsM is None:
            real_fake = self.modelD(d_in)
        else:
            i = 0
            weights = weightsM[1]
            for m in self.modelD.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data = weights[i]
                    m.bias.data = weights[i + 1]
                    i = i + 2
                if isinstance(m, nn.BatchNorm1d):
                    m.weight.data = weights[i]
                    m.bias.data = weights[i + 1]
                    i = i + 2
            real_fake = self.modelD(d_in)

        return real_fake


class mapping(nn.Module):
    def __init__(self):
        super(mapping, self).__init__()

        self.modelC = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            #
            # nn.Linear(512, 128)
        )

    #self.modelC.apply(init_weights)

    def forward(self, img_gen, weightsM=None):
        # Concatenate label embedding and image to produce input
        if weightsM is None:
            output = self.modelC(img_gen)
        else:
            i = 0
            weights = weightsM[2]
            for m in self.modelC.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data = weights[i]
                    m.bias.data = weights[i + 1]
                    i = i + 2
                if isinstance(m, nn.BatchNorm1d):
                    m.weight.data = weights[i]
                    m.bias.data = weights[i + 1]
                    i = i + 2
            output = self.modelC(img_gen)
        return output


class all_arguments():
    n_way = 10
    k_spt = 5
    k_qry = 3

    imgsz = 2048
    sigma_ts = 0.25
    sigma_tr = 0.5

    meta_lr = 1e-5
    meta_lrD = 1e-3
    update_lr = 1e-3
    update_step = 5

    input_shape = 2048
    num_class = 200
    attri_dim = 1024
    noise_dim = 512
    clssifier_weight = 0.05

    # Get options
    attributes = attributes
    cuda = True if torch.cuda.is_available() else False


args = all_arguments()
# Loss functions
adversarial_loss = torch.nn.MSELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
# generator = Generator()
# relationnetwork=RelationNetwork()
# discriminator = Discriminator()
# classifier = Classifier()
#
# if args.cuda:
#     generator.cuda()
#     discriminator.cuda()
#     classifier.cuda()
#     adversarial_loss.cuda()
#     auxiliary_loss.cuda()
#     relationnetwork.cuda()
#
# para = list(classifier.parameters())
# print(classifier)

#
# glen=len(list(generator.parameters()))
# gen_para=list(generator.parameters())+ list(classifier.parameters())
#
# disc_optim = optim.SGD(discriminator.parameters(), lr=args.meta_lrD,weight_decay=1e-6)
# gen_optim = optim.Adam(gen_para, lr=args.meta_lr,betas=(0.9, 0.99),weight_decay=1e-6)
#
# disc_schedular = StepLR(disc_optim,step_size=100,gamma=0.95)
# gen_schedular = StepLR(gen_optim,step_size=100,gamma=0.95)


class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self,args):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.cls_weight = args.clssifier_weight
        self.update_step = args.update_step
        cuda = args.cuda
        self.noise_dim = args.noise_dim
        # self.all_loss=all_loss()

        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

        #self.glen = glen
        #self.gen_para = gen_para
        #self.disc_optim = disc_optim
        #self.gen_optim = gen_optim

        #self.disc_schedular = disc_schedular
        #self.gen_schedular = gen_schedular

        self.generator = Generator()
        self.generator.cuda()
        self.relationnetwork = RelationNetwork()
        self.relationnetwork.cuda()
        self.discriminator = Discriminator()
        self.discriminator.cuda()
        self.mapping = mapping()
        self.mapping.cuda()

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def all_loss(self, img_feature, img_labels, qurey_feature, qurey_labels,fast_weight=None):
        batch_size = img_feature.shape[0]
        FloatTensor = self.FloatTensor
        LongTensor = self.LongTensor
        img_labels = img_labels.type(LongTensor)
        img_labels1 = qurey_labels.type(LongTensor)

        valid = Variable(FloatTensor(batch_size+30, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size+30, 1).fill_(0.0), requires_grad=False)

        labels = torch.cat((img_labels, img_labels1))

        # Configure input
        feature = torch.cat((img_feature, qurey_feature))
        real_imgs = Variable(feature.type(FloatTensor))
        gen_attri = Variable(attributes[labels].type(FloatTensor))

        z = Variable(FloatTensor(np.random.normal(0, args.sigma_tr, (batch_size + 30, self.noise_dim))))

        # Generate a batch of images
        gen_imgs = self.generator(z, gen_attri, fast_weight)
        validity = self.discriminator(gen_imgs, gen_attri, fast_weight)

        g_loss = adversarial_loss(validity, valid)
        # print('g_loss: '+str(g_loss))

        validity_real = self.discriminator(real_imgs, gen_attri, fast_weight)
        d_real_loss = adversarial_loss(validity_real, valid)
        # print('d_real_loss: '+str(d_real_loss))

        # Loss for fake images
        validity_fake = self.discriminator(gen_imgs.detach(), gen_attri, fast_weight)
        d_fake_loss = adversarial_loss(validity_fake, fake)
        # print('d_fake_loss: '+str(d_fake_loss))

        
        reconstruction_criterion = nn.L1Loss(size_average=False)
        lossrecon = reconstruction_criterion(gen_imgs, real_imgs)
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2


        inp1 = torch.cat((feature,gen_imgs), 0)
        inp = self.mapping(inp1)

        emb_all = inp.view(-1,512)
        N, d    = emb_all.shape[0], emb_all.shape[1]
        sigma = self.relationnetwork(inp, 300)

        eps = np.finfo(float).eps
        emb_all = emb_all / (sigma + eps)  # N*d
        emb1 = torch.unsqueeze(emb_all, 1)  # N*1*d
        emb2 = torch.unsqueeze(emb_all, 0)  # 1*N*d
        W = ((emb1 - emb2) ** 2).mean(2)  # N*N*d -> N*N
        W = torch.exp(-W / 2)

        ## keep top-k values
        k=20

        if k > 0:
            topk, indices = torch.topk(W, k)
            mask = torch.zeros_like(W)
            mask = mask.scatter(1, indices, 1)
            mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  # union, kNN graph
            # mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
            W = W * mask

            ## normalize
        D = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
        S = D1 * W * D2

        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        ys = img_labels
        ys = torch.eye(200)[ys, :].cuda()
        labels = torch.eye(200)[labels, :].cuda()

        yu = (torch.ones(30, 200)/200).cuda()
        y1 = torch.cat((ys, yu), 0)
        y = torch.cat((y1, labels), 0)
        F = torch.matmul(torch.inverse(torch.eye(N).cuda() - 0.99 * S + eps), y)
        Fq = F[50:80, :]  # query predictions

        # Step4: Cross-Entropy Loss
        ce = nn.CrossEntropyLoss().cuda()
        ## both support and query loss
        gt = torch.argmax(torch.cat((labels,labels), 0), 1)
        loss = ce(F, gt)

        ## acc
        predq = torch.argmax(Fq, 1)

        correct = (predq == qurey_labels).sum()
        total = 30
        acc = 1.0 * correct.float() / float(total)


        return g_loss, d_loss, loss,acc


    def one_hot(self,x):
        return torch.eye(200)[x, :]

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [b, setsz, d]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, d]
        :param y_qry:   [b, querysz]
        :return:
        """



        g_loss, d_loss, loss,acc = self.all_loss(x_spt, y_spt, x_qry, y_qry)



        return  g_loss, d_loss, loss,acc


def test_gzsl(genunseen_input, test_labels_repeat, test_features, testLabels,train_features):
    """
    """
    with torch.no_grad():
        gen_imgs = maml.generator(genunseen_input[:, args.attri_dim:], genunseen_input[:, :args.attri_dim])
        gen_imgs = maml.mapping(gen_imgs)
        FloatTensor = torch.cuda.FloatTensor
        train_features1= (torch.from_numpy(train_features)).type(FloatTensor)
        train_features = maml.mapping(train_features1)
        trainD = torch.cat((train_features,gen_imgs))
        trainD = normalize(trainD.cpu(), axis=1)
        # labels=np.vstack((train_label,test_labels_repeat))
        test_labels_repeat=test_labels_repeat.reshape(len(test_labels_repeat),1)
        trainD = normalize(trainD, axis=1)
        #c = [train_label, test_labels_repeat]
        labels = np.vstack((train_label,test_labels_repeat))
        #labels=np.hstack((train_label, test_labels_repeat))

        pseudoTrainLabels = labels

        clf5 = svm.SVC(C=1, kernel='linear', class_weight='balanced')

        clf5.fit(trainD, pseudoTrainLabels)
        # print 'Predicting...'
        x_test_seen1 = (torch.from_numpy(x_test_seen)).type(FloatTensor)
        x_test_s  = maml.mapping(x_test_seen1 )
        x_test_s = x_test_s.cpu().numpy()
        seenpred = clf5.predict(x_test_s)

        test_features1 = (torch.from_numpy(test_features)).type(FloatTensor)
        x_test_u = maml.mapping(test_features1)
        x_test_u = x_test_u.cpu().numpy()
        unseenpred = clf5.predict(x_test_u)
        #hormic accuracy
        # unseen_accuracy = compute_per_class_acc_gzsl(test_label_seen, seenpred, target_classes)
        #
        # test_feature_gzsl = np.vstack((test_features, x_test_seen))
        # test_label_gzsl = np.vstack((test_label, test_label_seen))

        seenacc = compute_per_class_acc_gzsl(test_label_seen, seenpred, seenlabel)
        unseenacc = compute_per_class_acc_gzsl(test_label_unseen, unseenpred, unseenlabel)
        #unpred = clf5.predict(test_features)


        H = (2 * seenacc * unseenacc) / (seenacc + unseenacc)

        #unseen_accuracy = accuracy_score(testLabels, pred)

    return seenacc,unseenacc,H

def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
    test_label = torch.tensor(test_label)
    test_label=test_label.reshape(len(test_label))
    predicted_label = torch.tensor(predicted_label)
    target_classes = torch.tensor(target_classes)

    per_class_accuracies = Variable(torch.zeros(target_classes.size()[0]).float()).detach()

    predicted_label = predicted_label

    for i in range(target_classes.size()[0]):

        is_class = test_label==target_classes[i]


        per_class_accuracies[i] = torch.div((predicted_label[is_class]==test_label[is_class]).sum().float(),is_class.sum().float())

    return per_class_accuracies.mean()


def test_zsl(genunseen_input, test_labels_repeat, test_features, testLabels):
    """
    """
    with torch.no_grad():
        gen_imgs = maml.generator(genunseen_input[:, args.attri_dim:], genunseen_input[:, :args.attri_dim])
        gen_imgs  = maml.mapping(gen_imgs)
        pseudoTrainData = normalize(gen_imgs.cpu(), axis=1)
        #testData = normalize(test_features, axis=1)
        pseudoTrainLabels = test_labels_repeat
        testLabels = testLabels

        clf5 = svm.SVC(C=1, kernel='linear', class_weight='balanced')

        clf5.fit(pseudoTrainData, pseudoTrainLabels)
        # print 'Predicting...'
        testf=torch.from_numpy(test_features).to(torch.float32).cuda()
        testData = maml.mapping(testf)
        testData = testData.cpu().numpy()
        pred = clf5.predict(testData)
        unseen_accuracy = accuracy_score(testLabels, pred)

    return unseen_accuracy

def gradient_penalty(D, xr, xf):
    """
    :param D:
    :param xr:[b,2]
    :param xf:[b,2]
    :return:
    """

    # only constrait for Discriminator
    #    xf = xf.detach()
    #    xr = xr.detach()

    # [b, 1] => [b, 2]
    t = torch.rand(batchsz, 1).cuda()
    t = t.expand_as(xr)


    mid = t * xr + ((1 - t) * xf)

    mid.requires_grad_()
    pred = D(mid)
    grads = autograd.grad(outputs=pred, inputs=mid,
                          grad_outputs=torch.ones_like(pred),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    gp = torch.pow((grads.norm(2, dim=1) - 1), 2).mean()

    return gp

def test():
    testunq_labels = np.unique(test_label_unseen)
    lab_attribute = np.reshape(attribute[testunq_labels[0]], [1, args.attri_dim])
    test_attri_repeat = []
    test_labels_repeat = []
    for i in testunq_labels:
        lab_attribute = np.reshape(attribute[i], [1, args.attri_dim])
        test_attri_repeat.append(np.repeat(lab_attribute, 50, axis=0))
        test_labels_repeat.append(np.repeat(i, 50, axis=0))
    test_attri_repeat = np.concatenate(test_attri_repeat, 0)
    test_labels_repeat = np.concatenate(test_labels_repeat, 0)
    z = np.random.normal(0, args.sigma_ts, (2500, args.noise_dim))
    genunseen_input = np.concatenate((test_attri_repeat, z), 1)

    FloatTensor = torch.cuda.FloatTensor
    genunseen_input = (torch.from_numpy(genunseen_input)).type(FloatTensor)

    seenacc,unseenacc,H= test_gzsl(genunseen_input, test_labels_repeat, test_features, test_label_unseen,train_features)
    return seenacc,unseenacc,H




db = zsl_NShot(train_data, test_data, batchsz=10, n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry)
x_spt, y_spt, x_qry, y_qry = db.next('train')
x_spt = torch.from_numpy(x_spt).cuda()
x_qry = torch.from_numpy(x_qry).cuda()
y_spt = torch.from_numpy(y_spt).cuda()
y_qry = torch.from_numpy(y_qry).cuda()
print('train :  ' + str(x_spt.shape))
print('train_labels :  ' + str(y_spt.shape))
print('test :  ' + str(x_qry.shape))
print('test_labels :  ' + str(y_qry.shape))

best = 0
maml = Meta(args).cuda()
# optimizer
# model_optim = torch.optim.Adam(maml.parameters(), lr=0.001)
# model_scheduler = StepLR(model_optim, step_size=10000, gamma=0.5)


gen_para=list(maml.generator.parameters())+ list(maml.relationnetwork.parameters())+ list(maml.mapping.parameters())
maml.discriminator.parameters =maml.discriminator.parameters()

disc_optim = optim.Adam(maml.discriminator.parameters, lr=1e-3,betas=(0.9, 0.99),weight_decay=1e-6)
gen_optim = optim.Adam(gen_para, lr=1e-5,betas=(0.9, 0.99),weight_decay=1e-6)




for itr in range(12000):

    x_spt, y_spt, x_qry, y_qry = db.next('train')
    x_spt = torch.from_numpy(x_spt).cuda()
    x_qry = torch.from_numpy(x_qry).cuda()
    y_spt = torch.from_numpy(y_spt).cuda()
    y_qry = torch.from_numpy(y_qry).cuda()

    maml.train()

    for i in range(10):

        maml.train()

        for _ in range(5):

            for _ in range(5):

                g_loss, d_loss, loss, accs = maml(x_spt[i], y_spt[i], x_qry[i], y_qry[i])

                disc_optim.zero_grad()

                d_loss.backward()

                disc_optim.step()

            #gp = gradient_penalty(maml.discriminator, xr, xf.detach())

            for p in maml.discriminator.parameters:
                p.requires_grad = False

            g_loss, d_loss, loss, accs = maml(x_spt[i], y_spt[i], x_qry[i], y_qry[i])

            loss1 = g_loss+10*loss
            #print(loss)
            #print(accs)
            gen_optim.zero_grad()
            loss1.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 4.0)
            gen_optim.step()



    if itr % 100 == 0:
        maml.eval()
        seenacc,unseenacc,H= test()
        # if unseen_accuracy >= best:
        #     best = unseen_accuracy
        print(iter,seenacc,unseenacc,H)
