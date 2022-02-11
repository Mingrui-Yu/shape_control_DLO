#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import os
import random
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import copy
import rospy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torch_rbf as rbf # reference: https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer



# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# state representation preprocess
# Note that the first and last feature points here are actually the left grasped end and the right grasped end
# The orientations of the robot end effectors are not included for simplification. However, the positions of features can imply the orientations of the ends to some extent.
def NNInputPreprocess(numFPs, fpsPositions):
    fpsPositions_aug = np.empty((fpsPositions.shape[0], 2 * fpsPositions.shape[1])).astype(np.float32)
    for i in range(numFPs): # 网络输入的fps positions统一为相对第一个fps的相对位置
        fpsPositions_aug[:, 3*i : 3*i+3] = normalize(fpsPositions[:, 3*i : 3*i+3] - fpsPositions[:, 0:3])
        fpsPositions_aug[:, 3*numFPs+3*i : 3*numFPs+3*i+3] = normalize(fpsPositions[:, 3*i : 3*i+3] - fpsPositions[:, 3*numFPs-3 : ])
        
    return fpsPositions_aug


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
class NNDataset(Dataset):
    def __init__(self, numFPs, state):
        self.dataNum = state.shape[0]
        self.fpsPositions = NNInputPreprocess(numFPs, state[:, 0 : 3*numFPs])
        self.fpsVelocities = state[:, 3*numFPs + 14 : 3*numFPs + 14 + 3*numFPs]
        self.controlEndVelocities = state[:, 3*numFPs + 14 + 3*numFPs : 3*numFPs + 14 + 3*numFPs + 12]
    
    def __getitem__(self, index):
        return self.fpsPositions[index], self.fpsVelocities[index], self.controlEndVelocities[index]

    def __len__(self):
        return self.dataNum



# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
class Net_J(nn.Module):
    def __init__(self, nFPs, bTrainMuBeta):
        super(Net_J, self).__init__()
        self.nFPs = nFPs
        self.numHidden = 256
        lw = [2 * self.nFPs * 3, self.numHidden, (self.nFPs * 3) * 12]
        basis_func = rbf.gaussian

        self.fc1 = rbf.RBF(lw[0], lw[1], basis_func, bTrainMuBeta=bTrainMuBeta)
        self.fc2 = nn.Linear(lw[1], lw[2], bias=False)


    def forward(self, x):
        phi = (self.fc1(x))
        output = (self.fc2(phi)) 
        output = torch.reshape(torch.reshape(output, (-1, self.nFPs, 12, 3)).transpose(2, 3), (-1, 3 * self.nFPs, 12)) # J: dimension: 30 * 12
        return output


    def CalcPhiAndJ(self, x):
        phi = (self.fc1(x))
        output = (self.fc2(phi))
        output = torch.reshape(torch.reshape(output, (-1, self.nFPs, 12, 3)).transpose(2, 3), (-1, 3 * self.nFPs, 12)) # J: dimension: 30 * 12
        return phi, output


    # use kmeans to calculate the initial value of mu and sigma in RBFN
    def GetMuAndBetaByKMeans(self, full_data):
        max_data_size = 600 * 60
        if(full_data.shape[0] > max_data_size):
            # randomly choose a subset of train data for kmeans
            index = np.random.choice(np.arange(full_data.shape[0]), size=max_data_size, replace=False)
            data = full_data[index, :]
        else:
            data = full_data

        # kmeans 聚类
        print("start kmeans ... ")
        kmeans = KMeans(n_clusters=self.numHidden, n_init=2, max_iter=100).fit(data)
        print("finish kmeans ... ")
        # 计算 \sigma
        nSamples = np.zeros((self.numHidden, ))
        variance = np.zeros((self.numHidden, ))
        for i, label in enumerate(kmeans.labels_):
            variance[label] += np.linalg.norm(data[i, :] - kmeans.cluster_centers_[label, :])**2
            nSamples[label] += 1
        variance = variance / nSamples
        sigma = np.sqrt(variance).astype('float32') * np.sqrt(2) * 10
        invSigma = 1.0 / (sigma + 1e-3)

        # 赋值给 \mu 和 \sigma
        self.fc1.centres.data = torch.tensor(kmeans.cluster_centers_).cuda()
        self.fc1.sigmas.data = torch.tensor(invSigma).cuda()



# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
class JacobianPredictor(object):

    numFPs = rospy.get_param("DLO/num_FPs")
    projectDir = rospy.get_param("project_dir")
    lr_approx_e = rospy.get_param("controller/online_learning/lr_approx_e")
    lr_task_e = rospy.get_param("controller/online_learning/lr_task_e")
    env = rospy.get_param("env/sim_or_real")
    env_dim = rospy.get_param("env/dimension")
    
    # ------------------------------------------------------
    def __init__(self):

        device = torch.device("cuda:0")
        
        self.bTrainMuBeta = True
        self.model_J = Net_J(self.numFPs, self.bTrainMuBeta).to(device)
        learningRate = 0.01
        self.optimizer = torch.optim.Adam([{'params': self.model_J.parameters()}], learningRate)
        torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99, last_epoch=-1, verbose=False)
        self.criterion = torch.nn.SmoothL1Loss(reduction='mean', beta=0.0005)
        self.online_optimizer = torch.optim.SGD([{'params': self.model_J.fc2.parameters()}], lr=0.1) # delta_t = 0.1s
        self.mse_criterion = torch.nn.MSELoss(reduction='sum')

        # 设置各类路径
        if rospy.get_param("learning/is_test"):
            self.nnWeightDir = self.projectDir + 'ws_dlo/src/dlo_manipulation_pkg/models_test/rbfWeights/' + self.env_dim + '/'
        else:
            self.nnWeightDir = self.projectDir + 'ws_dlo/src/dlo_manipulation_pkg/models/rbfWeights/' + self.env_dim + '/'
        self.resultsDir = self.projectDir + 'results/' + self.env + '/'
        self.dataDir = self.projectDir +'data/'

        
    # ------------------------------------------------------
    def LoadDataForTraining(self, train_dataset=None):
        # trainset
        if train_dataset is None:
            train_dataset = np.load(self.dataDir + 'train_data/' + self.env_dim + '/state_part.npy').astype(np.float32)
        self.trainDataset = NNDataset(self.numFPs, train_dataset.astype(np.float32))
        self.trainDataLoader = DataLoader(self.trainDataset, batch_size=512, shuffle=True, num_workers=4)

    # ------------------------------------------------------
    def LoadDataForTest(self, test_dataset=None):
        # testset
        if test_dataset is None:
            test_dataset = np.load(self.dataDir + 'train_data/' + self.env_dim + '/state.npy').astype(np.float32)[600*0 : 600*2, :]
        self.testDataset = NNDataset(self.numFPs, test_dataset.astype(np.float32))
        self.testDataLoader = DataLoader(self.testDataset, batch_size=test_dataset.shape[0], shuffle=False, num_workers=4)


    # ------------------------------------------------------
    def LoadModelWeights(self):
        if os.path.exists(self.nnWeightDir + "model_J.pth"):
            self.model_J.load_state_dict(torch.load(self.nnWeightDir + "model_J.pth"))
            print('Load previous model.')
        else:
            print('Warning: no model exists !')

    
    # ------------------------------------------------------
    def SaveModelWeights(self):
        torch.save(self.model_J.state_dict(), self.nnWeightDir + "model_J.pth")
        print("Save models to ", self.nnWeightDir)

    
    # ------------------------------------------------------
    def Train(self, loadPreModel=False):

        if loadPreModel == False:
            self.model_J.GetMuAndBetaByKMeans(self.trainDataset.fpsPositions)
        else:
            self.LoadModelWeights()

        # training
        for epoch in range(1, 51):
            accumLoss = 0.0
            numBatch = 0
            for batch_idx, (fpsPositions, fpsVelocities, controlEndVelocities) in enumerate(self.trainDataLoader):
                fpsPositions = fpsPositions.cuda()
                controlEndVelocities = controlEndVelocities.cuda()
                fpsVelocities = fpsVelocities.cuda()
                
                bmm_rV = torch.reshape(controlEndVelocities, (-1, 1, 12))
                bmm_yV = torch.reshape(fpsVelocities, (-1, 1, self.numFPs * 3))

                J_pred_T = self.model_J(fpsPositions).transpose(1, 2)
                loss = self.criterion(bmm_yV, torch.bmm(bmm_rV, J_pred_T))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                accumLoss += loss.item()
                numBatch += 1

            print("epoch: ", epoch, " , Loss/train: ", accumLoss/numBatch)

        # save model
        self.SaveModelWeights()


    # ------------------------------------------------------
    def TestAndSaveResults(self):

        self.LoadModelWeights()

        accumLoss = 0.0
        numBatch = 0
        for batch_idx, (fpsPositions, fpsVelocities, controlEndVelocities) in enumerate(self.testDataLoader):
            fpsPositions = fpsPositions.cuda()
            controlEndVelocities = controlEndVelocities.cuda()
            fpsVelocities = fpsVelocities.cuda()
            
            test_rV = torch.reshape(controlEndVelocities, (-1, 1, 12))
            test_yV = torch.reshape(fpsVelocities, (-1, 1, self.numFPs * 3))

            J_pred_T = self.model_J(fpsPositions).transpose(1, 2)
            # J_pinv_pred_T = torch.pinverse(J_pred_T)

            y_pred = torch.bmm(test_rV, J_pred_T)
            # r_pred = torch.bmm(test_yV, J_pinv_pred_T)

            testLoss = self.mse_criterion(y_pred, test_yV)

            accumLoss += testLoss.item()
            numBatch += 1

        print("Loss/test: ", accumLoss/numBatch)

        # test result 数据保存
        np.save(self.resultsDir + "nn_test/rbf/" + self.env_dim + "/dot_x_truth.npy", test_yV.cpu().detach().numpy())
        np.save(self.resultsDir + "nn_test/rbf/" + self.env_dim + "/dot_x_pred.npy", y_pred.cpu().detach().numpy())
        # np.save(self.resultsDir + "nn_test/rbf/" + self.env_dim + "/dot_r_truth.npy", test_rV.cpu().detach().numpy())
        # np.save(self.resultsDir + "nn_test/rbf/" + self.env_dim + "/dot_r_pred.npy", r_pred.cpu().detach().numpy())


    # ------------------------------------------------------
    def Predict_J(self, fpsPositions):
        fpsPositions = np.array(fpsPositions).astype(np.float32).reshape(1, -1)
        fpsPositions = NNInputPreprocess(self.numFPs, fpsPositions)

        fpsPositions_torch = torch.reshape(torch.tensor(fpsPositions), (1, 2 * self.numFPs * 3)).cuda()
        J = self.model_J(fpsPositions_torch)

        return J.cpu().detach().numpy() # J: 30 * 12

    
    # ------------------------------------------------------
    def OnlineLearningAndPredictJ(self, fpsPositions, fpsVelocities, controlEndVelocities, delta_x):
        fpsPositions = np.array(fpsPositions).astype(np.float32).reshape(1, -1) # one row matrix
        dot_x = np.array(fpsVelocities).astype(np.float32).reshape(-1, ) # vector
        dotR = np.array(controlEndVelocities).astype(np.float32).reshape(-1, ) # vector
        delta_x = np.array(delta_x).astype(np.float32).reshape(-1, )

        fpsPositions = NNInputPreprocess(self.numFPs, fpsPositions)
        fpsPositions_torch = torch.reshape(torch.tensor(fpsPositions), (1, 2 * self.numFPs * 3)).cuda()
        dot_r_torch = torch.reshape(torch.tensor(dotR), (1, 1, 12)).cuda()
        dot_x_torch = torch.reshape(torch.tensor(dot_x), (1, 1, 3 * self.numFPs)).cuda()
        delta_x_torch = torch.reshape(torch.tensor(delta_x), (1, 1, 3 * self.numFPs)).cuda()
        
        J = self.model_J(fpsPositions_torch)
        J_pred_T = J.transpose(1, 2)
        task_e = delta_x_torch
        approx_e = dot_x_torch - torch.bmm(dot_r_torch, J_pred_T)
        
        # Because of the imperfection of the simulator, sometimes the DLO will wiggle to the other side very fast.
        # We don't want to include these outlier data in training, so we just discard the online data with too fast speed.
        if np.linalg.norm(dot_x) < 0.3:
            # we use the SGD optimizer in PyTorch for online learning implementation to achieve faster computing speed. 
            # Note that the following computing is mathematically equivalent to the online updating law in the paper.
            lr_approx_e = np.sqrt(self.lr_approx_e / 2)
            lr_task_e = self.lr_task_e / 2 /  lr_approx_e

            loss = self.mse_criterion(lr_approx_e * approx_e  + lr_task_e * task_e,   torch.zeros(approx_e.shape).cuda())
            self.online_optimizer.zero_grad()
            loss.backward()
            self.online_optimizer.step()

        J = self.model_J(fpsPositions_torch)
        return J.cpu().detach().numpy()


    # # ------------------------------------------------------
    # # Here is an online learning implementation directly using the updating law in the paper. 
    # # It is mathematically equivalent to the above implementation.
    # # However, the accumulated numerical errors may cause the control results to be slightly different.
    # def OnlineLearningAndPredictJ2(self, fpsPositions, fpsVelocities, controlEndVelocities, delta_x):
    #     fpsPositions = np.array(fpsPositions).astype(np.float32).reshape(1, -1) # one row matrix
    #     dot_x = np.array(fpsVelocities).astype(np.float32).reshape(-1, ) # vector
    #     dotR = np.array(controlEndVelocities).astype(np.float32).reshape(-1, ) # vector
    #     delta_x = np.array(delta_x).astype(np.float32).reshape(-1, )

    #     fpsPositions = NNInputPreprocess(self.numFPs, fpsPositions)
    #     fpsPositions_torch = torch.reshape(torch.tensor(fpsPositions), (1, 2 * self.numFPs * 3)).cuda()
        
    #     phi, J = self.model_J.CalcPhiAndJ(fpsPositions_torch)
    #     phi = torch.reshape(phi, (self.model_J.numHidden, 1))
    #     J = J.cpu().detach().numpy()

    #     if np.linalg.norm(dot_x) < 0.3:
    #         W = self.model_J.state_dict()["fc2.weight"]
    #         dot_x_pred = np.dot(J, dotR.T)
    #         ew = self.lr_approx_e * (dot_x.reshape(-1, ) - dot_x_pred.reshape(-1, )) + self.lr_task_e * delta_x
    #         deltaTime = 0.1
    #         for objectIdx in range(self.numFPs):
    #             for i in range(12): # W_i 为 J_i 的权重
    #                 for j in range(3): # W_i 的第 j 行
    #                     # 原址操作，同时也改变了weight
    #                     W[objectIdx*12*3 + i * 3 + j, :] += deltaTime * torch.reshape((dotR[i] * phi * ew[3*objectIdx + j]), (self.model_J.numHidden, ))

    #     J = self.model_J(fpsPositions_torch)
    #     return J.cpu().detach().numpy()


    # ------------------------------------------------------
    def getW(self):
        W = self.model_J.state_dict()["fc2.weight"]
        return W.cpu().detach().numpy()


    # ------------------------------------------------------
    def PredNextFPsPositions(self, fps_positions, control_end_velocities, delta_t):
        current_fps_positions = np.array(fps_positions).astype(np.float32).reshape(-1, self.numFPs * 3)
        control_end_velocities = np.array(control_end_velocities).astype(np.float32).reshape(-1, 12)

        fps_positions_aug = NNInputPreprocess(self.numFPs, current_fps_positions)
        fps_positions_aug_torch = torch.reshape(torch.tensor(fps_positions_aug), (-1, 2 * self.numFPs * 3)).cuda()
        control_end_velocities_torch = torch.reshape(torch.tensor(control_end_velocities), (-1, 1, 12)).cuda()

        J_pred_T = self.model_J(fps_positions_aug_torch).transpose(1, 2)
        dot_x_pred = torch.bmm(control_end_velocities_torch, J_pred_T)
        dot_x_pred = dot_x_pred.cpu().detach().numpy().reshape(-1, self.numFPs * 3)

        next_fps_positions = (current_fps_positions + delta_t * dot_x_pred).reshape(-1, self.numFPs * 3)

        return next_fps_positions


    # ------------------------------------------------------
    def PredFPsVelocities(self, fps_positions, control_end_velocities):
        current_fps_positions = np.array(fps_positions).astype(np.float32).reshape(-1, self.numFPs * 3)
        control_end_velocities = np.array(control_end_velocities).astype(np.float32).reshape(-1, 12)
        fps_positions_aug = NNInputPreprocess(self.numFPs, current_fps_positions)
        fps_positions_aug_torch = torch.reshape(torch.tensor(fps_positions_aug), (-1, 2 * self.numFPs * 3)).cuda()
        control_end_velocities_torch = torch.reshape(torch.tensor(control_end_velocities), (-1, 1, 12)).cuda()

        J_pred_T = self.model_J(fps_positions_aug_torch).transpose(1, 2)
        dot_x_pred = torch.bmm(control_end_velocities_torch, J_pred_T)
        dot_x_pred = dot_x_pred.cpu().detach().numpy().reshape(-1, self.numFPs * 3)

        return dot_x_pred
    


# --------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    trainer = JacobianPredictor()

    trainer.LoadDataForTraining()
    trainer.Train(loadPreModel=False)

    print("Finished training. Start testing.")

    trainer.LoadDataForTest()
    trainer.TestAndSaveResults()





        


# ---------------------------------------------------- back up ----------------------------------------------------------