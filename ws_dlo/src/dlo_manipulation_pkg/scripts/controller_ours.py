#!/usr/bin/env python

# our proposed controller

import numpy as np
from matplotlib import pyplot as plt
import time
import sys
import rospy
from RBF import JacobianPredictor


class Controller(object):
    # --------------------------------------------------------------------
    def __init__(self):
        self.rodLength = rospy.get_param("DLO/length")
        self.numFPs = rospy.get_param("DLO/num_FPs")
        self.env_dim = rospy.get_param("env/dimension")
        self.env = rospy.get_param("env/sim_or_real")
        self.bEnableEndRotation = rospy.get_param("controller/enable_end_rotation")
        self.b_left_arm = rospy.get_param("controller/enable_left_arm")
        self.b_right_arm = rospy.get_param("controller/enable_right_arm")
        self.targetFPsIdx = rospy.get_param("controller/object_fps_idx")
        self.project_dir = rospy.get_param("project_dir")
        self.controlRate = 10

        # the non-zero dimension of the input (also the Jacobian)
        self.validJacoDim = self.getValidControlInputDim(self.env_dim, self.bEnableEndRotation, self.b_left_arm, self.b_right_arm)

        self.jacobianPredictor = JacobianPredictor()
        self.jacobianPredictor.LoadModelWeights()

        self.k = 0
        self.case_idx = 0
        self.state_save = []


    # --------------------------------------------------------------------
    def generateControlInput(self, state):
        self.state_save.append(state)

        fpsPositions = state[0 : 3*self.numFPs]
        fpsVelocities = state[3*self.numFPs + 14 : 3*self.numFPs + 14 + 3*self.numFPs]
        controlEndVelocities = state[3*self.numFPs + 14 + 3*self.numFPs : 3*self.numFPs + 14 + 3*self.numFPs + 12]
        desiredPositions = state[3*self.numFPs + 14 + 3*self.numFPs + 12 : ]

        deltaX = np.array(fpsPositions - desiredPositions).reshape(self.numFPs, 3)
        delta_x = deltaX.reshape(-1, ).tolist()
        
        # calcualte the current Jacobian, and do the online learning
        Jacobian = self.jacobianPredictor.OnlineLearningAndPredictJ(fpsPositions, fpsVelocities, controlEndVelocities, delta_x).reshape(3 * self.numFPs, 12)


        localDeltaX = np.zeros((3 * len(self.targetFPsIdx), 1))
        for i, targetIdx in enumerate(self.targetFPsIdx):
            localDeltaX[3*i : 3*i +3, :] = deltaX[targetIdx, :].reshape(3, 1)

        u_12DoF = np.zeros((12, ))

        localJ = np.zeros((3 * len(self.targetFPsIdx), len(self.validJacoDim)))
        for i, targetIdx in enumerate(self.targetFPsIdx):
            localJ[3*i : 3*i+3, :] = Jacobian[3*targetIdx : 3*targetIdx+3, self.validJacoDim]
            

        # use a stratety to improve the stability in 3D tasks
        if self.env_dim == '3D':
            U, S, Vh = np.linalg.svd(localJ)
            S[S < 0.01 * S[0]] = 0.0
            Smat = np.zeros(localJ.shape)
            Smat[:len(S), :len(S)] = np.diag(S)
            localJpinv = Vh.T @ np.linalg.pinv(Smat) @ U.T
        else:
            localJpinv = np.linalg.pinv(localJ)

        Kp = 0.3
        u = - Kp * np.dot(localJpinv, localDeltaX)
        u_12DoF[self.validJacoDim] = u.reshape(len(self.validJacoDim), )

        self.k += 1

        return u_12DoF


    # --------------------------------------------------------------------
    def reset(self, state):
        self.state_save.append(state)
        # np.save(self.project_dir + "results/" + self.env + "/control/ours/" + self.env_dim + "/state_" + str(self.case_idx) + ".npy", self.state_save)
        # print("Save control result ",  self.case_idx)
        
        self.case_idx += 1
        self.state_save = []
        self.k = 0

        if (self.case_idx == 101):
            rospy.signal_shutdown("finish.")

        self.jacobianPredictor.LoadModelWeights()


    # --------------------------------------------------------------------
    def getValidControlInputDim(self, env_dim, bEnableEndRotation, b_left_arm, b_right_arm):
        if env_dim == '2D':
            if bEnableEndRotation:
                if b_left_arm and b_right_arm:
                    validJacoDim = [0, 1, 5, 6, 7, 11]
                elif b_left_arm:
                    validJacoDim = [0, 1, 5]
                elif b_right_arm:
                    validJacoDim = [6, 7, 11]
                else:
                    validJacoDim = np.empty()
            else:
                if b_left_arm and b_right_arm:
                    validJacoDim = [0, 1, 6, 7]
                elif b_left_arm:
                    validJacoDim = [0, 1]
                elif b_right_arm:
                    validJacoDim = [6, 7]
                else:
                    validJacoDim = np.empty()
        elif env_dim == '3D':
            if bEnableEndRotation:
                if b_left_arm and b_right_arm:
                    validJacoDim = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                elif b_left_arm:
                    validJacoDim = [0, 1, 2, 3, 4, 5]
                elif b_right_arm:
                    validJacoDim = [6, 7, 8, 9, 10, 11]
                else:
                    validJacoDim = np.empty()
            else:
                if b_left_arm and b_right_arm:
                    validJacoDim = [0, 1, 2, 6, 7, 8]
                elif b_left_arm:
                    validJacoDim = [0, 1, 2]
                elif b_right_arm:
                    validJacoDim = [6, 7, 8]
                else:
                    validJacoDim = np.empty()
        else:
            print("Error: the environment dimension must be '2D' or '3D'.")

        return validJacoDim

