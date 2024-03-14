# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def postprocessing(data_name):

    def thumos_postprocessing(ground_truth, prediction, smooth=False, switch=False):
        """
        We follow (Shou et al., 2017) and adopt their perframe postprocessing method on THUMOS'14 datset.
        Source: https://bitbucket.org/columbiadvmm/cdc/src/master/THUMOS14/eval/PreFrameLabeling/compute_framelevel_mAP.m
        """

        # Simple temporal smoothing via NMS of 5-frames window
        if smooth:
            prob = np.copy(prediction)
            prob1 = prob.reshape(1, prob.shape[0], prob.shape[1])
            prob2 = np.append(prob[0, :].reshape(1, -1), prob[0: -1, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
            prob3 = np.append(prob[1:, :], prob[-1, :].reshape(1, -1), axis=0).reshape(1, prob.shape[0], prob.shape[1])
            prob4 = np.append(prob[0: 2, :], prob[0: -2, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
            prob5 = np.append(prob[2:, :], prob[-2:, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
            probsmooth = np.squeeze(np.max(np.concatenate((prob1, prob2, prob3, prob4, prob5), axis=0), axis=0))
            prediction = np.copy(probsmooth)

        # Assign cliff diving (5) as diving (8)
        if switch:
            switch_index = np.where(prediction[:, 5] > prediction[:, 8])[0]
            prediction[switch_index, 8] = prediction[switch_index, 5]

        # Remove ambiguous (21)
        valid_index = np.where(ground_truth[:, 21] != 1)[0]

        return ground_truth[valid_index], prediction[valid_index]

    return {'THUMOS': thumos_postprocessing}.get(data_name, None)
