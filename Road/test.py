# FILE: Road/test
# USER: mcfly
# IDE: PyCharm
# CREATE TIME: 2024/10/4 19:16
# DESCRIPTION:

import os
import torch
import cv2 as cv
import numpy as np

def test(unet):
    model_dict=unet.load_state_dict(torch.load('unet_road_model.pt'))
    root_dir = 'CrackForest-dataset-master/CrackForest-dataset-master/test/'
    fileNames = os.listdir(root_dir)
    for f in fileNames:
        image = cv.imread(os.path.join(root_dir, f), cv.IMREAD_GRAYSCALE)
        h, w = image.shape
        img = np.float32(image) /255.0
        img = np.expand_dims(img, 0)
        x_input = torch.from_numpy(img).view( 1, 1, h, w)
        probs = unet(x_input.cuda())
        m_label_out_ = probs.transpose(1, 3).transpose(1, 2).contiguous().view(-1, 2)
        grad, output = m_label_out_.data.max(dim=1)
        output[output > 0] = 255
        predic_ = output.view(h, w).cpu().detach().numpy()

        # print(predic_)
        # print(predic_.max())
        # print(predic_.min())

        # print(predic_)
        # print(predic_.shape)
        # cv.imshow("input", image)
        result = cv.resize(np.uint8(predic_), (w, h))

        cv.imshow("unet-segmentation-demo", result)
        cv.waitKey(0)
    cv.destroyAllWindows()
