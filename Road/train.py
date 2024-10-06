# FILE: Road/train
# USER: mcfly
# IDE: PyCharm
# CREATE TIME: 2024/10/4 19:14
# DESCRIPTION:
from torch.utils.data import DataLoader

import torch
from tqdm import tqdm

from data import SegmentationDataset
from model import UNetModel

if __name__ == '__main__':
    index = 0
    num_epochs = 50
    train_on_gpu = True
    unet = UNetModel().cuda()
    # model_dict = unet.load_state_dict(torch.load('unet_road_model-100.pt'))

    image_dir = "C:\\Users\\mcfly\Desktop\\CrackForest-dataset-master\\image"
    mask_dir = "C:\\Users\\mcfly\\Desktop\\CrackForest-dataset-master\\groundTruthPngImg"
    dataloader = SegmentationDataset(image_dir, mask_dir)
    optimizer = torch.optim.SGD(unet.parameters(), lr=0.01, momentum=0.9)
    train_loader = DataLoader(dataloader, batch_size=1, shuffle=False)

    for epoch in range(num_epochs):
        train_loss = 0.0
        for i_batch, sample_batched in tqdm(enumerate(train_loader)):
            images_batch, target_labels = \
                sample_batched['image'], sample_batched['mask']
            print(target_labels.min())
            print(target_labels.max())

            if train_on_gpu:
                images_batch, target_labels = images_batch.cuda(), target_labels.cuda()
            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model
            m_label_out_ = unet(images_batch)
            # print(m_label_out_)
            # calculate the batch loss
            target_labels = target_labels.contiguous().view(-1)
            m_label_out_ = m_label_out_.transpose(1,3).transpose(1, 2).contiguous().view(-1, 2)
            target_labels = target_labels.long()
            loss = torch.nn.functional.cross_entropy(m_label_out_, target_labels)
            print(loss)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

            # update training loss
            train_loss += loss.item()
            if index % 100 == 0:
                print('step: {} \tcurrent Loss: {:.6f} '.format(index, loss.item()))
            index += 1
            # test(unet)
        # 计算平均损失
        train_loss = train_loss / dataloader.num_of_samples()
        # 显示训练集与验证集的损失函数
        print('Epoch: {} \tTraining Loss: {:.6f} '.format(epoch, train_loss))
        # test(unet)
    # save model
    unet.eval()
    torch.save(unet.state_dict(), 'unet_road_model.pt')
