from datetime import datetime
from Gabor.RDNet import Generator, Discriminator
import random
from Vgg_loss import *
from datasets import get_loader
from option import opt
import logging
from src import adjust_lr, structure_loss
import os
import piq
import numpy as np

save_model_path = r"savepth.../"
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

file_handler = logging.FileHandler(save_model_path + 'log_dg_net.log', mode='a', encoding='utf-8')
formatter = logging.Formatter('[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', datefmt='%Y-%m-%d %I:%M:%S %p')
file_handler.setFormatter(formatter)
logging.getLogger('').addHandler(file_handler)
logging.getLogger('').setLevel(logging.INFO)
logging.info("GT-Train")
logging.info("Config")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

Generator = Generator()
Generator.freeze_rrcnn_blocks()
for name, param in Generator.named_parameters():
    if not param.requires_grad:
        print(f"{name} freeze")
Discriminator = Discriminator()
perceptual_loss = Perceptual_loss134()
criterion_GAN = torch.nn.L1Loss()
criterion_L1 = torch.nn.L1Loss()
criterion_L2 = torch.nn.MSELoss()

if cuda:
    Generator = Generator.cuda()
    Discriminator = Discriminator.cuda()
    criterion_GAN.cuda()
    criterion_L1.cuda()
    criterion_L2.cuda()

generator_losses = []
discriminator_losses = []

optimizer_G = torch.optim.Adam(Generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0002)
optimizer_D = torch.optim.Adam(Discriminator.parameters(), lr=0.00002, betas=(0.5, 0.999), weight_decay=0.0002)
logging.info('Generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0002')
logging.info('Discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0002')

# set path
image_root = opt.original_image_root
target_image_root = opt.target_image_root
gt_root = opt.gt_root

dataloader = get_loader(image_root, target_image_root, gt_root, batchsize=opt.batchsize,
                        trainsize=opt.trainsize)
total_step = len(dataloader)
# logger = Logger(opt.n_epochs, len(dataloader))
patch = (1, 256 // 2 ** 4, 256 // 2 ** 4)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

best_mae = 1
best_epoch = 0

best_ssim = 1
best_ssim_epoch = 0


def limit_zero2one(data_):
    data = data_.clone()
    data[data >= 1] = 1
    data[data <= -1] = -1
    data = (data + 1) / 2
    return data


def train(dataloader, Generator, Discriminator, epoch, save_model_path, optimizer_G, optimizer_D):
    global best_train_epoch
    if epoch == 50:
        Generator.unfreeze_rrcnn_blocks()
        print(" RRCNN_block  Recurrent_block")
    Generator.train()
    G_loss = 0
    D_loss = 0
    for i, (original_images, target_images, gts, filename) in enumerate(dataloader, start=1):
        optimizer_G.zero_grad()
        # --------------------------------------------------------
        # Model inputs
        # --------------------------------------------------------
        target_images = target_images.cuda()
        original_images = original_images.cuda()
        gts = gts.cuda()
        # Adversarial ground truths
        valid = torch.ones((target_images.size(0), *patch), dtype=torch.float32, device=target_images.device,
                           requires_grad=False)
        fake = torch.zeros((target_images.size(0), *patch), dtype=torch.float32, device=target_images.device,
                           requires_grad=False)

        # --------------------------------------------------------
        #  Train Generators
        # --------------------------------------------------------

        pre_gt, mid_fake_target = Generator(original_images)
        # gan loss
        pred_fake = Discriminator(original_images, mid_fake_target)
        loss_GAN = criterion_GAN(pred_fake, valid)

        # L1 loss
        loss_mae = criterion_L1(mid_fake_target, target_images)
        loss_mse = criterion_L2(mid_fake_target, target_images)

        # struct loss
        loss_struct = structure_loss(pre_gt, gts)

        g_loss = loss_GAN + (loss_mse + loss_mae) * 10 + loss_struct
        g_loss.backward()

        optimizer_G.step()
        G_loss += g_loss.item()

        # --------------------------------------------------------
        #  Train Discriminator
        # --------------------------------------------------------
        optimizer_D.zero_grad()
        # Real loss
        pred_fake = Discriminator(original_images, target_images)
        loss_D_fake = criterion_GAN(pred_fake, fake)
        # Fake loss
        pred_real = Discriminator(original_images, mid_fake_target.detach())
        loss_D_real = criterion_GAN(pred_real, valid)

        # Total discriminator loss
        disc_loss = (loss_D_fake + loss_D_real) * 0.5
        disc_loss.backward()
        optimizer_D.step()
        D_loss += disc_loss.item()

        # --------------------------------------------------------
        #  Logging Progress
        # --------------------------------------------------------
      

        logging.info(
            '#TRAIN#{}:Epoch [{:03d}/{:03d}], Step [{:04d}], gan loss: {:.4f}, loss_mae: {:.4f}, loss_mse: {:.4f}, g_loss:{:.4f}'.
            format(datetime.now(), epoch, opt.n_epochs, i, loss_GAN, loss_mae, loss_mse, g_loss,
                   ))

        
    logging.info("\n=============================================")
    logging.info('Epoch %d 生成器loss: %.10f' % (epoch, G_loss / (i + 1)))
    logging.info('Epoch %d 鉴别器loss: %.10f' % (epoch, D_loss / (i + 1)))
    logging.info("=============================================\n")

    # --------------------------------------------------------
    #  Save model
    # --------------------------------------------------------

    if epoch > 80:
        torch.save(Generator.state_dict(),
                   f"{save_model_path}/Generator_%d.pth" % (epoch))

def test(test_loader, Generator, epoch, save_model_path):
    global best_ssim, best_ssim_epoch
    Generator.eval()
    with torch.no_grad():
        ssim_sum = 0
        for i in range(test_loader.size):
            original_images, target_images, tar, gt, _, g = test_loader.load_data()

            original_images = original_images.cuda()
            target_images = target_images.cuda()
            predgt, res = Generator(original_images)

            limit_fake = limit_zero2one(res)
            limit_real = limit_zero2one(target_images)
            ssim = piq.ssim(limit_fake, limit_real, data_range=1.).cpu()
            ssim_sum += ssim

        average_ssim = ssim_sum / test_loader.size

        if epoch == 1:
            best_ssim = average_ssim
        else:
            if average_ssim > best_ssim:
                best_ssim = average_ssim
                best_ssim_epoch = epoch
                torch.save(Generator.state_dict(), save_model_path + '/Net_epoch_best_Tar_{}.pth'.format(epoch))
                

        logging.info(
            'Tar#TEST#:Epoch:{} average_ssim:{} best_ssim_epoch:{} best_ssim:{}'.format(epoch, average_ssim,
                                                                                        best_ssim_epoch, best_ssim))




if __name__ == '__main__':
    print("Start train...")
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    torch.cuda.manual_seed(17)
    for epoch in range(1, opt.n_epochs + 1):
        cur_lr = adjust_lr(optimizer_G, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(dataloader, Generator, Discriminator, epoch, save_model_path, optimizer_G, optimizer_D)
