from skimage.metrics import structural_similarity as ssim
import cv2
import torch
from openpyxl import Workbook
import pandas as pd
from Gabor.RDNet import Generator
from torchvision.utils import save_image
from tqdm import tqdm
from datasets import *
from option import opt


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr


def calculate_average_psnr(folder1, folder2):
    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))

    psnrlist = []
    for file1, file2 in zip(files1, files2):
        img1 = cv2.imread(os.path.join(folder1, file1))
        img2 = cv2.imread(os.path.join(folder2, file2))

        psnr_value = calculate_psnr(img1, img2)
        psnrlist.append(psnr_value)

    average_psnr = np.mean(psnrlist)
    std_psnr = np.std(psnrlist)

    return average_psnr, std_psnr


def calculate_ssim(img1, img2):
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    ssim_index, _ = ssim(gray1, gray2, full=True)

    return ssim_index


def calculate_average_ssim(folder1, folder2):
    images1 = [f for f in os.listdir(folder1) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images2 = [f for f in os.listdir(folder2) if f.endswith(('.png', '.jpg', '.jpeg'))]

    ssimlist = []

    for img1, img2 in zip(images1, images2):
        img_path1 = os.path.join(folder1, img1)
        img_path2 = os.path.join(folder2, img2)

        ssim_index = calculate_ssim(img_path1, img_path2)
        ssimlist.append(ssim_index)

    average_ssim = np.mean(ssimlist)
    std_ssim = np.std(ssimlist)

    return average_ssim, std_ssim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False


def sample_image():  # success
    with torch.no_grad():
        for i in tqdm(range(val_loader.size), desc="processing image...", unit="file"):
            original_images, target_images, tar, gt, name, g = val_loader.load_data()
            name = name.replace(".png", "")
            original_images = original_images.cuda()
            res = Generator(original_images)
            img_sample1 = res.data
            save_image(img_sample1, f"{save_path_generator}/{name}.png", nrow=5, normalize=True)


if __name__ == '__main__':
    save_img_pth = r"savepth.../"
    df = pd.DataFrame(columns=['epoch', 'aver_ssims', 'std_ssim', 'psnr', 'std_psnr'])
    Generator = Generator()
    xlsx_floder = save_img_pth + "data.xlsx"
    for i in range(91, 100):
        Generator.load_state_dict(
            torch.load(f"{save_img_pth}Generator_{i}.pth"))
        if cuda:
            Generator = Generator.cuda()

        val_image_root = opt.val_original_image_root
        val_target_image_root = opt.val_target_image_root
        val_gt_root = opt.val_gt_root

        val_loader = test_dataset(val_image_root, val_target_image_root, val_gt_root, opt.testsize)

        save_path_generator = save_img_pth + "fake_img"

        if not os.path.exists(save_path_generator):
            os.makedirs(save_path_generator)
        if not os.path.exists(xlsx_floder):
            wb = Workbook()
            wb.save(xlsx_floder)

        sample_image()
        print('done!')

        aver_ssims, std_ssim = calculate_average_ssim(save_path_generator, val_target_image_root)
        psnr, psnr_std = calculate_average_psnr(save_path_generator, val_target_image_root)
        print(aver_ssims)
        df = pd.concat([df, pd.DataFrame(
            [{'epoch': i, 'aver_ssims': aver_ssims, 'std_ssim': std_ssim, 'psnr': psnr, 'std_psnr': psnr_std}])],
                       ignore_index=True)
        print(df)
    df.to_excel(xlsx_floder, index=False)
    print("done")
