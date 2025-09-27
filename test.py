import os
import os.path as osp
import numpy as np
import torch
import nibabel as nib
import SimpleITK as sitk
from matplotlib import pyplot as plt

from NeBla.model import NeBlaModel
from NeBla.render import run_network

from src.dataset import TIGREDataset as Dataset
from src.config.configloading import load_config
from src.utils import get_psnr_3d, get_ssim_3d
from src.loss import get_dice


# 创建轴向MIP图像
def create_Axial_MIP(volume, slices_num=15):
    img_shape = volume.shape
    Axial_MIP = np.zeros((img_shape[1], img_shape[2]))

    for i in range(img_shape[0]):
        start = max(0, i - slices_num)
        Axial_MIP = np.maximum(Axial_MIP, volume[start:i + 1].max(axis=0))
    return Axial_MIP


# 创建冠状MIP图像
def create_Sagittal_MIP(volume, slices_num=15):
    img_shape = volume.shape
    Sagittal_MIP = np.zeros((img_shape[0], img_shape[2]))

    # 一层层遍历
    for i in range(img_shape[1]):
        Sagittal_MIP = np.maximum(Sagittal_MIP, volume[:, i, :])
    return Sagittal_MIP


# 创建矢状MIP图像
def create_Coronal_MIP(volume):
    img_shape = volume.shape
    Coronal_MIP = np.zeros((img_shape[0], img_shape[1]))

    for i in range(img_shape[2]):
        Coronal_MIP = np.maximum(Coronal_MIP, volume[:, :, i])
    return Coronal_MIP


def plot_2D(image, cmap="gray", title=""):
    # plt.title(title)
    # 获取图像的形状
    plt.imshow(np.squeeze(image), cmap=cmap)
    plt.axis('off')
    # plt.savefig("/data/pgz/CBCT/Image/" + title + ".png", bbox_inches='tight', pad_inches=0)
    plt.show()


def test_NeBla(test_datadir, config_path, checkpoint_path, testdir):
    cfg = load_config(config_path)
    n_rays = cfg["train"]["n_rays"]
    netchunk = cfg["render"]["netchunk"]
    device_id = cfg["train"]["local_rank"]
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    num = 0
    # 导入模型
    model = NeBlaModel(cfg, device, load_opt=not cfg["train"]["no_load_opt"], load_scheduler=not cfg["train"]["no_load_scheduler"])

    # 输出参数量
    num_params = 0
    for param in model.net.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
    for param in model.pixel_encoder.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
    for param in model.voxel_encoder.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
    for param in model.refiner.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
    for param in model.latent_encoder.parameters():
        num_params += param.numel()
    print('Total number of parameters : %.3f M' % (num_params / 1e6))
    exit()

    # 载入模型权重
    print("读取权重：", checkpoint_path)
    model.load_model(checkpoint_path)
    # 切换为验证
    model.switch_to_eval()
    # 遍历测试集
    with torch.no_grad():
        total_loss_sum = {"psnr_3d": 0.0, "ssim_3d": 0.0, "dice_3d": 0.0}
        for pickle in os.listdir(test_datadir):
            num += 1
            print(num)
            single_pickle = test_datadir + pickle
            single_eval = Dataset(single_pickle, n_rays, "val", device)
            # print("正在测试：", single_pickle)
            # 体密度值评估
            image = single_eval.image
            voxels = single_eval.voxels
            PX = single_eval.projs.unsqueeze(0)

            """1D latent code"""
            # latent_vector = model.latent_encode(PX)
            # projs_l_expand = latent_vector.expand(voxels.shape[0], voxels.shape[1], voxels.shape[2], -1)

            """2D Pixel value"""
            PX_f_expand = PX[:, :, single_eval.f_z, single_eval.f_x].squeeze(0).permute(1, 2, 3, 0)

            """2D Pixel feature"""
            # # BCHW, [1, 24, 256, 256]
            # featmaps = model.pixel_encode(PX).permute(0, 2, 3, 1)
            # # # BN2, [1, 256^3, 2]
            # projs_f_expand = featmaps[:, single_eval.f_z, single_eval.f_x, :].squeeze(0)

            input = torch.cat([voxels, PX_f_expand], -1)
            # 输入[坐标值，对应的二维像素值]
            image_pred = run_network(input, model.net, netchunk, None)
            image_pred = image_pred.squeeze()

            """DICE计算，参考论文，官方是0.2"""
            loss = {
                "psnr_3d": get_psnr_3d(image_pred, image),
                "ssim_3d": get_ssim_3d(image_pred, image),
                "dice_3d": get_dice(image_pred, image, threshold=0.15)
            }

            # Save
            test_save_dir = osp.join(testdir, pickle[:pickle.find(".")])
            os.makedirs(test_save_dir, exist_ok=True)
            np.save(osp.join(test_save_dir, "image_pred.npy"), image_pred.cpu().detach().numpy())
            np.save(osp.join(test_save_dir, "image_gt.npy"), image.cpu().detach().numpy())

            nifti_img_pred = nib.Nifti1Image(image_pred.cpu().detach().numpy(), np.eye(4))
            nifti_img_gt = nib.Nifti1Image(image.cpu().detach().numpy(), np.eye(4))
            nib.save(nifti_img_pred, osp.join(test_save_dir, "image_pred.nii.gz"))
            nib.save(nifti_img_gt, osp.join(test_save_dir, "image_gt.nii.gz"))

            with open(osp.join(test_save_dir, "stats.txt"), "w") as f:
                for key, value in loss.items():
                    f.write("%s: %f\n" % (key, value.item()))

            total_loss_sum["psnr_3d"] += loss["psnr_3d"].item()
            total_loss_sum["ssim_3d"] += loss["ssim_3d"].item()
            total_loss_sum["dice_3d"] += loss["dice_3d"].item()

    loss_avg = {k: v / num for k, v in total_loss_sum.items()}
    return loss_avg


if __name__ == "__main__":
    # 256: {'psnr_3d': 21.82722647346987, 'ssim_3d': 0.7125178456085617, 'dice_3d': 0.0}
    # 128: {'psnr_3d': 22.42292073294277, 'ssim_3d': 0.7189523410894718, 'dice_3d': 0.0}

    # NeBla_pixel-1:  {'psnr_3d': 21.276722443490936, 'ssim_3d': 0.7059807796717379, 'dice_3d': 0.7050027776757876}
    # NeBla:        {'psnr_3d': 21.848791223064406, 'ssim_3d': 0.7148045380917951, 'dice_3d': 0.719690831998984}
    # hash_nerf:    {'psnr_3d': 22.147835176809057, 'ssim_3d': 0.7224156662352018, 'dice_3d': 0.7299302021662394}
    # hash_nerf2:   {'psnr_3d': 22.320249065979368, 'ssim_3d': 0.7224738030314377, 'dice_3d': 0.7356266727050146}
    # hash_contact: {'psnr_3d': 21.943006403108487, 'ssim_3d': 0.7139503769050433, 'dice_3d': 0.7267354801297188}
    # NeBla_pixel-real_d13_300: {'psnr_3d': 21.26608984571212, 'ssim_3d': 0.6809059353864282, 'dice_3d': 0.5671628434211016}

    # mi-MLP+Pixel:
    # 40: {'psnr_3d': 22.53970003486334, 'ssim_3d': 0.7391200101583221, 'dice_3d': 0.7388961638013521}

    # NMF:
    # 290: {'psnr_3d': 21.876124162519048, 'ssim_3d': 0.7117530653426712, 'dice_3d': 0.7368536318341891}
    test_datadir = "/data/pgz/MFF-NAF/real_data/train/"
    config_path = "/data/pgz/CODE/NeBla_2/config/tooth_50.yaml"
    model = "hash_nerf"
    model2test = {"": test_NeBla, "NeBla": test_NeBla, "hash_nerf": test_NeBla, "hash_contact": test_NeBla, "NeBla_pixel": test_NeBla,
                  "mi-MLP": test_NeBla, "NMF": test_NeBla}
    checkpoint_path = '/data/pgz/CODE/NeBla_2/logs/tooth_50_' + model + '/ckpt_0300.tar'
    # 保存测试集测试结果
    # testdir = "/data/pgz/NMF/test_result/" + model + "/"
    # 保存训练集测试结果用于unet3d训练
    testdir = "/data/pgz/MFF-NAF/unet3d/" + model + "/"
    loss_avg = model2test[model](test_datadir, config_path, checkpoint_path, testdir)
    print(loss_avg)

    """遍历所有ckpt，找最优"""
    # for i in range(1, 31):
    #     epoch = i * 10
    #     ckpt = "/ckpt_" + '{:04d}'.format(epoch) + ".tar"
    #     checkpoint_path = '/data/pgz/CODE/NeBla_2/logs/tooth_50_' + model + ckpt
    #     # 保存测试机测试结果
    #     testdir = "/data/pgz/MFF-NAF/unet3d/" + model + "/"
    #     loss_avg = model2test[model](test_datadir, config_path, checkpoint_path, testdir)
    #     print(epoch, loss_avg)
