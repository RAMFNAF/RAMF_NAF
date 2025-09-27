import os
import os.path as osp
import torch
import numpy as np
import argparse
import nibabel as nib

from NeBla.trainer import Trainer
from NeBla.render import render, run_network
from src.config.configloading import load_config
from src.loss import calc_mse_loss, get_dice
from src.utils import get_psnr_3d, get_ssim_3d


class BasicTrainer(Trainer):
    def __init__(self):
        super().__init__(cfg)
        print(f"[Start] exp: {cfg['exp']['expname']}, net: RAMF-NAF")

    def compute_loss(self, data, global_step, idx_epoch):
        rays = data["rays"].reshape(-1, 8)
        select_coords = data["select_coords"]
        projs = data["projs"].reshape(-1)
        PX = self.train_dset.PX.unsqueeze(0)
        image = self.train_dset.image

        pixel_vector = self.model.pixel_encode(PX.unsqueeze(0)).permute(0, 2, 3, 1)

        # # 采样点对应GT体素值，采样点预测体素值
        raw_gt, raw_pred = render(self.device, rays, image, PX, select_coords, pixel_vector, self.model.net, **self.conf["render"])
        raw_pred = raw_pred.squeeze()

        loss = {"loss": 0.}
        calc_mse_loss(loss, raw_gt, raw_pred, lamda=1)

        # Log
        for ls in loss.keys():
            self.writer.add_scalar(f"train/{ls}", loss[ls].item(), global_step)
        return loss["loss"]

    def eval_step(self, eval_name, data, global_step, idx_epoch):
        ct_img = data["ct_img"].squeeze(0)
        voxels = data["voxels"].squeeze(0)
        dr_img = data["dr_img"].unsqueeze(0)

        # BCHW, [1, 24, 256, 256]
        featmaps = self.model.pixel_encode(dr_img).permute(0, 2, 3, 1)
        # BN2, [1, 256^3, 2]
        projs_f_expand = featmaps[:, self.eval_dset.f_z, self.eval_dset.f_x, :].squeeze(0)

        input = torch.cat([voxels, projs_f_expand], -1)
        image_pred = run_network(input, self.model.net, self.netchunk)
        image_pred = image_pred.squeeze()

        loss = {
            "psnr_3d": get_psnr_3d(image_pred, ct_img),
            "ssim_3d": get_ssim_3d(image_pred, ct_img),
            "dice_3d": get_dice(image_pred, ct_img),
        }

        for ls in loss.keys():
            self.writer.add_scalar(f"eval/{ls}", loss[ls], global_step)

        # Save
        eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}", eval_name)
        os.makedirs(eval_save_dir, exist_ok=True)
        np.save(osp.join(eval_save_dir, "image_pred.npy"), image_pred.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "image_gt.npy"), ct_img.cpu().detach().numpy())

        nifti_img_pred = nib.Nifti1Image(image_pred.cpu().detach().numpy(), np.eye(4))
        nifti_img_gt = nib.Nifti1Image(ct_img.cpu().detach().numpy(), np.eye(4))
        nib.save(nifti_img_pred, osp.join(eval_save_dir, "image_pred.nii.gz"))
        nib.save(nifti_img_gt, osp.join(eval_save_dir, "image_gt.nii.gz"))

        with open(osp.join(eval_save_dir, "stats.txt"), "w") as f:
            for key, value in loss.items():
                f.write("%s: %f\n" % (key, value.item()))

        return loss


def config_parser(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/" + config + "_50.yaml", help="configs file path")
    return parser


if __name__ == "__main__":
    parser = config_parser("tooth")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device_id = cfg["train"]["local_rank"]
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    trainer = BasicTrainer()
    trainer.start()
