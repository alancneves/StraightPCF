import argparse
import torch
import torch.utils.tensorboard
from tqdm.auto import tqdm
from pathlib import Path
from datasets.pcl import load_pcd

from datasets import *
from utils.misc import *
from utils.transforms import *
from utils.denoise import *
from utils.evaluate import *
from utils.logger import Logger
from models.straightpcf import *

def input_iter(path):
    fn = Path(path).name
    pcl_noisy = torch.FloatTensor(load_pcd(path))
    pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_noisy)
    return {
        'pcl_noisy': pcl_noisy,
        'name': fn[:-4],
        'center': center,
        'scale': scale
    }


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description="Inference script to denoise pointclouds")
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default='./pretrained_straightpcf/ckpt_straightpcf.pt')
    parser.add_argument('--velocity_ckpt', type=str, default='./pretrained_vm/ckpt_vm.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2020)
    # Denoiser parameters
    parser.add_argument('--patch_size', type=int, default=1000, help='Patch size to be sample, i.e., number of points to sample')
    parser.add_argument('--seed_k', type=int, default=6, help='Multiplier factor to sample from patches')
    parser.add_argument('--seed_k_alpha', type=int, default=10, help='Patches step multiplier factor. Increase it to reduce step size and memory usage.')
    parser.add_argument('--niters', type=int, default=1)
    args = parser.parse_args()
    print("Args: ", args)
    seed_all(args.seed)

    # Logging
    logger = Logger("straight-pcf")

    # Input/Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Model
    logger.info('Loading StraightPCF model and weights...')
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = StraightPCF(args=ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Denoise
    logger.info(f'Performing denoising on {args.input_file}...')
    data = input_iter(args.input_file)
    pcl_noisy = data['pcl_noisy'].to(args.device)
    try:
        with torch.no_grad():
            pcl_next = pcl_noisy
            for niter in range(args.niters):
                logger.info(f'Starting iteration {niter}!')
                pcl_next = patch_based_denoise(
                    model=model,
                    pcl_noisy=pcl_next,
                    seed_k=args.seed_k,
                    seed_k_alpha=args.seed_k_alpha,
                    patch_size=args.patch_size,
                )

        # Denormalize
        pcl_denoised = (pcl_next.cpu() * data['scale'] + data['center']).numpy()

        # Save result
        save_path = output_dir / (data['name'] + '.xyz')
        logger.info(f"Saving pointcloud at {save_path}")
        np.savetxt(save_path, pcl_denoised, fmt='%.8f')

    except Exception as e:
        logger.error(e)
        logger.error(f'Current niter is {niter}')

    finally:
        logger.warning(f"Finished!")

