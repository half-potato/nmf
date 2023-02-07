import os
from tqdm.auto import tqdm
from modules.tensor_nerf import TensorNeRF
from renderer import *
from utils import *
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import datetime
from omegaconf import DictConfig, OmegaConf
import math
from mutils import normalize

from dataLoader import dataset_dict
import sys
import hydra
from omegaconf import OmegaConf
from pathlib import Path
from loguru import logger
import functools
from modules.integral_equirect import IntegralEquirect

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

# torch.autograd.set_detect_anomaly(True)

# from torch.profiler import profile, record_function, ProfilerActivity


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = chunk_renderer


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self, batch=None):
        batch = self.batch if batch is None else batch
        self.curr+=batch
        if self.curr + batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        ids = self.ids[self.curr:self.curr+batch]
        return ids, ids

@torch.no_grad()
def render_test(args):
    params = args.model.params
    if not os.path.exists(args.ckpt):
        logger.info('the ckpt path does not exists!!')
        return

    expname = f"{args.dataset.dataset_name}_{args.expname}"

    # init dataset
    dataset = dataset_dict[args.dataset.dataset_name]
    test_dataset = dataset(os.path.join(args.datadir, args.dataset.scenedir), split='test', downsample=args.dataset.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.dataset.ndc_ray

    ckpt = torch.load(args.ckpt)
    tensorf = TensorNeRF.load(ckpt, args.model.arch, near_far=test_dataset.near_far, strict=False)

    if args.fixed_bg is not None:
        bg_sd = torch.load(args.fixed_bg)
        from modules import bg_modules
        # bg_module = bg_modules.HierarchicalCubeMap(bg_resolution=2048, num_levels=1, featureC=128, activation='softplus', power=2, lr=1e-2)
        bg_module = IntegralEquirect(
            bg_resolution = 1024,
            mipbias = 0,
            activation = 'exp',
            lr = 0.001,
            init_val = -1.897,
            mul_lr = 0.001,
            brightness_lr = 0,
            betas = [0.0, 0.0],
            mul_betas = [0.9, 0.9],
            mipbias_lr = 1e-4,
            mipnoise = 0.0
        )
        bg_module.load_state_dict(bg_sd, strict=False)
        bg_module.lr = 0
        bg_module.mul_lr = 0
        bg_module.brightness_lr = 0
        # a = bg_module.bg_mats[0].reshape(-1, 3).mean(dim=-1)
        # b = tensorf.bg_module.bg_mats[0].reshape(-1, 3).mean(dim=-1)
        # a.sort()
        # b.sort()
        # a0 = a[500]
        # a1 = a[-500]
        # b0 = b[500]
        # b1 = b[-500]
        # # a0 = torch.quantile(a, 0.05)
        # # a1 = torch.quantile(a, 0.95)
        # # b0 = torch.quantile(b, 0.05)
        # # b1 = torch.quantile(b, 0.95)
        # new_mul = (tensorf.bg_module.mul*(b1-b0)) / (bg_module.mul*(a1-a0))
        # new_mul = 3
        # bg_module.mul *= new_mul
        # offset = tensorf.bg_module.mean_color().mean() / bg_module.mean_color().mean()
        # ic(new_mul, offset, torch.log(offset))
        # bg_module.brightness += torch.log(offset)

        # bg_module.mul += 1
        tensorf.bg_module = bg_module
    tensorf = tensorf.to(device)
    tensorf.train()
    # tensorf.sampler.update(tensorf.rf, init=True)
    # if tensorf.bright_sampler is not None:
    #     tensorf.bright_sampler.update(tensorf.bg_module)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(os.path.join(args.datadir, args.dataset.scenedir), split='train', downsample=args.dataset.downsample_train, is_stack=True)
        test_res = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        logger.info(f'======> {expname} train all psnr: {np.mean(test_res["psnrs"])} <========================')

    if args.render_test:
        folder = f'{logfolder}/imgs_test_all'
        os.makedirs(folder, exist_ok=True)
        logger.info(f"Saving test to {folder}")
        evaluation(test_dataset,tensorf, args, renderer, folder,
                   N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    #  if args.render_path:
    #      c2ws = test_dataset.render_path
    #      os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
    #      evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
    #                      N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device, bundle_size=args.bundle_size)

def reconstruction(args):
    params = args.model.params
    ic(params)
    expname = f"{args.dataset.dataset_name}_{args.expname}"

    # init dataset
    dataset = dataset_dict[args.dataset.dataset_name]
    stack_norms = args.dataset.stack_norms if hasattr(args.dataset, 'stack_norms') else False
    white_bg = args.dataset.white_bg if hasattr(args.dataset, 'white_bg') else True
    train_dataset = dataset(os.path.join(args.datadir, args.dataset.scenedir), split='train',
                            downsample=args.dataset.downsample_train, is_stack=False, stack_norms=stack_norms, white_bg=white_bg)
    test_dataset = dataset(os.path.join(args.datadir, args.dataset.scenedir), split='test',
                           downsample=args.dataset.downsample_train, is_stack=True, white_bg=white_bg)
    white_bg = train_dataset.white_bg
    train_dataset.near_far = args.dataset.near_far
    near_far = train_dataset.near_far
    ndc_ray = args.dataset.ndc_ray

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{expname}'
    logger.add(logfolder + "/{time}.log", level="INFO", rotation="100 MB")
    

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    aabb_scale = 1 if not hasattr(args.dataset, "aabb_scale") else args.dataset.aabb_scale
    aabb = train_dataset.scene_bbox.to(device) * aabb_scale

    tensorf = hydra.utils.instantiate(args.model.arch)(aabb=aabb, near_far=train_dataset.near_far)
    if args.ckpt is not None:
        # TODO REMOVE
        ckpt = torch.load(args.ckpt)
        tensorf = TensorNeRF.load(ckpt, args.model.arch, strict=False)

        # del ckpt['state_dict']['bg_module.bg_mats.0']
        # del ckpt['state_dict']['bg_module.bg_mats.1']
        # del ckpt['state_dict']['bg_module.bg_mats.2']
        # tensorf2 = TensorNeRF.load(ckpt, strict=False)
        # tensorf.normal_module = tensorf2.normal_module
        # tensorf.rf = tensorf2.rf
        # tensorf.diffuse_module = tensorf2.diffuse_module
        # grid_size = N_to_reso(params.N_voxel_final, tensorf.rf.aabb)
        # tensorf.rf.update_stepSize(grid_size)

    # TODO REMOVE
    if args.fixed_bg is not None:
        bg_sd = torch.load(args.fixed_bg)
        # from modules import bg_modules
        # bg_module = bg_modules.HierarchicalCubeMap(bg_resolution=2048, num_levels=1, featureC=128, activation='softplus', power=2, lr=1e-2)
        # bg_module.load_state_dict(bg_sd, strict=False)
        # bg_module.lr = 0
        # tensorf.bg_module = bg_module
        # if tensorf.bright_sampler is not None:
        #     tensorf.bright_sampler.update(tensorf.bg_module)
        bg_module = IntegralEquirect(
            bg_resolution = 1024,
            mipbias = 0,
            activation = 'exp',
            lr = 0.001,
            init_val = -1.897,
            mul_lr = 0.001,
            brightness_lr = 0,
            betas = [0.0, 0.0],
            mul_betas = [0.9, 0.9],
            mipbias_lr = 1e-4,
            mipnoise = 0.0
        )
        bg_module.load_state_dict(bg_sd)
        bg_module.lr = 0
        bg_module.mul_lr = 0
        bg_module.brightness_lr = 0
        tensorf.bg_module = bg_module

    tensorf = tensorf.to(device)
    tensorf.train()

    lr_bg = 1e-5
    grad_vars = tensorf.get_optparam_groups()
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = params.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/params.n_iters)

    # smoothing_vals = [0.6, 0.7, 0.8, 0.7, 0.5]
    upsamp_bg = hasattr(params, 'bg_upsamp_res') and tensorf.bg_module is not None
    if upsamp_bg:
        res = params.bg_upsamp_res.pop(0)
        lr_bg = params.bg_upsamp_lr.pop(0)
        logger.info(f"Upsampling bg to {res}")
        tensorf.bg_module.upsample(res)
        ind = [i for i, d in enumerate(grad_vars) if 'name' in d and d['name'] == 'bg'][0]
        grad_vars[ind]['params'] = tensorf.bg_module.parameters()
        grad_vars[ind]['lr'] = lr_bg


    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not ndc_ray and args.filter_rays:
        allrays, allrgbs, mask = tensorf.filtering_rays(allrays, allrgbs, train_dataset.focal, bbox_only=True)
    else:
        mask = None
    trainingSampler = SimpleSampler(allrays.shape[0], params.batch_size)


    ortho_reg_weight = params.ortho_weight
    logger.info("initial ortho_reg_weight", ortho_reg_weight)

    L1_reg_weight = params.L1_weight_initial
    logger.info("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = params.TV_weight_density, params.TV_weight_app
    tvreg = TVLoss()
    logger.info(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    # allrgbs = allrgbs.to(device)
    # allrays = allrays.to(device)
    # ratio of meters to pixels at a distance of 1 meter
    focal = (train_dataset.focal[0] if ndc_ray else train_dataset.fx)
    # / train_dataset.img_wh[0]
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, record_shapes=True) as prof:
    logger.info(tensorf)
    ic(white_bg)
    # TODO REMOVE
    # if tensorf.bg_module is not None and not white_bg:
    #     if True:
    #         pbar = tqdm(range(args.n_bg_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    #         # warm up by training bg
    #         for _ in pbar:
    #             ray_idx, rgb_idx = trainingSampler.nextids()
    #             rays_train, rgba_train = allrays[ray_idx], allrgbs[rgb_idx].reshape(-1, allrgbs.shape[-1])
    #             rgb_train = rgba_train[..., :3]
    #             if rgba_train.shape[-1] == 4:
    #                 alpha_train = rgba_train[..., 3]
    #             else:
    #                 alpha_train = None
    #             roughness = 1e-16*torch.ones(rays_train.shape[0], 1, device=device)
    #             rgb = tensorf.render_just_bg(rays_train, roughness)
    #             loss = torch.sqrt((rgb - rgb_train) ** 2 + params.charbonier_eps**2).mean()
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             photo_loss = loss.detach().item()
    #             pbar.set_description(f'psnr={-10.0 * np.log(photo_loss) / np.log(10.0):.04f}')
        # tensorf.bg_module.save('test.png')

    # TODO REMOVE
    tensorf.sampler.update(tensorf.rf, init=True)
    if args.ckpt is None:
        if tensorf.rf.num_pretrain > 0:
            # dparams = tensorf.parameters()
            # space_optim = torch.optim.Adam(tensorf.rf.dbasis_mat.parameters(), lr=0.5, betas=(0.9,0.99))
            space_optim = torch.optim.Adam(tensorf.parameters(), lr=0.005, betas=(0.9,0.99))
            pbar = tqdm(range(tensorf.rf.num_pretrain))
            for _ in pbar:
                xyz = (torch.rand(20000, 3, device=device)*2-1) * tensorf.rf.aabb[1].reshape(1, 3)
                sigma_feat = tensorf.rf.compute_densityfeature(xyz)

                # step_size = 0.015
                step_size = tensorf.sampler.stepsize
                alpha = 1-torch.exp(-sigma_feat * step_size * tensorf.rf.distance_scale)
                # ic(alpha.mean(), sigma_feat.mean(), tensorf.rf.distance_scale)
                # sigma = 1-torch.exp(-sigma_feat)
                # loss = (sigma-torch.rand_like(sigma)*args.start_density).abs().mean()
                # target_alpha = (params.start_density+params.start_density*(2*torch.rand_like(alpha)-1))
                target_alpha = (params.start_density + 0.1*params.start_density*torch.randn_like(alpha))
                # target_alpha = target_alpha.clip(min=params.start_density/2, max=params.start_density*2)
                # target_alpha = params.start_density
                loss = (alpha-target_alpha).abs().mean()
                # loss = (-sigma[mask].clip(max=1).sum() + sigma[~mask].clip(min=1e-8).sum())
                space_optim.zero_grad()
                loss.backward()
                pbar.set_description(f"Mean alpha: {alpha.detach().mean().item():.06f}.")
                space_optim.step()
        else:
            # calculate alpha mean
            xyz = torch.rand(100000, 4, device=device)*2-1
            xyz[:, 3] *= 0
            sigma_feat = tensorf.rf.compute_densityfeature(xyz)

            # step_size = 0.015
            target_sigma = -math.log(1-params.start_density) / (tensorf.sampler.stepsize * tensorf.rf.distance_scale)

            # compute density_shift assume exponential activation
            density_shift = math.log(target_sigma) - math.log(sigma_feat.mean().item())
            ic(target_sigma, sigma_feat.mean(), density_shift, sigma_feat.mean())
            tensorf.rf.density_shift += density_shift
            args.field.density_shift = tensorf.rf.density_shift
    # tensorf.sampler.mark_untrained_grid(train_dataset.poses, train_dataset.intrinsics)
    torch.cuda.empty_cache()
    tensorf.sampler.update(tensorf.rf, init=True)
    torch.cuda.empty_cache()


    xyz = torch.rand(100000, 4, device=device)*2-1
    xyz[:, 3] *= 0
    sigma_feat = tensorf.rf.compute_densityfeature(xyz)
    alpha = 1-torch.exp(-sigma_feat * tensorf.sampler.stepsize * tensorf.rf.distance_scale)
    print(f"Mean alpha: {alpha.detach().mean().item():.06f}.")
    ic(sigma_feat.mean())
    feat = tensorf.rf.compute_appfeature(xyz)
    tensorf.model.diffuse_module.calibrate(xyz, normalize(torch.rand_like(xyz[:, :3])), feat)
    tensorf.model.brdf.calibrate(feat, tensorf.bg_module.mean_color().detach().mean())
    args.model.arch.model.brdf.bias = tensorf.model.brdf.bias
    args.model.arch.model.diffuse_module.diffuse_bias = tensorf.model.diffuse_module.diffuse_bias
    args.model.arch.model.diffuse_module.roughness_bias = tensorf.model.diffuse_module.roughness_bias

    pbar = tqdm(range(params.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    def init_optimizer(grad_vars):
        # optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.999), weight_decay=0, eps=1e-6)
        optimizer = torch.optim.Adam(grad_vars, betas=params.betas, eps=params.eps)
        if params.lr is not None:
            optimizer = torch.optim.Adam(tensorf.parameters(), lr=params.lr, betas=params.betas, eps=params.eps)
        else:
            optimizer = torch.optim.Adam(grad_vars, betas=params.betas, eps=params.eps, weight_decay=params.weight_decay)
        compute_lambda = functools.partial(
                learning_rate_decay, lr_init=params.lr_init, lr_final=params.lr_final, max_steps=params.n_iters,
                lr_delay_steps=params.lr_delay_steps, lr_delay_mult=params.lr_delay_mult)
        scheduler = lr_scheduler.LambdaLR(optimizer, compute_lambda)
        return optimizer, scheduler
    optimizer, scheduler = init_optimizer(grad_vars)
    ori_decay = math.exp(math.log(params.final_ori_lambda / params.ori_lambda) / params.n_iters) if params.ori_lambda > 0 and params.final_ori_lambda is not None else 1
    normal_decay = math.exp(math.log(params.final_pred_lambda / params.pred_lambda) / params.n_iters) if params.pred_lambda > 0 and params.final_pred_lambda is not None else 1
    ic(ori_decay)
    ic(normal_decay)

    OmegaConf.save(config=args, f=f'{logfolder}/config.yaml')
    num_rays = params.starting_batch_size
    prev_n_samples = None
    hist_n_samples = None
    gt_bg = cv2.imread(args.gt_bg) if args.gt_bg is not None else None
    if True:
    # with torch.profiler.profile(record_shapes=True, schedule=torch.profiler.schedule(wait=1, warmup=1, active=params.n_iters-1), with_stack=True) as p:
    # with torch.autograd.detect_anomaly():
        for iteration in pbar:
            optimizer.zero_grad(set_to_none=True)
            losses, roughnesses, envmap_regs, diffuse_regs = [],[],[],[]
            pred_losses, ori_losses = [], []
            TVs = []

            lbatch_size = params.min_batch_size if num_rays < params.min_batch_size else num_rays
            num_remaining = lbatch_size
            while num_remaining > 0:
                lnum_rays = min(num_rays, num_remaining)
                num_remaining -= lnum_rays
                ray_idx, rgb_idx = trainingSampler.nextids(lnum_rays)
                rays_train, rgba_train = allrays[ray_idx].to(device), allrgbs[rgb_idx].reshape(-1, allrgbs.shape[-1]).to(device)
                match params.bg_col:
                    case 'rand':
                        bg_col = torch.rand(3, device=device)
                    case 'white':
                        bg_col = torch.ones((3), device=device)
                    case 'black':
                        bg_col = torch.zeros((3), device=device)
                    case _:
                        raise Exception(f"Unknown bg col: {params.bg_col}")
                if rgba_train.shape[-1] == 4:
                    rgb_train = rgba_train[:, :3] * rgba_train[:, -1:] + (1 - rgba_train[:, -1:])*bg_col  # blend A to RGB
                    alpha_train = rgba_train[..., 3]
                else:
                    rgb_train = rgba_train
                    alpha_train = None
                gt_normal_map = train_dataset.all_norms[ray_idx].to(device) if train_dataset.stack_norms else None

                with torch.cuda.amp.autocast(enabled=args.fp16):
                    ims, stats = renderer(rays_train, tensorf, gt_normals=gt_normal_map,
                            keys = ['rgb_map', 'normal_err', 'distortion_loss', 'prediction_loss', 'ori_loss', 'diffuse_reg', 'roughness', 'whole_valid', 'envmap_reg', 'brdf_reg', 'n_samples'],
                            focal=focal, output_alpha=alpha_train, chunk=num_rays, bg_col=bg_col, is_train=True, ndc_ray=ndc_ray)

                    n_samples = stats['n_samples']
                    if n_samples[0] == 0:
                        continue
                    prediction_loss = stats['prediction_loss'].sum()
                    distortion_loss = stats['distortion_loss'].sum()
                    diffuse_reg = stats['diffuse_reg'].sum()
                    envmap_reg = stats['envmap_reg'].sum()
                    brdf_reg = stats['brdf_reg'].sum()
                    rgb_map = ims['rgb_map']
                    if not train_dataset.hdr:
                        rgb_map = rgb_map.clip(max=1)
                    whole_valid = stats['whole_valid'] 
                    if params.charbonier_loss:
                        loss = torch.sqrt((rgb_map - rgb_train[whole_valid]) ** 2 + params.charbonier_eps**2).sum()
                    else:
                        # loss = ((rgb_map - rgb_train[whole_valid]) ** 2).mean()
                        # loss = F.huber_loss(rgb_map.clip(0, 1), rgb_train[whole_valid], delta=1, reduction='mean')
                        loss = ((rgb_map.clip(0, 1) - rgb_train[whole_valid].clip(0, 1))**2).sum()
                        # loss = ((rgb_map.clip(0, 1) - rgb_train[whole_valid].clip(0, 1)).abs()).sum()
                    norm_err = sum(stats['normal_err']) if type(stats['normal_err']) == list else stats['normal_err'].sum()
                    # loss = torch.sqrt(F.huber_loss(rgb_map, rgb_train, delta=1, reduction='none') + params.charbonier_eps**2).mean()
                    # photo_loss = ((rgb_map.clip(0, 1) - rgb_train[whole_valid].clip(0, 1)) ** 2).mean().detach()
                    photo_loss = ((rgb_map.clip(0, 1) - rgb_train[whole_valid].clip(0, 1))**2).mean().detach()
                    ori_loss = stats['ori_loss'].sum()

                    # adjust number of rays
                    # need to store mean ratios if I have any hope of stabilizing this
                    mean_samples = n_samples
                    ratio = int(whole_valid.sum()) / mean_samples[0]
                    mean_ratio = ratio if prev_n_samples is None else min(0.1*ratio + 0.9*prev_n_samples, ratio)
                    prev_n_samples = mean_ratio
                    num_rays = int(mean_ratio * params.target_num_samples + 1)
                    tensorf.model.update_n_samples(n_samples[1:])
                    # tensorf.eval_batch_size = num_rays // 2

                    # rays_remaining -= rgb_map.shape[0]
                    # rays_train = rays_train[~whole_valid]
                    # rgb_train = rgb_train[~whole_valid]
                    # if gt_normal_map is not None:
                    #     gt_normal_map = gt_normal_map[~whole_valid]

                    # loss
                    # ori_lambda = params.ori_lambda if iteration > 1000 else params.ori_lambda * iteration / 1000
                    # pred_lambda = params.pred_lambda if iteration > 500 else params.pred_lambda * iteration / 500
                    ori_lambda = params.ori_lambda
                    pred_lambda = params.pred_lambda
                    # ic(pred_lambda, ori_lambda, loss, 
                    #     params.distortion_lambda*distortion_loss,
                    #     ori_lambda*ori_loss,
                    #     params.envmap_lambda * (envmap_reg-0.05).clip(min=0),
                    #     params.diffuse_lambda * diffuse_reg,
                    #     params.brdf_lambda * brdf_reg,
                    #     pred_lambda * prediction_loss)
                    total_loss = loss + \
                        params.distortion_lambda*distortion_loss + \
                        ori_lambda*ori_loss + \
                        params.envmap_lambda * (envmap_reg-0.05).clip(min=0) + \
                        params.diffuse_lambda * diffuse_reg + \
                        params.brdf_lambda * brdf_reg + \
                        pred_lambda * prediction_loss + \
                        params.normal_err_lambda * norm_err


                    # if tensorf.visibility_module is not None:
                        # pass
                        # if iteration % 1 == 0 and iteration > 250:
                        #     # if iteration < 100 or iteration % 1000 == 0:
                        #     if iteration % 250 == 0 and iteration < 2000:
                        #         tensorf.init_vis_module()
                        #         torch.cuda.empty_cache()
                        #     else:
                        #         tensorf.compute_visibility_loss(params.N_visibility_rays)

                    if ortho_reg_weight > 0:
                        loss_reg = tensorf.rf.vector_comp_diffs()
                        total_loss += ortho_reg_weight*loss_reg
                        summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
                    if L1_reg_weight > 0:
                        loss_reg_L1 = tensorf.rf.density_L1()
                        total_loss += L1_reg_weight*loss_reg_L1
                        summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

                    loss_tv = 0
                    if TV_weight_density>0:
                        TV_weight_density *= lr_factor
                        loss_tv = tensorf.rf.TV_loss_density(tvreg) * TV_weight_density
                        summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
                    if TV_weight_app>0:
                        TV_weight_app *= lr_factor
                        loss_tv = loss_tv + tensorf.rf.TV_loss_app(tvreg)*TV_weight_app
                        summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)
                    if params.TV_weight_bg > 0:
                        loss_tv = loss_tv + params.TV_weight_bg*tensorf.bg_module.tv_loss()
                    total_loss = total_loss + loss_tv

                    total_loss = total_loss / lbatch_size
                    total_loss.backward()

                    photo_loss = photo_loss.detach().item()
                
                    TVs.append(float(loss_tv))
                    ori_losses.append(params.ori_lambda * ori_loss.detach().item())
                    pred_losses.append(params.pred_lambda * prediction_loss.detach().item())
                    losses.append(total_loss.detach().item())
                    roughnesses.append(ims['roughness'].mean().detach().item())
                    diffuse_regs.append(params.diffuse_lambda * diffuse_reg.detach().item() / lbatch_size)
                    envmap_regs.append(envmap_reg.detach().item())
                    PSNRs.append(-10.0 * np.log(photo_loss) / np.log(10.0))

                    # summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
                    # summary_writer.add_scalar('train/mse', photo_loss, global_step=iteration)
                    # summary_writer.add_scalar('train/ori_loss', ori_loss.detach().item(), global_step=iteration)
                    # summary_writer.add_scalar('train/distortion_loss', distortion_loss.detach().item(), global_step=iteration)
                    # summary_writer.add_scalar('train/prediction_loss', prediction_loss.detach().item(), global_step=iteration)
                    # summary_writer.add_scalar('train/diffuse_loss', diffuse_reg.detach().item(), global_step=iteration)
                    #
                    # summary_writer.add_scalar('train/lr', list(optimizer.param_groups)[0]['lr'], global_step=iteration)
                del ray_idx, rgb_idx, rays_train, rgba_train, gt_normal_map, ims, stats

            if params.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(tensorf.parameters(), params.clip_grad)
            optimizer.step()
            scheduler.step()
            params.ori_lambda *= ori_decay
            params.pred_lambda *= normal_decay
                
            if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
                # tensorf.save(f'{logfolder}/{expname}_{iteration}.th', args.model.arch)
                test_res = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                        prtx=f'{iteration:06d}_', white_bg = white_bg, ndc_ray=ndc_ray,
                                        compute_extra_metrics=False, gt_bg=gt_bg)
                PSNRs_test = test_res['psnrs']
                summary_writer.add_scalar('test/psnr', np.mean(test_res['psnrs']), global_step=iteration)
                summary_writer.add_scalar('test/norm_err', np.mean(test_res['norm_errs']), global_step=iteration)
                logger.info(f'test_psnr = {float(np.mean(PSNRs_test)):.2f}')
                if args.save_often:
                    tensorf.save(f'{logfolder}/{expname}_{iteration:06d}.th', args.model.arch)

            # logger.info the current values of the losses.
            if iteration % args.progress_refresh_rate == 0:
                desc = f'psnr = {float(np.mean(PSNRs)):.2f}' + \
                    f' test_psnr = {float(np.mean(PSNRs_test)):.2f}' + \
                    f' loss = {float(np.sum(losses)):.5f}' + \
                    f' envmap = {float(np.mean(envmap_regs)):.5f}' + \
                    f' diffuse = {float(np.sum(diffuse_regs)):.5f}' + \
                    f' nrays = {[num_rays] + tensorf.model.max_retrace_rays}'
                    # f' rough = {float(np.mean(roughnesses)):.5f}' + \
                    # f' tv = {float(np.mean(TVs)):.5f}' + \
                    # f' ori loss = {float(np.mean(ori_losses) / num_rays):.5f}' + \
                    # f' pred loss = {float(np.mean(pred_losses) / num_rays):.5f}' + \
                    # + f' mse = {photo_loss:.6f}'
                if tensorf.bg_module is not None:
                    desc = desc + \
                    f' mipbias = {float(tensorf.bg_module.mipbias):.1e}'
                    # f' mul = {float(tensorf.bg_module.mul):.1e}' + \
                    # f' bright = {float(tensorf.bg_module.brightness):.1e}'
                pbar.set_description(desc)
                PSNRs = []

            if tensorf.check_schedule(iteration, 1):
                grad_vars = tensorf.get_optparam_groups()
                optimizer, scheduler = init_optimizer(grad_vars)
                # new_grad_vars = tensorf.get_optparam_groups()
                # for param_group, new_param_group in zip(optimizer.param_groups, new_grad_vars):
                #     param_group['params'] = new_param_group['params']

            # if iteration in update_alphamask_list:

                #  if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                    # tensorVM.alphaMask = None
                    # L1_reg_weight = params.L1_weight_rest
                    # logger.info("continuing L1_reg_weight", L1_reg_weight)


            # if not ndc_ray and iteration == update_AlphaMask_list[-1] and args.filter_rays:
            #     # filter rays outside the bbox
            #     allrays, allrgbs, mask = tensorf.filtering_rays(allrays, allrgbs, focal)
            #     trainingSampler = SimpleSampler(allrays.shape[0], params.batch_size)

    #         p.step()
    # p.export_chrome_trace('p.trace')


    # prof.export_chrome_trace('trace.json')
        

    tensorf.save(f'{logfolder}/{expname}.th', args.model.arch)


    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        test_res = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device, gt_bg=gt_bg)
        logger.info(f'======> {expname} test all psnr: {np.mean(test_res["psnrs"])} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        test_res = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device, gt_bg=gt_bg)
        summary_writer.add_scalar('test/psnr_all', np.mean(test_res["psnrs"]), global_step=iteration)
        logger.info(f'======> {expname} test all psnr: {np.mean(test_res["psnrs"])} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        logger.info('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                        N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device, gt_bg=gt_bg)


@hydra.main(version_base=None, config_path='configs', config_name='default')
def train(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    logger.info(cfg.dataset)
    logger.info(cfg.model)
    cfg.model.arch.rf = cfg.field

    if cfg.render_only:
        render_test(cfg)
    else:
        reconstruction(cfg)
        # reconstruction(args)

if __name__ == '__main__':
    train()
