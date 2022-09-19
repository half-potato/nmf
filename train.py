import os
from tqdm.auto import tqdm
from models.tensor_nerf import TensorNeRF
from renderer import *
from utils import *
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import datetime
from omegaconf import DictConfig, OmegaConf

from dataLoader import dataset_dict
import sys
import hydra
from omegaconf import OmegaConf
from pathlib import Path
from loguru import logger


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

    # init dataset
    dataset = dataset_dict[args.dataset.dataset_name]
    test_dataset = dataset(os.path.join(args.datadir, args.dataset.scenedir), split='test', downsample=args.dataset.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.dataset.ndc_ray

    ckpt = torch.load(args.ckpt)
    tensorf = TensorNeRF.load(ckpt, args.model.arch, near_far=test_dataset.near_far, strict=False)

    if args.fixed_bg:
        bg_sd = torch.load('log/mats360_bg.th')
        from models import bg_modules
        bg_module = bg_modules.HierarchicalCubeMap(bg_resolution=2048, num_levels=1, featureC=128, activation='softplus', power=2, lr=1e-2)
        bg_module.load_state_dict(bg_sd, strict=False)
        tensorf.bg_module = bg_module
    tensorf = tensorf.to(device)
    tensorf.sampler.update(tensorf.rf, init=True)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(os.path.join(args.datadir, args.dataset.scenedir), split='train', downsample=args.dataset.downsample_train, is_stack=True)
        test_res = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        logger.info(f'======> {args.expname} train all psnr: {np.mean(test_res["psnrs"])} <========================')

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

    # init dataset
    dataset = dataset_dict[args.dataset.dataset_name]
    train_dataset = dataset(os.path.join(args.datadir, args.dataset.scenedir), split='train', downsample=args.dataset.downsample_train, is_stack=False)
    test_dataset = dataset(os.path.join(args.datadir, args.dataset.scenedir), split='test', downsample=args.dataset.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    train_dataset.near_far = args.dataset.near_far
    near_far = train_dataset.near_far
    ndc_ray = args.dataset.ndc_ray

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    logger.add(logfolder + "/{time}.log", level="INFO", rotation="100 MB")
    

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    OmegaConf.save(config=args, f=f'{logfolder}/config.yaml')
    summary_writer = SummaryWriter(logfolder)

    aabb_scale = 1 if not hasattr(args.dataset, "aabb_scale") else args.dataset.aabb_scale
    aabb = train_dataset.scene_bbox.to(device) * aabb_scale

    tensorf = hydra.utils.instantiate(args.model.arch)(aabb=aabb, near_far=train_dataset.near_far)
    if args.ckpt is not None:
        # TODO REMOVE
        ckpt = torch.load(args.ckpt)
        tensorf = TensorNeRF.load(ckpt, args.model.arch)

    if args.fixed_bg:
        bg_sd = torch.load('log/mats360_bg.th')
        from models import bg_modules
        bg_module = bg_modules.HierarchicalCubeMap(bg_resolution=2048, num_levels=1, featureC=128, activation='softplus', power=2, lr=1e-2)
        bg_module.load_state_dict(bg_sd, strict=False)
        tensorf.bg_module = bg_module

    tensorf = tensorf.to(device)

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
    # optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.999), weight_decay=0, eps=1e-6)
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99), weight_decay=0)
    # optimizer = torch.optim.SGD(grad_vars, momentum=0.9, weight_decay=0)
    # optimizer = torch.optim.RMSprop(grad_vars, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0)
    # smoothing_vals = torch.linspace(0.5, 0.5, len(upsamp_list)+1).tolist()[1:]


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

    L1_reg_weight = params.L1_weight_inital
    logger.info("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = params.TV_weight_density, params.TV_weight_app
    tvreg = TVLoss()
    logger.info(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    allrgbs = allrgbs.to(device)
    allrays = allrays.to(device)
    # ratio of meters to pixels at a distance of 1 meter
    focal = (train_dataset.focal[0] if ndc_ray else train_dataset.focal)
    # / train_dataset.img_wh[0]
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, record_shapes=True) as prof:
    logger.info(tensorf)
    # TODO REMOVE
    if tensorf.bg_module is not None and not white_bg:
        if True:
            pbar = tqdm(range(args.n_bg_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
            # warm up by training bg
            for _ in pbar:
                ray_idx, rgb_idx = trainingSampler.nextids()
                rays_train, rgba_train = allrays[ray_idx], allrgbs[rgb_idx].reshape(-1, allrgbs.shape[-1])
                rgb_train = rgba_train[..., :3]
                if rgba_train.shape[-1] == 4:
                    alpha_train = rgba_train[..., 3]
                else:
                    alpha_train = None
                roughness = 1e-16*torch.ones(rays_train.shape[0], 1, device=device)
                rgb = tensorf.render_just_bg(rays_train, roughness)
                loss = torch.sqrt((rgb - rgb_train) ** 2 + params.charbonier_eps**2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                photo_loss = loss.detach().item()
                pbar.set_description(f'psnr={-10.0 * np.log(photo_loss) / np.log(10.0):.04f}')
        # tensorf.bg_module.save('test.png')

    # TODO REMOVE
    if args.ckpt is None:
        # dparams = tensorf.parameters()
        # space_optim = torch.optim.Adam(tensorf.rf.dbasis_mat.parameters(), lr=0.5, betas=(0.9,0.99))
        space_optim = torch.optim.Adam(tensorf.parameters(), lr=0.005, betas=(0.9,0.99))
        pbar = tqdm(range(1000))
        for _ in pbar:
            xyz = torch.rand(20000, 3, device=device)*2-1
            sigma_feat = tensorf.rf.compute_densityfeature(xyz)

            alpha = 1-torch.exp(-sigma_feat * 0.015 * tensorf.rf.distance_scale)
            # target_alpha = (params.start_density+params.start_density*torch.randn_like(alpha)).clip(min=1e-3)

            # sigma = 1-torch.exp(-sigma_feat)
            # loss = (sigma-torch.rand_like(sigma)*args.start_density).abs().mean()
            # target_alpha = (params.start_density+params.start_density*(2*torch.rand_like(alpha)-1))
            # target_alpha = (params.start_density+params.start_density*torch.randn_like(alpha))
            # target_alpha = target_alpha.clip(min=params.start_density/2, max=params.start_density*2)
            target_alpha = params.start_density
            loss = (alpha-target_alpha).abs().mean()
            # loss = (-sigma[mask].clip(max=1).sum() + sigma[~mask].clip(min=1e-8).sum())
            space_optim.zero_grad()
            loss.backward()
            pbar.set_description(f"Mean alpha: {alpha.detach().mean().item():.06f}.")
            space_optim.step()
    # tensorf.sampler.mark_untrained_grid(train_dataset.poses, train_dataset.intrinsics)
    torch.cuda.empty_cache()
    tensorf.sampler.update(tensorf.rf, init=True)


    pbar = tqdm(range(params.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    old_decay = False
    # T_max = 30000
    # scheduler1 = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    # scheduler2 = lr_scheduler.ChainedScheduler([
    #         lr_scheduler.ConstantLR(optimizer, factor=0.25, total_iters=600000),
    #         lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10000, T_mult=1)
    # ])
    # scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[3000])
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=params.n_iters, T_mult=1, eta_min=1e-3)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1, eta_min=1e-3)
    if True:
    # with torch.profiler.profile(record_shapes=True, schedule=torch.profiler.schedule(wait=1, warmup=1, active=20), with_stack=True) as p:
    # with torch.autograd.detect_anomaly():
        for iteration in pbar:

            if iteration < 50:
                ray_idx, rgb_idx = trainingSampler.nextids(batch=params.batch_size//8)
            elif iteration < 500:
                ray_idx, rgb_idx = trainingSampler.nextids(batch=params.batch_size//4)
            else:
                ray_idx, rgb_idx = trainingSampler.nextids()

            # patches = allrgbs[ray_idx].reshape(-1, args.bundle_size, args.bundle_size, 3)
            # plt.imshow(patches[0])
            # plt.show()

            rays_train, rgba_train = allrays[ray_idx], allrgbs[rgb_idx].reshape(-1, allrgbs.shape[-1])
            rgb_train = rgba_train[..., :3]
            if rgba_train.shape[-1] == 4:
                alpha_train = rgba_train[..., 3]
            else:
                alpha_train = None

            #rgb_map, alphas_map, depth_map, weights, uncertainty
            with torch.cuda.amp.autocast(enabled=args.fp16):
            # if True:
                data = renderer(rays_train, tensorf,
                        keys = ['rgb_map', 'floater_loss', 'normal_loss', 'backwards_rays_loss', 'diffuse_reg', 'bounce_count', 'color_count', 'roughness', 'whole_valid'],
                        focal=focal, output_alpha=alpha_train, chunk=params.batch_size, white_bg = white_bg, is_train=True, ndc_ray=ndc_ray)

                # loss = torch.mean((rgb_map[:, 1, 1] - rgb_train[:, 1, 1]) ** 2)
                normal_loss = data['normal_loss'].mean()
                floater_loss = data['floater_loss'].mean()
                diffuse_reg = data['diffuse_reg'].mean()
                rgb_map = data['rgb_map']
                if not train_dataset.hdr:
                    rgb_map = rgb_map.clip(max=1)
                whole_valid = data['whole_valid'] 
                if params.charbonier_loss:
                    loss = torch.sqrt((rgb_map - rgb_train[whole_valid]) ** 2 + params.charbonier_eps**2).mean()
                else:
                    # loss = ((rgb_map - rgb_train[whole_valid]) ** 2).mean()
                    loss = F.huber_loss(rgb_map, rgb_train[whole_valid], delta=1, reduction='mean')
                # loss = torch.sqrt(F.huber_loss(rgb_map, rgb_train, delta=1, reduction='none') + params.charbonier_eps**2).mean()
                photo_loss = ((rgb_map.clip(0, 1) - rgb_train[whole_valid].clip(0, 1)) ** 2).mean().detach()
                backwards_rays_loss = data['backwards_rays_loss']

                # loss
                total_loss = loss + \
                    params.normal_lambda*normal_loss + \
                    params.floater_lambda*floater_loss + \
                    params.backwards_rays_lambda*backwards_rays_loss + \
                    params.diffuse_lambda * diffuse_reg
                # ic(total_loss, params.normal_lambda*normal_loss, params.floater_lambda*floater_loss, params.backwards_rays_lambda*backwards_rays_loss, params.diffuse_lambda*diffuse_reg)

                if tensorf.visibility_module is not None:
                    pass
                    if iteration % 1 == 0 and iteration > 1000:
                        # if iteration < 100 or iteration % 1000 == 0:
                        if iteration % 500 == 0 and iteration < 5000:
                            tensorf.init_vis_module()
                        else:
                            tensorf.compute_visibility_loss(params.N_visibility_rays)

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
                    total_loss = total_loss + loss_tv
                    summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
                if TV_weight_app>0:
                    TV_weight_app *= lr_factor
                    loss_tv = loss_tv + tensorf.rf.TV_loss_app(tvreg)*TV_weight_app
                    total_loss = total_loss + loss_tv
                    summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

            optimizer.zero_grad()
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(tensorf.parameters(), 1e-3)
            optimizer.step()
            if not old_decay:
                scheduler.step()

            photo_loss = photo_loss.detach().item()
            
            PSNRs.append(-10.0 * np.log(photo_loss) / np.log(10.0))
            summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
            summary_writer.add_scalar('train/mse', photo_loss, global_step=iteration)
            summary_writer.add_scalar('train/backwards_rays_loss', backwards_rays_loss.detach().item(), global_step=iteration)
            summary_writer.add_scalar('train/floater_loss', floater_loss.detach().item(), global_step=iteration)
            summary_writer.add_scalar('train/normal_loss', normal_loss.detach().item(), global_step=iteration)
            summary_writer.add_scalar('train/diffuse_loss', diffuse_reg.detach().item(), global_step=iteration)
            summary_writer.add_scalar('train/color_count', data['color_count'].sum(), global_step=iteration)
            summary_writer.add_scalar('train/bounce_count', data['bounce_count'], global_step=iteration)

            if old_decay:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * lr_factor
            summary_writer.add_scalar('train/lr', list(optimizer.param_groups)[0]['lr'], global_step=iteration)

            # logger.info the current values of the losses.
            if iteration % args.progress_refresh_rate == 0:
                pbar.set_description(
                    f'psnr = {float(np.mean(PSNRs)):.2f}'
                    + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                    + f' rough = {data["roughness"].mean().item():.5f}'
                    + f' nerr = {float(normal_loss):.1e}'
                    + f' back = {backwards_rays_loss:.5e}'
                    + f' float = {floater_loss:.1e}'
                    + f' mipbias = {float(tensorf.bg_module.mipbias):.1e}'
                    # + f' mse = {photo_loss:.6f}'
                )
                PSNRs = []
                

            if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
                # tensorf.save(f'{logfolder}/{args.expname}_{iteration}.th', args.model.arch)
                test_res = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                        prtx=f'{iteration:06d}_', white_bg = white_bg, ndc_ray=ndc_ray,
                                        compute_extra_metrics=False)
                PSNRs_test = test_res['psnrs']
                summary_writer.add_scalar('test/psnr', np.mean(test_res['psnrs']), global_step=iteration)
                summary_writer.add_scalar('test/norm_err', np.mean(test_res['norm_errs']), global_step=iteration)
                logger.info(f'test_psnr = {float(np.mean(PSNRs_test)):.2f}')
                if args.save_often:
                    tensorf.save(f'{logfolder}/{args.expname}_{iteration:06d}.th', args.model.arch)

            if tensorf.check_schedule(iteration):
                new_grad_vars = tensorf.get_optparam_groups()
                for param_group, new_param_group in zip(optimizer.param_groups, new_grad_vars):
                    param_group['params'] = new_param_group['params']

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
        

    tensorf.save(f'{logfolder}/{args.expname}.th', args.model.arch)


    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        test_res = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        logger.info(f'======> {args.expname} test all psnr: {np.mean(test_res["psnrs"])} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        test_res = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(test_res["psnrs"]), global_step=iteration)
        logger.info(f'======> {args.expname} test all psnr: {np.mean(test_res["psnrs"])} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        logger.info('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                        N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


@hydra.main(version_base=None, config_path='configs', config_name='default')
def train(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)
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
