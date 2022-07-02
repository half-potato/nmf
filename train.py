import os
from tqdm.auto import tqdm
from models.tensor_nerf import TensorNeRF
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from omegaconf import DictConfig, OmegaConf

from dataLoader import dataset_dict
import sys
import hydra
from omegaconf import OmegaConf

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
    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt)
    ckpt['config']['bg_module']['bg_resolution'] = ckpt['state_dict']['bg_module.bg_mat'].shape[-1]
    tensorf = TensorNeRF.load(ckpt).to(device)
    tensorf.rf.set_smoothing(1)

    # init dataset
    dataset = dataset_dict[args.dataset.dataset_name]
    test_dataset = dataset(os.path.join(args.datadir, args.dataset.scenedir), split='test', downsample=args.dataset.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.dataset.ndc_ray

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(os.path.join(args.datadir, args.dataset.scenedir), split='train', downsample=args.dataset.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device,
                                render_mode=args.render_mode)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        folder = f'{logfolder}/imgs_test_all'
        os.makedirs(folder, exist_ok=True)
        print(f"Saving test to {folder}")
        evaluation(test_dataset,tensorf, args, renderer, folder,
                   N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device,
                   render_mode=args.render_mode)

    #  if args.render_path:
    #      c2ws = test_dataset.render_path
    #      os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
    #      evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
    #                      N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device,
    #                      bundle_size=args.bundle_size, render_mode=args.render_mode)

def reconstruction(args):
    assert(args.batch_size % args.params.num_rays_per_envmap == 0)

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
    

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    OmegaConf.save(config=args, f=f'{logfolder}/config.yaml')
    summary_writer = SummaryWriter(logfolder)

    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.params.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        tensorf = TensorNeRF.load(ckpt, aabb).to(device)
    else:
        tensorf = hydra.utils.instantiate(args.model)(aabb=aabb, grid_size=reso_cur).to(device)


    lr_bg = 0.03
    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis, lr_bg)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))

    # Set up schedule
    upsamp_list = args.params.upsamp_list
    uplambda_list = args.params.uplambda_list
    update_AlphaMask_list = args.params.update_AlphaMask_list
    bounce_n_list = args.params.bounce_n_list
    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.params.N_voxel_init), np.log(args.params.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]
    l_list = torch.linspace(1.0, 0.0, len(uplambda_list)+1).tolist()
    tensorf.l = l_list.pop(0)
    tensorf.max_bounce_rays = bounce_n_list.pop(0)

    # smoothing_vals = [0.6, 0.7, 0.8, 0.7, 0.5]
    smoothing_vals = torch.linspace(args.params.smoothing_start, args.params.smoothing_end, len(upsamp_list)+1).tolist()
    tensorf.rf.set_smoothing(smoothing_vals.pop(0))
    upsamp_bg = hasattr(args.params, 'bg_upsamp_res') and tensorf.bg_module is not None
    if upsamp_bg:
        res = args.params.bg_upsamp_res.pop(0)
        lr_bg = args.params.bg_upsamp_lr.pop(0)
        print(f"Upsampling bg to {res}")
        tensorf.bg_module.upsample(res)
        ind = [i for i, d in enumerate(grad_vars) if 'name' in d and d['name'] == 'bg'][0]
        grad_vars[ind]['params'] = tensorf.bg_module.parameters()
        grad_vars[ind]['lr'] = lr_bg
        optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
    # smoothing_vals = torch.linspace(0.5, 0.5, len(upsamp_list)+1).tolist()[1:]


    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not ndc_ray and args.filter_rays:
        allrays, allrgbs, mask = tensorf.filtering_rays(allrays, allrgbs, train_dataset.focal, bbox_only=True)
    else:
        mask = None
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)


    Ortho_reg_weight = args.params.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.params.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.params.TV_weight_density, args.params.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    allrgbs = allrgbs.to(device)
    allrays = allrays.to(device)
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    # ratio of meters to pixels at a distance of 1 meter
    focal = (train_dataset.focal[0] if ndc_ray else train_dataset.focal)
    # / train_dataset.img_wh[0]
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, record_shapes=True) as prof:
    if True:
    # with torch.autograd.detect_anomaly():
        for iteration in pbar:

            if iteration < 16:
                ray_idx, rgb_idx = trainingSampler.nextids(args.batch_size // 4)
            else:
                ray_idx, rgb_idx = trainingSampler.nextids()

            # patches = allrgbs[ray_idx].reshape(-1, args.bundle_size, args.bundle_size, 3)
            # plt.imshow(patches[0])
            # plt.show()

            # rays_train, rgb_train = allrays[ray_idx], allrgbs[rgb_idx].to(device).reshape(-1, args.bundle_size, args.bundle_size, 3)
            rays_train, rgba_train = allrays[ray_idx], allrgbs[rgb_idx].reshape(-1, allrgbs.shape[-1])
            rgb_train = rgba_train[..., :3]
            if rgba_train.shape[-1] == 4:
                alpha_train = rgba_train[..., 3]
            else:
                alpha_train = None
                
            if iteration <= args.params.num_bw_iters:
                # convert rgb to bw
                rgb_train = rgb_train[..., :1].expand(rgb_train.shape)

            #rgb_map, alphas_map, depth_map, weights, uncertainty
            with torch.cuda.amp.autocast(enabled=args.fp16):
                data = renderer(rays_train, tensorf,
                        keys = ['rgb_map', 'floater_loss', 'normal_loss', 'roughness', 'backwards_rays_loss', 'termination_xyz', 'normal_map', 'diffuse_loss'],
                        focal=focal, output_alpha=alpha_train, chunk=args.batch_size,
                        N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, is_train=True)

                # loss = torch.mean((rgb_map[:, 1, 1] - rgb_train[:, 1, 1]) ** 2)
                normal_loss = data['normal_loss'].mean()
                diffuse_loss = data['diffuse_loss'].mean()
                floater_loss = data['floater_loss'].mean()
                roughness = data['roughness'].mean()
                loss = torch.sqrt((data['rgb_map'] - rgb_train) ** 2 + args.params.charbonier_eps**2).mean()
                photo_loss = ((data['rgb_map'] - rgb_train) ** 2).mean().detach()
                backwards_rays_loss = data['backwards_rays_loss']


                # loss
                total_loss = loss + \
                    args.params.normal_lambda*normal_loss + \
                    args.params.floater_lambda*floater_loss + \
                    args.params.backwards_rays_lambda*backwards_rays_loss + \
                    args.params.diffuse_lambda*diffuse_loss

                if Ortho_reg_weight > 0:
                    loss_reg = tensorf.rf.vector_comp_diffs()
                    total_loss += Ortho_reg_weight*loss_reg
                    summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
                if L1_reg_weight > 0:
                    loss_reg_L1 = tensorf.rf.density_L1()
                    total_loss += L1_reg_weight*loss_reg_L1
                    summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

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
                optimizer.step()
            # now train on the environment maps
            env_loss = torch.tensor(1.0)
            torch.cuda.empty_cache()
            if args.params.num_env_train_iters > 0 and \
               iteration > args.params.envmap_train_start and args.params.envmap_train_start > 0 and \
               tensorf.ref_module is not None:
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    # sample termination_xyz
                    with torch.no_grad():
                        termination_xyz = data['termination_xyz']
                        normal_map = data['normal_map']
                        #  termination_xyz = rays_train[:, :4]
                        L = args.params.num_rays_per_envmap
                        H = 1
                        M = args.batch_size // L

                        # num per ray to allow smoothing
                        inds = np.random.permutation(args.batch_size)[:M]
                        # remember the difference between tile and repeat
                        env_rays_o = termination_xyz[inds][:, :3] \
                            .reshape(M, 1, 1, 3) \
                            .expand(M, L, H, 3) \
                            .reshape(-1, 3) \
                            .to(device)
                        outward = normal_map[inds] \
                            .reshape(M, 1, 3) \
                            .expand(M, L, 3) \
                            .reshape(-1, 3) \
                            .to(device)

                        env_rays_d = torch.rand(M*L, 3, device=device)
                        env_rays_d /= torch.linalg.norm(env_rays_d, dim=1, keepdim=True)+1e-6

                        # project env_rays_d onto half space
                        similarity = torch.sum(env_rays_d * outward, dim=1, keepdim=True)
                        ortho_component = env_rays_d - similarity * outward
                        env_rays_d = ortho_component + similarity.abs() * outward

                        env_rays_d = env_rays_d.reshape(M, L, 1, 3)
                        env_rays_d = env_rays_d / torch.linalg.norm(env_rays_d, axis=-1, keepdim=True)
                        env_rays_d = env_rays_d.reshape(-1, 3)

                        env_rays_up = torch.rand(args.batch_size, 3, device=device)
                        env_rays_up /= torch.linalg.norm(env_rays_up, dim=1, keepdim=True)

                        env_rays = torch.cat([env_rays_o, env_rays_d, env_rays_up], dim=1)

                    # right? I don't want a gradient attached
                    with torch.no_grad():
                        env_data = renderer(env_rays, tensorf, keys = ['rgb_map'],
                                focal=focal, output_alpha=alpha_train, chunk=args.batch_size,
                                N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, is_train=True)

                    for _ in range(args.params.num_env_train_iters):
                        render_rgb = env_data['rgb_map']
                        pred_rgb = tensorf.render_env_sparse(
                                termination_xyz[inds][:, :3].to(device),
                                env_rays_d.reshape(M, L * H, 3), roughness=20)

                        env_loss = torch.sqrt((render_rgb-pred_rgb)**2+args.params.charbonier_eps**2).mean()
                        optimizer.zero_grad()
                        env_loss.backward()
                        optimizer.step()

                    """
                    if iteration % 100 == 0:

                        res = 200
                        ele_grid, azi_grid = torch.meshgrid(
                            torch.linspace(-np.pi/2, np.pi/2, res, dtype=torch.float32),
                            torch.linspace(-np.pi, np.pi, 2*res, dtype=torch.float32), indexing='ij')
                        # each col of x ranges from -pi/2 to pi/2
                        # each row of y ranges from -pi to pi
                        ang_vecs = torch.stack([
                            torch.cos(ele_grid) * torch.cos(azi_grid),
                            torch.cos(ele_grid) * torch.sin(azi_grid),
                            -torch.sin(ele_grid),
                        ], dim=-1).to(device)
                        env_rays_d = ang_vecs.reshape(-1, 3)
                        mask = env_rays_d @ outward[0] < 0
                        env_rays_d[mask, 0] = 0
                        env_rays_d[mask, 1] = 0
                        env_rays_d[mask, 2] = 1

                        #  similarity = torch.sum(env_rays_d * outward[0], dim=1, keepdim=True)
                        #  ortho_component = env_rays_d - similarity * outward[0:1]
                        #  env_rays_d = ortho_component + similarity.abs() * outward[0:1]

                        env_rays = torch.cat([
                            env_rays_o[0].reshape(1, 3).repeat(env_rays_d.shape[0], 1),
                            env_rays_d, torch.rand(*env_rays_d.shape, device=env_rays_o.device)], dim=1)
                        with torch.no_grad():
                            env_data = renderer(env_rays, tensorf,
                                    keys = ['rgb_map'],
                                    focal=focal, output_alpha=alpha_train, chunk=args.batch_size,
                                    N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, is_train=True)
                        raw_env_map = env_data['rgb_map'].detach()
                        env_map = (raw_env_map.cpu().numpy().reshape(res, 2*res, 3)*255).astype(np.uint8)
                        env_map = cv2.cvtColor(env_map, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(f'env_map{iteration}.png', env_map)

                        pbar2 = tqdm(range(500), miniters=args.progress_refresh_rate, file=sys.stdout)
                        for _ in pbar2:
                            pred_rgb = tensorf.render_env_sparse(
                                    env_rays_o[0].reshape(1, 3),
                                    env_rays_d[~mask].reshape(1, -1, 3), roughness=20)
                            env_loss = torch.sqrt((raw_env_map[~mask]-pred_rgb)**2+args.params.charbonier_eps**2).mean()
                            optimizer.zero_grad()
                            env_loss.backward()
                            optimizer.step()
                            pbar2.set_description(f'loss: {env_loss.detach().item():.4f}')

                        pred_env_map, _ = tensorf.recover_envmap(res, termination_xyz[inds][0])
                        pred_env_map = (pred_env_map.detach().cpu().numpy() * 255).astype('uint8')
                        pred_env_map = cv2.cvtColor(pred_env_map, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(f'pred_env_map{iteration}.png', pred_env_map)
                        #  pred_env_map = pred_rgb.reshape(M, -1, 3)[0].detach().cpu().numpy().reshape(res, 2*res, 3)
                        #  fig, axs = plt.subplots(2)
                        #  axs[0].imshow(env_map)
                        #  axs[1].imshow(pred_env_map)
                        #  plt.show()
                    """

            #  total_loss += 2*env_loss
            env_loss = env_loss.detach().item()

            #  optimizer.zero_grad()
            #  total_loss.backward()
            #  optimizer.step()

            photo_loss = photo_loss.detach().item()
            
            PSNRs.append(-10.0 * np.log(photo_loss) / np.log(10.0))
            summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
            summary_writer.add_scalar('train/mse', photo_loss, global_step=iteration)

            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_factor

            # Print the current values of the losses.
            if iteration % args.progress_refresh_rate == 0:
                pbar.set_description(
                    f'Iteration {iteration:05d}:'
                    + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                    + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                    + f' env_psnr = {-10 * np.log(float(env_loss**2))/np.log(10.0):.4f}'
                    + f' normal_loss = {normal_loss:.5f}'
                    + f' floater_loss = {floater_loss:.6f}'
                    + f' mse = {photo_loss:.6f}'
                )
                PSNRs = []
                

            if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
                tensorf.save(f'{logfolder}/{args.expname}_{iteration}.th', args.model)
                PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                        prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray,
                                        compute_extra_metrics=False, render_mode=args.render_mode)
                summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)


            if upsamp_bg and iteration in args.params.bg_upsamp_list:
                res = args.params.bg_upsamp_res.pop(0)
                lr_bg = args.params.bg_upsamp_lr.pop(0)
                print(f"Upsampling bg to {res}")
                tensorf.bg_module.upsample(res)
                ind = [i for i, d in enumerate(grad_vars) if 'name' in d and d['name'] == 'bg'][0]
                grad_vars[ind]['params'] = tensorf.bg_module.parameters()
                grad_vars[ind]['lr'] = lr_bg
                optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

            if iteration in args.params.bounce_iter_list:
                print(f"Max bounces {tensorf.max_bounce_rays} -> {bounce_n_list[0]}")
                tensorf.max_bounce_rays = bounce_n_list.pop(0)
            if iteration in update_AlphaMask_list:

                #  if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                reso_mask = reso_cur
                new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
                if iteration == update_AlphaMask_list[0]:
                    apply_correction = not torch.all(tensorf.alphaMask.grid_size == tensorf.rf.grid_size)
                    tensorf.shrink(new_aabb, apply_correction)
                    # tensorVM.alphaMask = None
                    L1_reg_weight = args.params.L1_weight_rest
                    print("continuing L1_reg_weight", L1_reg_weight)


                if not ndc_ray and iteration == update_AlphaMask_list[1] and args.filter_rays:
                    # filter rays outside the bbox
                    allrays,allrgbs,mask = tensorf.filtering_rays(allrays, allrgbs, focal)
                    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

            if iteration in args.params.smoothing_list:
                sval = smoothing_vals.pop(0)
                tensorf.rf.set_smoothing(sval)

            if iteration in uplambda_list:
                tensorf.l = l_list.pop(0)
                print(f"Setting normal lambda to {tensorf.l}")

            if iteration in upsamp_list:
                n_voxels = N_voxel_list.pop(0)
                reso_cur = N_to_reso(n_voxels, tensorf.rf.aabb)
                nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio/tensorf.rf.density_res_multi))
                tensorf.rf.upsample_volume_grid(reso_cur)


                if args.lr_upsample_reset:
                    print("reset lr to initial")
                    lr_scale = 1 #0.1 ** (iteration / args.n_iters)
                else:
                    lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
                grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale, lr_bg=lr_bg)
                optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
    # prof.export_chrome_trace('trace.json')
        

    tensorf.save(f'{logfolder}/{args.expname}.th', args.model)


    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device, render_mode=args.render_mode)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device, render_mode=args.render_mode)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                        N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device,
                        render_mode=args.render_mode)


@hydra.main(version_base=None, config_path='configs', config_name='default')
def train(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)
    print(cfg.dataset)
    print(cfg.model)
    print(cfg.params)

    if cfg.render_only:
        render_test(cfg)
    else:
        reconstruction(cfg)
        # reconstruction(args)

if __name__ == '__main__':
    train()
