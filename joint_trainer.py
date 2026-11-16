import os
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from kornia.morphology import dilation, erosion
from kornia.filters import gaussian_blur2d

from dataset.cameras import Camera
from networks.gshair.hairwrapper import GSHairWrapper
from networks.meshface.facewrapper import MeshFaceWrapper
from utils import AverageMeter, CUDA_Timer, color_mask, directory, img2mask, restore_model, ssim, update_lambda, visimg

from exp.lfimg_loss import compute_lfimg_loss
from exp.lfuv_loss import compute_lfuv_loss
from exp.uv_seam_loss import compute_seam_loss
from exp.hfimg_loss import compute_hfimg_loss
from exp.masks import get_face_vis_mask
from exp.uv_bake import bake_to_uv_functional, get_uv_resolution, make_face_px_mask
from exp.cross_view_uv_loss import bake_attribute_to_uv, compute_cross_view_uv_loss
from exp.config_utils import validate_hf_image_config
from exp.fresco_stage_schedule import apply_subject460_stage, schedule_enabled
from geom.seam_pairs import build_seam_pair_maps, save_seam_maps, load_seam_maps


class JointTrainer:
    def __init__(self, config, logger, spatial_lr_scale, all_flame_params=None, painting=False, is_val=False):
        print("[DEBUG] JointTrainer.__init__ started")
        logger.info("[DEBUG] JointTrainer.__init__ started")

        self.config = config
        self.neural = config.get("training.neural_texture", True)
        self.img_h, self.img_w = config["data.img_h"], config["data.img_w"]
        self.rate_h, self.rate_w = self.img_h / 802.0, self.img_w / 550.0
        self.rate = min(self.rate_h, self.rate_w)
        self.nan_detect = False
        self.is_val = is_val
        self.spatial_lr_scale = spatial_lr_scale
        self.gs_pretrain = config["gs.pretrain"]
        self.lr = config["training.learning_rate"]
        self.alter_hair = False
        self.stages = config["training.stages"]
        config["training.stages_epoch"] = (
            [] if None in config["training.stages_epoch"] else config["training.stages_epoch"]
        )
        self.stages_epoch = [0] + config["training.stages_epoch"] + [1e10]
        assert (
            len(self.stages_epoch) - len(self.stages)
        ) >= -1, (
            "[ERROR] The length of 'training.stages_epoch' should be larger than the length of 'training.stages' - 1."
        )
        assert (
            self.gs_pretrain is not None or config["bald"]
        ), "[ERROR] You need set 'gs.pretrain' to pretrained neutral hair ckpt."

        self.xyz_cond = config.get("flame.xyz_cond", False)
        self.move_eyes = config.get("flame.move_eyes", False)

        self.parameters_to_train = []
        print("[DEBUG] About to call _init_nets")
        logger.info("[DEBUG] About to call _init_nets")
        self._init_nets(painting)
        print("[DEBUG] _init_nets completed")
        logger.info("[DEBUG] _init_nets completed")

        # set optimizer
        print("[DEBUG] About to create optimizer")
        logger.info("[DEBUG] About to create optimizer")
        self.optimizer = torch.optim.Adam(self.parameters_to_train, eps=1e-15)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=config["training.step"], gamma=0.1
        )
        print("[DEBUG] Optimizer created")
        logger.info("[DEBUG] Optimizer created")
        if all_flame_params is not None:
            self.init_all_flame_params(all_flame_params)

        # Restore checkpoint
        checkpoint_path = (
            os.path.join(config["local_workspace"], "checkpoint_latest.pth")
            if config["training.pretrained_checkpoint_path"] is None
            else config["training.pretrained_checkpoint_path"]
        )

        # The Subject 460 curriculum is explicitly 0-based. Preserve the
        # original 1-based epoch semantics for unrelated training runs.
        self.current_epoch = 0 if schedule_enabled(config) else 1
        self.global_step = 0
        self.stage = self.stages[0]
        self.stage_step = 0
        explicit_checkpoint = config["training.pretrained_checkpoint_path"] is not None
        restore_training_state = not self.is_val
        strict_resume = restore_training_state and bool(
            config.get("training.require_resume_state", False)
        )
        if explicit_checkpoint and strict_resume and not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Requested training checkpoint does not exist: {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            self.current_epoch, self.global_step, stage, stage_step = restore_model(
                checkpoint_path, self.hairwrapper, self.facewrapper, self.optimizer, logger,
                scheduler=self.scheduler,
                keep_configured_lrs=bool(
                    config.get("training.resume_keep_configured_lrs", explicit_checkpoint)
                ),
                require_training_state=strict_resume,
                restore_training_state=restore_training_state,
            )

            # load optimized flame params
            dir_name = os.path.dirname(checkpoint_path)
            opt_flame_params = np.load(os.path.join(dir_name, "flame_params.npz"))
            if not self.is_val:
                self.load_all_flame_params(opt_flame_params)
            else:
                self.all_flame_params["shape"].data = torch.from_numpy(opt_flame_params["shape"]).cuda()

            if stage is not None:
                self.stage = stage
                self.stage_step = stage_step

        self.logger = logger
        self.tb_writer = config.get("tb_writer", None)
        
        # J1/J2: Add is_main flag for distributed training
        self.is_main = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)

        print("[DEBUG] JointTrainer: About to call _init_data")
        self._init_data()
        print("[DEBUG] JointTrainer: _init_data completed")
        print("[DEBUG] JointTrainer: About to call _init_losses")
        self._init_losses()
        print("[DEBUG] JointTrainer: _init_losses completed")
        print("[DEBUG] JointTrainer: About to call _set_stage")
        self._set_stage(self.stage)
        print("[DEBUG] JointTrainer: _set_stage completed")

        # find all lambda
        self.all_lambdas = {}
        prelen = len("training.lambda_")
        for k, v in self.config.items():
            if "lambda" not in k or "lambda_update_list" in k:
                continue
            self.all_lambdas[k[prelen:]] = v

    def _freeze(self, label):
        for group in self.optimizer.param_groups:
            if label in group["name"] or label == "all":
                group["params"][0].requires_grad = False

    def _unfreeze(self, label):
        for group in self.optimizer.param_groups:
            if label in group["name"] or label == "all":
                group["params"][0].requires_grad = True

    def _set_stage(self, stage):
        if stage == "joint":
            self._freeze("all")

            # load canonical hair
            state_dict = torch.load(self.gs_pretrain, map_location=lambda storage, loc: storage.cpu())
            _state_dict = {
                k.replace("module.", "") if k.startswith("module.") else k: v
                for k, v in state_dict["canonical_gs"].items()
            }
            self.hairwrapper.get_model("canonical_gs").load_state_dict(
                _state_dict, self.optimizer, self.global_step, self.config["gs.upSH"]
            )
            # learn deformation field & head tex
            self._unfreeze("hair")
            self._unfreeze("head_tex")
        elif stage == "head":
            # learn facial mesh
            self._freeze("all")
            self._unfreeze("head")
        elif stage == "painting":
            self._freeze("all")
            self._unfreeze("head_tex_basic")
            self._unfreeze("head_tex_mlp")
        elif stage == "painting_code":
            self._freeze("all")
            self._unfreeze("head_tex_basic")
        elif stage == "painting_mlp":
            self._freeze("all")
            self._unfreeze("head_tex_mlp")
        else:
            self.logger.info("Unknown training stage: {}".format(stage))
            exit(1)

    def _init_nets(self, painting=False):
        print("[DEBUG] _init_nets: Loading init_pts")
        init_pts = np.load(self.config["gs.init_pts"])
        print("[DEBUG] _init_nets: Creating GSHairWrapper")
        self.hairwrapper = GSHairWrapper(self.config, init_pts, self.spatial_lr_scale)
        print("[DEBUG] _init_nets: Creating MeshFaceWrapper")
        self.facewrapper = MeshFaceWrapper(self.config, self.move_eyes, self.xyz_cond, painting=painting)
        print("[DEBUG] _init_nets: Both wrappers created")

        # Initialize seam maps for UV seam regularization
        self._init_seam_maps()

        self.parameters_to_train = self.hairwrapper.get_optim_params() + self.facewrapper.get_optim_params()

    def _init_seam_maps(self):
        """
        Initialize seam maps for UV seam regularization.
        Loads pre-computed seam maps or generates them if not available.
        """
        if not self.config.get("exp.lf_uv.seam.enable", False):
            self.seam_maps = {}
            return
        try:
            # Get UV resolution from config
            uv_res = self.config.get('exp.lf_uv.uv_res', 1024)
            uv_hw = (uv_res, uv_res)
            
            # Define cache path
            cache_dir = self.config.get("exp.lf_uv.seam.cache_dir", "cache/uv")
            os.makedirs(cache_dir, exist_ok=True)
            # v2 fixes cross-chart endpoint orientation and stores discrete
            # paired texel centers. Do not reuse maps generated by the old rule.
            cache_path = os.path.join(cache_dir, f"seam_maps_v2_{uv_res}x{uv_res}.pt")
            
            # Try to load existing seam maps
            if os.path.exists(cache_path):
                print(f"[SEAM_MAPS] Loading existing seam maps from {cache_path}")
                self.seam_maps = load_seam_maps(cache_path)
            else:
                print(f"[SEAM_MAPS] Generating new seam maps for resolution {uv_hw}")
                
                # Get mesh data from facewrapper
                face_dec = getattr(self.facewrapper, 'flame_dec', None)
                if face_dec is None:
                    raise RuntimeError("FLAME decoder is unavailable")
                
                # Get mesh data
                faces = getattr(face_dec, 'faces', None)
                vertex_uvs = getattr(face_dec, 'vertex_uvs', None)
                faces_uvs = getattr(face_dec, 'faces_uvs', None)
                
                if any(value is None for value in (faces, vertex_uvs, faces_uvs)):
                    raise RuntimeError("FLAME mesh topology or UV data is unavailable")
                
                # Generate seam maps
                self.seam_maps = build_seam_pair_maps(
                    faces=faces,
                    vertex_uvs=vertex_uvs,
                    faces_uvs=faces_uvs,
                    uv_hw=uv_hw,
                    max_step_px=1
                )
                
                # Save for future use
                save_seam_maps(self.seam_maps, cache_path)
                print(f"[SEAM_MAPS] Generated and saved seam maps to {cache_path}")
            
            # Move to GPU if needed (will be done when used)
            print(f"[SEAM_MAPS] Seam maps initialized with shape: {self.seam_maps['pair_valid'].shape}")
            
        except Exception as e:
            raise RuntimeError(f"Seam loss is enabled but seam maps could not be initialized: {e}") from e

    def _init_data(self):
        B, H, W = (
            self.config["data.per_gpu_batch_size"],
            self.config["data.img_h"],
            self.config["data.img_w"],
        )

        self.img = torch.zeros((B, H, W, 3), dtype=torch.float32).cuda()
        self.view = torch.zeros((B, 3, 8, 8), dtype=torch.float32).cuda()

        self.mask = {}
        self.mask["full"] = torch.zeros((B, H, W), dtype=torch.float32).cuda()
        self.mask["hair"] = torch.zeros((B, H, W), dtype=torch.float32).cuda()
        self.mask["head"] = torch.zeros((B, H, W), dtype=torch.float32).cuda()
        self.mask["erode_hair"] = torch.zeros((B, H, W), dtype=torch.float32).cuda()

        self.depth_map = torch.zeros((B, H, W), dtype=torch.float32).cuda()

        # uv --> mesh vertices, barycentric
        self.uv2verts_ids = np.load("/app/data/face-data/unwrap_uv_idx_v_idx.npy")
        self.uv2verts_bw = np.load("/app/data/face-data/unwrap_uv_idx_bw.npy")

    def _init_losses(self):
        self.train_losses = {
            "loss": AverageMeter("train_loss"),
            "loss_pho/rgb.obj": AverageMeter("train_rgb_obj_loss"),
            "loss_pho/rgb.hair": AverageMeter("train_rgb_hair_loss"),
            "loss_pho/rgb.head": AverageMeter("train_rgb_head_loss"),
            "loss_pho/rgb.basic_head": AverageMeter("train_rgb_basic_head_loss"),
            "loss_geo/silh.hair": AverageMeter("train_silh_hair_loss"),
            "loss_geo/depth.head": AverageMeter("train_depth_head_loss"),
            "loss_geo/normal.head": AverageMeter("train_normal_head_loss"),
            "loss_pho/ssim.obj": AverageMeter("train_ssim_obj_loss"),
            "loss_pho/ssim.hair": AverageMeter("train_ssim_hair_loss"),
            "loss_pho/ssim.head": AverageMeter("train_ssim_head_loss"),
            "loss_reg/mesh.laplacian": AverageMeter("train_mesh_laplacian_loss"),
            "loss_reg/mesh.normal": AverageMeter("train_mesh_normal_loss"),
            "loss_reg/mesh.edges": AverageMeter("train_mesh_edges_loss"),
            "loss_reg/mesh.vscale": AverageMeter("train_mesh_vscale_loss"),
            "loss_reg/silh.solid_hair": AverageMeter("train_silh_solid_hair_loss"),
        }
        self.val_losses = {
            "loss": AverageMeter("val_loss"),
            "metrics/mse": AverageMeter("metrics.mse"),
            "metrics/psnr": AverageMeter("metrics.psnr"),
            "loss_pho/rgb.obj": AverageMeter("val_rgb_obj_loss"),
            "loss_pho/rgb.hair": AverageMeter("val_rgb_hair_loss"),
            "loss_pho/rgb.head": AverageMeter("val_rgb_head_loss"),
            "loss_pho/rgb.basic_head": AverageMeter("val_rgb_basic_head_loss"),
            "loss_geo/silh.hair": AverageMeter("val_silh_hair_loss"),
            "loss_geo/depth.head": AverageMeter("val_depth_head_loss"),
            "loss_geo/normal.head": AverageMeter("val_normal_head_loss"),
            "loss_pho/ssim.obj": AverageMeter("val_ssim_obj_loss"),
            "loss_pho/ssim.hair": AverageMeter("val_ssim_hair_loss"),
            "loss_pho/ssim.head": AverageMeter("val_ssim_head_loss"),
            "loss_reg/mesh.laplacian": AverageMeter("val_mesh_laplacian_loss"),
            "loss_reg/mesh.normal": AverageMeter("val_mesh_normal_loss"),
            "loss_reg/mesh.edges": AverageMeter("val_mesh_edges_loss"),
            "loss_reg/mesh.vscale": AverageMeter("val_mesh_vscale_loss"),
            "loss_reg/silh.solid_hair": AverageMeter("val_silh_solid_hair_loss"),
        }
        self.fresco_losses = {
            "lf_img_raw": AverageMeter("train_lf_img_raw"),
            "hf_img_raw": AverageMeter("train_hf_img_raw"),
            "lf_uv/raw": AverageMeter("train_cross_view_uv_raw"),
            "lf_uv_seam/raw": AverageMeter("train_uv_seam_raw"),
        }

    def set_train(self):
        """Convert models to training mode"""
        self.hairwrapper.set_train()
        self.facewrapper.set_train()

    def set_eval(self):
        """Convert models to evaluation mode"""
        self.hairwrapper.set_eval()
        self.facewrapper.set_eval()

    def load_hair(self, ckpt):
        self.logger.info("\n\nLoading hairstyle ...")

        state_dict = torch.load(ckpt, map_location=lambda storage, loc: storage.cpu())
        for key in ["canonical_gs", "deform_mlp"]:
            model = self.hairwrapper.get_model(key)
            _state_dict = {
                k.replace("module.", "") if k.startswith("module.") else k: v for k, v in state_dict[key].items()
            }
            missing_in_model = set(_state_dict.keys()) - set(model.state_dict().keys())
            missing_in_ckp = set(model.state_dict().keys()) - set(_state_dict.keys())

            if self.logger:
                self.logger.info("[MODEL_RESTORE] missing keys in %s checkpoint: %s" % (key, missing_in_ckp))
                self.logger.info("[MODEL_RESTORE] missing keys in %s model: %s" % (key, missing_in_model))

            try:
                if key == "canonical_gs":
                    model.load_state_dict(_state_dict, self.optimizer, self.global_step, self.config["gs.upSH"])
                else:
                    model.load_state_dict(_state_dict, strict=False)
            except Exception as error:
                if self.logger:
                    self.logger.info("[warning] {} weights are not loaded.".format(key))
                else:
                    print("[warning] {} weights are not loaded.".format(key))
        self.alter_hair = True

    def update_stage(self):
        try:
            cur_id = next(i for i, v in enumerate(self.stages_epoch) if v > self.current_epoch)
        except:
            cur_id = -1
        new_stage = self.stages[cur_id - 1]
        if new_stage != self.stage:
            # save the final ckpt of the current stage
            savepath = os.path.join(
                self.config["local_workspace"], "checkpoint_{}_it{}.pth".format(self.stage, self.global_step)
            )
            self.save_ckpt(savepath)

            self.stage = new_stage
            self.stage_step = 0
            self._set_stage(new_stage)

    def _apply_epoch_fresco_schedule(self):
        if not schedule_enabled(self.config):
            return None
        previous_seam = bool(self.config.get("exp.lf_uv.seam.enable", False))
        stage = apply_subject460_stage(self.config, self.current_epoch)
        seam_enabled = bool(self.config.get("exp.lf_uv.seam.enable", False))
        if seam_enabled and not self.seam_maps:
            self._init_seam_maps()
        elif previous_seam and not seam_enabled:
            self.seam_maps = {}
        return stage

    def _log_fresco_stage(self, stage, train_loader):
        if stage is None or getattr(self, "_fresco_stage_label", None) == stage.label:
            return
        self._fresco_stage_label = stage.label
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        sampler_name = type(batch_sampler).__name__ if batch_sampler is not None else type(train_loader.sampler).__name__
        self.logger.info(
            "FReSCO stage %s epoch=%d LF=%s/%.4f UV=%s/%.4f "
            "Seam=%s/pair=%.4f HF=%s/%.4f sampler=%s checkpoint_epoch=%d "
            "global_step=%d optimizer_lrs=%s",
            stage.label, self.current_epoch,
            self.config["exp.lf_img.enable"], self.config["exp.lf_img.lambda"],
            self.config["exp.lf_uv.loss.enable"], self.config["exp.lf_uv.loss.weight"],
            self.config["exp.lf_uv.seam.enable"], self.config["exp.lf_uv.seam.weight_pair"],
            self.config["exp.hf_img.enable"], self.config["exp.hf_img.lambda"],
            sampler_name, self.current_epoch, self.global_step,
            [group["lr"] for group in self.optimizer.param_groups],
        )

    def train(self, train_loader, val_loader, show_time=False, train_loader_provider=None):
        print("[DEBUG] JointTrainer.train: Starting training")
        self.logger.info("[DEBUG] JointTrainer.train: Starting training")
        torch.cuda.empty_cache()
        def training_not_finished():
            if schedule_enabled(self.config):
                return self.current_epoch < self.config["training.epochs"]
            return self.current_epoch <= self.config["training.epochs"]

        while training_not_finished():
            fresco_stage = self._apply_epoch_fresco_schedule()
            if train_loader_provider is not None:
                train_loader = train_loader_provider(self.current_epoch, self.config)
            self._log_fresco_stage(fresco_stage, train_loader)
            print("[DEBUG] JointTrainer.train: About to call update_stage")
            self.logger.info("[DEBUG] JointTrainer.train: About to call update_stage")
            self.update_stage()
            print("[DEBUG] JointTrainer.train: update_stage completed")
            self.logger.info("[DEBUG] JointTrainer.train: update_stage completed")
            print("[DEBUG] JointTrainer.train: About to call train_epoch")
            self.logger.info("[DEBUG] JointTrainer.train: About to call train_epoch")
            success = self.train_epoch(train_loader, val_loader, show_time)
            if not success:
                return

            self.scheduler.step()
            self.logger.info("Epoch finished, average losses: ")
            for v in self.train_losses.values():
                self.logger.info("    {}".format(v))
            for v in self.fresco_losses.values():
                self.logger.info("    {}".format(v))
            self.current_epoch += 1
            # Stage scripts resume from checkpoint_latest. Save after incrementing
            # so the next phase starts at the next epoch rather than repeating the
            # just-completed boundary epoch.
            checkpoint_path = os.path.join(self.config["local_workspace"], "checkpoint_latest.pth")
            self.save_ckpt(checkpoint_path)
            self.logger.info(
                "Epoch boundary checkpoint saved: next_epoch=%d path=%s",
                self.current_epoch, checkpoint_path,
            )

    def set_data(self, items):
        self.batch_size = items["view"].shape[0]

        self.view.resize_as_(items["view"]).copy_(items["view"])
        self.img.resize_as_(items["img"]).copy_(items["img"])

        self.mask["full"].resize_as_(items["obj_mask"]).copy_(items["obj_mask"])
        self.mask["hair"].resize_as_(items["hair_mask"]).copy_(items["hair_mask"])
        self.mask["head"].resize_as_(items["head_mask"]).copy_(items["head_mask"])
        self.mask["erode_hair"].resize_as_(items["erode_hair_mask"]).copy_(items["erode_hair_mask"])

        self.depth_map.resize_as_(items["depth_map"]).copy_(items["depth_map"])

        # driving flame params
        self.flame_params = {}
        frame_idx = items["frame_idx"]
        self.frame_idx = frame_idx
        self.camera_ids = list(items["cam"])
        for key, val in self.all_flame_params.items():
            if key == "shape":
                self.flame_params[key] = val.expand(self.batch_size, -1)
            else:
                self.flame_params[key] = val[frame_idx]

        # build cameras
        self.camera = []
        for i in range(self.batch_size):
            camera = Camera(
                R=items["w2c"][i, :3, :3],
                t=items["w2c"][i, :3, 3],
                intr=items["intr"][i],
                zfar=100.0,
                znear=0.01,
                img_h=self.img_h,
                img_w=self.img_w,
                name=items["cam"][i],
            )
            self.camera.append(camera)
        self.proj_R = items["proj_w2c"][:, :3, :3].float().cuda()
        self.proj_t = items["proj_w2c"][:, :3, 3].float().cuda()
        self.intr = items["intr"].float().cuda()
        self.intr[:, 0, 0] *= -1

        self.name = items["name"]

    def load_cameras(self, T_c2w, items):
        """Used to render free views"""
        for key, val in items.items():
            if isinstance(val, np.ndarray):
                items[key] = torch.from_numpy(val).cuda()

        self.camera = []
        for i in range(T_c2w.shape[0]):
            w2c = torch.from_numpy(np.linalg.inv(T_c2w[i])).float().cuda()
            camera = Camera(
                R=w2c[:3, :3],
                t=w2c[:3, 3],
                intr=items["intr"][i],
                zfar=100.0,
                znear=0.01,
                img_h=self.img_h,
                img_w=self.img_w,
                name=items["cam"][i],
            )
            self.camera.append(camera)

        views = []
        for c2w in T_c2w:
            campos = c2w[:3, 3]
            view = campos / np.linalg.norm(campos)
            views.append(np.tile(view, (8, 8, 1)).transpose((2, 0, 1)))
        self.view = torch.from_numpy(np.stack(views, axis=0)).float().cuda()

    def init_all_flame_params(self, flame_params):
        # learnable flame params
        T = max(list(flame_params.keys())) + 1
        m_id = min(list(flame_params.keys()))
        self.all_flame_params = {
            "shape": torch.from_numpy(flame_params[m_id]["shape"])[None],
            "expr": torch.zeros([T, flame_params[m_id]["expr"].shape[1]]),
            "rotation": torch.zeros([T, 3]),
            "neck_pose": torch.zeros([T, 3]),
            "jaw_pose": torch.zeros([T, 3]),
            "eyes_pose": torch.zeros([T, 6]),
            "translation": torch.zeros([T, 3]),
        }

        for i, param in flame_params.items():
            self.all_flame_params["expr"][i] = torch.from_numpy(param["expr"])
            self.all_flame_params["rotation"][i] = torch.from_numpy(param["rotation"])
            self.all_flame_params["neck_pose"][i] = torch.from_numpy(param["neck_pose"])
            self.all_flame_params["jaw_pose"][i] = torch.from_numpy(param["jaw_pose"])
            self.all_flame_params["eyes_pose"][i] = torch.from_numpy(param["eyes_pose"])
            self.all_flame_params["translation"][i] = torch.from_numpy(param["translation"])

        for k, v in self.all_flame_params.items():
            self.all_flame_params[k] = v.float().cuda()

        optimize_params = self.config.get("flame.optimize_params", False)
        if (not self.is_val) and optimize_params:
            flame_lrs = {"shape": 1e-5, "expr": 1e-3, "pose": 1e-5, "translation": 1e-6}

            # shape
            self.all_flame_params["shape"].requires_grad = True
            param_shape = {
                "params": [self.all_flame_params["shape"]],
                "lr": flame_lrs["shape"],
                "name": "head.flame_shape",
            }
            self.optimizer.add_param_group(param_shape)

            # expression
            self.all_flame_params["expr"].requires_grad = True
            param_expr = {"params": [self.all_flame_params["expr"]], "lr": flame_lrs["expr"], "name": "head.flame_expr"}
            self.optimizer.add_param_group(param_expr)

            # pose
            self.all_flame_params["rotation"].requires_grad = True
            self.all_flame_params["neck_pose"].requires_grad = True
            self.all_flame_params["jaw_pose"].requires_grad = True
            self.all_flame_params["eyes_pose"].requires_grad = True
            params = [
                self.all_flame_params["rotation"],
                self.all_flame_params["neck_pose"],
                self.all_flame_params["jaw_pose"],
                self.all_flame_params["eyes_pose"],
            ]
            param_pose = {"params": params, "lr": flame_lrs["pose"], "name": "head.flame_pose"}
            self.optimizer.add_param_group(param_pose)

            # translation
            self.all_flame_params["translation"].requires_grad = True
            param_trans = {
                "params": [self.all_flame_params["translation"]],
                "lr": flame_lrs["translation"],
                "name": "head.flame_trans",
            }
            self.optimizer.add_param_group(param_trans)

    def load_all_flame_params(self, all_flame_params):
        if not hasattr(self, "all_flame_params"):
            self.all_flame_params = {
                key: torch.as_tensor(value, dtype=torch.float32, device="cuda")
                for key, value in all_flame_params.items()
            }
            return
        missing = set(self.all_flame_params) - set(all_flame_params)
        if missing:
            raise RuntimeError(f"Restored FLAME parameters are missing keys: {sorted(missing)}")
        with torch.no_grad():
            for key, destination in self.all_flame_params.items():
                source = torch.as_tensor(
                    all_flame_params[key], dtype=destination.dtype, device=destination.device
                )
                if source.shape != destination.shape:
                    raise RuntimeError(
                        f"Restored FLAME shape mismatch for {key}: "
                        f"checkpoint={tuple(source.shape)} current={tuple(destination.shape)}"
                    )
                # Preserve the Tensor objects referenced by optimizer groups and
                # their requires_grad flags while restoring checkpoint values.
                destination.copy_(source)

    def load_flame_params(self, flame_params):
        """Used to driving from another flame params"""
        self.flame_params = flame_params
        for key, val in flame_params.items():
            if isinstance(val, np.ndarray):
                self.flame_params[key] = torch.from_numpy(val).float().cuda()
            elif isinstance(val, torch.Tensor):
                self.flame_params[key] = val.float().cuda()

    def neural2rgb(self, neural_features, valid):
        """Decode neural image to RGB image

        Args:
            neural_features: Nx15, N is the num of valid pixels.
            valid:
        """
        B, H, W = valid.shape
        colors = self.models["head_mlp"](neural_features)

        rgb = torch.ones((B, H, W, 3)).float().cuda()
        rgb[valid] = colors

        return rgb

    def compare_depth(self, hair_depth, head_depth):
        hair_nz, head_nz = (
            torch.ones_like(hair_depth).cuda() * 1e10,
            torch.ones_like(head_depth).cuda() * 1e10,
        )
        valid_hair, valid_head = hair_depth > 0, head_depth > 0
        hair_nz[valid_hair] = hair_depth[valid_hair]
        head_nz[valid_head] = head_depth[valid_head]
        hair_mask = (head_nz > hair_nz).int()

        return hair_mask

    def fuse(self, rasterized_hair, rasterized_face, is_val=False):
        """To fuse hair image and head image"""
        # ablation options
        gsdepth = self.config.get("ab.gsdepth", False)
        hardblend = self.config.get("ab.hardblend", False)
        usemorph = self.config.get("training.usemorph", False)

        has_face = rasterized_face is not None
        has_hair = rasterized_hair is not None
        need_fusion = has_hair and has_face
        assert has_face or has_hair, "Please render face, or hair, or both of them. "

        outputs = {}
        # 1. Hair processing, [B, H, W, 3]
        if has_hair:
            rendered_hair = rasterized_hair["render"].permute((0, 2, 3, 1))

        # 2. Head processing
        if has_face:
            valid_pixels = rasterized_face["valid_nograd"]
            s_id = 3 if self.xyz_cond else 0
            rasterized_xyz, rasterized_uv = (
                rasterized_face["neural_img"][..., :s_id],
                rasterized_face["neural_img"][..., s_id : s_id + 2],
            )  # [N, 3], [N, 2], [N, tex_ch], N is the num of valid pixels.
            tex_ch = self.config["training.tex_ch"] if self.neural else 3
            neural_features = rasterized_face["neural_features"]
            rasterized_features = F.grid_sample(neural_features, rasterized_uv, mode="bilinear", align_corners=True)

            valid_xyz = rasterized_xyz[valid_pixels]
            valid_uv = rasterized_uv[valid_pixels]
            valid_features = rasterized_features.permute((0, 2, 3, 1))[valid_pixels]
            valid_tex = valid_features[..., :tex_ch]
            valid_basic_tex = valid_features[..., tex_ch:]

            # uv_pe
            if self.neural:
                rendered_face = self.facewrapper.feats2rgbs(valid_xyz, valid_uv, valid_tex, valid_pixels)
                rendered_basic_face = self.facewrapper.feats2rgbs(valid_xyz, valid_uv, valid_basic_tex, valid_pixels)
            else:
                B, H, W = valid_pixels.shape
                rendered_face = torch.ones((B, H, W, 3)).float().cuda()
                rendered_face[valid_pixels] = valid_tex

                rendered_basic_face = torch.ones((B, H, W, 3)).float().cuda()
                rendered_basic_face[valid_pixels] = valid_basic_tex

        # 3. Compute fusing mask & fuse
        if need_fusion:
            hair_depth = rasterized_hair["near_z"] if not gsdepth else rasterized_hair["depth"]
            head_depth = rasterized_face["depth"]

            hair_mask = self.compare_depth(hair_depth, head_depth)

            # TRY: different ways to use hair mask
            if is_val:
                if self.alter_hair:
                    processed_hair_mask = hair_mask
                else:
                    processed_hair_mask = self.compare_depth(rasterized_hair["near_z2"], head_depth)
                processed_hair_mask = erosion(processed_hair_mask.unsqueeze(1), torch.ones(3, 3).cuda())
                processed_hair_mask = dilation(processed_hair_mask, torch.ones(3, 3).cuda())
                processed_hair_mask = dilation(processed_hair_mask, torch.ones(5, 5).cuda())
                processed_hair_mask = erosion(processed_hair_mask, torch.ones(5, 5).cuda())
                processed_hair_mask = processed_hair_mask[:, 0]
            else:
                processed_hair_mask = hair_mask

            processed_hair_mask = gaussian_blur2d(
                processed_hair_mask[:, None].float(), kernel_size=(5, 5), sigma=(1.5, 1.5)
            )[:, 0]

            if hardblend:
                outputs["raster_hairmask"] = processed_hair_mask
            else:
                outputs["raster_hairmask"] = rasterized_hair["silhoutte"] * processed_hair_mask

            outputs["raster_headmask"] = torch.clip(
                rasterized_face["valid_nograd"].float() - outputs["raster_hairmask"], min=0.0, max=1.0
            )
            outputs["fullmask"] = outputs["raster_headmask"] + outputs["raster_hairmask"]

            head_part = outputs["raster_headmask"][..., None].expand(-1, -1, -1, 3)
            hair_part = outputs["raster_hairmask"][..., None].expand(-1, -1, -1, 3)
            render_fuse = head_part * rendered_face + hair_part * rendered_hair

            bg = torch.ones_like(render_fuse).float().cuda()
            render_fuse = (1 - outputs["fullmask"])[..., None] * bg + render_fuse
        else:
            render_fuse = rendered_face if has_face else rendered_hair

        # 4. Output results
        outputs["render_hair"] = rendered_hair if has_hair else None
        outputs["render_face"] = rendered_face if has_face else None
        outputs["render_basic_face"] = rendered_basic_face if has_face else None
        outputs["render_fuse"] = render_fuse
        outputs["hair_depth"] = rasterized_hair["depth"] if has_hair else None
        outputs["head_depth"] = rasterized_face["depth"] if has_face else None
        outputs["hair_silhoutte"] = rasterized_hair["silhoutte"] if has_hair else None
        outputs["occlussion_mask"] = hair_mask if need_fusion else None  # no gradients
        outputs["head_geomap"] = rasterized_face["rgba"][..., :3] if has_face else None
        outputs["head_normal"] = rasterized_face["normal"] if has_face else None
        outputs["view_confidence"] = rasterized_face["view_confidence"] if has_face else None

        if not self.config["bald"]:
            outputs["raster_hairmask"] = None if "raster_hairmask" not in outputs else outputs["raster_hairmask"]
            outputs["raster_headmask"] = None if "raster_headmask" not in outputs else outputs["raster_headmask"]
            outputs["fullmask"] = None if "fullmask" not in outputs else outputs["fullmask"]
        else:
            outputs["raster_hairmask"] = None
            outputs["raster_headmask"] = None
            outputs["fullmask"] = torch.clip(rasterized_face["valid_nograd"].float(), min=0.0, max=1.0)

        # 传递face_id、face_bw和valid_nograd给J1/J2使用
        if has_face:
            if 'face_id' in rasterized_face:
                outputs['face_id'] = rasterized_face['face_id']
            else:
                print(f"[FUSE DEBUG] face_id not found in rasterized_face keys: {list(rasterized_face.keys())}")
            if 'face_bw' in rasterized_face:
                outputs['face_bw'] = rasterized_face['face_bw']
            else:
                print(f"[FUSE DEBUG] face_bw not found in rasterized_face keys: {list(rasterized_face.keys())}")
            if 'valid_nograd' in rasterized_face:
                outputs['valid_nograd'] = rasterized_face['valid_nograd']
            else:
                print(f"[FUSE DEBUG] valid_nograd not found in rasterized_face keys: {list(rasterized_face.keys())}")
            # 文件断点：记录fuse阶段的关键可用性
            try:
                import os, json
                os.makedirs('reports', exist_ok=True)
                fuse_trace = {
                    'phase': 'fuse_out',
                    'step': int(getattr(self, 'global_step', -1)),
                    'has_face': bool(has_face),
                    'has_face_id': bool('face_id' in rasterized_face),
                    'has_face_bw': bool('face_bw' in rasterized_face),
                    'has_valid': bool('valid_nograd' in rasterized_face),
                }
                if 'face_id' in rasterized_face:
                    try:
                        fuse_trace['face_id_shape'] = list(rasterized_face['face_id'].shape)
                    except Exception:
                        pass
                if 'valid_nograd' in rasterized_face:
                    try:
                        fuse_trace['valid_shape'] = list(rasterized_face['valid_nograd'].shape)
                    except Exception:
                        pass
                import os
                os.makedirs('reports/306_J4', exist_ok=True)
                with open('reports/306_J4/j1_withj4_trace.log', 'a') as f:
                    f.write(json.dumps(fuse_trace) + "\n")
            except Exception:
                pass

        return outputs

    def remap_tex_from_2dmask(self, verts, input_img, face_ids):
        B, N_v, _ = verts.shape
        C = input_img.shape[-1]

        verts_cam = self.transform(verts, self.proj_R, self.proj_t)
        verts_proj = verts_cam.bmm(self.intr.transpose(1, 2))
        coords_screen = verts_proj[:, :, :2] / verts_proj[:, :, 2:]  # [B, N_v, 2]

        ver_XY = coords_screen[0].unsqueeze(1).detach().cpu().numpy()
        unwrap_uv_idx_v_idx = self.uv2verts_ids.astype(np.float32)
        unwrap_uv_idx_bw = self.uv2verts_bw.astype(np.float32)

        # test ver_XY
        # input_img[ver_XY[:, 0, 1].astype(np.int64), ver_XY[:, 0, 0].astype(np.int64), :] = 0.0

        uv_ver_map_y0 = unwrap_uv_idx_v_idx[:, :, 0].astype(np.float32)
        uv_ver_map_y1 = unwrap_uv_idx_v_idx[:, :, 1].astype(np.float32)
        uv_ver_map_y2 = unwrap_uv_idx_v_idx[:, :, 2].astype(np.float32)
        uv_ver_map_x = np.zeros_like(uv_ver_map_y0).astype(np.float32)

        uv_XY_0 = cv2.remap(ver_XY, uv_ver_map_x, uv_ver_map_y0, cv2.INTER_NEAREST)
        uv_XY_1 = cv2.remap(ver_XY, uv_ver_map_x, uv_ver_map_y1, cv2.INTER_NEAREST)
        uv_XY_2 = cv2.remap(ver_XY, uv_ver_map_x, uv_ver_map_y2, cv2.INTER_NEAREST)
        uv_XY = (
            uv_XY_0 * unwrap_uv_idx_bw[:, :, 0:1]
            + uv_XY_1 * unwrap_uv_idx_bw[:, :, 1:2]
            + uv_XY_2 * unwrap_uv_idx_bw[:, :, 2:3]
        )

        remap_tex = cv2.remap(input_img.astype(np.float32), uv_XY[:, :, 0], uv_XY[:, :, 1], cv2.INTER_LINEAR)
        remap_tex = np.clip(remap_tex, 0.0, 255.0)

        ver_vis_mask = np.zeros((N_v, 1, 1)).astype(np.float32)
        vis_vert_ids = list(set(self.facewrapper.flame_dec.faces[face_ids].reshape(-1).tolist()))
        vis_vert_ids.sort()
        ver_vis_mask[vis_vert_ids] = 1.0
        remap_vis_mask0 = cv2.remap(ver_vis_mask, uv_ver_map_x, uv_ver_map_y0, cv2.INTER_NEAREST)
        remap_vis_mask1 = cv2.remap(ver_vis_mask, uv_ver_map_x, uv_ver_map_y1, cv2.INTER_NEAREST)
        remap_vis_mask2 = cv2.remap(ver_vis_mask, uv_ver_map_x, uv_ver_map_y2, cv2.INTER_NEAREST)
        remap_vis_mask = (
            remap_vis_mask0 * unwrap_uv_idx_bw[:, :, 0]
            + remap_vis_mask1 * unwrap_uv_idx_bw[:, :, 1]
            + remap_vis_mask2 * unwrap_uv_idx_bw[:, :, 2]
        )
        thres = 0.5
        if C == 1:
            remap_vis_mask = (remap_vis_mask > thres).astype(np.float32)
        else:
            remap_vis_mask = img2mask(remap_vis_mask, thre=thres)

        remap_tex = remap_tex * remap_vis_mask

        return remap_tex

    def transform(self, pts, R, t):
        translation = deepcopy(t)
        if len(translation.shape) == 2:
            translation = translation[:, None]
        R_inv = R.transpose(1, 2)

        return pts.bmm(R_inv) + translation

    def network_forward(self, is_val=False):
        rasterized_hair, rasterized_face = None, None
        bg_color = [1.0, 1.0, 1.0]  # white

        rasterized_face, rigid_trans = self.facewrapper.render(self.camera, self.flame_params, self.view, bg_color)
        if self.stage == "joint":
            rasterized_hair = self.hairwrapper.render_with_trans(
                self.camera, self.flame_params, rigid_trans[:, :3, :3], rigid_trans[:, :3, 3], bg_color
            )

        outputs = self.fuse(rasterized_hair, rasterized_face, is_val=is_val)

        return outputs

    def update_x(self, lambda_name):
        return update_lambda(
            self.config["training.lambda_{}".format(lambda_name)],
            self.config["training.lambda_{}.slope".format(lambda_name)],
            self.config["training.lambda_{}.end".format(lambda_name)],
            self.global_step,
            self.config["training.lambda_{}.interval".format(lambda_name)],
        )

    def update_lambda(self):
        update_names = self.config["training.lambda_update_list"]
        for k, _ in self.all_lambdas.items():
            if k in update_names:
                self.all_lambdas[k] = self.update_x(k)

    def get_lambda(self, key):
        return self.all_lambdas.get(key, 0.0)
    
    def _get_report_dir(self):
        """生成基于 local_workspace 的报告目录，用于隔离不同训练版本的日志"""
        workspace = self.config.get("local_workspace", "")
        parts = workspace.split('/')
        # 解析路径结构，假设格式为：.../subject_id/version_id/...
        if len(parts) >= 3:
            subject_id = parts[-2]  # 倒数第二级目录作为 subject
            version = parts[-1]      # 最后一级目录作为 version
            return f"reports/{subject_id}/{version}"
        return "reports/default"

    def _legacy_maybe_uv_bake(self, batch, outputs):
        """Inactive same-view render-to-GT baker retained for checkpoint-era diagnostics."""
        try:
            print(f"[J2 UV Bake] Called at step {self.global_step}")
            # 构建正确的配置字典
            cfg = {
                'enable': self.config.get('exp.lf_uv.enable', False),
                'uv_res': self.config.get('exp.lf_uv.uv_res', 1024),
                'force_bake_now': self.config.get('exp.lf_uv.force_bake_now', False),
                'every': self.config.get('exp.lf_uv.every', 1),
                'vis_every': self.config.get('exp.lf_uv.vis_every', 10000),
                'cache': {
                    'enable': self.config.get('exp.lf_uv.cache.enable', False),
                    'keep_uv_gt_mask': self.config.get('exp.lf_uv.cache.keep_uv_gt_mask', False)
                },
                'loss': {
                    'enable': self.config.get('exp.lf_uv.loss.enable', False),
                    'type': self.config.get('exp.lf_uv.loss.type', 'charbonnier'),
                    'colorspace': self.config.get('exp.lf_uv.loss.colorspace', 'rgb'),
                    'gaussian_kernel': self.config.get('exp.lf_uv.loss.gaussian_kernel', 7),
                    'gaussian_sigma': self.config.get('exp.lf_uv.loss.gaussian_sigma', 1.2),
                    'weight': self.config.get('exp.lf_uv.loss.weight', 0.05),
                    'vis_mask_min_cov': self.config.get('exp.lf_uv.loss.vis_mask_min_cov', 0.02),
                    'scale_by_every': self.config.get('exp.lf_uv.loss.scale_by_every', True)
                },
                'seam': {
                    'enable': self.config.get('exp.lf_uv.seam.enable', False),
                    'weight': self.config.get('exp.lf_uv.seam.weight', 0.0),
                    'band_px': self.config.get('exp.lf_uv.seam.band_px', 2),
                    'tv_type': self.config.get('exp.lf_uv.seam.tv_type', 'l1'),
                    'edge': self.config.get('exp.lf_uv.seam.edge', 'sobel'),
                    'edge_tau': self.config.get('exp.lf_uv.seam.edge_tau', 0.05),
                    'min_cov': self.config.get('exp.lf_uv.seam.min_cov', 0.02),
                    'vis_every': self.config.get('exp.lf_uv.seam.vis_every', 10000)
                }
            }
            if not bool(cfg.get('enable', False)):
                if getattr(self, 'is_main', True) and self.global_step % 100 == 0:
                    print(f"[J2 UV Bake] Skipping: lf_uv not enabled")
                return
            
            # ✅ 修复1：添加 every 频率控制
            every = cfg.get('every', 1)
            if self.global_step % every != 0:
                if getattr(self, 'is_main', True) and self.global_step % 500 == 0:
                    print(f"[J2 UV Bake] Skipping step {self.global_step} (every={every})")
                return

            # 前置: 关键输出是否存在
            if 'render_face' not in outputs or outputs['render_face'] is None:
                if getattr(self, 'is_main', True) and self.global_step % 100 == 0:
                    print(f"[J2 UV Bake] Skipping: missing render_face")
                return

            # 资产（来自 facewrapper / FLAME 解码器）
            face_dec = getattr(self.facewrapper, 'flame_dec', None)
            vertex_uvs = getattr(face_dec, 'vertex_uvs', None)
            faces_uvs  = getattr(face_dec, 'faces_uvs',  None)
            if vertex_uvs is None or faces_uvs is None:
                return

            # 获取face_id和face_bw（现在从fuse函数传递过来）
            face_id = outputs['face_id']                     # [B,H,W] int64
            face_bw = outputs['face_bw']                     # [B,H,W,2 or 3]
            print(f"[J2 UV Bake] Using face_id/face_bw: face_id.shape={face_id.shape}, face_bw.shape={face_bw.shape}")

            # 像素可见掩膜（face_id>=0 与 valid_nograd，与 head_mask 对齐）
            valid   = outputs.get('valid_nograd', None)      # [B,H,W] bool or None
            head_m  = self.mask.get('head', None)            # [B,H,W] bool

            px_mask = (face_id >= 0)
            if valid is not None: px_mask = px_mask & valid
            if head_m is not None:
                # 保证尺寸一致
                if head_m.shape[-2:] != face_id.shape[-2:]:
                    head_m = torch.nn.functional.interpolate(
                        head_m.float().unsqueeze(1),
                        size=face_id.shape[-2:], mode='nearest'
                    ).squeeze(1).bool()
                # 确保head_m是bool类型
                if head_m.dtype != torch.bool:
                    head_m = head_m.bool()
                px_mask = px_mask & head_m

            # 调用 UV_BAKE（使用正确的函数签名）
            print(f"[J2 UV Bake] Calling bake_to_uv with:")
            print(f"  img.shape: {self.img.shape}")
            print(f"  pred.shape: {outputs['render_face'].shape}")
            print(f"  face_id.shape: {face_id.shape}")
            print(f"  face_bw.shape: {face_bw.shape}")
            print(f"  vertex_uvs.shape: {vertex_uvs.shape}")
            print(f"  faces_uvs.shape: {faces_uvs.shape}")
            print(f"  px_mask.shape: {px_mask.shape}")
            print(f"  uv_hw: {(cfg.get('uv_res', 1024), cfg.get('uv_res', 1024))}")
            
            uv_result = bake_to_uv_functional(
                img= self.img,                                # [B,H,W,3] float
                pred= outputs['render_face'],                 # [B,H,W,3] float
                face_id= face_id,                             # [B,H,W] int (可能来自outputs或fallback)
                face_bw= face_bw,                             # [B,H,W,2 or 3] (可能来自outputs或fallback)
                vertex_uvs= vertex_uvs,                       # [V,2]
                faces_uvs= faces_uvs,                         # [F,3]
                px_mask= px_mask,                             # [B,H,W] bool
                uv_hw= (cfg.get('uv_res', 1024), cfg.get('uv_res', 1024)),  # Tuple[int, int]
                is_main= getattr(self, 'is_main', True),      # 只主进程落图
                report_dir= "reports",                        # 可视化输出目录
                step= self.global_step,                       # 当前训练步数
                vis_every= cfg.get('vis_every', 10000)       # 可视化保存间隔
            )
            
            print(f"[J2 UV Bake] bake_to_uv returned successfully")
            print(f"  uv_result keys: {list(uv_result.keys())}")
            
            # 提取结果
            uv_gt = uv_result['uv_gt']
            uv_pred = uv_result['uv_pred']
            uv_mask = uv_result['uv_mask']

            # 绑回 outputs（遵循只读约定）
            outputs['__lfuv_gt__']   = uv_gt.detach()
            outputs['__lfuv_pred__'] = uv_pred               # 保持梯度
            outputs['__lfuv_mask__'] = uv_mask.detach()
            
            # 调试：检查梯度状态
            print(f"[J2 UV Bake] uv_pred.requires_grad: {uv_pred.requires_grad}")
            print(f"[J2 UV Bake] uv_gt.requires_grad: {uv_gt.requires_grad}")
            print(f"[J2 UV Bake] uv_mask.requires_grad: {uv_mask.requires_grad}")
            
            # 调试信息
            if getattr(self, 'is_main', True) and self.global_step % 100 == 0:
                print(f"[J2 UV Bake] Success: stored __lfuv_*__ keys in outputs")
                print(f"[J2 UV Bake] uv_gt.shape={uv_gt.shape}, uv_pred.shape={uv_pred.shape}, uv_mask.shape={uv_mask.shape}")

        except Exception as e:
            # 详细错误信息，避免静默失败
            print(f"[J2 UV Bake] ERROR at step {self.global_step}: {e}")
            print(f"[J2 UV Bake] Exception type: {type(e).__name__}")
            import traceback
            print(f"[J2 UV Bake] Traceback: {traceback.format_exc()}")
            if getattr(self, 'is_main', True) and self.global_step % 50 == 0:
                print(f"[J2 UV Bake] skipped at step {self.global_step}: {e}")

    def _prepare_cross_view_uv(self, outputs):
        """Bake rendered head colors and Eq. (11) confidence into UV space."""
        uv_enabled = (
            self.config.get("exp.lf_uv.enable", False)
            and self.config.get("exp.lf_uv.loss.enable", False)
            and float(self.config.get("exp.lf_uv.loss.weight", 0.0)) > 0.0
        )
        seam_enabled = self.config.get("exp.lf_uv.seam.enable", False)
        if not (uv_enabled or seam_enabled):
            return
        every = int(self.config.get("exp.lf_uv.every", 1))
        if self.global_step % max(every, 1):
            return
        required = ("render_face", "face_id", "face_bw", "valid_nograd", "view_confidence")
        missing = [key for key in required if outputs.get(key) is None]
        if missing:
            raise RuntimeError(f"Cross-view UV supervision is enabled but render outputs are missing {missing}")
        face_dec = self.facewrapper.flame_dec
        uv_res = int(self.config.get("exp.lf_uv.uv_res", 1024))
        visibility = outputs["valid_nograd"].bool() & (outputs["face_id"] >= 0)
        uv_render, uv_visibility = bake_attribute_to_uv(
            outputs["render_face"], outputs["face_id"], outputs["face_bw"],
            face_dec.vertex_uvs, face_dec.faces_uvs, visibility, (uv_res, uv_res)
        )
        confidence = outputs["view_confidence"].detach()
        if confidence.ndim == 3:
            confidence = confidence.unsqueeze(-1)
        uv_confidence, confidence_visibility = bake_attribute_to_uv(
            confidence, outputs["face_id"], outputs["face_bw"],
            face_dec.vertex_uvs, face_dec.faces_uvs, visibility, (uv_res, uv_res)
        )
        outputs["__uv_render__"] = uv_render
        outputs["__uv_visibility__"] = (uv_visibility * confidence_visibility).detach()
        outputs["__uv_confidence__"] = uv_confidence.detach().clamp(0.0, 1.0)

    def compute_loss(self, outputs, include_fresco=True):
        render_rgb = outputs["render_fuse"]
        render_face = outputs["render_face"]

        # update hyper-parameters
        self.update_lambda()

        # RGB Loss
        hair_mask = self.mask["hair"]
        erode_hair_mask = self.mask["erode_hair"]
        head_mask = self.mask["head"]

        gt_hair = self.img.clone()
        gt_head = self.img.clone()
        gt_hair[(1 - hair_mask).bool()] = 1.0
        gt_head[hair_mask.bool()] = 1.0

        # L2 Loss
        rgb_loss = torch.linalg.norm((render_rgb - self.img), dim=-1).mean()
        whole_head_loss = torch.linalg.norm((render_face - self.img), dim=-1).mean()
        whole_head_ssim_loss = 1.0 - ssim(render_face.permute((0, 3, 1, 2)), self.img.permute((0, 3, 1, 2)))

        # SSIM Loss
        if self.get_lambda("ssim") > 0:
            ssim_loss = 1.0 - ssim(render_rgb.permute((0, 3, 1, 2)), self.img.permute((0, 3, 1, 2)))
        else:
            ssim_loss = render_rgb.new_zeros(())  # 标量 0.0，同设备同 dtype

        loss_head, loss_head_dict = self.facewrapper.compute_losses(
            outputs, self.img, gt_head, self.depth_map, hair_mask, head_mask, self.global_step
        )
        loss_hair, loss_hair_dict = self.hairwrapper.compute_losses(
            outputs, gt_hair, hair_mask, erode_hair_mask, self.global_step
        )
        loss_joint = (
            self.get_lambda("rgb") * rgb_loss
            + self.get_lambda("ssim") * ssim_loss
            + self.get_lambda("rgb.head") * whole_head_loss
            + self.get_lambda("ssim") * whole_head_ssim_loss
        )

        loss_all = {"head": loss_head, "hair": loss_hair, "joint": loss_joint}

        # 初始化总损失（避免从 float 开始）
        loss = next(iter(loss_all.values())).new_zeros(())  # 标量0，设备/dtype一致
        for name in self.config["training.{}_stage_loss".format(self.stage)]:
            loss = loss + loss_all[name]

        outputs["gt_hair"] = gt_hair
        outputs["gt_head"] = gt_head
        loss_dict = {"loss": loss, "loss_pho/rgb.obj": rgb_loss, "loss_pho/ssim.obj": ssim_loss}

        # Update with head & hair loss
        loss_dict.update(loss_head_dict)
        loss_dict.update(loss_hair_dict)

        # FReSCO paper losses. Scheduling remains external (shell/config weights).
        loss_dict["__fresco_paper_losses__"] = True
        lf_enable = include_fresco and bool(self.config.get("exp.lf_img.enable", False))
        lf_weight = float(self.config.get("exp.lf_img.lambda", 0.0))
        if lf_enable and lf_weight > 0:
            lf_cfg = {
                "kernel": int(self.config.get("exp.lf_img.kernel", 9)),
                "sigma": float(self.config.get("exp.lf_img.sigma", -1.0)),
            }
            lf_stats = compute_lfimg_loss(render_rgb, self.img, None, lf_cfg, self.global_step)
            loss = loss + lf_weight * lf_stats["loss"]
            loss_dict.update({
                "lf_img_raw": lf_stats["raw"], "lf_img_cov": lf_stats["cov"],
                "lf_img_weighted": (lf_weight * lf_stats["loss"]).detach(),
                "lf_img_k": lf_stats["k"], "lf_img_w": lf_weight,
            })

        hf_enable = include_fresco and bool(self.config.get("exp.hf_img.enable", False))
        hf_weight = float(self.config.get("exp.hf_img.lambda", 0.0))
        if hf_enable and hf_weight > 0:
            hf_cfg = validate_hf_image_config(self.config, context="Trainer HF-IMG")
            hf_stats = compute_hfimg_loss(render_rgb, self.img, None, hf_cfg, self.global_step)
            loss = loss + hf_weight * hf_stats["loss"]
            loss_dict.update({
                "hf_img_raw": hf_stats["raw"], "hf_img_edge": hf_stats["edge_hf"],
                "hf_img_gdl": hf_stats["gdl"], "hf_img_edge_cov": hf_stats["edge_coverage"],
                "hf_img_weighted": (hf_weight * hf_stats["loss"]).detach(), "hf_img_w": hf_weight,
            })

        uv_enable = include_fresco and bool(self.config.get("exp.lf_uv.enable", False))
        uv_loss_enable = (
            bool(self.config.get("exp.lf_uv.loss.enable", False))
            and float(self.config.get("exp.lf_uv.loss.weight", 0.0)) > 0.0
        )
        uv_keys = ("__uv_render__", "__uv_visibility__", "__uv_confidence__")
        uv_due = self.global_step % max(int(self.config.get("exp.lf_uv.every", 1)), 1) == 0
        if uv_enable and uv_loss_enable and uv_due:
            if not all(key in outputs for key in uv_keys):
                raise RuntimeError("Cross-view UV loss is enabled but this step has no baked UV tensors")
            uv_stats = compute_cross_view_uv_loss(
                outputs["__uv_render__"], outputs["__uv_visibility__"], outputs["__uv_confidence__"],
                self.frame_idx, self.camera_ids,
            )
            uv_base_weight = float(self.config.get("exp.lf_uv.loss.weight", 0.0))
            uv_every = max(int(self.config.get("exp.lf_uv.every", 1)), 1)
            scale_by_every = bool(self.config.get("exp.lf_uv.loss.scale_by_every", True))
            uv_weight = uv_base_weight * uv_every if scale_by_every else uv_base_weight
            loss = loss + uv_weight * uv_stats["loss"]
            loss_dict.update({
                "lf_uv/raw": uv_stats["loss"].detach(), "lf_uv/weighted": (uv_weight * uv_stats["loss"]).detach(),
                "lf_uv/base_weight": uv_base_weight, "lf_uv/effective_weight": uv_weight,
                "lf_uv/cov": uv_stats["coverage"], "lf_uv/pairs": uv_stats["pair_count"],
                "lf_uv/mean_confidence": uv_stats["mean_confidence"],
                "lf_uv/mutual_texels": uv_stats["mutual_texels"],
                "lf_uv/skipped": uv_stats["skipped"], "lf_uv/reason": uv_stats["reason"],
            })

        seam_enable = include_fresco and bool(self.config.get("exp.lf_uv.seam.enable", False))
        if seam_enable and uv_due:
            if "__uv_render__" not in outputs:
                raise RuntimeError("UV seam loss is enabled but this step has no baked rendered UV")
            seam_cfg = {
                "weight_pair": float(self.config.get("exp.lf_uv.seam.weight_pair", 0.02)),
                "weight_tv": float(self.config.get("exp.lf_uv.seam.weight_tv", 0.004)),
                "band_px": int(self.config.get("exp.lf_uv.seam.band_px", 2)),
            }
            seam_stats = compute_seam_loss(
                outputs["__uv_render__"], seam_cfg, self.global_step, self.is_main,
                outputs["__uv_visibility__"], self.seam_maps.get("pair_u"),
                self.seam_maps.get("pair_v"), self.seam_maps.get("pair_valid"),
            )
            loss = loss + seam_stats["loss"]
            loss_dict.update({
                "lf_uv_seam/raw": seam_stats["loss"].detach(), "lf_uv_seam/pair": seam_stats["pair"],
                "lf_uv_seam/tv": seam_stats["tv"], "lf_uv_seam/cov": seam_stats["effective_cov"],
                "lf_uv_seam/weighted": seam_stats["loss"].detach(),
                "lf_uv_seam/pair_cov": seam_stats["pair_cov"],
                "lf_uv_seam/skipped": seam_stats["skipped"], "lf_uv_seam/reason": seam_stats["reason"],
            })
        loss_dict["loss"] = loss

        # LF-UV Loss (J2-2) - 带护栏的调用
        lf_uv_enabled = include_fresco and self.config.get('exp.lf_uv.enable', False)
        lf_uv_loss_enabled = self.config.get('exp.lf_uv.loss.enable', False)
        has_lfuv_keys = ('__lfuv_pred__' in outputs and '__lfuv_gt__' in outputs and '__lfuv_mask__' in outputs)
        
        if lf_uv_enabled and lf_uv_loss_enabled and has_lfuv_keys:
            try:
                uv_pred = outputs['__lfuv_pred__']  # Keep gradients
                uv_gt = outputs['__lfuv_gt__'].detach()
                uv_mask = outputs['__lfuv_mask__'].detach()
                
                # Compute LF-UV loss (带内置护栏) - 传入 report_dir
                lfuv_result = compute_lfuv_loss(
                    uv_pred=uv_pred,
                    uv_gt=uv_gt,
                    uv_mask=uv_mask,
                    cfg=self.config,
                    step=self.global_step,
                    is_main=self.is_main,
                    report_dir=self._get_report_dir()
                )
                
                # 提取结果
                vis_cov = lfuv_result['vis_cov']
                denom = lfuv_result['denom']
                lfuv_loss_raw = lfuv_result['loss']
                
                # 检查覆盖率阈值
                vis_mask_min_cov = self.config.get('exp.lf_uv.loss.vis_mask_min_cov', 0.02)
                
                # 提取原始损失值
                lfuv_raw = lfuv_loss_raw.detach().item()
                
                # ✅ 修复2：添加 scale_by_every 权重补偿
                lfuv_weight = self.config.get('exp.lf_uv.loss.weight', 0.0)
                every = self.config.get('exp.lf_uv.every', 1)
                scale_by_every = self.config.get('exp.lf_uv.loss.scale_by_every', True)
                
                final_weight = lfuv_weight
                if scale_by_every and every > 1:
                    final_weight = lfuv_weight * every
                    if self.global_step % 500 == 0 and self.is_main:
                        print(f"[J2] Applying scale_by_every: {lfuv_weight:.4f} × {every} = {final_weight:.4f}")
                
                # 记录到loss_dict
                loss_dict['lf_uv/raw'] = lfuv_raw
                loss_dict['lf_uv/cov'] = vis_cov
                
                if vis_cov < vis_mask_min_cov:
                    # 跳过本步并记录日志
                    loss_dict['lf_uv/weighted'] = 0.0  # ✅ 跳过时显示0
                    if self.global_step % 50 == 0:  # 每50步记录一次
                        self.logger.info(f"[LF-UV] Skipping step {self.global_step}: vis_cov={vis_cov:.4f} < min_cov={vis_mask_min_cov:.4f}")
                    # 记录跳过状态
                    if self.is_main:
                        print(f"LF-UV: raw={lfuv_raw:.4f} w={final_weight:.3f} cov={vis_cov:.3f} skipped=True")
                else:
                    # 添加LF-UV损失到总损失 (保持张量，不detach) - 使用 final_weight
                    lfuv_loss_tensor = lfuv_result['loss'] * final_weight  # ✅ 保持 Tensor 计算图，应用补偿后的权重
                    loss += lfuv_loss_tensor                                # ✅ 参与反传
                    loss_dict['lf_uv/weighted'] = lfuv_loss_tensor.detach().item()  # ✅ 实际加权值
                    # 注入loss_dict用于TensorBoard/日志观察
                    loss_dict['loss'] = loss
                    loss_dict['lf_uv.loss'] = lfuv_loss_tensor.detach().item()  # ✅ 日志用标量
                    loss_dict['lf_uv.vis_cov'] = vis_cov
                    loss_dict['lf_uv.denom'] = denom
                    loss_dict['lf_uv.mode'] = lfuv_result['mode']
                    loss_dict['lf_uv.k'] = lfuv_result['k']
                    loss_dict['lf_uv.sigma'] = lfuv_result['sigma']
                    
                    # 主进程打印控制台信息
                    if self.is_main:
                        print(f"LF-UV: raw={lfuv_raw:.4f} w={final_weight:.3f} cov={vis_cov:.3f}")
                    
                    # 在主训练日志中记录J2损失
                    if self.global_step % 50 == 0:  # 每50步记录一次
                        self.logger.info(f"[J2 LF-UV] step={self.global_step}, loss={lfuv_raw:.6f}, weight={final_weight:.3f}, cov={vis_cov:.6f}, mode={lfuv_result['mode']}")
                        
            except Exception as e:
                # 异常处理：记录但不中断训练
                if self.global_step % 100 == 0:
                    self.logger.warning(f"[LF-UV] Error at step {self.global_step}: {str(e)}")
        else:
            # 缺少必要键或未启用时跳过
            if lf_uv_enabled and lf_uv_loss_enabled and not has_lfuv_keys:
                if self.global_step % 100 == 0:
                    self.logger.info(f"[LF-UV] Skipping step {self.global_step}: missing __lfuv_*__ keys in outputs")

        # UV Seam Loss (J2-3) - 接缝正则化
        lf_uv_seam_enabled = include_fresco and self.config.get('exp.lf_uv.seam.enable', False)
        has_seam_keys = ('__lfuv_pred__' in outputs and '__lfuv_mask__' in outputs)
        
        if lf_uv_enabled and lf_uv_seam_enabled and has_seam_keys:
            try:
                uv_pred = outputs['__lfuv_pred__']  # 不要 detach
                uv_mask = outputs['__lfuv_mask__'].detach()  # 只当权重/可视化
                is_main = bool(getattr(self, 'is_main', True))  # 兼容单卡
                
                # 获取接缝配置
                seam_cfg = {
                    'enable': self.config.get('exp.lf_uv.seam.enable', False),
                    'weight': self.config.get('exp.lf_uv.seam.weight', 0.0),
                    'weight_pair': self.config.get('exp.lf_uv.seam.weight_pair', 0.02),  # 主项权重
                    'weight_tv': self.config.get('exp.lf_uv.seam.weight_tv', 0.004),     # 辅项权重
                    'min_vis': self.config.get('exp.lf_uv.seam.min_vis', 0.5),           # 最小可见性阈值
                    'band_px': self.config.get('exp.lf_uv.seam.band_px', 2),
                    'apply_on': self.config.get('exp.lf_uv.seam.apply_on', 'rgb'),      # 应用目标
                    'tv_type': self.config.get('exp.lf_uv.seam.tv_type', 'l1'),
                    'edge': self.config.get('exp.lf_uv.seam.edge', 'sobel'),
                    'edge_tau': self.config.get('exp.lf_uv.seam.edge_tau', 0.05),
                    'min_cov': self.config.get('exp.lf_uv.seam.min_cov', 0.02),
                    'vis_every': self.config.get('exp.lf_uv.seam.vis_every', 10000)
                }
                
                # 计算接缝损失
                # 获取 seam_maps 中的配对映射
                pair_u = self.seam_maps.get('pair_u', None)
                pair_v = self.seam_maps.get('pair_v', None) 
                pair_valid = self.seam_maps.get('pair_valid', None)
                
                stats = compute_seam_loss(
                    uv_texture=uv_pred,
                    config=seam_cfg,
                    step=self.global_step,
                    is_main=is_main,
                    uv_mask=uv_mask,
                    pair_u=pair_u,
                    pair_v=pair_v,
                    pair_valid=pair_valid
                )
                
                # 提取原始损失值
                seam_raw = stats["loss"].detach().item()
                seam_weight = seam_cfg.get('weight', 0.0)
                
                # 记录到loss_dict
                loss_dict['lf_uv_seam/raw'] = seam_raw
                loss_dict['lf_uv_seam/weighted'] = seam_raw * seam_weight
                loss_dict['lf_uv_seam/cov'] = stats['cov']
                loss_dict['lf_uv_seam/band_px'] = stats['band_px']
                
                if not stats.get("skipped", False):
                    # 前后对账：记录累加前的总损失
                    tot_before = loss.detach().item()
                    seam_loss = seam_weight * stats["loss"]
                    loss += seam_loss
                    # 前后对账：记录累加后的总损失
                    tot_after = loss.detach().item()
                    
                    # 调试开关：显示损失分解
                    show_breakdown = self.config.get('exp', {}).get('debug_run', {}).get('show_loss_breakdown', True)
                    if self.is_main and show_breakdown:
                        print(f"[LOSS BREAKDOWN] before={tot_before:.4f} seam={seam_raw:.4f}*{seam_weight:.3f} -> after={tot_after:.4f}")
                    
                    # 注入loss_dict用于日志观察
                    loss_dict['loss'] = loss
                    loss_dict['seam.loss'] = seam_loss.item()
                    loss_dict['seam.cov'] = stats['cov']
                    loss_dict['seam.band_px'] = stats['band_px']
                    loss_dict['seam.tv'] = stats['tv']
                    loss_dict['seam.edge'] = stats['edge']
                    loss_dict['seam.tau'] = stats['tau']
                    loss_dict['seam.skipped'] = stats['skipped']
                    
                    # 主进程打印控制台信息
                    if self.is_main:
                        print(f"LF-UV-SEAM: raw={seam_raw:.4f} w={seam_weight:.3f} cov={stats['cov']:.3f} band={stats['band_px']}")
                else:
                    # 记录跳过信息
                    if self.global_step % 100 == 0:
                        self.logger.info(f"[Seam] Skipping step {self.global_step}: cov={stats['cov']:.4f} < min_cov or no valid band")
                    # 主进程打印跳过状态
                    if self.is_main:
                        print(f"LF-UV-SEAM: raw={seam_raw:.4f} w={seam_weight:.3f} cov={stats['cov']:.3f} band={stats['band_px']} skipped=True")
                        
            except Exception as e:
                # 异常处理：记录但不中断训练
                if self.global_step % 100 == 0:
                    self.logger.warning(f"[Seam] Error at step {self.global_step}: {str(e)}")
        else:
            # 缺少必要键或未启用时跳过
            if lf_uv_enabled and lf_uv_seam_enabled and not has_seam_keys:
                if self.global_step % 100 == 0:
                    self.logger.info(f"[Seam] Skipping step {self.global_step}: missing __lfuv_*__ keys in outputs")

        # LF-IMG Loss (J1) - 图像域低频损失
        try:
            # 构建正确的J1配置字典
            cfg_lf = {
                'enable': self.config.get('exp.lf_img.enable', False),
                'kernel': self.config.get('exp.lf_img.kernel', 5),
                'loss': self.config.get('exp.lf_img.loss', 'charbonnier'),
                'lambda': self.config.get('exp.lf_img.lambda', 0.1)
            }
            enable = bool(self.config.get("exp.lf_img.enable", cfg_lf.get("enable", True))) and not loss_dict.get("__fresco_paper_losses__", False)
            
            # 权重优先级：exp.lf_img.lambda > training.lambda_lf_img
            w = self.config.get("exp.lf_img.lambda", None)
            if w is None:
                w = cfg_lf.get("lambda", None)
            if w is None:
                w = self.config.get("training.lambda_lf_img", 0.0)
            w = float(w)
            
            # 调试：检查J1是否被触发
            if enable and self.global_step % 10 == 0:
                print(f"[J1 DEBUG] step={self.global_step}, enable={enable}, w={w}")
                # 强制输出到文件，不受静默模式影响
                import os
                os.makedirs("reports/306_J4", exist_ok=True)
                with open("reports/306_J4/j1_withj4debug.log", "a") as f:
                    f.write(f"[J1 DEBUG] step={self.global_step}, enable={enable}, w={w}\n")
            
            if enable and w > 0.0 and not loss_dict.get("__fresco_paper_losses__", False):
                # 调试：J1被调用
                if self.global_step % 10 == 0:  # 改为每10步输出一次
                    print(f"[J1 CALLED] step={self.global_step}, J1 is being called!")
                    import os
                    os.makedirs("reports/306_J4", exist_ok=True)
                    with open("reports/306_J4/j1_debug.log", "a") as f:
                        f.write(f"[J1 CALLED] step={self.global_step}, J1 is being called!\n")
                
                # 确保数据范围一致: 都归一化到 [0, 1]
                gt = self.img.to(device=outputs["render_face"].device, dtype=torch.float32)
                # 如果gt是[0,255]范围，归一化到[0,1]
                if gt.max() > 1.0:
                    gt = gt / 255.0
                pred = outputs["render_face"]  # float32, cuda, [0,1]
                
                # 构造batch数据用于get_face_vis_mask
                batch_data = {
                    'head_mask': self.mask["head"],
                    'img': self.img,
                    'global_step': self.global_step
                }
                # 在outputs中也添加global_step
                outputs['global_step'] = self.global_step

                # 文件断点：记录J1前置可见性检查
                try:
                    import os, json
                    os.makedirs('reports', exist_ok=True)
                    j1_pre = {
                        'phase': 'j1_pre',
                        'step': int(self.global_step),
                        'has_face_id': bool('face_id' in outputs),
                        'has_valid': bool('valid_nograd' in outputs),
                    }
                    if 'face_id' in outputs:
                        try:
                            j1_pre['face_id_min'] = int(outputs['face_id'].min().item())
                            j1_pre['face_id_max'] = int(outputs['face_id'].max().item())
                            j1_pre['face_id_shape'] = list(outputs['face_id'].shape)
                        except Exception:
                            pass
                    import os
                    os.makedirs('reports/306_J4', exist_ok=True)
                    with open('reports/306_J4/j1_trace.log', 'a') as f:
                        f.write(json.dumps(j1_pre) + "\n")
                except Exception:
                    pass
                face_mask = get_face_vis_mask(batch_data, outputs)  # [B,H,W] bool
                
                # 调试信息：检查face_id和face_mask的值
                if self.global_step % 10 == 0:
                    face_id = outputs.get('face_id')
                    if face_id is not None:
                        face_id_min = face_id.min().item()
                        face_id_max = face_id.max().item()
                        face_id_unique = torch.unique(face_id).cpu().numpy()
                        face_mask_count = face_mask.sum().item()
                        print(f"[J1 DEBUG] face_id: min={face_id_min}, max={face_id_max}, unique={face_id_unique[:10]}")
                        print(f"[J1 DEBUG] face_mask: count={face_mask_count}, shape={face_mask.shape}")
                    else:
                        print(f"[J1 DEBUG] face_id is None in outputs")

                # 文件断点：记录J1面罩统计
                try:
                    import os, json
                    os.makedirs('reports', exist_ok=True)
                    j1_mask_stat = {
                        'phase': 'j1_mask',
                        'step': int(self.global_step),
                        'face_mask_sum': int(face_mask.sum().item()),
                    }
                    if 'face_id' in outputs:
                        try:
                            j1_mask_stat['face_id_min'] = int(outputs['face_id'].min().item())
                            j1_mask_stat['face_id_max'] = int(outputs['face_id'].max().item())
                        except Exception:
                            pass
                    import os
                    os.makedirs('reports/306_J4', exist_ok=True)
                    with open('reports/306_J4/j1_trace.log', 'a') as f:
                        f.write(json.dumps(j1_mask_stat) + "\n")
                except Exception:
                    pass
                
                stat = compute_lfimg_loss(pred, gt, face_mask, cfg_lf, int(self.global_step),
                                         report_dir=self._get_report_dir())
                
                # 计入总损失 & 轻量日志
                loss_dict["lf_img_raw"] = stat["raw"]
                loss_dict["lf_img_cov"] = stat["cov"]
                loss_dict["lf_img_k"] = stat["k"]
                loss_dict["lf_img_w"] = w
                loss += w * stat["loss"]
                
                # 在主训练日志中记录J1损失
                if self.global_step % 50 == 0:  # 每50步记录一次
                    self.logger.info(f"[J1 LF-IMG] step={self.global_step}, loss={stat['loss']:.6f}, weight={w:.3f}, raw={stat['raw']:.6f}, cov={stat['cov']:.6f}")
                # 统一刷新总损失
                loss_dict["loss"] = loss
                
                if (self.global_step % 50 == 0) and (not self.config.get("exp", {}).get("logging", {}).get("silent", False)):
                    print(f"[LF-IMG] step={self.global_step} raw={stat['raw'].item():.4f} "
                          f"w={w:.3f} cov={stat['cov']:.3f} k={stat['k']}")
        except Exception as e:
            if not self.config.get("exp", {}).get("logging", {}).get("silent", False):
                print(f"[LF-IMG] skipped due to error: {e}")

        # HF-IMG Loss (J4) - 高频图像损失
        try:
            # 构建J4配置字典
            cfg_hf = {
                'enable': bool(self.config.get('exp.hf_img.enable', False)),
                'mode': self.config.get('exp.hf_img.mode', 'charbonnier'),
                'ksize': int(self.config.get('exp.hf_img.ksize', 5)),
                'sigma': float(self.config.get('exp.hf_img.sigma', 1.2)),
                'edge_method': self.config.get('exp.hf_img.edge_method', 'sobel'),
                'edge_threshold': float(self.config.get('exp.hf_img.edge_threshold', 0.1)),
                'edge_weight': float(self.config.get('exp.hf_img.edge_weight', 2.0)),
                'base_weight': float(self.config.get('exp.hf_img.base_weight', 1.0)),
                'pre_smooth_kernel': int(self.config.get('exp.hf_img.pre_smooth_kernel', 5)),
                'pre_smooth_sigma': float(self.config.get('exp.hf_img.pre_smooth_sigma', 1.0)),
                'edge_percentile': float(self.config.get('exp.hf_img.edge_percentile', 95.0)),
                'edge_sharpness': float(self.config.get('exp.hf_img.edge_sharpness', 10.0)),
                'edge_dilate_kernel': int(self.config.get('exp.hf_img.edge_dilate_kernel', 3)),
                'cov_target': float(self.config.get('exp.hf_img.cov_target', 0.05)),
                'p_init': float(self.config.get('exp.hf_img.p_init', 94.0)),
                'p_min': float(self.config.get('exp.hf_img.p_min', 85.0)),
                'p_max': float(self.config.get('exp.hf_img.p_max', 99.0)),
                'band_dilate_kernel': int(self.config.get('exp.hf_img.band_dilate_kernel', 3)),
                'soft_span': float(self.config.get('exp.hf_img.soft_span', 0.1)),
                'edge_gamma': float(self.config.get('exp.hf_img.edge_gamma', 2.0)),
                'apply_valid_mask': bool(self.config.get('exp.hf_img.apply_valid_mask', True)),
            }
            enable = bool(self.config.get("exp.hf_img.enable", cfg_hf.get("enable", False))) and not loss_dict.get("__fresco_paper_losses__", False)
            
            # 权重优先级：exp.hf_img.lambda > training.lambda_hf_img
            w = self.config.get("exp.hf_img.lambda", None)
            if w is None:
                w = self.config.get("training.lambda_hf_img", 0.0)
            w = float(w)
            
            # 调试：检查J4是否被触发
            if enable and self.global_step % 10 == 0:
                print(f"[J4 DEBUG] step={self.global_step}, enable={enable}, w={w}")
                # 强制输出到文件
                import os
                os.makedirs("reports/306_J4", exist_ok=True)
                with open("reports/306_J4/j4_debug.log", "a") as f:
                    f.write(f"[J4 DEBUG] step={self.global_step}, enable={enable}, w={w}\n")
            
            if enable and w > 0.0 and not loss_dict.get("__fresco_paper_losses__", False):
                # 调试：J4被调用
                if self.global_step % 10 == 0:
                    print(f"[J4 CALLED] step={self.global_step}, J4 is being called!")
                    import os
                    os.makedirs("reports/306_J4", exist_ok=True)
                    with open("reports/306_J4/j4_debug.log", "a") as f:
                        f.write(f"[J4 CALLED] step={self.global_step}, J4 is being called!\n")
                
                # 确保数据范围一致: 都归一化到 [0, 1]
                gt = self.img.to(device=outputs["render_face"].device, dtype=torch.float32)
                # 如果gt是[0,255]范围，归一化到[0,1]
                if gt.max() > 1.0:
                    gt = gt / 255.0
                gt = gt.clamp(0.0, 1.0)
                if gt.dim() == 4 and gt.shape[1] == 3:  # BCHW -> BHWC
                    gt = gt.permute(0, 2, 3, 1).contiguous()
                pred = outputs["render_face"]  # float32, cuda, [0,1]
                if pred.dim() == 4 and pred.shape[1] == 3:  # BCHW
                    pred = pred.permute(0, 2, 3, 1).contiguous()
                # 构造batch数据用于get_face_vis_mask
                batch_data = {
                    'head_mask': self.mask["head"],
                    'img': self.img,
                    'global_step': self.global_step
                }
                # 在outputs中也添加global_step
                outputs['global_step'] = self.global_step

                # 获取面部可见性mask
                face_mask = get_face_vis_mask(batch_data, outputs)  # [B,H,W] bool
                face_mask = face_mask.to(device=pred.device)
                if face_mask.dtype != torch.bool:
                    face_mask = face_mask.bool()
                # 调试信息：检查face_mask的值
                if self.global_step % 10 == 0:
                    face_mask_count = face_mask.sum().item()
                    print(f"[J4 DEBUG] face_mask: count={face_mask_count}, shape={face_mask.shape}")
                if face_mask.sum().item() == 0:
                    if self.global_step % 10 == 0:
                        print(f"[J4 DEBUG] step={self.global_step} skip J4: empty face_mask")
                    loss_dict["hf_img_raw"] = torch.tensor(0.0, device=pred.device)
                    loss_dict["hf_img_face_cov"] = 0.0
                    loss_dict["hf_img_edge_cov"] = 0.0
                    loss_dict["hf_img_ksize"] = cfg_hf["ksize"]
                    loss_dict["hf_img_edge_method"] = cfg_hf["edge_method"]
                    loss_dict["hf_img_w"] = w
                   
                else:
    # 计算高频图像损失（只有在前景非空时才调用） - 传入 report_dir
                    stat = compute_hfimg_loss(pred, gt, face_mask, cfg_hf, int(self.global_step),
                                             report_dir=self._get_report_dir())               # 计算高频图像损失
                    if not torch.isfinite(stat["loss"]):
                        if self.global_step % 10 == 0:
                            print(f"[J4 WARN] non-finite loss at step {self.global_step}, skip this term.")
                        loss_dict["hf_img_raw"] = torch.tensor(0.0, device=pred.device)
                        loss_dict["hf_img_face_cov"] = 0.0
                        loss_dict["hf_img_edge_cov"] = 0.0
                        loss_dict["hf_img_ksize"] = cfg_hf["ksize"]
                        loss_dict["hf_img_edge_method"] = cfg_hf["edge_method"]
                        loss_dict["hf_img_w"] = w
                    else:
                        # 正常记账
                        loss_dict["hf_img_raw"] = stat["raw"]
                        loss_dict["hf_img_face_cov"] = stat["face_coverage"]
                        loss_dict["hf_img_edge_cov"] = stat["edge_coverage"]
                        loss_dict["hf_img_ksize"] = stat["ksize"]
                        loss_dict["hf_img_edge_method"] = stat["edge_method"]
                        loss_dict["hf_img_w"] = w
                        loss += w * stat["loss"]
                        if self.global_step % 50 == 0:
                            self.logger.info(
                                f"[J4 HF-IMG] step={self.global_step}, loss={stat['loss'].item():.6f}, "
                                f"weight={w:.3f}, raw={stat['raw'].item():.6f}, "
                                f"face_cov={stat['face_coverage']:.6f}, edge_cov={stat['edge_coverage']:.6f}"
                            )
                        loss_dict["loss"] = loss
               
        except Exception as e:
            if not self.config.get("exp", {}).get("logging", {}).get("silent", False):
                print(f"[HF-IMG] skipped due to error: {e}")

        # 返回前统一覆盖总损失
        loss_dict["loss"] = loss
        for key, meter in self.fresco_losses.items():
            if key in loss_dict:
                value = loss_dict[key]
                meter.update(float(value.detach()) if isinstance(value, torch.Tensor) else float(value))
        return loss_dict

    def log_training(self, epoch, step, global_step, dataset_length, loss_dict):
        loss = loss_dict["loss"]
        loss_rgb_obj = loss_dict["loss_pho/rgb.obj"]
        loss_rgb_hair = loss_dict["loss_pho/rgb.hair"]
        loss_rgb_head = loss_dict["loss_pho/rgb.head"]
        loss_rgb_basic_head = loss_dict["loss_pho/rgb.basic_head"]
        loss_silh_hair = loss_dict["loss_geo/silh.hair"]
        loss_depth_head = loss_dict["loss_geo/depth.head"]
        loss_normal_head = loss_dict["loss_geo/normal.head"]
        loss_ssim_obj = loss_dict["loss_pho/ssim.obj"]
        loss_ssim_hair = loss_dict["loss_pho/ssim.hair"]
        loss_ssim_head = loss_dict["loss_pho/ssim.head"]
        loss_mesh_laplacian = loss_dict["loss_reg/mesh.laplacian"]
        loss_mesh_normal = loss_dict["loss_reg/mesh.normal"]
        loss_mesh_edges = loss_dict["loss_reg/mesh.edges"]
        loss_mesh_vscale = loss_dict["loss_reg/mesh.vscale"]
        loss_silh_solid_hair = loss_dict["loss_reg/silh.solid_hair"]
        
        # J1 LF-IMG loss
        loss_lf_img = loss_dict.get("lf_img_raw", 0.0)
        if isinstance(loss_lf_img, torch.Tensor):
            loss_lf_img = loss_lf_img.item()
        loss_lf_img_w = loss_dict.get("lf_img_w", 0.0)
        
        # J2 LF-UV loss - 显示原始损失（raw）而不是加权损失（weighted）
        loss_lf_uv = loss_dict.get("lf_uv/raw", 0.0)  # 使用原始损失，更清晰
        if isinstance(loss_lf_uv, torch.Tensor):
            loss_lf_uv = loss_lf_uv.item()
        loss_lf_uv_w = self.config.get("exp.lf_uv.loss.weight", 0.0)
        
        # J4 HF-IMG loss
        loss_hf_img = loss_dict.get("hf_img_raw", 0.0)
        if isinstance(loss_hf_img, torch.Tensor):
            loss_hf_img = loss_hf_img.item()
        loss_hf_img_w = loss_dict.get("hf_img_w", 0.0)

        lr = self.scheduler.get_last_lr()[0]
        self.logger.info(
            "stage [%s] epoch [%.3d] step [%d/%d] global_step = %d loss = %.4f lr = %.6f\n"
            "        rgb = %.4f                     w: %.4f\n"
            "           hair = %.4f                 w: %.4f\n"
            "           head = %.4f                 w: %.4f\n"
            "           basic_head = %.4f           w: %.4f\n"
            "        silh:                                 \n"
            "           hair = %.4f                 w: %.4f\n"
            "        depth:                                \n"
            "           head = %.4f                 w: %.4f\n"
            "        normal:                               \n"
            "           head = %.4f                 w: %.4f\n"
            "        ssim = %.4f                    w: %.4f\n"
            "           hair = %.4f                 w: %.4f\n"
            "           head = %.4f                 w: %.4f\n"
            "        reg:                                  \n"
            "           mesh_laplacian = %.4f       w: %.4f\n"
            "           mesh_normal = %.4f          w: %.4f\n"
            "           mesh_edges = %.4f           w: %.4f\n"
            "           mesh_vscale = %.4f          w: %.4f\n"
            "           silh_binary = %.4f          w: %.4f\n"
            "        lf_img:                               \n"
            "           loss = %.4f                 w: %.4f\n"
            "        lf_uv:                                \n"
            "           loss = %.4f                 w: %.4f\n"
            "        hf_img:                               \n"
            "           loss = %.4f                 w: %.4f\n"
            % (
                self.stage,
                epoch,
                step,
                dataset_length,
                self.global_step,
                loss.item(),
                lr,
                loss_rgb_obj.item(),
                self.get_lambda("rgb"),
                loss_rgb_hair.item(),
                self.get_lambda("rgb.hair"),
                loss_rgb_head.item(),
                self.get_lambda("rgb.head"),
                loss_rgb_basic_head.item(),
                self.get_lambda("rgb.head"),
                loss_silh_hair.item(),
                self.get_lambda("silh.hair"),
                loss_depth_head.item(),
                self.get_lambda("depth.head"),
                loss_normal_head.item(),
                self.get_lambda("normal.head"),
                loss_ssim_obj.item(),
                self.get_lambda("ssim"),
                loss_ssim_hair.item(),
                self.get_lambda("ssim"),
                loss_ssim_head.item(),
                self.get_lambda("ssim"),
                loss_mesh_laplacian.item(),
                self.get_lambda("mesh.laplacian"),
                loss_mesh_normal.item(),
                self.get_lambda("mesh.normal"),
                loss_mesh_edges.item(),
                self.get_lambda("mesh.edges"),
                loss_mesh_vscale.item(),
                self.get_lambda("mesh.verts_scale"),
                loss_silh_solid_hair.item(),
                self.get_lambda("silh.solid_hair"),
                loss_lf_img,
                loss_lf_img_w,
                loss_lf_uv,
                loss_lf_uv_w,
                loss_hf_img,
                loss_hf_img_w,
            )
        )

        optional = {
            "lf_img_raw": loss_dict.get("lf_img_raw"),
            "hf_img_edge": loss_dict.get("hf_img_edge"),
            "hf_img_gdl": loss_dict.get("hf_img_gdl"),
            "lf_uv_raw": loss_dict.get("lf_uv/raw"),
            "lf_uv_seam_pair": loss_dict.get("lf_uv_seam/pair"),
            "lf_uv_seam_tv": loss_dict.get("lf_uv_seam/tv"),
        }
        if schedule_enabled(self.config):
            optional.update({
                "lf_img_weighted": loss_dict.get("lf_img_weighted"),
                "hf_img_weighted": loss_dict.get("hf_img_weighted"),
                "lf_uv_weighted": loss_dict.get("lf_uv/weighted"),
                "lf_uv_seam_weighted": loss_dict.get("lf_uv_seam/weighted"),
            })
        visible = {key: (float(value.detach()) if isinstance(value, torch.Tensor) else float(value))
                   for key, value in optional.items() if value is not None}
        if visible:
            self.logger.info("FReSCO components: %s", visible)
            if self.tb_writer:
                for key, value in visible.items():
                    self.tb_writer.add_scalar("fresco/" + key, value, global_step)

        # Write losses to tensorboard
        # Update avg meters
        for key, value in self.train_losses.items():
            if self.tb_writer:
                self.tb_writer.add_scalar(key, loss_dict[key].item(), global_step)
            value.update(loss_dict[key].item())

    def run_eval(self, val_loader):
        self.logger.info("Start running evaluation on validation set:")
        self.set_eval()

        # clear train losses average meter
        for val_loss_item in self.val_losses.values():
            val_loss_item.reset()

        batch_count = 0
        with torch.no_grad():
            for step, items in enumerate(val_loader):
                batch_count += 1
                if batch_count % 20 == 0:
                    self.logger.info("    Eval progress: {}/{}".format(batch_count, len(val_loader)))

                self.set_data(items)
                outputs = self.network_forward(is_val=True)
                # The curriculum uses training-only paired UV batches. Keep
                # legacy validation behavior when the curriculum is disabled.
                loss_dict = self.compute_loss(
                    outputs, include_fresco=not schedule_enabled(self.config)
                )

                mse, psnr = self.compute_metrics(outputs)

                loss_dict["metrics/mse"] = mse
                loss_dict["metrics/psnr"] = psnr

                self.log_val(step, loss_dict)

            # log evaluation result
            self.logger.info("Evaluation finished, average losses: ")
            for v in self.val_losses.values():
                self.logger.info("    {}".format(v))

            # Write val losses to tensorboard
            if self.tb_writer:
                for key, value in self.val_losses.items():
                    self.tb_writer.add_scalar(key + "//val", value.avg, self.global_step)

        self.set_train()

    def log_val(self, step, loss_dict):
        B = self.batch_size
        # loss logging
        for key, value in self.val_losses.items():
            value.update(loss_dict[key].item(), n=B)

    def compute_metrics(self, outputs):
        if outputs["fullmask"] is None:
            return np.array(0.0), np.array(0.0)

        valid_mask = outputs["fullmask"] * self.mask["full"]

        gt_img = (self.img[0] * valid_mask[0, ..., None]).detach().cpu().numpy() * 255
        pred_img = (outputs["render_fuse"][0] * valid_mask[0, ..., None]).detach().cpu().numpy() * 255
        mse = ((pred_img - gt_img) ** 2).mean()
        psnr = 10 * np.log10(65025 / mse)

        return mse, psnr

    def visualization(self, outputs, step, label="log"):
        # create dirs
        logdir = os.path.join(self.config["local_workspace"], label)
        directory(logdir)
        if label == "log":
            savedir = os.path.join(logdir, "it{}".format(step))
            directory(savedir)
        elif label == "eval":
            savedir = os.path.join(logdir, self.name[0])
            directory(savedir)

        valid_mask = self.mask["full"]  # foreground mask
        hair_mask = self.mask["hair"]
        face_mask = self.mask["head"]
        valid_mask, face_mask, hair_mask = valid_mask.bool(), face_mask.bool(), hair_mask.bool()

        # gt_img
        savepath = os.path.join(savedir, "gt_it{}.png".format(step))
        gt_img = self.img[0].detach().cpu().numpy()
        cv2.imwrite(savepath, gt_img * 255)

        # image
        savepath = os.path.join(savedir, "rendering_it{}.png".format(step))
        render_fuse = outputs["render_fuse"][0].detach().cpu().numpy()
        cv2.imwrite(savepath, render_fuse * 255)

        face_visuals = self.facewrapper.visualize(savedir, outputs, step)
        hair_visuals = self.hairwrapper.visualize(savedir, outputs, step)
        gt_head, gt_head_normal, render_head, raster_headmask, head_normal, head_geomap, colored_mask = (
            face_visuals["gt_head"],
            face_visuals["gt_head_normal"],
            face_visuals["render_head"],
            face_visuals["raster_headmask"],
            face_visuals["head_normal"],
            face_visuals["head_geomap"],
            face_visuals["colored_mask"],
        )
        gt_hair, raster_hairmask, render_hair = (
            hair_visuals["gt_hair"],
            hair_visuals["raster_hairmask"],
            hair_visuals["render_hair"],
        )

        # concat img
        white_img = np.ones((self.img_h, self.img_w, 3))

        savepath = os.path.join(savedir, "combined_it{}.png".format(step))
        gt = np.concatenate([gt_img, gt_head, gt_head_normal[..., ::-1], gt_hair], axis=1)

        alpha = 0.4
        color = [1.0, 0.0, 0.0]
        head_withmask = cv2.addWeighted(render_head, 1.0, color_mask(raster_hairmask, color=color), alpha, 0)
        hair_withmask = cv2.addWeighted(render_hair, 1.0, color_mask(raster_headmask, color=color), alpha, 0)
        pred = np.concatenate([render_fuse, head_withmask, head_normal[..., ::-1], hair_withmask], axis=1)

        alpha = 0.5
        color_pred = [0.0, 0.0, 0.8]
        color_gt = [0.0, 0.6, 0.3]
        headmask_gt = face_mask[0, ..., None].detach().cpu().numpy().repeat(3, axis=-1)
        hairmask_gt = hair_mask[0, ..., None].detach().cpu().numpy().repeat(3, axis=-1)
        headmask_withgt = cv2.addWeighted(
            color_mask(raster_headmask, color=color_pred, bg_white=True),
            1.0,
            color_mask(headmask_gt, color=color_gt),
            alpha,
            0,
        )
        hairmask_withgt = cv2.addWeighted(
            color_mask(raster_hairmask, color=color_pred, bg_white=True),
            1.0,
            color_mask(hairmask_gt, color=color_gt),
            alpha,
            0,
        )
        mask = np.concatenate([head_geomap, headmask_withgt, colored_mask, hairmask_withgt], axis=1)
        combined_img = np.concatenate([gt, pred, mask], axis=0)
        cv2.imwrite(savepath, combined_img * 255)

        self.facewrapper.visualize_textures(savedir, step)
        # fuse mask
        if outputs["fullmask"] is not None:
            savepath = os.path.join(savedir, "fullmask_it{}.png".format(step))
            fullmask = outputs["fullmask"] * self.mask["full"]
            cv2.imwrite(savepath, fullmask[0].detach().cpu().numpy() * 255)

        # metrics
        if label == "eval":
            savepath = os.path.join(savedir, "metrics.txt")
            mse, psnr = self.compute_metrics(outputs)

            with open(savepath, "w") as f:
                f.write("MSE: {}\n".format(mse))
                f.write("PSNR: {}\n".format(psnr))

            print("MSE: {}\nPSNR: {}\n".format(mse, psnr))

    def clip_grad(self, max_norm=0.01):
        for dict in self.parameters_to_train:
            torch.nn.utils.clip_grad_norm_(dict["params"], max_norm)

    def save_ckpt(self, savepath, stage=None):
        save_dict = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "stage": self.stage if stage is None else stage,
            "stage_step": self.stage_step,
        }
        save_dict.update(self.facewrapper.state_dict())
        save_dict.update(self.hairwrapper.state_dict())

        torch.save(save_dict, savepath)

        basedir = os.path.dirname(savepath)
        npz_path = os.path.join(basedir, "flame_params.npz")
        if os.path.exists(npz_path):
            os.remove(npz_path)

        flame_params = {k: v.detach().cpu().numpy() for k, v in self.all_flame_params.items()}
        np.savez(str(npz_path), **flame_params)

    def nan_debug(self, loss):
        if not self.nan_detect and loss.isnan().any():
            self.nan_detect = True
            checkpoint_path = os.path.join(self.config["local_workspace"], "nan_break.pth")
            self.save_ckpt(checkpoint_path)
            print("NaN break checkpoint has saved at {}".format(checkpoint_path))
            print("Data {}".format(self.name))
            return False

    def train_epoch(self, train_loader, val_loader, show_time=False):
        print("[DEBUG] JointTrainer.train_epoch: Starting train_epoch")
        self.logger.info("[DEBUG] JointTrainer.train_epoch: Starting train_epoch")
        # convert models to traning mode
        self.set_train()
        for meter in self.fresco_losses.values():
            meter.reset()
        print("[DEBUG] JointTrainer.train_epoch: set_train completed")
        self.logger.info("[DEBUG] JointTrainer.train_epoch: set_train completed")

        warmup_steps = 10
        ld_timer = CUDA_Timer("load data", self.logger, valid=show_time, warmup_steps=warmup_steps)
        sd_timer = CUDA_Timer("set data", self.logger, valid=show_time, warmup_steps=warmup_steps)
        f_timer = CUDA_Timer("forward", self.logger, valid=show_time, warmup_steps=warmup_steps)
        cl_timer = CUDA_Timer("compute loss", self.logger, valid=show_time, warmup_steps=warmup_steps)
        b_timer = CUDA_Timer("backward", self.logger, valid=show_time, warmup_steps=warmup_steps)
        pp_timer = CUDA_Timer("post process", self.logger, valid=show_time, warmup_steps=warmup_steps)
        up_timer = CUDA_Timer("update params", self.logger, valid=show_time, warmup_steps=warmup_steps)
        ld_timer.start(0)
        print("[DEBUG] JointTrainer.train_epoch: About to start training loop")
        self.logger.info("[DEBUG] JointTrainer.train_epoch: About to start training loop")

        print("[DEBUG] JointTrainer.train_epoch: About to enumerate train_loader")
        for step, items in enumerate(train_loader):
            print(f"[DEBUG] JointTrainer.train_epoch: Got batch {step}")
            step += 1
            self.global_step += 1
            self.stage_step += 1
            print(f"[DEBUG] JointTrainer.train_epoch: Processing step {step}, global_step {self.global_step}")

            if show_time and self.global_step > warmup_steps:
                ld_timer.end(step - 1)

            if self.stage in ["joint"]:
                self.hairwrapper.update_xyz_lr(self.stage_step, self.optimizer)

            # 1. Set data for trainer
            sd_timer.start(step)
            self.set_data(items)
            sd_timer.end(step)

            # 2. Run the network
            f_timer.start(step)
            outputs = self.network_forward()
            f_timer.end(step)
            
            # J2 UV Bake trigger (before loss computation) - 从epoch 0开始执行
            self._prepare_cross_view_uv(outputs)

            # 3. Compute losses
            cl_timer.start(step)
            loss_dict = self.compute_loss(outputs)
            cl_timer.end(step)
            loss = loss_dict["loss"]

            self.nan_debug(loss)

            # 4. Backprop
            b_timer.start(step)
            loss.backward()
            b_timer.end(step)

            # 5. update parameters
            up_timer.start(step)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            up_timer.end(step)

            # logging
            if step > 0 and (step % 10 == 0 or step == len(train_loader)):
                self.log_training(
                    self.current_epoch,
                    step,
                    self.global_step,
                    len(train_loader),
                    loss_dict,
                )

            # Visualize
            if self.stage_step == 1 or self.global_step % self.config["training.visual_interval"] == 0:
                self.visualization(outputs, self.global_step)

            checkpoint_interval = int(self.config.get("training.checkpoint_interval", 2000))
            if checkpoint_interval > 0 and self.global_step % checkpoint_interval == 0:
                # Save model
                checkpoint_path = os.path.join(self.config["local_workspace"], "checkpoint_latest.pth")
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                self.save_ckpt(checkpoint_path)
                self.logger.info("Latest checkpoint saved at {}".format(checkpoint_path))

            if self.global_step % 5000 == 0:
                # Save adapted mesh
                obj_path = os.path.join(
                    self.config["local_workspace"], "mesh_with_offsets_it{}.obj".format(self.global_step)
                )
                self.facewrapper.save_mesh(obj_path)

            if self.global_step > 0 and self.global_step % self.config["training.eval_interval"] == 0:
                self.run_eval(val_loader)

                # Save model
                checkpoint_path = os.path.join(
                    self.config["local_workspace"],
                    "checkpoint_%012d.pth" % self.global_step,
                )
                self.save_ckpt(checkpoint_path)

            if show_time and self.global_step > warmup_steps:
                ld_timer.start(step)

        return True
