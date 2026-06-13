# Standalone KIP config: refine FoundationPose init poses on our Anker_Kurz frames.
# RefineTestDataset (NO GT) — crop by ref (FP) pose bbox. Units: mm everywhere.
image_scale = 256
normalize_mean = [0., 0., 0.]
normalize_std = [255., 255., 255.]
mesh_dir = 'data/kip/models'
mesh_diameter = [128.58, 143.057]            # obj_000001, obj_000002 (mm)
symmetry_types = {}                          # unseen mode: no symmetry canonicalization
file_client_args = dict(backend='disk')

test_pipeline = [
    dict(type='LoadImages', color_type='unchanged', file_client_args=file_client_args),
    dict(type='LoadMasks'),
    dict(type='ComputeBbox', mesh_dir=mesh_dir, clip_border=False, filter_invalid=False,
         pose_field=['ref_rotations', 'ref_translations'], bbox_field='ref_bboxes'),
    dict(type='Crop', size_range=(1.1, 1.1), crop_bbox_field='ref_bboxes',
         clip_border=False, pad_val=128),
    dict(type='Resize', img_scale=image_scale, keep_ratio=True),
    dict(type='Pad', size=(image_scale, image_scale), center=True,
         pad_val=dict(img=(128, 128, 128), mask=0)),
    dict(type='RemapPose', keep_intrinsic=False),
    dict(type='GetPointCloud', filter_depth=False, filter_point_cloud=False,
         minimum_points=32, depth_sample_num=1024, filter_rgb=False),
    dict(type='Normalize', mean=normalize_mean, std=normalize_std, to_rgb=True),
    dict(type='ToTensor', stack_keys=[]),
    dict(type='Collect',
         annot_keys=['ref_rotations', 'ref_translations', 'labels', 'k', 'ori_k',
                     'transform_matrix', 'depths', 'model_list', 'cloud_list', 'gt_masks'],
         meta_keys=('img_path', 'ori_shape', 'img_shape', 'img_norm_cfg',
                    'scale_factor', 'keypoints_3d', 'geometry_transform_mode')),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    test_samples_per_gpu=1,
    test=dict(
        type='RefineTestDataset',
        data_root='data/kip/test',
        ref_annots_root='data/kip/init',
        image_list='data/kip/image_list.txt',
        keypoints_json='data/kip/keypoints.json',
        keypoints_num=8,
        pipeline=test_pipeline,
        class_names=('anker_kurz', 'anker_lang'),
        label_mapping=None,
        target_label=None,
        load_depth=True,
        load_mask=True,
        load_point_clouds=True,
        crop_depth=True,
        meshes_eval=mesh_dir,
        mesh_symmetry=symmetry_types,
        mesh_diameter=mesh_diameter,
        mesh_sample_num=1024,
    ),
)
num_gpus = 1

model = dict(
    type='SCFlow2Refiner',
    cxt_channels=384, h_channels=128, seperate_encoder=False, cxt_feat_detach=True,
    max_flow=400., solve_type='reg', add_dense_fusion=True, filter_invalid_flow=True,
    encoder=dict(type='DINOv2Encoder', in_channels=3, out_channels=256, net_type='basic',
                 norm_cfg=dict(type='IN'),
                 init_cfg=[dict(type='Kaiming', layer=['Conv2d'], mode='fan_out', nonlinearity='relu'),
                           dict(type='Constant', layer=['InstanceNorm2d'], val=1, bias=0)]),
    cxt_encoder=dict(type='SCFlow2Decoder', in_channels=3, out_channels=256, net_type='basic',
                     norm_cfg=dict(type='BN'),
                     init_cfg=[dict(type='Kaiming', layer=['Conv2d'], mode='fan_out', nonlinearity='relu'),
                               dict(type='Constant', layer=['SyncBatchNorm2d'], val=1, bias=0)]),
    decoder=dict(type='SCFlow2Decoder', net_type='Basic', num_levels=4, radius=4, iters=8,
                 cxt_channels=384, detach_flow=True, detach_mask=True, detach_pose=True,
                 detach_depth_for_xy=True, depth_based_upsample=False, mask_flow=False, mask_corr=False,
                 pose_head_cfg=dict(type='SceneFlowPoseHead', in_channels=16, net_type='Basic',
                                    rotation_mode='ortho6d',
                                    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                                    act_cfg=dict(type='ReLU')),
                 corr_lookup_cfg=dict(align_corners=True), gru_type='SeqConv', act_cfg=dict(type='ReLU')),
    freeze_bn=False, freeze_encoder=False,
    train_cfg=dict(rendered_mask_filte=True, online_image_renderer=True),
    test_cfg=dict(iters=8),
    renderer=dict(mesh_dir=mesh_dir, image_size=(image_scale, image_scale), shader_type='Phong',
                  soft_blending=False, render_mask=False, render_image=True, seperate_lights=True,
                  faces_per_pixel=1, bin_size=-1, blur_radius=0., sigma=1e-12, gamma=1e-12,
                  background_color=(.5, .5, .5)),
)
