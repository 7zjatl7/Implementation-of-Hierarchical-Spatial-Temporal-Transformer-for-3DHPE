# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import math
import matplotlib.pyplot as plt


from einops import rearrange, repeat
from copy import deepcopy

from common.camera import *
import collections

# from common.diffusionpose import *
from model.HPFormer import HPFormer

from common.loss import *
from common.generators import ChunkedGenerator_Seq, UnchunkedGenerator_Seq
from time import time
from common.utils import *
from common.logging import Logger
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
#cudnn.benchmark = True       
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 3'
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# args.evaluate = 'best_epoch_100.bin'
if args.evaluate != '':
    description = "Evaluate!"
elif args.evaluate == '':
    description = "Train!"

# args.checkpoint = 'log/default_20231017T13-26-38'

# initial setting
TIMESTAMP = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())
# tensorboard
if not args.nolog:
    writer = SummaryWriter(args.log+'_'+TIMESTAMP)
    writer.add_text('description', description)
    writer.add_text('command', 'python ' + ' '.join(sys.argv))
    # logging setting
    logfile = os.path.join(args.log+'_'+TIMESTAMP, 'logging.log')
    sys.stdout = Logger(logfile)
print(description)
print('python ' + ' '.join(sys.argv))
print("CUDA Device Count: ", torch.cuda.device_count())
print(args)

manualSeed = 1
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

# if not assign checkpoint path, Save checkpoint file into log folder
if args.checkpoint=='':
    args.checkpoint = args.log+'_'+TIMESTAMP
try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

# dataset loading
print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('humaneva'):
    from common.humaneva_dataset import HumanEvaDataset
    dataset = HumanEvaDataset(dataset_path)
elif args.dataset.startswith('custom'):
    from common.custom_dataset import CustomDataset
    dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
else:
    raise KeyError('Invalid dataset')

print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

###################
for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):

            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])



for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps

subjects_train = args.subjects_train.split(',')
subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
if not args.render:
    subjects_test = args.subjects_test.split(',')
else:
    subjects_test = [args.viz_subject]


def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    
    # 캠마다 2d랑 3d pose만 따로 담는 리스트 생성
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue
            
            # cam : 0, 1, 2, 3 이 있나보다
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
            
            # dataset에 cam : 0, 1, 2, 3이랑 pose_2d의 0, 1, 2, 3을 맞춤
            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]


    return out_camera_params, out_poses_3d, out_poses_2d

action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

# set receptive_field as number assigned
receptive_field = args.number_of_frames
print('INFO: Receptive field: {} frames'.format(receptive_field))
if not args.nolog:
    writer.add_text(args.log+'_'+TIMESTAMP + '/Receptive field', str(receptive_field))
pad = (receptive_field -1) // 2 # Padding on each side
min_loss = args.min_loss
width = cam['res_w']
height = cam['res_h']
num_joints = keypoints_metadata['num_joints']

######################################################################
print('receptive_field : ', receptive_field)
model_pos_train = HPFormer(args=args, num_joints=num_joints, num_frame=receptive_field, is_train=True)
# model_pos_test = HPDF(args,joints_left, joints_right)
model_pos_test = HPFormer(args=args, num_joints=num_joints, num_frame=receptive_field, is_train=False)
######################################################################

causal_shift = 0
model_params = 0
for parameter in model_pos_test.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params/1000000, 'Million')
if not args.nolog:
    writer.add_text(args.log+'_'+TIMESTAMP + '/Trainable parameter count', str(model_params/1000000) + ' Million')

# make model parallel
######################################################################
if torch.cuda.is_available():

    model_pos_train = nn.DataParallel(model_pos_train)
    model_pos_train = model_pos_train.cuda()
    model_pos_test = nn.DataParallel(model_pos_test)
    model_pos_test = model_pos_test.cuda()
######################################################################

if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    # chk_filename = args.resume or args.evaluate
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)
    model_pos_test.load_state_dict(checkpoint['model_pos'], strict=False)


test_generator = UnchunkedGenerator_Seq(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))
if not args.nolog:
    writer.add_text(args.log+'_'+TIMESTAMP + '/Testing Frames', str(test_generator.num_frames()))

def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):

    assert inputs_2d.shape[:-1] == inputs_3d.shape[:-1], "2d and 3d inputs shape must be same! "+str(inputs_2d.shape)+str(inputs_3d.shape)
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = torch.squeeze(inputs_3d)

    if inputs_2d_p.shape[0] / receptive_field > inputs_2d_p.shape[0] // receptive_field: 
        out_num = inputs_2d_p.shape[0] // receptive_field+1
    elif inputs_2d_p.shape[0] / receptive_field == inputs_2d_p.shape[0] // receptive_field:
        out_num = inputs_2d_p.shape[0] // receptive_field

    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    eval_input_3d = torch.empty(out_num, receptive_field, inputs_3d_p.shape[1], inputs_3d_p.shape[2])

    for i in range(out_num-1):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
        eval_input_3d[i,:,:,:] = inputs_3d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
    if inputs_2d_p.shape[0] < receptive_field:
        from torch.nn import functional as F
        pad_right = receptive_field-inputs_2d_p.shape[0]
        inputs_2d_p = rearrange(inputs_2d_p, 'b f c -> f c b')
        inputs_2d_p = F.pad(inputs_2d_p, (0,pad_right), mode='replicate')
        # inputs_2d_p = np.pad(inputs_2d_p, ((0, receptive_field-inputs_2d_p.shape[0]), (0, 0), (0, 0)), 'edge')
        inputs_2d_p = rearrange(inputs_2d_p, 'f c b -> b f c')
    if inputs_3d_p.shape[0] < receptive_field:
        pad_right = receptive_field-inputs_3d_p.shape[0]
        inputs_3d_p = rearrange(inputs_3d_p, 'b f c -> f c b')
        inputs_3d_p = F.pad(inputs_3d_p, (0,pad_right), mode='replicate')
        inputs_3d_p = rearrange(inputs_3d_p, 'f c b -> b f c')
    eval_input_2d[-1,:,:,:] = inputs_2d_p[-receptive_field:,:,:]
    eval_input_3d[-1,:,:,:] = inputs_3d_p[-receptive_field:,:,:]

    return eval_input_2d, eval_input_3d


###################

# Training start
if not args.evaluate:
    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)

    lr = args.learning_rate
    optimizer = optim.AdamW(model_pos_train.parameters(), lr=lr, weight_decay=0.1)

#########################################################
    lr_decay = args.lr_decay
    losses_3d_train = []
    losses_3d_pos_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []
    losses_3d_depth_valid = []

    epoch = 0
    best_epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001
    
#########################################################

    # get training data
    train_generator = ChunkedGenerator_Seq(args.batch_size//args.stride, cameras_train, poses_train, poses_train_2d, args.number_of_frames,
                                       pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    train_generator_eval = UnchunkedGenerator_Seq(cameras_train, poses_train, poses_train_2d,
                                              pad=pad, causal_shift=causal_shift, augment=False)
    print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))
    if not args.nolog:
        writer.add_text(args.log+'_'+TIMESTAMP + '/Training Frames', str(train_generator_eval.num_frames()))

    if args.resume:
        epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_generator.set_random_state(checkpoint['random_state'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
        if not args.coverlr:
            lr = checkpoint['lr']

    print('** Note: reported losses are averaged over all frames.')
    print('** The final evaluation will be carried out after the last training epoch.')
    
    print('======================================= Starting Train =======================================')
    # Pos model only
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_3d_pos_train = 0
        epoch_loss_3d_diff_train = 0
        epoch_loss_traj_train = 0
        epoch_loss_2d_train_unlabeled = 0
        N = 0
        N_semi = 0
        model_pos_train.train()
        iteration = 0

        num_batches = train_generator.num_frames()
        
        # Just train 1 time, for quick debug
        # quickdebug=args.debug
        for cameras_train, batch_3d, batch_2d in tqdm(train_generator.next_epoch()):
            # if notrain:break
            # notrain=True

            if iteration % 100 == 0:
                print("%d/%d"% (iteration, num_batches))

            if cameras_train is not None:
                cameras_train = torch.from_numpy(cameras_train.astype('float32'))
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
                if cameras_train is not None:
                    cameras_train = cameras_train.cuda()
            # 배치마다 맨 첫번째 pelvis 부분의 초기 Pose Clone
            inputs_traj = inputs_3d[:, :, :1].clone()
            # 초기포즈 0으로 초기화
            inputs_3d[:, :, 0] = 0

            optimizer.zero_grad()
            
            predicted_3d_pos = model_pos_train(inputs_2d)
            
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)

            loss_total = loss_3d_pos
            
            loss_total.backward(loss_total.clone().detach())

            loss_total = torch.mean(loss_total)

            epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_total.item()
            epoch_loss_3d_pos_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]
            
            
            optimizer.step()
            # del inputs_3d, loss_3d_pos, predicted_3d_pos
            # torch.cuda.empty_cache()
            print('Iteration Loss :', loss_total.item())

            iteration += 1

            # if quickdebug:
            #     if N==inputs_3d.shape[0] * inputs_3d.shape[1]:
            #         break
        print('Epoch Loss 3d train:', epoch_loss_3d_train / N)
        
        print('Epoch Loss 3d pose train :', epoch_loss_3d_pos_train / N)
        losses_3d_train.append(epoch_loss_3d_train / N)
        losses_3d_pos_train.append(epoch_loss_3d_pos_train / N)
        # torch.cuda.empty_cache()



        # End-of-epoch evaluation
        with torch.no_grad():
            print('======================================= Starting Test =======================================')
            model_pos_test.load_state_dict(model_pos_train.state_dict(), strict=False)
            model_pos_test.eval()

            epoch_loss_3d_valid = None
            epoch_loss_3d_depth_valid = 0
            epoch_loss_traj_valid = 0
            epoch_loss_2d_valid = 0
            epoch_loss_3d_vel = 0
            N = 0
            iteration = 0
            if not args.no_eval:
                # Evaluate on test set
                for cam, batch, batch_2d in tqdm(test_generator.next_epoch()):
                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

                    ##### apply test-time-augmentation (following Videopose3d)
                    inputs_2d_flip = inputs_2d.clone()
                    inputs_2d_flip[:, :, :, 0] *= -1
                    inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

                    ##### convert size
                    inputs_3d_p = inputs_3d
                    inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d_p)
                    inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d_p)

                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                        inputs_2d_flip = inputs_2d_flip.cuda()
                    inputs_3d[:, :, 0] = 0

                    predicted_3d_pos = model_pos_test(inputs_2d)  # b, t, h, f, j, c
                    print('Test')
                    print('predicted_3d_pos shape: ', predicted_3d_pos.shape)
                    print('predicted_3d_pos type: ', type(predicted_3d_pos))

                    # predicted_3d_pos[:, :, :, 0] = 0
                    # predicted_3d_pos.shape, inputs_3d.shape 확인
                    error = mpjpe(predicted_3d_pos, inputs_3d)
                    # error = mpjpe_diffusion(predicted_3d_pos, inputs_3d)

                    if iteration == 0:
                        epoch_loss_3d_valid = inputs_3d.shape[0] * inputs_3d.shape[1] * error.clone()
                    else:
                        epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * error.clone()

                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    iteration += 1

                print('Validation Epoch Loss :', epoch_loss_3d_valid / N)
                losses_3d_valid.append(epoch_loss_3d_valid / N)

        elapsed = (time() - start_time) / 60

        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000,
            ))

            log_path = os.path.join(args.checkpoint, 'training_log.txt')
            f = open(log_path, mode='a')
            f.write('[%d] time %.2f lr %f 3d_train %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000,
            ))
            f.close()

        else:
            print('[%d] time %.2f lr %f 3d_train %f 3d_pos_valid %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000,
                losses_3d_valid[-1].item() * 1000
            ))

            log_path = os.path.join(args.checkpoint, 'training_log.txt')
            f = open(log_path, mode='a')
            f.write('[%d] time %.2f lr %f 3d_train %f 3d_pos_valid %f\n' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000,
                losses_3d_valid[-1].item() * 1000
            ))
            f.close()

            if not args.nolog:
                # writer.add_scalar("Loss/3d training eval loss", losses_3d_train_eval[-1] * 1000, epoch+1)
                writer.add_scalar("Loss/3d validation loss", losses_3d_valid[-1].item() * 1000, epoch+1)
        if not args.nolog:
            writer.add_scalar("Loss/3d training loss", losses_3d_train[-1] * 1000, epoch+1)
            writer.add_scalar("Parameters/learing rate", lr, epoch+1)
            writer.add_scalar('Parameters/training time per epoch', elapsed, epoch+1)
        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        # Decay BatchNorm momentum
        # momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
        # model_pos_train.set_bn_momentum(momentum)

        # Save checkpoint if necessary
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)

            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                # 'min_loss': min_loss
                # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
            }, chk_path)

        #### save best checkpoint
        best_chk_path = os.path.join(args.checkpoint, 'best_epoch_%d.bin' % (best_epoch+1))
        if losses_3d_valid[-1].item() * 1000 < min_loss:
            min_loss = losses_3d_valid[-1].item() * 1000
            best_epoch = epoch
            print("save best checkpoint")
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
            }, best_chk_path)

            f = open(log_path, mode='a')
            f.write('best epoch\n')
            f.close()

        # Save training curves after every epoch, as .png images (if requested)
        if args.export_training_curves and epoch > 3:
            if 'matplotlib' not in sys.modules:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            print('epoch_x : ', epoch_x)
            print('losses_3d_train : ',losses_3d_train)
            print('losses_3d_train_eval : ',losses_3d_train_eval)
            print('losses_3d_valid : ',losses_3d_valid)
            plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
            plt.plot(epoch_x, losses_3d_train_eval[3:].item(), color='C0')
            plt.plot(epoch_x, losses_3d_valid[3:].item(), color='C1')
            plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))
            plt.close('all')
# Training end

# Evaluate
def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False, newmodel=None):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0

    with torch.no_grad():
        if newmodel is not None:
            print('Loading comparison model')
            model_eval = newmodel
            chk_file_path = '/data_disk/home/donghoon/research/Now/HPDFormer/checkpoint/epoch_100.bin'
            print('Loading evaluate checkpoint of comparison model', chk_file_path)
            checkpoint = torch.load(chk_file_path, map_location=lambda storage, loc: storage)
            model_eval.load_state_dict(checkpoint['model_pos'], strict=False)
            model_eval.eval()
        else:
            model_eval = model_pos_test
            if not use_trajectory_model:
                # load best checkpoint
                if args.evaluate == '':
                    chk_file_path = os.path.join(args.checkpoint, 'best_epoch_%d.bin' % (best_epoch))
                    print('Loading best checkpoint', chk_file_path)
                elif args.evaluate != '':
                    chk_file_path = os.path.join(args.checkpoint, args.evaluate)
                    print('Loading evaluate checkpoint', chk_file_path)
                checkpoint = torch.load(chk_file_path, map_location=lambda storage, loc: storage)
                print('This model was trained for {} epochs'.format(checkpoint['epoch']))
                # model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)
                model_eval.load_state_dict(checkpoint['model_pos'])
                model_eval.eval()
        # else:
            # model_traj.eval()
        N = 0
        iteration = 0

        #num_batches = test_generator.batch_num()
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_3d = torch.from_numpy(batch.astype('float32'))

            ##### apply test-time-augmentation (following Videopose3d)
            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip [:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right,:] = inputs_2d_flip[:, :, kps_right + kps_left,:]

            ##### convert size
            inputs_3d_p = inputs_3d
            if newmodel is not None:
                def eval_data_prepare_pf(receptive_field, inputs_2d, inputs_3d):
                    inputs_2d_p = torch.squeeze(inputs_2d)
                    inputs_3d_p = inputs_3d.permute(1,0,2,3)
                    padding = int(receptive_field//2)
                    inputs_2d_p = rearrange(inputs_2d_p, 'b f c -> f c b')
                    inputs_2d_p = F.pad(inputs_2d_p, (padding,padding), mode='replicate')
                    inputs_2d_p = rearrange(inputs_2d_p, 'f c b -> b f c')
                    out_num = inputs_2d_p.shape[0] - receptive_field + 1
                    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
                    for i in range(out_num):
                        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
                    return eval_input_2d, inputs_3d_p
                
                inputs_2d, inputs_3d = eval_data_prepare_pf(81, inputs_2d, inputs_3d_p)
                inputs_2d_flip, _ = eval_data_prepare_pf(81, inputs_2d_flip, inputs_3d_p)
            else:
                inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d_p)
                inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d_p)


            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_2d_flip = inputs_2d_flip.cuda()
                inputs_3d = inputs_3d.cuda()

            inputs_3d[:, :, 0] = 0

            predicted_3d_pos = model_eval(inputs_2d) #b, t, h, f, j, c
            predicted_3d_pos_flip = model_eval(inputs_2d_flip)
            predicted_3d_pos_flip[:, :, :, 0] *= -1
            predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                    joints_right + joints_left]
            for i in range(predicted_3d_pos.shape[0]):
                predicted_3d_pos[i,:,:,:] = (predicted_3d_pos[i,:,:,:] + predicted_3d_pos_flip[i,:,:,:])/2
            # predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1, keepdim=True)

            # del inputs_2d, inputs_2d_flip
            # torch.cuda.empty_cache()

            if return_predictions:
                return predicted_3d_pos.squeeze().cpu().numpy()

            error = mpjpe(predicted_3d_pos, inputs_3d)

            # error, joints_err = mpjpe(predicted_3d_pos, inputs_3d, return_joints_err=True)
            # joints_errs.append(joints_err)

            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    if action is None:
        print('----------')
    else:
        print('----'+action+'----')
    e1 = (epoch_loss_3d_pos / N)*1000
    e2 = (epoch_loss_3d_pos_procrustes / N)*1000
    e3 = (epoch_loss_3d_pos_scale / N)*1000
    ev = (epoch_loss_3d_vel / N)*1000
    print('Test time augmentation:', test_generator.augment_enabled())
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    return e1, e2, e3, ev


if args.render:
    print('Rendering...')

    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    ground_truth = None
    if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
        if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
            ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
    if ground_truth is None:
        print('INFO: this action is unlabeled. Ground truth will not be rendered.')

    gen = UnchunkedGenerator_Seq(None, [ground_truth], [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, return_predictions=True)
    if args.compare:
        from common.model_poseformer import PoseTransformer
        model_pf = PoseTransformer(num_frame=81, num_joints=17, in_chans=2, num_heads=8, mlp_ratio=2., qkv_bias=False, qk_scale=None,drop_path_rate=0.1)
        if torch.cuda.is_available():
            model_pf = nn.DataParallel(model_pf)
            model_pf = model_pf.cuda()
        prediction_pf = evaluate(gen, newmodel=model_pf, return_predictions=True)
        
        # ### reshape prediction_pf as ground truth
        # if ground_truth.shape[0] / receptive_field > ground_truth.shape[0] // receptive_field: 
        #     batch_num = (ground_truth.shape[0] // receptive_field) +1
        #     prediction_pf_2 = np.empty_like(ground_truth)
        #     for i in range(batch_num-1):
        #         prediction_pf_2[i*receptive_field:(i+1)*receptive_field,:,:] = prediction_pf[i,:,:,:]
        #     left_frames = ground_truth.shape[0] - (batch_num-1)*receptive_field
        #     prediction_pf_2[-left_frames:,:,:] = prediction_pf[-1,-left_frames:,:,:]
        #     prediction_pf = prediction_pf_2
        # elif ground_truth.shape[0] / receptive_field == ground_truth.shape[0] // receptive_field:
        #     prediction_pf.reshape(ground_truth.shape[0], 17, 3)

    # if model_traj is not None and ground_truth is None:
    #     prediction_traj = evaluate(gen, return_predictions=True, use_trajectory_model=True)
    #     prediction += prediction_traj
    ### reshape prediction as ground truth
    if ground_truth.shape[0] / receptive_field > ground_truth.shape[0] // receptive_field: 
        batch_num = (ground_truth.shape[0] // receptive_field) +1
        prediction2 = np.empty_like(ground_truth)
        for i in range(batch_num-1):
            prediction2[i*receptive_field:(i+1)*receptive_field,:,:] = prediction[i,:,:,:]
        left_frames = ground_truth.shape[0] - (batch_num-1)*receptive_field
        prediction2[-left_frames:,:,:] = prediction[-1,-left_frames:,:,:]
        prediction = prediction2
    elif ground_truth.shape[0] / receptive_field == ground_truth.shape[0] // receptive_field:
        prediction.reshape(ground_truth.shape[0], 17, 3)

    if args.viz_export is not None:
        print('Exporting joint positions to', args.viz_export)
        # Predictions are in camera space
        np.save(args.viz_export, prediction)

    if args.viz_output is not None:
        if ground_truth is not None:
            # Reapply trajectory
            trajectory = ground_truth[:, :1]
            ground_truth[:, 1:] += trajectory
            prediction += trajectory
            if args.compare:
                prediction_pf += trajectory

        # Invert camera transformation
        cam = dataset.cameras()[args.viz_subject][args.viz_camera]
        if ground_truth is not None:
            if args.compare:
                prediction_pf = camera_to_world(prediction_pf, R=cam['orientation'], t=cam['translation'])
            prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
            ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
        else:
            # If the ground truth is not available, take the camera extrinsic params from a random subject.
            # They are almost the same, and anyway, we only need this for visualization purposes.
            for subject in dataset.cameras():
                if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
                    rot = dataset.cameras()[subject][args.viz_camera]['orientation']
                    break
            if args.compare:
                prediction_pf = camera_to_world(prediction_pf, R=rot, t=0)
                prediction_pf[:, :, 2] -= np.min(prediction_pf[:, :, 2])
            prediction = camera_to_world(prediction, R=rot, t=0)
            # We don't have the trajectory, but at least we can rebase the height
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])
        
        if args.compare:
            anim_output = {'PoseFormer': prediction_pf}
            anim_output['Ours'] = prediction
            # print(prediction_pf.shape, prediction.shape)
        else:
            # anim_output = {'Reconstruction': prediction}
            anim_output = {'Reconstruction': ground_truth + np.random.normal(loc=0.0, scale=0.1, size=[ground_truth.shape[0], 17, 3])}
        
        if ground_truth is not None and not args.viz_no_ground_truth:
            anim_output['Ground truth'] = ground_truth

        input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

        from common.visualization import render_animation
        render_animation(input_keypoints, keypoints_metadata, anim_output,
                        dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                        limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                        input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                        input_video_skip=args.viz_skip)

else:
    print('Evaluating...')
    all_actions = {}
    all_actions_flatten = []
    all_actions_by_subject = {}
    for subject in subjects_test:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}

        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_flatten.append((subject, action))
            all_actions_by_subject[subject][action_name].append((subject, action))

    def fetch_actions(actions):
        out_poses_3d = []
        out_poses_2d = []
        out_camera_params = []

        for subject, action in actions:
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            poses_3d = dataset[subject][action]['positions_3d']
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            for i in range(len(poses_3d)): # Iterate across cameras
                out_poses_3d.append(poses_3d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

        stride = args.downsample
        if stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d


    def run_evaluation(actions, action_filter=None):
        errors_p1 = []
        errors_p2 = []
        errors_p3 = []
        if args.dataset.startswith('mpi'):
            errors_p4 = []
            errors_p5 = []
        errors_vel = []

        for action_key in actions.keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            cameras_act, poses_act, poses_2d_act = fetch_actions(actions[action_key])
            gen = UnchunkedGenerator_Seq(cameras_act, poses_act, poses_2d_act,
                                     pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                     kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                     joints_right=joints_right)
            if args.dataset.startswith('mpi'):
                e1, e2, e3, e4, e5, ev = evaluate(gen, action_key)
                errors_p4.append(e4)
                errors_p5.append(e5)
            else:
                e1, e2, e3, ev = evaluate(gen, action_key)
            errors_p1.append(e1)
            errors_p2.append(e2)
            errors_p3.append(e3)
            errors_vel.append(ev)

        print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
        print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
        print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
        if args.dataset.startswith('mpi'):
            print('PCK:', round(np.mean(errors_p4), 1))
            print('AUC:', round(np.mean(errors_p5), 1))
        print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')



    if not args.by_subject:
        run_evaluation(all_actions, action_filter)
    else:
        for subject in all_actions_by_subject.keys():
            print('Evaluating on subject', subject)
            run_evaluation(all_actions_by_subject[subject], action_filter)
            print('')
if not args.nolog:
    writer.close()