import argparse
import copy
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import loss
import network
from data_list import ImageList_idx
from object.center_loss import CenterLoss
from object.image_source import print_top_evals, save_linear_net
from swin.config import get_config
from swin.data import build_loader
from swin.logger import create_logger
from swin.models import build_model
from swin.utils import load_pretrained, save_checkpoint

import object.image_eval as image_eval

TOP_N = 3


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def train_target(args):
    logger = create_logger(output_dir=args.output_dir_src, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if args.dset == 'rareplanes':
        config.defrost()
        config.DATA.DATA_PATH = args.test_dset_path
        config.DATA.IDX_DATASET = True
        config.OUTPUT = args.output_dir_src
        config.AMP_OPT_LEVEL = "O0"
        config.freeze()
        dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
        dset_loaders = {}
        dset_loaders["target"] = data_loader_train
        dset_loaders["test"] = data_loader_val
    else:
        dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()
    elif args.net == 'swin-b':
        netF = build_model(
            config)  # If config.MODEL.SOURCE_NUM_CLASSES == 0 then classification head is an identity layer
        num_features = netF.num_features

        # Load pretrained weights
        netF.head = nn.Identity()
        if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
            load_pretrained(config, netF, logger)
        netF.cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=num_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    # modelpath = args.output_dir_src + '/ckpt_epoch_0.pth'
    # netF.load_state_dict(torch.load(modelpath))
    file_name = config.MODEL.PRETRAINED
    eval_num_str = file_name[file_name.rfind('_') + 1:file_name.find('.')]
    pretrained_dir = os.path.dirname(config.MODEL.PRETRAINED)
    netB_path = args.netB
    netC_path = args.netC
    for file in os.listdir(pretrained_dir):
        if ('eval_%s' % eval_num_str) in file:
            if 'source_B' in file and netB_path == '':
                netB_path = os.path.join(pretrained_dir, file)
            elif 'source_C' in file and netC_path == '':
                netC_path = os.path.join(pretrained_dir, file)

    netB.load_state_dict(torch.load(netB_path))
    netC.load_state_dict(torch.load(netC_path))

    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    learning_rate = config.TRAIN.BASE_LR
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': learning_rate * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': learning_rate * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.AdamW(param_group, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                            lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * len(dset_loaders["target"]))
    interval_iter = len(dset_loaders["target"]) // 10
    pseudo_label_iter = len(dset_loaders['target'])
    # interval_iter = 10
    iter_num = 0
    epoch_num = 0
    acc_s_best = 0
    eval_num = 0

    validation_accuracy = []

    cosine_lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=max_iter,
        cycle_mul=1.,
        lr_min=config.TRAIN.MIN_LR,
        warmup_lr_init=config.TRAIN.WARMUP_LR,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    if args.center_loss:
        cent_lr = args.cent_lr
        cent_alpha = args.cent_alpha
        center_loss_func = CenterLoss(num_classes=args.class_num, feat_dim=netF.num_features, use_gpu=True)
        optimizer_centloss = torch.optim.AdamW(center_loss_func.parameters(), lr=cent_lr)

    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            print('Starting Epoch Number %d' % epoch_num)
            tqdm_iter = tqdm(dset_loaders['target'], file=sys.stdout)
            iter_test = iter(tqdm_iter)
            inputs_test, _, tar_idx = next(iter_test)
            epoch_num += 1

        if inputs_test.size(0) == 1:
            continue

        if iter_num % pseudo_label_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            mem_label = obtain_label(dset_loaders['target'], netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        # lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features = netF(inputs_test)
        features_test = netB(features)
        outputs_test = netC(features_test)

        if args.cls_par > 0:
            pred = mem_label[tar_idx]  # Due to a bug, the batch size must evenly divide the training dataset or
                                       # this line will trigger an index out of bounds error.  Could fix at some point...
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        if iter_num > warmup_steps and args.center_loss:
            center_loss = center_loss_func(features, pred)
            classifier_loss = (cent_alpha*center_loss) + classifier_loss
            optimizer_centloss.zero_grad()

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        cosine_lr_scheduler.step_update(iter_num)

        writer.add_scalar('Loss/Train', classifier_loss.item(), global_step=iter_num)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step=iter_num)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            if args.dset == 'VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter,
                                                                            acc_s_te) + '\n' + acc_list
            else:
                acc_s_te = image_eval.cal_acc(dset_loaders['test'], netF, netB, netC, name=('eval/eval_%d' % eval_num),
                                                 eval_psuedo_labels=False, out_path=config.OUTPUT)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)
                tqdm_iter.set_description(log_str)
                writer.add_scalar('Validation Accuracy', scalar_value=acc_s_te, global_step=iter_num)
            logger.info(log_str + '\n')
            validation_accuracy.append(acc_s_te)

            best_netF = copy.deepcopy(netF)
            best_netB = netB.state_dict()
            best_netC = netC.state_dict()
            print_top_evals(np.asarray(validation_accuracy), n=TOP_N, logger=logger)
            save_checkpoint(config, epoch_num, best_netF, acc_s_best, optimizer, cosine_lr_scheduler, logger,
                            eval_num=eval_num, validation_accuracy=np.asarray(validation_accuracy), top_n=TOP_N)
            save_linear_net(best_netB, 'source_B', epoch_num, eval_num, np.asarray(validation_accuracy),
                            args.output_dir_src, top_n=TOP_N)
            save_linear_net(best_netC, 'source_C', epoch_num, eval_num, np.asarray(validation_accuracy),
                            args.output_dir_src, top_n=TOP_N)
            # torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
            # torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))
            eval_num += 1

            netF.train()
            netB.train()


    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    return pred_label.astype('int')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--source', type=int, default=0, help="source")
    parser.add_argument('--target', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'rareplanes'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)

    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--netB', default='')
    parser.add_argument('--netC', default='')
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--transfer-dataset', action='store_true', help='Transfer the model to a new dataset')
    parser.add_argument('--name', type=str, default='test',
                        help='Unique name for the run')

    # Args needed to load swin.  Not necessarily used
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    parser.add_argument('--center-loss', default=False)
    parser.add_argument('--cent-lr', default=0.01, type=float)
    parser.add_argument('--cent-alpha', default=0.3, type=float)

    args = parser.parse_args()
    args.eval_period = -1

    config = get_config(args)

    torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank)

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'rareplanes':
        names = ['real', 'synth']
        args.class_num = config.MODEL.NUM_CLASSES

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.source:
            continue
        args.target = i

        print('Training on target: %s' % names[i])

        if args.data_path is None:
            folder = './data/'
            args.s_dset_path = folder + args.dset + '/' + names[args.source] + '_list.txt'
            args.t_dset_path = folder + args.dset + '/' + names[args.target] + '_list.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.target] + '_list.txt'
        else:
            args.s_dset_path = os.path.join(args.data_path, names[args.source])
            args.test_dset_path = os.path.join(args.data_path, names[args.target])

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        # args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        # args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper() + names[args.t][0].upper())

        args.output_dir_src = osp.join(args.output, args.name, names[args.target][0].upper())
        args.name_str = names[args.source][0].upper() + names[args.target][0].upper()

        if not osp.exists(args.output_dir_src):
            os.system('mkdir -p ' + args.output_dir_src)
        if not osp.exists(args.output_dir_src):
            os.mkdir(args.output_dir_src)
        if not osp.exists(os.path.join(args.output_dir_src, 'eval')):
            os.mkdir(os.path.join(args.output_dir_src, 'eval'))

        args.savename = 'par_' + str(args.cls_par)
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir_src, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_target(args)
