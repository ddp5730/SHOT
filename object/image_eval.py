import argparse
import os
import os.path as osp
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm import tqdm

import loss
import network
import image_target
from swin.config import get_config
from swin.data import build_loader
from swin.logger import create_logger
from swin.models import build_model
from swin.utils import load_pretrained


def cal_acc(loader, netF, netB, netC, name, eval_psuedo_labels=False, out_path='', print_out=False):
    start_test = True

    num_features = netF.num_features
    embeddings = np.zeros((0, num_features))

    if eval_psuedo_labels:
        mem_label = image_target.obtain_label(loader, netF, netB, netC, args)

    with torch.no_grad():
        iter_test = iter(loader)
        for i in tqdm(range(len(loader))):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            pseudo_idx = data[2]
            inputs = inputs.cuda()
            feat_embeddings = netF(inputs)
            outputs = netC(netB(feat_embeddings))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                if eval_psuedo_labels:
                    all_psuedo = mem_label[pseudo_idx]
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                if eval_psuedo_labels:
                    all_psuedo = np.concatenate((all_psuedo, mem_label[pseudo_idx]), 0)
            embeddings = np.concatenate([embeddings, feat_embeddings.detach().cpu().numpy()], axis=0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    all_preds = torch.squeeze(predict).float()

    plt.clf()
    cf_matrix = confusion_matrix(all_label, all_preds)
    cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    acc = cf_matrix.diagonal() / cf_matrix.sum(axis=1) * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=loader.dataset.classes)
    disp.plot()
    plt.title('CF acc=%.2f%%' % acc.mean())
    plt.savefig(os.path.join(out_path, '%s_cf.png' % name))
    plt.clf()

    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(embeddings)

    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 8))
    num_categories = len(loader.dataset.classes)
    for lab in range(num_categories):
        indices = all_label == lab
        ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], label=loader.dataset.classes[lab],
                   alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.title('TSNE acc=%.2f%%' % acc.mean())
    plt.savefig(os.path.join(out_path, '%s_tsne.png' % name))
    plt.clf()

    if eval_psuedo_labels:
        fig, ax = plt.subplots(figsize=(8, 8))
        num_categories = len(loader.dataset.classes)
        for lab in range(num_categories):
            indices = all_psuedo == lab
            ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], label=lab,
                       alpha=0.5)
        ax.legend(fontsize='large', markerscale=2)
        plt.savefig(os.path.join(out_path, '%s_pseudo_tnse.png' % name))
        plt.clf()

        log_str = classification_report(all_label, all_psuedo, target_names=loader.dataset.classes, digits=4)
        print_all(args.out_file, 'Performance of pseudo labels')
        print_all(args.out_file, log_str)

    # class_labels = [int(i) for i in test_loader.dataset.classes]
    log_str = classification_report(all_label, all_preds, target_names=loader.dataset.classes, digits=4)
    if(print_out):
        print_all(args.out_file, 'Performance on: %s' % name)
        print_all(args.out_file, log_str)
        print_all(args.out_file, '------------------------------\n\n')

    plt.close()

    return acc.mean()

def print_all(outfile, string):
    print(string)
    outfile.write(string)
    outfile.flush()


def evaluate_models(args, config):
    logger = create_logger(output_dir=args.output_dir_src, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if args.dset == 'rareplanes-synth' or args.dset == 'xview' or args.dset == 'dota':
        config.defrost()
        config.DATA.IDX_DATASET = True
        config.freeze()

        _, _, _, data_loader_val_source, _ = build_loader(config)

        # TODO: Dynamically select target dataset
        # Validating on target dataset so no longer as unsupervised
        config.defrost()
        config.DATA.DATASET = args.t_dset
        config.DATA.DATA_PATH = args.t_data_path
        config.OUTPUT = args.output_dir_src
        config.AMP_OPT_LEVEL = "O0"
        config.freeze()
        _, _, _, data_loader_val_target, _ = build_loader(config)

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

    netF.eval()
    netB.eval()
    netC.eval()

    # Evaluate model on both test and training dataset
    cal_acc(data_loader_val_source, netF, netB, netC, 'source', out_path=args.output_dir_src)
    cal_acc(data_loader_val_target, netF, netB, netC, 'target', out_path=args.output_dir_src, eval_psuedo_labels=True)



def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--source', type=int, default=0, help="source")
    parser.add_argument('--target', type=int, default=1, help="target")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'rareplanes-synth', 'dota', 'xview'])
    parser.add_argument('--t-dset', type=str, default='rareplanes-real')
    parser.add_argument('--t-data-path', type=str, default='/home/poppfd/data/RarePlanesCrop/chipped/real')
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101, swin-b")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])

    # Pseudo-label parameters
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)

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
    if args.dset == 'rareplanes-synth':
        names = ['train', 'validation']
        args.class_num = config.MODEL.NUM_CLASSES
    if args.dset == 'dota' or args.dset == 'xview':
        names = ['train', 'val']
        args.class_num = config.MODEL.NUM_CLASSES

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    if args.dset != 'rareplanes-synth' and args.dset != 'dota' and args.dset != 'xview':
        if args.dset_root is None:
            folder = './data/'
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        else:
            args.s_dset_path = os.path.join(args.dset_root, names[args.s])
            args.test_dset_path = os.path.join(args.dset_root, names[args.t])

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]
            if args.da == 'oda':
                args.class_num = 25
                args.src_classes = [i for i in range(25)]
                args.tar_classes = [i for i in range(65)]

    file_name = config.MODEL.PRETRAINED
    eval_num_str = file_name[file_name.rfind('_') + 1:file_name.find('.')]
    args.output_dir_src = osp.join(args.output, 'eval', args.name, 'eval_%s' % eval_num_str)
    args.name_src = names[args.source][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    path = os.path.join(args.output_dir_src, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    evaluate_models(args, config)
