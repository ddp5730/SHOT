import argparse
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch import nn
from tqdm import tqdm

import network
from object.image_source import data_load, cal_acc_oda, cal_acc, print_args
from transfg.models.modeling import CONFIGS
from transfg.train import count_parameters
from transfg.utils import data_utils


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089
    elif args.dataset == 'RarePlanes':
        num_classes = 7

    model = network.VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes,
                                      smoothing_value=args.smoothing_value)

    model.load_from(np.load(args.pretrained_dir))
    if args.pretrained_model is not None:
        pretrained_model = torch.load(args.pretrained_model)['model']
        model.load_state_dict(pretrained_model)
    model.to(args.device)
    num_params = count_parameters(model)

    print("{}".format(config))
    print("Training parameters %s", args)
    print("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def test_target(args):
    if args.dset == 'rareplanes-synth':
        args.dataset = 'RarePlanes'
        args.data_root = args.s_dset_path
        args.local_rank = -1
        args.train_batch_size = args.batch_size
        args.eval_batch_size = args.batch_size
        args.balance_classes = True
        dset_loaders = {}
        dset_loaders["source_tr"], dset_loaders["source_te"] = data_utils.get_loader(args, use_validation=True)

        args.data_root = args.test_dset_path
        _, dset_loaders["test"] = data_utils.get_loader(args)
    else:
        dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()
    elif args.net[0:3] == 'tra':
        args.device = torch.device('cuda')
        args, netF = setup(args)

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    if args.da == 'oda':
        acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB, netC)
        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.trte, args.name,
                                                                                            acc_os2, acc_os1,
                                                                                            acc_unknown)
    else:
        if args.dset == 'VISDA-C':
            acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc) + '\n' + acc_list
        else:
            orig_name = args.name
            args.name = orig_name + '_target'
            log_str = 'Validation on Target Domain'
            print(log_str)
            args.out_file.write(log_str)
            args.out_file.flush()
            acc = cal_acc(dset_loaders['test'], netF, netB, netC, False)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc)
            print(log_str)
            print('------------------------------------------------------------------\n')

            args.name = orig_name + '_source'
            log_str = '---------------------------------------------------------------\nValidation on Source Domain'
            print(log_str)
            args.out_file.write(log_str)
            args.out_file.flush()
            acc = cal_acc(dset_loaders['source_te'], netF, netB, netC, False)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc)
            print(log_str)
            print('------------------------------------------------------------------\n')
            args.name = orig_name

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in tqdm(range(len(loader)), file=sys.stdout):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    all_preds = torch.squeeze(predict).float()

    cf_matrix = confusion_matrix(all_label, all_preds)
    acc = cf_matrix.diagonal() / cf_matrix.sum(axis=1) * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=loader.dataset.classes)
    disp.plot()
    plt.savefig(os.path.join(args.output_dir_src, '%s_cf.png' % args.name))

    # class_labels = [int(i) for i in test_loader.dataset.classes]
    log_str = classification_report(all_label, all_preds, target_names=loader.dataset.classes, digits=4)
    print(log_str)
    args.out_file.write(log_str)
    args.out_file.flush()

    return acc.mean()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'rareplanes-synth'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101, transfg")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--dset_root', type=str, default=None, help='Path to the target dataset.  Directory should '
                                                                    'contain folder for different domains')

    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")
    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")
    parser.add_argument("--pretrained_dir", type=str, default="/opt/tiger/minist/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="load pretrained model")
    parser.add_argument('--name', type=str, default='test',
                        help='Unique name for the run')

    args = parser.parse_args()
    args.workers = args.worker

    names = []
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
        names = ['real', 'synth']
        args.class_num = 3

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

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

    args.output_dir_src = osp.join(args.output, args.name, names[args.s][0].upper())
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
    test_target(args)
