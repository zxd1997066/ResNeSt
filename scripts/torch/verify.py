##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import print_function
import os
import argparse
from tqdm import tqdm
import time

import torch
import torch.nn as nn

import PIL
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

class Options():
    def __init__(self):
        # data settings
        parser = argparse.ArgumentParser(description='Deep Encoding')
        parser.add_argument('--base-size', type=int, default=None,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=224,
                            help='crop image size')
        # model params 
        parser.add_argument('--model', type=str, default='densenet',
                            help='network model type (default: densenet)')
        # training hyper params
        parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                            help='batch size for training (default: 128)')
        parser.add_argument('--workers', type=int, default=32,
                            metavar='N', help='dataloader threads')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', 
                            default=False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--verify', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--ipex', action='store_true', default=False,
                    help='use ipex')
        parser.add_argument('--jit', action='store_true', default=False,
                            help='use ipex')
        parser.add_argument('--precision', default="float32",
                                help='precision, "float32" or "bfloat16"')
        parser.add_argument('--warmup', type=int, default=10,
                            help='number of warmup')
        parser.add_argument('--max_iters', type=int, default=500,
                            help='max number of iterations to run')
        parser.add_argument('--dummy', action='store_true', default=False,
                            help='use dummy data')
        parser.add_argument('--channels_last', type=int, default=1, help='use channels last format')
        parser.add_argument('--profile', action='store_true', help='Trigger profile on current topology.')
        parser.add_argument("--compile", action='store_true', default=False,
                    help="enable torch.compile")
        parser.add_argument("--backend", type=str, default='inductor',
                    help="enable torch.compile backend")
        parser.add_argument("--triton_cpu", action='store_true', default=False,
                    help="enable triton_cpu")
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args


def save_profile_result(filename, table):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    keys = ["Name", "Self CPU total %", "Self CPU total", "CPU total %" , "CPU total", \
            "CPU time avg", "Number of Calls"]
    for j in range(len(keys)):
        worksheet.write(0, j, keys[j])

    lines = table.split("\n")
    for i in range(3, len(lines)-4):
        words = lines[i].split(" ")
        j = 0
        for word in words:
            if not word == "":
                worksheet.write(i-2, j, word)
                j += 1
    workbook.close()


def main():
    # init the args
    args = Options().parse()
    if args.triton_cpu:
        print("run with triton cpu backend")
        import torch._inductor.config
        torch._inductor.config.cpu_backend="triton"
    if args.ipex:
        import intel_extension_for_pytorch as ipex
        print("Running with IPEX...")

    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if not args.dummy:
        # init dataloader
        interp = PIL.Image.BILINEAR if args.crop_size < 320 else PIL.Image.BICUBIC
        base_size = args.base_size if args.base_size is not None else int(1.0 * args.crop_size / 0.875)
        transform_val = transforms.Compose([
            ECenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        valset = ImageNetDataset(transform=transform_val, train=False)
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True if args.cuda else False)
    
    # init the model
    model_kwargs = {}

    assert args.model in torch.hub.list('zhanghang1989/ResNeSt', force_reload=False)
    model = torch.hub.load('zhanghang1989/ResNeSt', args.model, pretrained=False)
    # print(model)

    if args.cuda:
        model.cuda()
        # Please use CUDA_VISIBLE_DEVICES to control the number of gpus
        model = nn.DataParallel(model)

    # checkpoint
    if args.verify:
        if os.path.isfile(args.verify):
            print("=> loading checkpoint '{}'".format(args.verify))
            model.module.load_state_dict(torch.load(args.verify))
        else:
            raise RuntimeError ("=> no verify checkpoint found at '{}'".\
                format(args.verify))
    elif args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            raise RuntimeError ("=> no resume checkpoint found at '{}'".\
                format(args.resume))

    model.eval()

    if args.channels_last:
        model_oob = model
        try:
            model_oob.to(memory_format=torch.channels_last)
            print("Use NHWC model.")
        except:
            print("Use normal model.")
        if args.jit:
            data = torch.randn(args.batch_size, 3, args.crop_size, args.crop_size)
            model_oob = torch.jit.trace(model_oob, data.to(memory_format=torch.channels_last))
        model = model_oob
    if args.compile:
        model = torch.compile(model, backend=args.backend, options={"freezing": True})

    if args.ipex:
        if args.precision == "bfloat16":
            model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
            print('Running with bfloat16...')
        else:
            model = ipex.optimize(model, dtype=torch.float32, inplace=True)
            print('Running with float32...')

    if args.jit:
        data = torch.randn(args.batch_size, 3, args.crop_size, args.crop_size)
        model = torch.jit.trace(model, data)
        if args.ipex:
            model = torch.jit.freeze(model)
        #warmup
        for i in range(10):
            model(data)

    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    batch_time_list = []
    is_best = False
    if args.dummy:
        max_iters = args.max_iters + args.warmup
        if args.precision == "bfloat16":
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.bfloat16):
                for batch_idx in range(max_iters):
                    data = torch.randn(args.batch_size, 3, args.crop_size, args.crop_size)
                    target = torch.arange(1, args.batch_size + 1).long()
                    if args.cuda:
                        data, target = data.cuda(), target.cuda()
                    if args.channels_last:
                        try:
                            data, target = data.to(memory_format=torch.channels_last), target.to(memory_format=torch.channels_last)
                        except:
                            pass
                    start = time.time()
                    with torch.no_grad():
                        if args.profile:
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
                                output = model(data)
                        else:
                            output = model(data)
                    end = time.time()
                    print("Iteration: {}, inference time: {} sec.".format(batch_idx, end - start), flush=True)
                    if batch_idx >= args.warmup:
                        batch_time.update(end - start)
                        batch_time_list.append((end - start) * 1000)

                    if batch_idx % 10 == 0:
                        print('iters: {:d}/{:d}, {:0.3f}({:0.3f}).'.format(batch_idx + 1, max_iters, batch_time.val, batch_time.avg))

                    if batch_idx >= max_iters -1:
                        break
        elif args.precision == "float16":
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.half):
                for batch_idx in range(max_iters):
                    data = torch.randn(args.batch_size, 3, args.crop_size, args.crop_size)
                    target = torch.arange(1, args.batch_size + 1).long()
                    if args.cuda:
                        data, target = data.cuda(), target.cuda()
                    if args.channels_last:
                        try:
                            data, target = data.to(memory_format=torch.channels_last), target.to(memory_format=torch.channels_last)
                        except:
                            pass
                    start = time.time()
                    with torch.no_grad():
                        if args.profile:
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
                                output = model(data)
                        else:
                            output = model(data)
                    end = time.time()
                    print("Iteration: {}, inference time: {} sec.".format(batch_idx, end - start), flush=True)
                    if batch_idx >= args.warmup:
                        batch_time.update(end - start)
                        batch_time_list.append((end - start) * 1000)

                    if batch_idx % 10 == 0:
                        print('iters: {:d}/{:d}, {:0.3f}({:0.3f}).'.format(batch_idx + 1, max_iters, batch_time.val, batch_time.avg))

                    if batch_idx >= max_iters -1:
                        break
        else:
            for batch_idx in range(max_iters):
                data = torch.randn(args.batch_size, 3, args.crop_size, args.crop_size)
                target = torch.arange(1, args.batch_size + 1).long()
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                if args.channels_last:
                    try:
                        data, target = data.to(memory_format=torch.channels_last), target.to(memory_format=torch.channels_last)
                    except:
                        pass
                start = time.time()
                with torch.no_grad():
                    if args.profile:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
                            output = model(data)
                    else:
                        output = model(data)
                end = time.time()
                print("Iteration: {}, inference time: {} sec.".format(batch_idx, end - start), flush=True)
                if batch_idx >= args.warmup:
                    batch_time.update(end - start)
                    batch_time_list.append((end - start) * 1000)

                if batch_idx % 10 == 0:
                    print('iters: {:d}/{:d}, {:0.3f}({:0.3f}).'.format(batch_idx + 1, max_iters, batch_time.val, batch_time.avg))

                if batch_idx >= max_iters -1:
                    break
    else:
        tbar = tqdm(val_loader, desc='\r')
        max_iters = min(args.max_iters + args.warmup, len(tbar)) if args.max_iters > 0 else len(tbar)
        for batch_idx, (data, target) in enumerate(tbar):
            if batch_idx >= args.warmup:
                start = time.time()
            elif args.cuda:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                output = model(data)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], data.size(0))
                top5.update(acc5[0], data.size(0))
            if batch_idx >= args.warmup:
                batch_time.update(time.time() - start)
                batch_time_list.append((time.time() - start) * 1000)

            tbar.set_description('Top1: %.3f | Top5: %.3f'%(top1.avg, top5.avg))

            if batch_idx >= max_iters -1:
                break

        print('Top1 Acc: %.3f | Top5 Acc: %.3f '%(top1.avg, top5.avg))
    latency = batch_time.avg / args.batch_size * 1000
    throughput = args.batch_size / batch_time.avg
    print("\n", "-"*20, "Summary", "-"*20)
    print("inference latency:\t {:.3f} ms".format(latency))
    print("inference Throughput:\t {:.2f} samples/s".format(throughput))
    # P50
    batch_time_list.sort()
    p50_latency = batch_time_list[int(len(batch_time_list) * 0.50) - 1]
    p90_latency = batch_time_list[int(len(batch_time_list) * 0.90) - 1]
    p99_latency = batch_time_list[int(len(batch_time_list) * 0.99) - 1]
    print('Latency P50:\t %.3f ms\nLatency P90:\t %.3f ms\nLatency P99:\t %.3f ms\n'\
            % (p50_latency, p90_latency, p99_latency))

    #
    if args.profile:
        import pathlib
        timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
        if not os.path.exists(timeline_dir):
            os.makedirs(timeline_dir)
        timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                    "resnest" + str(batch_idx) + '-' + str(os.getpid()) + '.json'
        print(timeline_file)
        prof.export_chrome_trace(timeline_file)
        table_res = prof.key_averages().table(sort_by="cpu_time_total")
        print(table_res)
        # self.save_profile_result(timeline_dir + torch.backends.quantized.engine + "_result_average.xlsx", table_res)

class ECenterCrop:
    """Crop the given PIL Image and resize it to desired size.
    Args:
        img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
        output_size (sequence or int): (height, width) of the crop box. If int,
            it is used for both directions
    Returns:
        PIL Image: Cropped image.
    """
    def __init__(self, imgsize):
        self.imgsize = imgsize
        self.resize_method = transforms.Resize((imgsize, imgsize), interpolation=PIL.Image.BICUBIC)

    def __call__(self, img):
        image_width, image_height = img.size
        image_short = min(image_width, image_height)

        crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short

        crop_height, crop_width = crop_size, crop_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        img = img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
        return self.resize_method(img)

class ImageNetDataset(datasets.ImageFolder):
    BASE_DIR = "ILSVRC2012"
    def __init__(self, root=os.path.expanduser('~/.encoding/data'), transform=None,
                 target_transform=None, train=True, **kwargs):
        split='train' if train == True else 'val'
        root = os.path.join(root, self.BASE_DIR, split)
        super(ImageNetDataset, self).__init__(root, transform, target_transform)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        avg = 0 if self.count == 0 else self.sum / self.count
        return avg

if __name__ == "__main__":
    main()

