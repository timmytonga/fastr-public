import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from global_vars import ROOT_DIR, DATA_ROOT_DIR, GLUE_DATASETS, MLM_DATASETS
from data.nlp_datasets import get_glue_dataset, get_mlm_dataset
import os


def get_dataset(dataset_name, args):
    """
        Returns standard (if applicable) train/val/split of the given dataset_name

    """
    assert ROOT_DIR != "", "ROOT_DIR is empty! Must set environment variable RESEARCH_ROOT_DIR to " \
                           "root_dir containing datasets. Do this by adding " \
                           "'export RESEARCH_ROOT_DIR=<root_dir>' in your ~/.bashrc file."
    val_split_proportion = args.val_fraction
    collator = None
    n_classes = None

    assert 0 <= val_split_proportion < 1, "val_split_proportion must be in [0,1)."
    if val_split_proportion == 0:
        val = None
    else:
        raise NotImplementedError("Have to fix some augmentation stuff first!")

    if dataset_name == "cifar10":
        # CIFAR10 dataset needs to be renormalized
        if args.no_augment_data:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        # transform = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train = torchvision.datasets.CIFAR10(root=DATA_ROOT_DIR, train=True,
                                             download=True, transform=transform_train)
        n_classes = len(train.classes)
        if val_split_proportion == 0:
            val = None
        else:
            raise NotImplementedError("Have to fix some augmentation stuff first!")
            # num_val = int(len(train) * val_split_proportion)
            # num_train = len(train) - num_val
            # train, val = torch.utils.data.random_split(train, [num_train, num_val])
        test = torchvision.datasets.CIFAR10(root=DATA_ROOT_DIR, train=False,
                                            download=True, transform=transform_test)
    elif dataset_name == "cifar100":
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[n / 255.
                                                                         for n in [129.3, 124.1, 112.4]],
                                                                   std=[n / 255. for n in [68.2, 65.4, 70.4]])])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[n / 255.
                                                                        for n in [129.3, 124.1, 112.4]],
                                                                  std=[n / 255. for n in [68.2, 65.4, 70.4]])])
        train = torchvision.datasets.CIFAR100(root=DATA_ROOT_DIR, train=True, download=True, transform=transform_train)
        n_classes = len(train.classes)
        test = torchvision.datasets.CIFAR100(root=DATA_ROOT_DIR, train=False, download=True, transform=transform_test)
    elif dataset_name == "mnist":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        train = torchvision.datasets.MNIST(root=DATA_ROOT_DIR, download=True, train=True, transform=transform)
        n_classes = len(train.classes)
        test = torchvision.datasets.MNIST(root=DATA_ROOT_DIR, download=True, train=False, transform=transform)
    elif dataset_name == "imagenet":
        # instructions: https://github.com/pytorch/examples/tree/e0d33a69bec3eb4096c265451dbb85975eb961ea/imagenet
        # instructions 2: https://csinva.io/blog/misc/imagenet_quickstart/readme
        data_dir = os.path.join(DATA_ROOT_DIR, 'imagenet')

        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')
        testdir = os.path.join(data_dir, 'test')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        valtest_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        train = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        n_classes = len(train.classes)
        if val_split_proportion == 0:
            print("[imagenet] Using validation as test set. Need to fix certain things to use validation.")
            val = None
            test = datasets.ImageFolder(valdir, valtest_transforms)
        else:  # actually this is not yet implemented???
            val = datasets.ImageFolder(valdir, valtest_transforms)
            test = datasets.ImageFolder(testdir, valtest_transforms)
    elif dataset_name in GLUE_DATASETS:
        train, val, test, n_classes, collator = get_glue_dataset(dataset_name, args)
    elif dataset_name in MLM_DATASETS:
        train, val, test, n_classes, collator = get_mlm_dataset(dataset_name, args)
    else:
        raise ValueError(f"{dataset_name} not recognized.")

    assert n_classes is not None, "We haven't set the n_classes right! Something has gone wrong. "
    return train, val, test, n_classes, collator
