import torch
from matplotlib import pyplot as plt
from torch.utils.data import random_split, DataLoader
from minist import get_mnist



def prepare_dataset(num_partitions: int,
                    batch_size: int,
                    val_ratio: float = 0.1):

    """This function partitions the training set into N disjoint
    subsets, each will become the local dataset of a client. This
    function also subsequently partitions each traininset partition
    into train and validation. The test set is left intact and will
    be used by the central server to asses the performance of the
    global model. """

    # get the MNIST dataset
    trainset, testset = get_mnist()

    # split trainset into `num_partitions` trainsets
    num_images = len(trainset) // num_partitions

    partition_len = [num_images] * num_partitions

    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))

    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        # 将训练集划分为训练集和验证集
        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

        # 创建训练集和验证集的dataloader
        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=0))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=0))

    # create dataloader for the test set
    testloader = DataLoader(testset, batch_size=128, num_workers=0)

    return trainloaders, valloaders, testloader

trainloaders, valloaders, testloader = prepare_dataset(num_partitions=100,
                                                       batch_size=32)

if __name__ == "__main__":
    # first partition
    train_partition = trainloaders[0].dataset

    # count data points
    partition_indices = train_partition.indices
    print(f"number of images: {len(partition_indices)}")

    # visualise histogram
    plt.hist(train_partition.dataset.dataset.targets[partition_indices], bins=10)
    plt.grid()
    plt.xticks(range(10))
    plt.xlabel('Label')
    plt.ylabel('Number of images')
    plt.title('Class labels distribution for MNIST')
    plt.show()
