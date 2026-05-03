import os

from py_src.util import expand_path

default_path_mnist = expand_path('~/dataset/mnist')
default_path_cifar10 = expand_path('~/dataset/cifar10')
default_path_cifar100 = expand_path('~/dataset/cifar100')
default_path_flowers102 = expand_path('~/dataset/flowers102')
default_path_svhn = expand_path('~/dataset/svhn')
default_path_imagenet1k = expand_path('~/dataset/imagenet1k')
default_path_imagenet100 = expand_path('~/dataset/imagenet100')
default_path_imagenet10 = expand_path('~/dataset/imagenet10')
default_path_flickr30k = expand_path('~/dataset/flickr30k')
default_path_arithmetic = expand_path('~/dataset/arithmetic')

default_path_random_mnist = expand_path('~/dataset/random_mnist')
default_path_random_cifar10 = expand_path('~/dataset/random_cifar10')
default_path_random_cifar100 = expand_path('~/dataset/random_cifar100')
default_path_random_imagenet10 = expand_path('~/dataset/random_imagenet10')
default_path_random_imagenet100 = expand_path('~/dataset/random_imagenet100')
default_path_random_imagenet1k = expand_path('~/dataset/random_imagenet1k')

""" Load env override file """
imagenet1k_path = None
imagenet100_path = None
imagenet10_path = None
flickr30k_path = None
flowers102_path = None

# load override dataset path from dataset_env.py
dataset_env_file_path = f"{os.path.dirname(os.path.abspath(__file__))}/dataset_env.py"
if os.path.exists(dataset_env_file_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("dataset_env", dataset_env_file_path)
    assert spec is not None, f"error to load dataset_env_file from {dataset_env_file_path}"
    assert spec.loader is not None, f"error: loader is missing for {dataset_env_file_path}"
    env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env)

    if hasattr(env, "imagenet1k_path"):
        imagenet1k_path = env.imagenet1k_path
        print("override imagenet1k_path: ", imagenet1k_path)
    if hasattr(env, "imagenet100_path"):
        imagenet100_path = env.imagenet100_path
        print("override imagenet100_path: ", env.imagenet100_path)
    if hasattr(env, "imagenet10_path"):
        imagenet10_path = env.imagenet10_path
        print("override imagenet10_path: ", env.imagenet10_path)
    if hasattr(env, "flickr30k_path"):
        flickr30k_path = env.flickr30k_path
    if hasattr(env, "flowers102_path"):
        flowers102_path = env.flowers102_path
        print("override flowers102_path: ", env.flowers102_path)
if imagenet1k_path is None:
    imagenet1k_path = default_path_imagenet1k
if imagenet100_path is None:
    imagenet100_path = default_path_imagenet100
if imagenet10_path is None:
    imagenet10_path = default_path_imagenet10
if flickr30k_path is None:
    flickr30k_path = default_path_flickr30k
if flowers102_path is None:
    flowers102_path = default_path_flowers102
