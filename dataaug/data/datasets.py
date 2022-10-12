"""Additional torchvision-like datasets."""

import torch
import os
import glob
from PIL import Image

import numpy as np

from torchvision.datasets.utils import download_and_extract_archive, extract_archive
import hashlib


class TinyImageNet(torch.utils.data.Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

    This is a TinyImageNet variant to the code of Meng Lee, mnicnc404 / Date: 2018/06/04
    References:
        - https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel.html
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    cached: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    download: bool
        Set to true to automatically download the dataset in to the root folder.
    """

    EXTENSION = "JPEG"
    NUM_IMAGES_PER_CLASS = 500
    CLASS_LIST_FILE = "wnids.txt"
    VAL_ANNOTATION_FILE = "val_annotations.txt"
    CLASSES = "words.txt"

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    archive = "tiny-imagenet-200.zip"
    folder = "tiny-imagenet-200"
    train_md5 = "c77c61d662a966d2fcae894d82df79e4"
    val_md5 = "cef44e3f1facea2ea8cd5e5a7a46886c"
    test_md5 = "bc72ebd5334b12e3a7ba65506c0f8bc0"

    def __init__(self, root, split="train", transform=None, target_transform=None, cached=True, download=True):
        """Init with split, transform, target_transform."""
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.cached = cached

        self.split_dir = os.path.join(root, self.folder, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, "**", "*.%s" % self.EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping

        if download:
            self.download()

        self._parse_labels()

        if self.cached:
            self._build_cache()

    def _check_integrity(self):
        """This only checks if all files are there."""
        string_rep = "".join(self.image_paths).encode("utf-8")
        hash = hashlib.md5(string_rep)
        if self.split == "train":
            return hash.hexdigest() == self.train_md5
        elif self.split == "val":
            return hash.hexdigest() == self.val_md5
        else:
            return hash.hexdigest() == self.test_md5

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.archive)

    def _parse_labels(self):
        with open(os.path.join(self.root, self.folder, self.CLASS_LIST_FILE), "r") as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == "train":
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(self.NUM_IMAGES_PER_CLASS):
                    self.labels["%s_%d.%s" % (label_text, cnt, self.EXTENSION)] = i
        elif self.split == "val":
            with open(os.path.join(self.split_dir, self.VAL_ANNOTATION_FILE), "r") as fp:
                for line in fp.readlines():
                    terms = line.split("\t")
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # Build class names
        label_text_to_word = dict()
        with open(os.path.join(self.root, self.folder, self.CLASSES), "r") as file:
            for line in file:
                label_text, word = line.split("\t")
                label_text_to_word[label_text] = word.split(",")[0].rstrip("\n")
        self.classes = [label_text_to_word[label] for label in self.label_texts]

        # Prepare index - label mapping
        self.targets = [self.labels[os.path.basename(file_path)] for file_path in self.image_paths]

    def _build_cache(self):
        """Cache images in RAM."""
        self.cache = []
        for index in range(len(self)):
            img = Image.open(self.image_paths[index])
            img = img.convert("RGB")
            self.cache.append(img)

    def __len__(self):
        """Return length via image paths."""
        return len(self.image_paths)

    def __getitem__(self, index):
        """Return image, label."""
        if self.cached:
            img = self.cache[index]
        else:
            img = Image.open(self.image_paths[index])
            img = img.convert("RGB")
        target = self.targets[index]

        img = self.transform(img) if self.transform else img
        target = self.target_transform(target) if self.target_transform else target
        if self.split == "test":
            return img, None
        else:
            return img, target


class CIFAR10C(torch.utils.data.Dataset):
    """CIFAR-10 C downloader. Use only for testing."""

    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"
    archive = "CIFAR-10-C.tar"
    folder = "CIFAR-10-C"
    data_md5s = [
        "b426e7929e9ac645b56c5081b5747d20",
        "6626762e374da748dd5d9ff22b295f4d",
        "877ce80552da27255f652ae63b30bad2",
    ]  # whywhywhy (probably order)

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_transform=None,
        download=True,
        categories="all",
        severity=5,
        mmap=False,
    ):
        """Init with split, transform, target_transform."""
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        self._unpack(categories, severity, mmap)

    def _check_integrity(self):
        """This only checks if all files are there."""
        file_hash = hashlib.md5()
        try:
            for category_file in sorted(os.listdir(os.path.join(self.root, self.folder))):
                with open(os.path.join(self.root, self.folder, category_file), "rb") as f:
                    while chunk := f.read(8192):
                        file_hash.update(chunk)
        except FileNotFoundError:
            return False
        if file_hash.hexdigest() in self.data_md5s:
            return True
        else:
            print(f"Hash {file_hash.hexdigest()} not matching.")
            return False

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.archive, remove_finished=False)

    def _unpack(self, categories="all", severity=5, mmap=False):
        """Unpack and load .npy files. Todo: Actually implement the "categories" option."""
        data = []
        targets = []
        num_categories = 0
        if mmap:
            mmap_mode = "c"  # copy on write
        else:
            mmap_mode = None

        for category_file in os.listdir(os.path.join(self.root, self.folder)):
            X = np.load(os.path.join(self.root, self.folder, category_file))
            if "labels" in category_file:
                targets.append(X[10_000 * (severity - 1) : 10_000 * (severity)])
            else:
                num_categories += 1
                data.append(X[10_000 * (severity - 1) : 10_000 * (severity)])

        self.data = np.concatenate(data, axis=0)
        self.targets = np.concatenate(targets * num_categories)

    def __len__(self):
        """Return length via image paths."""
        return len(self.data)

    def __getitem__(self, index):
        """Return image, label."""
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class CINIC10(torch.utils.data.Dataset):
    """Deduplicated CINIC-10. Train/Valid/Test are repartitioned into a nicer format.
    full data = train/val/test [minus duplicates in itself and to both CIFAR-10 train and test]
    new train -> first 260k images
    new test -> CINIC
    """

    EXTENSION = "png"
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    url = "https://datashare.ed.ac.uk/download/DS_10283_3192.zip"
    archive = "cinic-10.zip"
    folder = "cinic-10"
    data_md5s = [
        "265e9a3d259e8560a7be6c1049f2fd9c",
        "9f9a2951a00b3fe47db086eeb5868eed",
        "26cd64820ea6e5b2ebbdd47e49179a06",
    ]  # I dont even know ... # ...
    valid_size = 10_000

    def __init__(self, root, split="train", transform=None, target_transform=None, cached=True, download=True, deduplicate=False):
        """Init with split, transform, target_transform."""
        self.root = os.path.expanduser(root)

        self.transform = transform
        self.target_transform = target_transform
        self.cached = cached
        self.deduplicate = deduplicate

        self.dir = os.path.join(root, self.folder)
        self.pre_cache = dict()
        self.image_paths = sorted(
            glob.iglob(os.path.join(os.path.join(self.root, self.folder, "**", f"*{self.EXTENSION}")), recursive=True)
        )

        if download:
            self.download()

        self.image_paths = self._remove_cifar10_traintest()
        self.image_paths = self._remove_invalid_data()

        if deduplicate:
            self.image_paths = self._deduplicate()

        self.image_paths = self._split_traintest(split)

        self._parse_labels()

        if self.cached:
            self._build_cache()

    def _check_integrity(self):
        """This only checks if all files are there."""
        string_rep = "".join(sorted(self.image_paths)).encode("utf-8")
        file_hash = hashlib.md5(string_rep)
        if file_hash.hexdigest() in self.data_md5s:
            return True
        else:
            print(f"Hash {file_hash.hexdigest()} not matching.")
            return False

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.archive, remove_finished=False)
        # CINIC is double packed: ...
        extract_archive(os.path.join(self.root, "CINIC-10.tar.gz"), os.path.join(self.root, self.folder))
        self.image_paths = sorted(
            glob.iglob(os.path.join(os.path.join(self.root, self.folder, "**", f"*{self.EXTENSION}")), recursive=True)
        )

    def _remove_cifar10_traintest(self):
        """Remove CIFAR-10 test images."""
        train_ids = []
        for idx, img_path in enumerate(self.image_paths):
            if "cifar" in img_path.split(os.sep)[-1]:  # and "test" in img_path:
                pass
            else:
                train_ids.append(idx)
        return [self.image_paths[idx] for idx in train_ids]

    def _remove_invalid_data(self):
        """This is a terrible dataset..."""
        train_ids = []
        self.pre_cache = dict()  # CML is slow in loading files. Cache only once.
        for idx, path in enumerate(self.image_paths):
            try:
                img = Image.open(self.image_paths[idx], formats=("PNG",))
                img = img.convert("RGB")
                train_ids.append(idx)
                self.pre_cache[self.image_paths[idx]] = img
            except Exception as e:  # I don't care why it doesn't load
                pass
        return [self.image_paths[idx] for idx in train_ids]

    def _deduplicate(self):
        """Remove duplicate images via hashing. Code mostly from https://github.com/BayesWatch/cinic-10/issues/5"""

        def hash_file(filepath):
            with open(filepath, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()

        uniques = []
        num_duplicate = 0
        hash_keys = set()
        file_list = self.image_paths
        for idx, img_path in enumerate(self.image_paths):
            if os.path.isfile(img_path):
                file_hash = hash_file(img_path)
                if file_hash not in hash_keys:
                    hash_keys.add(file_hash)
                    uniques.append(idx)
        return [self.image_paths[idx] for idx in uniques]

    def _split_traintest(self, split):
        """Declare last 10k as test data."""
        generator = torch.Generator().manual_seed(233)
        indices = torch.randperm(len(self.image_paths), generator=generator).tolist()
        if split == "train":
            return [self.image_paths[idx] for idx in indices[: -self.valid_size]]
        else:
            return [self.image_paths[idx] for idx in indices[-self.valid_size :]]

    def _parse_labels(self):
        self.labels = []
        for idx, path in enumerate(self.image_paths):
            for cls_idx, class_name in enumerate(self.classes):
                if class_name in path:
                    self.labels.append(cls_idx)

    def _build_cache(self):
        """Cache images in RAM."""
        self.cache = []
        for index in range(len(self)):
            if self.image_paths[index] in self.pre_cache:
                img = self.pre_cache[self.image_paths[index]]
            else:
                img = Image.open(self.image_paths[index])
                img = img.convert("RGB")
            self.cache.append(img)
        self.pre_cache = dict()

    def __len__(self):
        """Return length via image paths."""
        return len(self.image_paths)

    def __getitem__(self, index):
        """Return image, label."""
        if self.cached:
            img = self.cache[index]
        else:
            img = Image.open(self.image_paths[index])
            img = img.convert("RGB")
        target = self.labels[index]

        img = self.transform(img) if self.transform else img
        target = self.target_transform(target) if self.target_transform else target
        return img, target


from torchvision.datasets import CIFAR10


class CIFAR10stoch(CIFAR10):
    def __init__(
        self,
        *args,
        rounds=1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.rounds = rounds - 1

    def __getitem__(self, index):
        img, img_aug, target = None, None, None

        img, target = Image.fromarray(self.data[index]), self.targets[index]
        if self.train:
            img_aug = [self.transform(img) for _ in range(self.rounds)]
            img_aug = torch.stack(img_aug)
            img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, img_aug, target
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target
