import argparse
from icgen.icgen import ICDatasetGenerator, save_dataset

parser = argparse.ArgumentParser(description="Experiment runner")
parser.add_argument(
    "--data_path",
)
parser.add_argument(
    "--dataset",
    default="cifar10",
)
args = parser.parse_args()

dataset_generator = ICDatasetGenerator(
  data_path=args.data_path,
  min_resolution=64,
  max_resolution=64,
  max_log_res_deviation=3,  # Sample only 1 log resolution from the native one
  min_classes=8,
  max_classes=8,
  min_examples_per_class=20,
  max_examples_per_class=100_000,
)
dev_data, test_data, dataset_info = dataset_generator.get_dataset(
    dataset=args.dataset, augment=False, download=True
)
print(len(dev_data), len(test_data), dataset_info)

from pathlib import Path
from PIL import Image


for max_dim in [32, 64, 128]:
    print(max_dim)
    save_dataset(dev_data, test_data, dataset_info, Path(args.data_path), valid_fraction=0.2, new_max_dim=max_dim)
save_dataset(dev_data, test_data, dataset_info, Path(args.data_path), valid_fraction=0.2)

from icgen.vision_dataset import ICVisionDataset

d = ICVisionDataset("colorectal_histology", Path(args.data_path), "train")

print(len(d))
