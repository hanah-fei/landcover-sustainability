import shutil 
import numpy
import os
import random
random.seed(123)

"""Two modes:
   1) MODE = '3_channels'
   Assumes you have downloaded the 3 band jpg data from
   http://madm.dfki.de/downloads

   2) MODE = '13_channels'
   Assumes you have downloaded the 13 band tif data from 
   http://madm.dfki.de/downloads 
  
   For both modes, data should have following directory structure
   where the sub-directory is the image label.
   
   data_dir/Forest/Forest_864.tif
   data_dir/Forest/Forest_2917.tif
   ...
   data_dir/Pasture/Pasture_1212.tif
   data_dir/Pasture/Pasture_276.tif
   data_dir/Pasture/Pasture_1574.tif 

   This script creates a 80/10/10 split of the (shuffled) data
   and writes the corresponding image names to a train, dev, and
   test folder.

   out_dir/train/PermanentCrop_1198.tif
   out_dir/train/'Residential_1497.tif
   ...
   out_dir/dev/HerbaceousVegetation_919.tif
   out_dir/dev/River_1066.tif
   ...
   out_dir/test/AnnualCrop_2173.tif
   out_dir/test/Highway_1678.tif
"""


## Set your data_dir and out_dir
data_dir = '../satellite_data/rgb/'
out_dir = '../satellite_data/rgb_dataset/'
MODE = '3_channels'

def main(data_dir, out_dir, MODE):
  filenames = get_and_shuffle(data_dir)
  train_images, dev_images, test_images = create_splits(filenames)
  for split in ['train', 'dev', 'test']:
    output_dir_split = os.path.join(out_dir, split)
    if not os.path.exists(output_dir_split):
      os.mkdir(output_dir_split)
    else:
      print("Warning: dir {} already exists".format(output_dir_split))
  
  print("Copying train data.")
  write_data(train_images, 'train', MODE)
  print("Copying dev data.")
  write_data(dev_images, 'dev', MODE)
  print("Copying test data.")
  write_data(test_images, 'test', MODE)   

 
def write_data(filenames, split, MODE):
  """Copy images to new outdir."""
  for name in filenames:
    source = data_dir + name.split('_')[0] + '/' + name
    if MODE == '3_channels':
      label = name.split('_')[0]
      destination = out_dir + split + '/' + label + '/' +  name
      if not os.path.exists(out_dir + split + '/' + label):
        os.mkdir(out_dir + split + '/' + label)
    else:
      destination = out_dir + split + '/' + name
    dest = shutil.copyfile(source, destination)  
 

def create_splits(filenames):
  """Split shuffled data 80/10/10."""
  split_1 = int(0.8 * len(filenames))
  split_2 = int(0.9 * len(filenames))
  train_images = filenames[:split_1]
  dev_images = filenames[split_1:split_2]
  test_images = filenames[split_2:]
  return train_images, dev_images, test_images


def get_and_shuffle(data_dir):
  """Build a shuffled list of image names."""
  image_name_list = []
  for label_name in os.listdir(data_dir):
    if label_name != '.DS_Store':
      for image_name in os.listdir(os.path.join(data_dir, label_name)):
        image_name_list.append(image_name)
  # Shuffle the data
  shuffled_index = list(range(len(image_name_list)))
  random.shuffle(shuffled_index)
  filenames = [image_name_list[i] for i in shuffled_index]
  return filenames

if __name__ == "__main__":
   main(data_dir, out_dir, MODE)

