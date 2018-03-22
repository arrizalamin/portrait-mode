output_location=models

echo "Downloading monodepth model"
output_file=$output_location/model_city2kitti.zip
extract_location=$output_location/model_city2kitti/

wget -nc http://visual.cs.ucl.ac.uk/pubs/monoDepth/models/model_city2kitti.zip -O $output_file
mkdir -p $extract_location
unzip $output_file -d $extract_location

echo "Downloading DeepLabv3+ model"
output_file=$output_location/deeplabv3_pascal_train_aug_2018_01_04.tar.gz

wget -nc http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz -O $output_file
tar xvf $output_file -C $output_location
