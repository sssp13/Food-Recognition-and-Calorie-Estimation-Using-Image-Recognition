import splitfolders

# Split folders to train and test
# Current directory
dataraw = r"C:\Users\sptio\Documents\SEM 8 (Sep 2022)\TSE30910 SE PROJECT\test"
# Directories for training and test splits
datasplit = r"C:\Users\sptio\Documents\SEM 8 (Sep 2022)\TSE30910 SE PROJECT\data_split"

# Copy 85% of images to train folder, and 15% of images to test folder
splitfolders.ratio(dataraw, output=datasplit, seed=1337, ratio=(.80, .0, .20), group_prefix=None)


# Split folders from train to validation
# Current directory
data_train = r"C:\Users\sptio\Documents\SEM 8 (Sep 2022)\TSE30910 SE PROJECT\data_split_8_2\train"
# Directories for training and test splits
data_val = r"C:\Users\sptio\Documents\SEM 8 (Sep 2022)\TSE30910 SE PROJECT\data_split_8_2\val"
# Copy 10% of images to validation folder from test folder
splitfolders.ratio(data_train, output=data_val, ratio=(0.9, 0.1, 0.0), group_prefix=None)
