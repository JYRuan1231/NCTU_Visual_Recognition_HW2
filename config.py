##################################################
# Config
##################################################

# generate testing/valid dataset


generate_dataset_mat = './data/train/digitStruct.mat'
train_img_folder = './data/train/' # training dataset path
test_img_folder = './data/test/'  # testing images
annotation_file = './model/SVHN_annotation.pkl'  # training label

##################################################
# Training Config
##################################################

# model parameter
train_path = './data/train/'
model_folder = './model/'
model_name = 'faster_rcnn_efficientnetb4_v1.pth'
batch_size = 32  # batch size
workers = 4  # number of Dataloader workers
anchor_sizes = (32, 64, 128, 256, 512)
aspect_ratios = ((0.5, 1, 2),) * len(anchor_sizes)
num_classes = 11
min_size = 380
max_size = 380

# traning parameter

epochs = 18
learning_rate = 0.01
momentum = 0.9
weight_decay = 5e-4

milestones = [5,12]
gamma = 0.1


model_pth = '.'


##################################################
# Testing Config
##################################################
test_path = './data/test/'

score_threshold = 0.3
IoU_threshold = 1
plot_img = False
result_pth = './results/'
json_name = '0856566.json' #submit json name