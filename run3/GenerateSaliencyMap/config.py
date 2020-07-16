#########################################################################
# MODEL PARAMETERS														#
#########################################################################
# batch size
b_s = 4
# number of epochs
nb_epoch = 12
# number of timestep
nb_timestep = 3
# number of learned priors
nb_gaussian = 16

img_channel_mean = [103.939, 116.779, 123.68]
#########################################################################
# DATA SETTINGS										            	#
#########################################################################
# path of datasets
# datasest_path = './DUTSET/'
datasest_path = r'C:\Users\ian_c\Desktop\ECS\COMP6223CV\CourseWork3\testing\\'
saliency_output = r'C:\Users\ian_c\Desktop\ECS\COMP6223CV\CourseWork3\SaliencyMap\testing\\'
VGG_TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'