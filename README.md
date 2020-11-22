# Semi-Supervised-GANs-For-Melanoma-Detection

In this liveProject, you will assume the role of a computer vision engineer working on a proof of concept for a mobile app identifying melanomas among photos of melanocytic nevi (more commonly known simply as moles). Melanoma is the most dangerous type of skin cancer, and its early detection is the key factor in determining the patient’s long term survival rate. Your task is to construct a model that would perform malignant vs. benign image classification on low resolution photos, typical of cellphone cameras. You have 200 training images, 100 of which have been labeled as melanomas, and the other half as benign. In addition to a balanced test set composed of 600 images, you are also provided with 7018 unlabeled images of moles, some of which may be malignant.

In the scenarios where annotating data has to be done manually by a highly trained (in this case, medical) professional, not having enough labeled data is a common problem. There are several techniques available to help you achieve higher accuracy for your model, the most common being data augmentation (artificially inflating the size of your dataset by applying select transformations to existing data) and transfer learning (fine-tuning a model that has been pre-trained on a different dataset). Transfer learning is particularly beneficial when your dataset and the original data come from the same, or at least similar, domains. This presents an additional challenge for the highly specialized sectors, such as medical applications. Fortunately, there is another approach one can take to take advantage of unlabeled data, which is typically much cheaper to obtain and thus often available in large quantities. This approach is called semi-supervised learning. You are requested to combine it with data augmentation techniques in order to build a classifier that will take in a 32x32 pixel photo of a melanocytic nevus, and output the probability of that image being melanoma-positive.

You will be using a popular deep learning framework called Pytorch. The machine learning task can be broken down into the following steps:

Setup an image pre-processing pipeline that handles data augmentation and prepares your data to be fed as input to a Pytorch model.

Train a fully supervised melanoma classifier on the labeled data that you have, and test it to serve as a baseline.

Train a semi-supervised GAN model to make use of the unlabeled training data. Run the trained model on the test set and compare its performance to the supervised baseline. 
