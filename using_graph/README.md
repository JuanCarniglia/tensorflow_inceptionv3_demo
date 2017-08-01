
# Tensorflow Demo using a pre-trained GRAPH

In this example use, I take a pre-trained graph and use a script that is shipped with Tensorflow to re-train the model, using my own images.

The images have to be the same size as those originally used to train the graph (299x299), and JPEG.

The images have to be on the tf_files/train_images folder, inside each category floder.

The number of categories will be taken from a file called retrained_labels.txt
