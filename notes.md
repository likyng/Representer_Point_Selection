# Representer Point Selection

## File structure

Of the git repository for Representer Point Selection.

### Representer Point Selection

    .
    |-- compute_representer_vals.py

Calculates the Representer values. The final output for CIFAR10 is
`output/weight_matrix.npz` which is a numpy array with the shape `50000 x 10`.

For CIFAR10, model, training, and test data is loaded from
`data/weight_323436.pkl`. Then, that model is trained until the condition
`min_loss < init_grad/200` is reached, where `min_loss = 10000`. That usually
happens after epoch 456.

    |-- experiments
    |   |-- class0.png
    |   |-- class1.png
    |   |-- correlation.png
    |   |-- euclideaninfcompare.png
    |   |-- fig1_correlation.ipynb
    |   |-- fig2_dataset_debugging.ipynb
    |   |-- fig2_dataset_debugging.py
    |   |-- fig3_4_visualize_samples_awa.ipynb
    |   |-- fig6_sens_decomposition.ipynb
    |   |-- fig7_distribution.ipynb
    |   |-- fig8_toymodel_appendix.ipynb
    |   |-- fig9_euclidean_vs_representer_appendix.ipynb
    |   |-- img_3092.png
    |   |-- inf_distribution.png
    |   |-- labelfix.png
    |   |-- toy_inf_ours.png
    |   |-- toy_inf.png
    |   |-- zeroinf_but_bigreps.png
    |   `-- zeroreps_but_biginf.png

### Influence Functions

The following files contain code used in the paper regarding influence
functions. Repo originally cloned from the influence function release repo and
then modified.

    |-- influence-release-mod
    |   |-- influence
    |   |   |-- awa_mlp.py
    |   |   |-- binaryLogisticRegressionWithLBFGS.py
    |   |   |-- cifar_mlp.py

Defines VGG16 using `genericNeuralNet` and then redefined the last three layers
of VGG16 depending on which layer was given as `idx`, meaning where to pull
the output from (usually 31, 32, or 34).

    |   |   |-- dataset.py

Contains the definition of the `DataSet` class used as base for CIFAR10 and
AwA. Also contains the functions which are used for random subsampling of the
dataset. This is for example used to generate Figure 2 of the paper.

    |   |   |-- experiments.py
    |   |   |-- genericNeuralNet.py
    |   |   |-- hessians.py
    |   |   |-- __init__.py
    |   |   |-- logisticRegressionWithLBFGS.py
    |   |   `-- toy_mlp.py
    |   |-- __init__.py
    |   |-- README.md
    |   `-- scripts
    |       |-- data -> ../../data
    |       |-- gen_vgg_features.py

Defines VGG16 as network to be trained. Loads the `fine_tune_relu_cifar.h5`
weights. Then trains the network and saves the output of the layer `idx` given
to the function. In most cases the output of `layer 34` will be saved. The file
saved to is `data/vgg_features_cifar_IDX.h5`. Layer indices can be seen
[here](https://www.mathworks.com/help/deeplearning/ref/vgg16.html).  

An excerpt:

    29   'relu5_2'   ReLU
    30   'conv5_3'   Convolution (last conv layer)
    31   'relu5_3'   ReLU
    32   'pool5'     Max Pooling
    33   'fc6'       Fully Connected
    34   'relu6'     ReLU
    35   'drop6'     Dropout

Continuing the file structure:

    |       |-- load_cifar.py
    |       |-- load_mnist.py
    |       |-- load_toy.py
    |       |-- output -> ../../output
    |       |-- train_awa_resnet.py
    |       |-- train_cifar_vgg.py

Trains VGG16 from features obtained by `gen_vgg_features.py` loaded from
`data/vgg_features_cifar_34.h5`. The model is defined in class `CIFAR_MLP` in
file `influence-release-mod/influence/cifar_mlp.py`. After training the model,
the latest checkpoint is loaded and the influence function values for the first
1000 test points from the test dataset are calculated and saved in:
`output/cifar_inf_test_XXX.npz`. The Representer Point Selection vales are *not*
calculated.

    |       |-- train_toy_mlp.py
    |       `-- utils.py

### Misc files

    |-- notes.md
    |-- README.md
    |-- Representer Point Selection Supplementary.pdf

### Data used in the experiments

    |-- data
    |   |-- checkpoint.pth.tar
    |   |-- fine_tune_relu_cifar.h5
    |   |-- intermediate34_test.pkl
    |   |-- intermediate34_train.pkl
    |   |-- intermediate36_test.pkl
    |   |-- intermediate36_train.pkl
    |   |-- output_test_cifar.pkl
    |   |-- presoft_test.pkl
    |   |-- toy_2d.npz
    |   |-- train_feature_awa.npy
    |   |-- train_input-005.npy
    |   |-- train_label.npy
    |   |-- train_output_awa.npy
    |   |-- train_output.npy
    |   |-- val_feature_awa.npy
    |   |-- val_input-004.npy
    |   |-- val_label.npy
    |   |-- val_output_awa.npy
    |   |-- vgg_features_cifar_34.h5
    |   |-- weight_323436.pkl
    |   `-- weight_bias.pickle
    `-- output
        |-- awa_inf_test_1000.npz
        |-- cifar_inf_test_1000.npz
        |-- cifar_inputcheck_results.npz
        |-- idx_inf_awa_1000.npz
        |-- toy2d_mlp_inf.npz
        |-- toy2d_outputs.npz
        |-- weight_matrix_AwA.npz
        |-- weight_matrix_AwA.pkl
        |-- weight_matrix_Cifar.npz
        |-- weight_matrix_Cifar.pkl
        `-- weight_matrix_out_toy.pkl
