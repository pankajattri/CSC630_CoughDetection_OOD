The models folder contains following files:

1. FENet.py: Creates FENet network. This file is a copy of the FENet.py file present in the Cough-Detection/models/ folder.
2. disc_entropic_ood_Resnet.py: Entropy based framework. Creates a Resnet based architecture with the last layer changed from a linear layer to the one that contains IsoMAx loss.
3. disc_entropic_ood_FENet.py: Entropy based framework. Creates FENet based architecture with the last layer changed from a linear layer to the one that contains IsoMAx loss.
4. disc_with_confidence.py: Confidence based network. This file is unchanged.

The main folder contains following files:

1. evaluate_entropic_ood_FENet.py: Run this file to run entropy based FENet model.
2. evaluate_entropic_ood_Resnet.py: Run this file to run entropy based Resnet model.