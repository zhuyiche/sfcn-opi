# sfcn-opi
This is an implmentation of SFCN-OPI on Python3, Keras and TensorFlow. 
For details about the SFCN-OPI architecture please refer to the paper [SFCN-OPI: Detection and Fine-grained Classification of Nuclei Using
Sibling FCN with Objectness Prior Interaction.](https://arxiv.org/pdf/1712.08297.pdf)

### Requirement
In addition to python3.5, tensorflow and keras, 
you can installed the required packages by running `pip install -r requirements.txt`
 under virtualenv
 
 
 
 
### Dataset
We use CRCHistoPhenotypes to test our model.
This dataset involves 100 H&E stained histology images of colorectal adenocarcinomas.
 A total of 29,756 nuclei were marked at/around the center for detection purposes. 
 Out of these, there were 22,444 nuclei that also have an associated class label,
 i.e. epithelial, inflammatory, fibroblast, and miscellaneous.
 
 For more detail and download the dataset, please check 
 https://warwick.ac.uk/fac/sci/dcs/research/tia/data/crchistolabelednucleihe/
 
 You can use masks_creation.py and function in utils.py to do data preprocessing for you.