# Oilseed rape / service plants Segmentation

This repository contains code for transfer learning of [DeepLabV3 ResNet101](https://arxiv.org/abs/1706.05587) in PyTorch. The model is from the [torchvision module](https://pytorch.org/docs/stable/torchvision/models.html#semantic-segmentation). 

Pretrained DeepLabV3 Resnet101 on Pascal VOC dataset is fine tuned on 50 field images of oilseed intercrop data-set over 25 epochs and achieves an testing mean accuracy of 0.96 and mean IoU value of 0.86.

The segmentation output of the model on a sample image are shown below.

<img src="https://live.staticflickr.com/65535/51691461915_dc945f0c04_o_d.png" width=60% >


#### Aknowledgement

This research was funded by Agroscope (Swiss confederation), UFA Samen, Florin and Nurtriswiss as part of the ICARO project. 



#### Credits & References

- M. S. Minhas, “Transfer Learning for Semantic Segmentation using PyTorch DeepLab v3,” GitHub.com/msminhas93, 12-Sep-2019. [Online]. Available: [https://github.com/aureliedj/DeepLabv3Finetuning](https://github.com/aureliedj/DeepLabv3Finetuning)

- L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, ‘Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation’, arXiv:1802.02611 [cs], Aug. 2018, Available: [http://arxiv.org/abs/1802.02611](http://arxiv.org/abs/1802.02611)






