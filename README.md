# DeepLabv3 Oilseed Rape Segmentation

This repository contains code for Fine Tuning [DeepLabV3 ResNet101](https://arxiv.org/abs/1706.05587) in PyTorch. The model is from the [torchvision module](https://pytorch.org/docs/stable/torchvision/models.html#semantic-segmentation). 

The model is fine tunned on field images of oilseed intercrop data-set. [X. Bousselin, 2021](https://www.mdpi.com/2073-4395/11/8/1493#cite)

The model was fine tuned for 25 epochs and achieves an testing mean Accuracy value of 0.968.

The segmentation output of the model on a sample image are shown below.

![Sample segmentation output](https://live.staticflickr.com/65535/51688455931_53f2b85a8a_o_d.jpg)

## Key Features

* LivePreview - Make changes, See changes
  - Instantly see what your Markdown documents look like in HTML as you create them.
* Sync Scrolling
  - While you type, LivePreview will automatically scroll to the current location you're editing.
* GitHub Flavored Markdown  
* Syntax highlighting
* [KaTeX](https://khan.github.io/KaTeX/) Support
* Dark/Light mode
* Toolbar for basic Markdown formatting
* Supports multiple cursors
* Save the Markdown preview as PDF
* Emoji support in preview :tada:
* App will keep alive in tray for quick usage
* Full screen mode
  - Write distraction free.
* Cross platform
  - Windows, macOS and Linux ready.

## How To Use

To clone and run this application, you'll need [Git](https://git-scm.com) and [Node.js](https://nodejs.org/en/download/) (which comes with [npm](http://npmjs.com)) installed on your computer. From your command line:

```bash
# Clone this repository
$ git clone https://github.com/aureliedj/DeepLabv3---Oilseed-Rape-segmentation.git

# Go into the repository
$ cd DeepLabv3---Oilseed-Rape-segmentation

# Install dependencies
$ npm install

# Run the app
$ npm start
```

Note: If you're using Linux Bash for Windows, [see this guide](https://www.howtogeek.com/261575/how-to-run-graphical-linux-desktop-applications-from-windows-10s-bash-shell/) or use `node` from the command prompt.


## Credits

- M. S. Minhas, “Transfer Learning for Semantic Segmentation using PyTorch DeepLab v3,” GitHub.com/msminhas93, 12-Sep-2019. [Online]. Available: [https://github.com/aureliedj/DeepLabv3Finetuning](https://github.com/aureliedj/DeepLabv3Finetuning)

## Citations

- L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, ‘Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation’, arXiv:1802.02611 [cs], Aug. 2018, Available: http://arxiv.org/abs/1802.02611{http://arxiv.org/abs/1802.02611}



