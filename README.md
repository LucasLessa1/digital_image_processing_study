# Image Processing - Projects

This repository contains three image processing projects utilizing different techniques and algorithms. Below are the details of each project.

## Table of Contents

1. [Style Transfer](#style-transfer)
2. [Mathematical Morphology](#mathematical-morphology)
   - [Hole Counting in PCB Boards](#hole-counting-in-pcb-boards)
   - [Segmentation using Watershed](#segmentation-using-watershed)
   - [Grayscale Image Binarization](#grayscale-image-binarization)
4. [Image Processing Tools](#image-processing-tools)
   - [CDF/Histogram](#cdfhistogram)
   - [Gamma Correction](#gamma-correction)
   - [Image Augmentation](#image-augmentation)

## Style Transfer

In this project, a pre-trained neural network is used to perform style transfer between two images. Style transfer allows combining the content of one image with the style of another, thus creating a new stylized image.

- **Model Used:** Magenta Arbitrary Image Stylization v1-256
- **Framework:** TensorFlow Hub

## Mathematical Morphology

### Hole Counting in PCB Boards

In this project, mathematical morphology techniques are applied to count the number of holes present in a printed circuit board (PCB). Accurate identification of holes is crucial for the analysis and manufacturing of PCBs.

- **Techniques Used:** Morphological operations such as dilation and erosion.

### Segmentation using Watershed

This project involves segmenting images using the watershed algorithm. Watershed segmentation is particularly useful in situations where objects in an image overlap or are adjacent to each other.

- **Technique Used:** Watershed transform.

### Grayscale Image Binarization

An algorithm was developed to binarize grayscale images. Binarization is a process that converts a grayscale image into a binary image, where the pixels are represented by only two possible values: black and white.

- **Objective:** Highlight specific features of an image to facilitate subsequent analysis.

## Image Processing Tools

### CDF/Histogram

Tools to calculate and visualize the cumulative distribution function (CDF) and histograms of images. These techniques are essential for statistical analysis of images and for applying histogram equalization.

### Gamma Correction

Implementation of gamma correction to adjust the brightness of images. Gamma correction is a nonlinear operation used to encode and decode luminance or tristimulus values in images.

### Image Augmentation

Various techniques were applied to augment the size of images while preserving quality. Augmentation techniques are crucial for improving image resolution and for preparing data for machine learning models.

- **Techniques Used:** Bilinear interpolation, bicubic interpolation, among others.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
