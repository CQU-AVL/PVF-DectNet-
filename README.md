# PVF-DectNet++

**Adaptive Multi-Modal Fusion with Perspective Voxels for 3D Object Detection**

This repository corresponds to the paper:

> **PVF-DectNet++: Adaptive Multi-Modal Fusion with Perspective Voxels for 3D Object Detection**  
> Ke Wang, Weilin Gao, Kai Chen, Tianyi Shao, Liyang Li, Tianqiang Zhou, Jianbo Lu  
> *IEEE Transactions on Circuits and Systems for Video Technology*

---

## Overview

PVF-DectNet++ is a multi-modal 3D object detection framework that fuses LiDAR point clouds and RGB images through **perspective voxel projection** and **learning-aware feature fusion**.  
It addresses limitations of fixed-weight fusion and insufficient image depth semantics in existing LiDAR–camera fusion methods.

Key highlights:
- Perspective voxel-based geometric–semantic alignment
- Adaptive RGB-I image semantic extraction
- Learnable fusion with channel attention and cross-attention
- Strong performance on KITTI, nuScenes, and Waymo benchmarks

---

## Method

The framework consists of four main components:

1. **Dual-hash dynamic voxelization** for LiDAR geometry
2. **Adaptive image semantic extraction** using RGB-I representation
3. **Learning-aware fusion module** (channel + cross attention)
4. **Sparse convolution backbone with RPN detection head**

![PVF-DectNet++ Framework](Files/Figures/fig2.png)

---

## Experimental Results

PVF-DectNet++ achieves consistent improvements over prior methods:

- **KITTI**: +3.56% mAP over PVF-DectNet
- **nuScenes**: +3.8% mAP, +2.6% NDS
- **Waymo**: Significant gains on Pedestrian and Cyclist AP/APH

