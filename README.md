# 🎨 Poisson Image Editing

A Python implementation of seamless image blending using Poisson equations for Computer Graphics.

## 🎯 Overview

This project implements the Poisson Image Editing method that enables seamless image blending without visible borders. The implementation follows the mathematical framework from Pérez et al. (2003) paper [1], solving discrete Poisson equations for optimal gradient domain editing.

## 🧮 Mathematical Foundation

The method minimizes the energy functional:
```
min ∫∫ |∇f - v|² with f|∂Ω = f*|∂Ω
```

Which discretizes to:
```
|Np|fp - Σ fq = Σ f*q + Σ vpq
```

Where `v = ∇g` (gradient field from source image).

## 🔧 Core Function

```python
poisson_editing(source, target, mask, offset=(0,0), mixing=False)
```

### Parameters
- **📸 source**: Source image array (patch to copy)
- **🎯 target**: Target image array (destination)
- **🎭 mask**: Binary mask selecting pixels of interest
- **📍 offset**: Vertical/horizontal displacement (default: (0,0))
- **🔄 mixing**: Flag to blend gradients from both images

## ✨ Key Features

### 🔧 Algorithm Steps
1. **📐 Region calculation** - Determines ROI based on mask and offset
2. **🔢 Matrix construction** - Builds coefficient matrix A and vector b
3. **⚡ System solving** - Solves Ax = b for each color channel
4. **🎨 Blending** - Applies solution to create seamless result

### 🎭 Techniques Supported
- **🖼️ Object insertion** - Seamless placement with irregular borders
- **🔄 Feature exchange** - Transfer characteristics between objects
- **🌫️ Gradient mixing** - Preserve target texture while adding source
- **⚫ Grayscale transfer** - Monochromatic blending for natural colors
- **🕳️ Transparent objects** - Handle objects with holes/transparency
- **🎨 Local color changes** - Selective color manipulation
- **📏 Texture flattening** - Remove textures while preserving edges

## 🏗️ Implementation Details

### Matrix Construction
- **A**: Sparse coefficient matrix (size×size)
- **b**: Right-hand side vector with boundary conditions
- **4-connected neighbors** for discrete Laplacian
- **Boundary handling** for seamless transitions

### Gradient Mixing
```python
if abs(diff_target) > abs(diff_source):
    b[index] += diff_target
else:
    b[index] += diff_source
```

## 🎨 Applications

- **🖼️ Photo montage** - Combine multiple images seamlessly
- **🎭 Object removal/insertion** - Clean image editing
- **🌈 Color grading** - Selective color manipulation
- **✨ Texture editing** - Modify surface properties
- **🔧 Image restoration** - Fill missing regions

## 📊 Results

The implementation successfully demonstrates:
- ✅ Seamless object insertion with irregular boundaries
- ✅ Feature exchange between different objects
- ✅ Gradient mixing for texture preservation
- ✅ Transparent object handling
- ✅ Local color modifications
- ✅ Texture flattening capabilities

## 🔬 Technical Notes

- **Discrete Poisson equation** solving using sparse linear algebra
- **4-connected neighborhood** for gradient computation
- **Per-channel processing** for color images
- **Edge detection integration** for texture flattening

---

*Developed by Cristian Casali for Computer Graphics Course, October 2024*

---
[1] [Poisson Image Editing - Original Paper](https://doi.org/10.1145%2F1201775.882269)
