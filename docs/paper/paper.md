---
title: 'LCPP: Learning Curve Plus Plus'
tags:
  - C++
  - hyper-parameter tuning
  - mlpack
  - ensmallen
  - Armadillo
  - learning curve
  - generalization performance
  - OpenMP
  - pattern recognition
  - machine learning

authors:
  - name: Ozgur Taylan Turan
    orcid: 0009-0004-1022-6334
    corresponding: true 
    affiliation: 1
  - name: David M.J. Tax
    affiliation: 1
affiliations:
 - name: Delft University of Technology, The Netherlands
   index: 1
   ror: 02e2c7k09
date: 09 January 2026
bibliography: ref.bib
--- 
# Summary

A learning algorithm is said to learn if its performance on a given task improves with experience\ [@mitchell]. This fundamental definition links the size of training data to the generalization performance of the model. In supervised learning, a **learning curve** depicts how generalization performance evolves as a function of the training set size. Collections of such data, known as **learning curve databases**, track the performance of diverse machine learning algorithms (learners) across multiple tasks as they observe increasing amounts of training data.

Learning curve databases are valuable for **model selection** and for **estimating the amount of data needed** to achieve a target performance. These applications typically assume learning curves are monotonic and convex. However, findings of @yan2025, @mohr2023, and @viering2022 suggest that learning curves often exhibit more complex and irregular behavior. Sparse sampling of training sizes limits the ability to fully characterize these behaviors, highlighting the need for **high-fidelity learning curves** to investigate them.

**LCPP (Learning Curve Plus Plus)** is a C++ library that allows for learning curve creation of machine learning models. `LCPP` enables its users to obtain learning curves for a variety of learners on any supervised learning dataset with or without hyper-parameter tuning, enabling model selection and training data requirement determination.

# Statement of Need

Generally, creating learning curves is computationally expensive because it requires repeatedly training algorithms on many subsets of varying training sizes. Consequently, learning curves are often computed for a limited number of training set sizes. For example, while creating learning curve databases, @mohr2023 and @yan2025 used a limited number of training set sizes. Moreover, these generations are computed only for fixed learners without hyper-parameter tuning.

To empower the machine learning community to generate richer, more detailed learning curves, we propose `LCPP`: a C++ library for scalable learning curve generation. `LCPP` offers several features; first, it offers several approaches for splitting a given dataset into training and test sets of varying sizes (where training sets can be drawn randomly or incrementally, where test sets can be fixed or vary in size). Next, unlike most existing tools that fix hyper-parameters during learning curve creation, `LCPP` integrates hyper-parameter optimization routines from `mlpack` [@mlpack2023], enabling optimized learner evaluations.

`LCPP` also includes a simple dataset container for access to `OpenML` datasets [@openml], with built-in support for complete dataset transformations and train/test splits, allowing users to directly measure the generalization performance of models available in `mlpack` and some other learning algorithms included in `LCPP`, such as kernel ridge regression, discriminant classifiers, multi-class classification extensions of binary classifiers. 

# State of the Field 

Several tools are available in the Python ecosystem for learning curve generation. `scikit-learn` [@scikit] provides a flexible interface for constructing learning curves, allowing cross-validation strategies to be combined with a learner. However, its extensibility and suitability for constructing high-fidelity learning curves remains limited. `LCDB 1.0` and `1.1`\ [@yan2025; @mohr2023] primarily serve as wrappers around existing learning curve databases. While learning curve generation is possible, it requires additional modification of the provided scripts and is not a central design focus.

In the C++ domain, we are not aware of any tool explicitly designed for learning curve generation. `LCPP` addresses this gap. It is modular by design, can be extended to support a wide range of learning curve research workflows, and can be deployed in high-performance computing environments with minimal overhead. In addition, similar to `mlpack`, it can be valuable for embedded and low-resource environments, for model selection, and hyper-parameter tuning purposes.

# Software Design

`LCPP` is designed for easy deployment on high-performance computing (HPC) environments. With little effort it can efficiently run large-scale experiments in parallel, ensuring reproducibility and scalability. Moreover, it supports easy and lightweight check-pointing, allowing high-fidelity (both in terms of the training set size resolution and also the times the training set is resampled) learning curves to be created in multiple sessions. This structure also enables the missing experiments to be investigated easily. 

It is also designed with future-proofing in mind. Adoption of the `mlpack` conventions means `LCPP` has access to a wide range of learning algorithms. And as `mlpack` continues to expand the number of supported models, so will `LCPP`. In addition, `LCPP` is not restricted by this: any model that is using the same conventions as `mlpack` and relies on `Armadillo` [@arma2016] and `ensmallen` [@ensmallen2021] can also be used without extra effort. 

# Research Impact Statement

`LCPP` is used by @turan2025 and @turan2026 to generate large-scale learning curve databases by considering many degrees of freedom involved in this process. By enabling tracking of the generalization performance across machine learning models, `LCPP` facilitates a systematic learning curve creation with a fast development and deployment cycle. We hope that it will serve as a foundation for future learning curve research.

# AI Usage Disclosure

We did not use AI for the development of this software or the writing of this paper.

# Acknowledgements 

We thank Marco Loog, and Tom Viering for discussions and collaborations that helped shape the broader context of this work, and Gijs van Tulder for his help with documentation and rebasing the repository.

# References

