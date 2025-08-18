# algo

This directory contains implementations of machine learning algorithms organized into two main categories: **regressors** and **classifiers**. Additionally, it provides a **neural network wrapper** that offers a unified interface for deep learning models, simplifying the use of *mlpack* neural networks.  

The main function signatures of all models follow those of *mlpack*. Classifiers use the `Train/Classify` interface, while regressors use `Train/Predict`. If you plan to use **LCPP**, it is recommended to adhere to this structure, since most of the metrics rely on this convention.  

---

## 1. classifier

For classification tasks, the following models are available: Linear and Quadratic Discriminant classifiers (`LDC`, `QDC`), Nearest Mean and Nearest Neighbor classifiers (`NMC`, `NNC`), and the Non-linear SVM classifier (`SVM`). Additionally, a templated One-vs-All (`OnevAll`) framework is provided for multi-class classification, allowing you to train any binary classifier that follows the *mlpack* classifier signature. All classifiers are located in the `algo::classification` namespace. Example usage of `OnevALL` is

```cpp
  OnevAll<mlpack::LogisticRegression> model;
  model.Train(...);
  model.Classify(...);
```

---

## 2. regressor

For regression tasks, the following models are available: Kernel Ridge Regression (`KernelRidge`), Semi-Parametric Kernel Ridge (`SemiParamKernelRidge`), and Gaussian Process (`GaussianProcess`). These are implemented in the `algo::regression` namespace. Furthermore, a kernel smoothing function (`kernelsmoothing`) with internal bandwidth optimization is provided in the `algo::functional` namespace.  

---

## 3. dimred

For dimensionality reduction, the available method is Univariate Functional Principal Component Analysis (`ufpca`), which can be found in the `algo::functional` namespace.  

---

## 4. nn

For those seeking a simple training procedure with *mlpack* artificial neural networks, this wrapper provides task-invariant training and prediction/classification functionality. Users can easily define a neural network architecture and train it with built-in early stopping, where 2% of the training data is randomly separated for validation. This class is templated over the network type, optimizer, and loss function. It is located in the `algo` namespace. Example usages of the `ANN` class for both classification and regression tasks can be found in the `examples` directory.  


```cpp
using OPT = ens::Adam;
using LOSS = mlpack::Accuracy;
using NetworkType = mlpack::FFN<mlpack::CrossEntropyError>;
using MODEL = algo::ANN<NetworkType,OPT,LOSS>;
... 
  // Define the architecture
  NetworkType network;
  network.Add<mlpack::Linear>(...);
  network.Add<mlpack::ReLU>();
  network.Add<mlpack::Linear>(...);
  network.Add<mlpack::ReLU>();
  network.Add<mlpack::Linear>(...);
  network.Add<mlpack::Softmax>();
...
  MODEL ann(network);
  ann.Train(...);
  ann.Classify(...);
```

