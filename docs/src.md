# src

This is the main application of this project. It provides a scalable and reproducible way of obtaining the generalization performance of machine learning models with respect to the provided training set size.

Its main benefits include:  
- **Parallel execution**  
- **Progress tracking**  
- **Safe interruption handling**  
- **Support for hyperparameter tuning**  
- **Ability to resume generation seamlessly**  

---

## 1. `lcurve`

`MODEL` must follow the mlpack interface (`Train()`/`Classify()` for classification and `Train()`/`Predict()` for regression). Similarly, `LOSS` should also adhere to the mlpack constraints. For more information about mlpack, visit [mlpack.org](https://www.mlpack.org/).  

The `DATASET` and `SPLIT` use the dataset containers provided by *LCPP*, or you can create your own data containers with a similar signature. Learning curves can be generated using `double` or `float` types if the machine learning models support these data types.  

An example code snippet is given below:


```cpp
template<
  class MODEL,   // The machine learning model type
  class DATASET, // Dataset type with inputs_ and labels_
  class SPLIT,   // Data splitting strategy
  class LOSS,    // Loss function type
  class O        // Output type (e.g., float, double)
>
LCurve<MODEL, DATASET, SPLIT, LOSS, O> curve( dataset,Ns,reps,parallel,progres,dir,name,seed );
```

In the above example, all the data points are utilized to create learning curves, which results in a decreased test set size as the training set size increases. However, if you want to have a separate testing set, you still need to provide a dataset container of a similar `DATASET` type. In [data](docs/data.md), you can see how to split and transform `DATASET` types.

```cpp
template<
  class MODEL,   // The machine learning model type
  class DATASET, // Dataset type with inputs_ and labels_
  class SPLIT,   // Data splitting strategy
  class LOSS,    // Loss function type
  class O        // Output type (e.g., float, double)
>
LCurve<MODEL, DATASET, SPLIT, LOSS, O> curve( trainset,testset,Ns,reps,parallel,progres,dir,name, seed );
``` 


| Argument        | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `dataset`       | The whole dataset used to generate learning curves.                         |
| `trainset`      | The training set used to generate learning curves.                          |
| `testset`       | The testing set used to generate learning curves.                           |
| `Ns`            | Vector of training set sizes for evaluation.                                |
| `reps`          | Number of repetitions per training set size.                                |
| `parallel`      | Enables parallel execution using OpenMP.                                    |
| `progress`      | Enables the progress bar.                                                   |
| `directory`     | Directory path where results and serialized objects will be saved.          |
| `name`          | Name of the learning curve experiment, used for logging and file naming.    |
| `seed`          | Seed used for splitting the dataset.                                        |

To create a learning curve for models with fixed hyperparameters, as shown below, it is important to provide the same inputs in the same order as they appear in the `Train` method of your learning algorithm.  

In the case of classification methods, most require `num_class` (the number of classes). Here, we automatically deduce that it is a classification problem and include the necessary number of classes, so no additional input is needed from the user.


```cpp
  template<class... Ts>
  LCurve::Generate ( const Ts&... args );
``` 

If you also want to tune the hyperparameters of the model, you need to provide a cross-validation scheme (`CV`) and an optimizer (`OPT`). By default, these are `mlpack::SimpleCV` and `ens::GridSearch`.  

In this case, you must specify a cross-validation parameter (either the number of folds or the percentage of data to be used for testing) and provide iterables for the variables that the `Train` method takes, in the same order. This approach is quite similar to the hyperparameter tuning class in mlpack.


```cpp
  template<template<class,class,class,class,class> class CV = mlpack::SimpleCV,
           class OPT = ens::GridSearch,
           class T = typename std::conditional<
                        std::is_same<CV<MODEL,LOSS,OPT,O,O>,
           mlpack::SimpleCV<MODEL,LOSS,OPT,O,O>>::value,O,size_t>::type,
           class... Ts>
  void GenerateHpt ( const T cvp,
                     const Ts&... args );
``` 

Internally, the generalization performances are computed, and if something unexpected occurs, the binarized version of the current state is saved, and the program exits safely. This allows you to investigate what went wrong. Moreover, external signals can be used to terminate the program. In other words, you can easily impose time limits on your code and resume execution later.

You can access the generalization performances for every training size using the `GetResults` function, which returns an Armadillo matrix of type `arma::Mat<O>`. You can save it or manipulate it as needed.

## 2. `metrics`

Here, we provide some alternative metrics similar to the mlpack signature. You can continue adding your own metrics as needed.

