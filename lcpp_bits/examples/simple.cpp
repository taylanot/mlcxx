/**
 * @file simple.cpp
 * @author Ozgur Taylan Turan
 *  This is a simple program that create a learning curve with and without 
 * hyper-parameter tunining for a simple case.
 */
#include <lcpp.h>

// Since it is a classification task we use the template parameter 
// arma::Row<size_t>. DTYPE is "double" by default for some mlpack learners
// do not have the option to be run in "float" unfortunately.
using DATASET = data::Dataset<arma::Row<size_t>,DTYPE>;
// Let's choose the model as Decision Tree Classifier.
using MODEL = mlpack::DecisionTree<>;
// Our loss measure is accuracy.
using LOSS = mlpack::Accuracy;
// We will be sampling randomly from the finite dataset without replacement.
// We are in the classification setting however, we will not be using 
// stratification since we want to observe the learning curve for each training
// sample size possible.
using SAMPLE = data::RandomSelect<>;

int main ( ) 
{
  // We will be generating our own data with banana dataset aka moons by 
  // sklearn people
  DATASET data(2); // 2 represents the dimension of the dataset
  data.Banana(200); // We are drawing 100 samples from each class.
  // You can split this dataset and used training and testing splits in 
  // LCurve. This operation will be using only the testset to test it.
                             
  // Let's look at the portion of [10,100] of the learning curve
  auto Ns = arma::regspace<arma::Row<size_t>>(10,1,350);
  {
    lcurve::LCurve<MODEL,DATASET,
                   SAMPLE, LOSS,DTYPE> curve(data,  // dataset
                                             Ns,  // training set sizes
                                             size_t(1000),  // repititions
                                             true,  // parallel 
                                             true,  // progress 
                                             "build", // directory
                                             "fixedLCurve");// name
                                                          
    // This will create a learning curve with 
    // minimumleafsize=50
    // DecisionTree<...>.Train function signiture it should match exactly.
    // If you wanted to use deafult parameters you can also do 
    // curve.Generate( )
    curve.Generate(50); 
    // GetResults returns an arma::Mat<DTYPE> which you can save or do whatever.
    // We will save it for plotting.
    curve.GetResults().save("build/nottuned.csv",arma::csv_ascii); 
  }

  {
    lcurve::LCurve<MODEL,DATASET,
                   SAMPLE, LOSS,DTYPE> curve(data,  // dataset
                                             Ns,  // training set sizes
                                             size_t(1000),  // repititions
                                             true,  // parallel 
                                             true,  // progress 
                                             "build", // directory
                                             "tunedLCurve");// name
                                             
    // minimumleafsize range [1,50] with incriments of 5. 
    auto leafs= arma::regspace<arma::Row<size_t>>(1,5,50);

    // Here we will obtain the learning curve with hyper-parameter tuning
    // DecisionTree<...>.Train function signiture it should match exactly.
    // Here you can also choose to use other cross validation and 
    // optimization procedures with template paramters of this function
    // curve.GenerateHpt<CV,OPT> Default is mlpack::SimpleCV and ens::GridSearch
    curve.GenerateHpt(0.2, // validation set percent 
                      leafs// Hyper-parameter values that we search from
                      ); 
    // GetResults returns an arma::Mat<DTYPE> which you can save or do whatever.
    // We will save it for plotting.
    curve.GetResults().save("build/tuned.csv",arma::csv_ascii); 

  }

  DEPEND_INFO( );

  return 0;
}

