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
  // We will be generating our own data which are two Gaussian blobs 
  // with identity cocvariance matrices and means placed at -1 and 1
  DATASET data(2); // 2 represents the dimension of the dataset
  data.Gaussian(100,{-1,1}); // We are drawing 100 samples from each class.
                             
  // Let's look at the portion of [10,100] of the learning curve
  auto Ns = arma::regspace<arma::Row<size_t>>(10,1,100);
  {
    lcurve::LCurve<MODEL,DATASET,
                   SAMPLE, LOSS,DTYPE> curve(data,        // Provide the dataset
                                             Ns,          // Training set sizes
                                             size_t(100), // number of repititions
                                             true,        // parallel or not?
                                             true);       // Show the progress 
                                                          
    // This will create a learning curve with 
    // maxIterations=100, tolerance=1e-6 parameters required by the 
    // DecisionTree<...>.Train function signiture it should match exactly.
    // Note that, you have to specify the parameters the way 
    // If you wanted to use deafult parameters you can also do 
    // curve.Generate( )
    curve.Generate(100,1e-6); 
    // GetResults returns an arma::Mat<DTYPE> which you can save or do whatever.
    // We will save it for plotting.
    curve.GetResults().save("nottuned.csv",arma::csv_ascii); 
  }

  {
    lcurve::LCurve<MODEL,DATASET,
                   SAMPLE, LOSS,DTYPE> curve(data,        // Provide the dataset
                                             Ns,          // Training set sizes
                                             size_t(100), // number of repititions
                                             true,        // parallel or not?
                                             true);       // Show the progress 
                                                          
    // maxIterations range [10,100] with incriments of 10                                                       
    auto iters = arma::regspace<arma::Row<size_t>>(10,10,100);

    // Here we will obtain the learning curve with hyper-parameter tuning
    // DecisionTree<...>.Train function signiture it should match exactly.
    // Note that, you have to specify the parameters the way 
    // If you wanted to use deafult parameters you can also do 
    // curve.Generate( )
    curve.GenerateHpt(0.2, // validation set percent 
                      iters); 
    // GetResults returns an arma::Mat<DTYPE> which you can save or do whatever.
    // We will save it for plotting.
    curve.GetResults().save("tuned.csv",arma::csv_ascii); 

  }




  DEPEND_INFO( );
  return 0;
}

