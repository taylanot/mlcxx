/**
 * @file myxval.cpp
 * @author Ozgur Taylan Turan
 *
 * Let's check trying to create xval with repitition so that I have 
 * fixed datasets for this i can just use ids to call the dataset portion
 *
 */

#include <headers.h>


using LREG  = algo::classification::LogisticRegression<>;  
using LSVC  = algo::classification::SVM<mlpack::LinearKernel>; 
using GSVC  = algo::classification::SVM<mlpack::GaussianKernel>; 
using ESVC  = algo::classification::SVM<mlpack::EpanechnikovKernel>; 
using LDC   = algo::classification::LDC<>; 
using QDC   = algo::classification::QDC<>; 
using NNC   = algo::classification::NNC<>; 
using OpenML = data::classification::oml::Dataset<>;
using Dataset = data::classification::Dataset<>;
using NMC   = algo::classification::NMC<>; 
using RFOR  = mlpack::RandomForest<>; 
using DT    = mlpack::DecisionTree<>; 
using NB    = mlpack::NaiveBayesClassifier<>;
typedef mlpack::FFN<mlpack::CrossEntropyError,mlpack::HeInitialization> Network;
using ANN = algo::ANN<Network> ;

//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
/* int main ( int argc, char** argv ) */
/* { */
/*   arma::wall_clock timer; */

/*   size_t did = 1046;size_t rep = 1000; */
/*   OpenML dataset(did); */
/*   arma::Mat<DTYPE> X; */ 
/*   arma::Row<size_t> y; */ 

/*   timer.tic(); */
/*   arma::uvec a; */
/*   for(size_t i=0;i<rep;i++) */
/*     a = arma::randi<arma::uvec>(dataset.size_,arma::distr_param(0,dataset.size_)); */
/*   PRINT(timer.toc()); */

/*   timer.tic(); */
/*   for(size_t i=0;i<rep;i++) */
/*     mlpack::ShuffleData(dataset.inputs_, dataset.labels_,X,y); */
/*   PRINT(timer.toc()); */

/*   /1* xval::KFoldCV<LDC, mlpack::Accuracy> xv(2, dataset.inputs_, dataset.labels_,true); *1/ */
/*   /1* PRINT(xv.TrainAndEvaluate(dataset.num_class_,0.)); *1/ */
/* } */

template <class Vector=arma::Row<DTYPE>, class T=DTYPE>
Vector resample ( const Vector& vec )
{
  return vec.cols(arma::randi<arma::uvec>(vec.n_elem,
                                          arma::distr_param(0,vec.n_elem-1)));
}

template <class Vector=arma::Row<DTYPE>, class T=DTYPE>
Vector bmean_ci ( const Vector& x, size_t nboot = 100000 )
{
  size_t size = x.n_elem;
  Vector means(nboot);
  arma::uvec idx;

  #pragma omp parallel for
  for (size_t i=0; i<nboot;i++)
    means[i] = arma::accu(resample(x))/size;

  Vector qs = {0.25,0.5,0.95};
  means.save("means.csv",arma::csv_ascii);
  return arma::quantile(means,qs);
}

int main ( int argc, char** argv )
{
  arma::wall_clock timer;
  auto x = arma::randu<arma::Row<DTYPE>>(10000);
  PRINT_VAR(arma::mean(x));
  PRINT_VAR(bmean_ci(x));
}


