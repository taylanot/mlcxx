/**
 * @file check.cpp
 * @author Ozgur Taylan Turan
 *
 * Let's check the dataset 39
 */

#include <headers.h>


using LREG  = algo::classification::LogisticRegression<>;  
using LSVC  = algo::classification::SVM<mlpack::LinearKernel>; 
using GSVC  = algo::classification::SVM<mlpack::GaussianKernel>; 
using ESVC  = algo::classification::SVM<mlpack::EpanechnikovKernel>; 
using LDC   = algo::classification::LDC<>; 
using QDC   = algo::classification::QDC<>; 
using NMC   = algo::classification::NMC<>; 

//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{
  using Dataset = data::classification::oml::Dataset<>;
  /* arma::irowvec ids = {1046}; // 11 37 39 53 61 */
  arma::irowvec ids = {11}; // 11 37 39 53 61
  Dataset dataset(ids[0]);
  PRINT_VAR(dataset.size_);
  PRINT_VAR(dataset.dimension_);
  PRINT_VAR(dataset.num_class_);
  /* GSVC model(dataset.inputs_,dataset.labels_,dataset.num_class_); */
  /* mlpack::Accuracy metric; */
  /* PRINT(metric.Evaluate(model,dataset.inputs_,dataset.labels_)); */

  xval::KFoldCV<GSVC, mlpack::Accuracy> xv(10, dataset.inputs_, dataset.labels_);
  xval::KFoldCV<LSVC, mlpack::Accuracy> xv2(10, dataset.inputs_, dataset.labels_);
  PRINT(xv.TrainandEvaluate(dataset.num_class_,double(10)));
  PRINT(xv2.TrainandEvaluate(dataset.num_class_,double(10)));
  /* PRINT(xv.TrainAndEvaluate(dataset.num_class_)); */
  /* PRINT(xv.Evaluate(dataset.num_class_)); */
}
