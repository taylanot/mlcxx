/**
 * @file svm.cpp
 * @author Ozgur Taylan Turan
 *
 * Trying to create kernelized svm
 */

#include <headers.h>
// Define the RBF kernel function

// Main function
int main() {
  arma::mat X = {{0, 2}, {0, 0}, {2, 1}, {3, 4}, {4, 3}};
  arma::inplace_trans(X);
  /* arma::Row<int> y = {-1, -1, -1, 1, 1}; */
  arma::Row<size_t> y = {2,0,0,1,1};
 
  double C = 1.;
  double l = 1.;

  // Create SVM object and train

  algo::classification::SVM<mlpack::GaussianKernel> model(X,y,3,C,l);
  /* algo::classification::OnevAll<mlpack::LogisticRegression<>> model(X,y); */
  arma::Mat<DTYPE> prob;
  arma::Row<size_t> ypred;
  model.Classify(X,ypred,prob);
  PRINT(ypred);
  PRINT(prob);
  PRINT(model.ComputeAccuracy(X, y));

  /* PRINT(svm.ComputeAccuracy(X, y)); */

  /* data::classification::Dataset trainset(2,4,2); */
  /* data::classification::Dataset testset(2,3,2); */
  /* trainset.Generate("Simple"); */
  /* testset.Generate("Simple"); */
  

  /* algo::classification::SVM<mlpack::GaussianKernel> */ 
  /*   svm2(trainset.inputs_,trainset.labels_,2,C); */
  /* PRINT(svm2.ComputeAccuracy(testset.inputs_, testset.labels_)); */

  /* arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,2); */
  /* src::LCurve<algo::classification::LDC<>, */
  /*             mlpack::Accuracy> */ 
  /*   lcurve(Ns,100,true,false,true); */
  /* src::LCurve<algo::classification::KernelSVM<mlpack::GaussianKernel>, */
  /* /1* src::LCurve<algo::classification::KernelSVM<mlpack::LinearKernel>, *1/ */
  /*             mlpack::Accuracy, */
  /*             data::N_StratSplit> */ 
  /*   lcurve(Ns,100,true,false,true); */

  /* /1* lcurve.Bootstrap(trainset.inputs_,trainset.labels_,1.,1.); *1/ */
  /* lcurve.Split(trainset,testset,2); */
  /* lcurve.test_errors_.save("ldc.csv",arma::csv_ascii); */



  return 0;
}


