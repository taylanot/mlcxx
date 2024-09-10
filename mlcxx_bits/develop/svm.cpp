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
  arma::Row<size_t> y = {0,0,0,1,1};
 
  double C = 1.;
  double l = 1.;

  // Create SVM object and train
  /* algo::classification::KernelSVM svm(C); */
  /* svm.Train(X, y); */

  algo::classification::KernelSVM<mlpack::GaussianKernel> svm(X,y,C);
  PRINT(svm.ComputeAccuracy(X, y));

  /* data::classification::Dataset trainset(2,100,2); */
  /* data::classification::Dataset testset(2,1000,2); */
  /* trainset.Generate("Harder"); */
  /* testset.Generate("Harder"); */
  

  /* algo::classification::KernelSVM<mlpack::GaussianKernel> */ 
  /*   svm2(trainset.inputs_,trainset.labels_,C); */
  /* PRINT(svm2.ComputeAccuracy(testset.inputs_, testset.labels_)); */

  /* arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,10); */
  /* src::LCurve<algo::classification::KernelSVM<mlpack::GaussianKernel>, */
  /* /1* src::LCurve<algo::classification::KernelSVM<mlpack::LinearKernel>, *1/ */
  /*             mlpack::Accuracy, */
  /*             data::N_StratSplit> */ 
  /*   lcurve(Ns,100,true,false,true); */

  /* /1* lcurve.Bootstrap(trainset.inputs_,trainset.labels_,1.,1.); *1/ */
  /* lcurve.Split(trainset,testset); */
  /* lcurve.test_errors_.save("svc.csv",arma::csv_ascii); */



  return 0;
}


