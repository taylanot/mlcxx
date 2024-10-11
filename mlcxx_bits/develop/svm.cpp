/**
 * @file svm.cpp
 * @author Ozgur Taylan Turan
 *
 * Trying to create kernelized svm
 */

#include <headers.h>

// Main function
int main() {
  arma::mat X = {{0, 2}, {0, 0}, {2, 1}, {3, 4}, {4, 3}};
  arma::inplace_trans(X);
  /* arma::Row<int> y = {-1, -1, -1, 1, 1}; */
  /* arma::Row<size_t> y = {1,1,1,1,1}; */
  arma::Row<size_t> y = {1,0,0,1,1};

  arma::mat X2 = {{3.1,4.0}, {4.1,3.1}};
  arma::inplace_trans(X2);
  arma::Row<size_t> y2 = {1,1};
  /* arma::Row<int> y = {-1, -1, -1, 1, 1}; */

  double C = 1.;
  double l = 1.;

  // Create SVM object and train

  /* algo::classification::SVM<mlpack::LinearKernel> model(X,y,3,C); */
  /* mlpack::LogisticRegression<> model(X.n_rows,1.e-8); */
  algo::classification::LogisticRegression<> model(X,y,3);
  /* model.Train(X,y); */
  /* /1* algo::classification::OnevAll<mlpack::LogisticRegression<>> model(X,y); *1/ */
  /* arma::Mat<DTYPE> prob; */
  /* arma::Row<size_t> ypred; */
  /* model.Classify(X,ypred,prob); */
  /* arma::Mat<DTYPE> cmat; */
  /* mlpack::data::ConfusionMatrix(y,ypred,cmat,3); */
  /* PRINT_VAR(cmat) */
  /* /1* LOG("FP"<<arma::accu(arma::trimatu(cmat))); *1/ */ 
  /* /1* LOG("TP"<<arma::accu(cmat.diag())); *1/ */ 
  /* /1* PRINT_VAR(cmat); *1/ */
  /* /1* PRINT_VAR(ypred); *1/ */
  /* /1* PRINT_VAR(prob); *1/ */
  /* /1* PRINT_VAR(model.ComputeAccuracy(X2, y2)); *1/ */
  /* /1* PRINT_VAR(model.ComputeAccuracy(X, y)); *1/ */
  /* arma::Mat<DTYPE> encoded; */
  /* mlpack::data::OneHotEncoding(y,encoded); */
  /* PRINT_VAR(encoded); */
  /* PRINT_VAR(prob); */ 
  /* /1* PRINT_VAR(encoded); *1/ */ 
  /* PRINT_VAR(-arma::accu(encoded%arma::log(prob))/double(prob.n_cols)); */ 

  PRINT(model.ComputeAccuracy(X, y));

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


