/**
 * @file svm.cpp
 * @author Ozgur Taylan Turan
 *
 * Trying to create kernelized svm
 */

#define DTYPE float

#include <headers.h>

// Main function
/* int main() { */
/*   /1* data::classification::oml::Dataset dataset(8); *1/ */
/*   /1* arma::Mat<DTYPE> X = dataset.inputs_.cols(0,20); *1/ */
/*   /1* arma::Row<size_t> y = dataset.labels_.cols(0,20); *1/ */
/*   /1* PRINT_VAR(y); *1/ */
/*   /1* arma::mat X = {{0, 2}, {0, 0}, {2, 1}, {3, 4}, {4, 3}}; *1/ */
/*   arma::Mat<DTYPE> X = {{-1, -1}, {-1, 1}, {1, 1}, {3, -1}}; */
/*   arma::inplace_trans(X); */
/*   /1* arma::Row<> y = {-1, -1, -1, 1, 1}; *1/ */
/*   /1* /2* arma::Row<size_t> y = {1,1,1,1,1}; *2/ *1/ */
/*   arma::Row<size_t> y = {1,1,0,0}; */

/*   /1* arma::mat X2 = {{3.1,4.0}, {4.1,3.1}}; *1/ */
/*   /1* arma::inplace_trans(X2); *1/ */
/*   /1* arma::Row<size_t> y2 = {1,1}; *1/ */
/*   /1* /2* arma::Row<int> y = {-1, -1, -1, 1, 1}; *2/ *1/ */

/*   DTYPE C = 1.; */
/*   /1* double l = 1.; *1/ */

/*   // Create SVM object and train */

/*   algo::classification::SVM<mlpack::LinearKernel> model(X,y,2,C); */
/*   /1* mlpack::LogisticRegression<> model(X.n_rows,1.e-8); *1/ */
/*   /1* algo::classification::SVM<> model(X,y,arma::unique(dataset.labels_).eval().n_elem); *1/ */
/*   /1* algo::classification::LogisticRegression<> model(X,y,arma::unique(dataset.labels_).eval().n_elem); *1/ */
/*   /1* model.Train(X,y); *1/ */
/*   /1* /2* algo::classification::OnevAll<mlpack::LogisticRegression<>> model(X,y); *2/ *1/ */
/*   /1* arma::Mat<DTYPE> prob; *1/ */
/*   /1* arma::Row<size_t> ypred; *1/ */
/*   /1* model.Classify(X,ypred,prob); *1/ */
/*   /1* arma::Mat<DTYPE> cmat; *1/ */
/*   /1* mlpack::data::ConfusionMatrix(y,ypred,cmat,3); *1/ */
/*   /1* PRINT_VAR(cmat) *1/ */
/*   /1* /2* LOG("FP"<<arma::accu(arma::trimatu(cmat))); *2/ *1/ */ 
/*   /1* /2* LOG("TP"<<arma::accu(cmat.diag())); *2/ *1/ */ 
/*   /1* /2* PRINT_VAR(cmat); *2/ *1/ */
/*   /1* /2* PRINT_VAR(ypred); *2/ *1/ */
/*   /1* /2* PRINT_VAR(prob); *2/ *1/ */
/*   /1* /2* PRINT_VAR(model.ComputeAccuracy(X2, y2)); *2/ *1/ */
/*   /1* /2* PRINT_VAR(model.ComputeAccuracy(X, y)); *2/ *1/ */
/*   /1* arma::Mat<DTYPE> encoded; *1/ */
/*   /1* mlpack::data::OneHotEncoding(y,encoded); *1/ */
/*   /1* PRINT_VAR(encoded); *1/ */
/*   /1* PRINT_VAR(prob); *1/ */ 
/*   /1* /2* PRINT_VAR(encoded); *2/ *1/ */ 
/*   /1* PRINT_VAR(-arma::accu(encoded%arma::log(prob))/double(prob.n_cols)); *1/ */ 

/*   PRINT(model.ComputeAccuracy(X, y)); */

/*   /1* data::classification::Dataset trainset(2,4,2); *1/ */
/*   /1* data::classification::Dataset testset(2,3,2); *1/ */
/*   /1* trainset.Generate("Simple"); *1/ */
/*   /1* testset.Generate("Simple"); *1/ */
  

/*   /1* algo::classification::SVM<mlpack::GaussianKernel> *1/ */ 
/*   /1*   svm2(trainset.inputs_,trainset.labels_,2,C); *1/ */
/*   /1* PRINT(svm2.ComputeAccuracy(testset.inputs_, testset.labels_)); *1/ */

/*   /1* arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,2); *1/ */
/*   /1* src::LCurve<algo::classification::LDC<>, *1/ */
/*   /1*             mlpack::Accuracy> *1/ */ 
/*   /1*   lcurve(Ns,100,true,false,true); *1/ */
/*   /1* src::LCurve<algo::classification::KernelSVM<mlpack::GaussianKernel>, *1/ */
/*   /1* /2* src::LCurve<algo::classification::KernelSVM<mlpack::LinearKernel>, *2/ *1/ */
/*   /1*             mlpack::Accuracy, *1/ */
/*   /1*             data::N_StratSplit> *1/ */ 
/*   /1*   lcurve(Ns,100,true,false,true); *1/ */

/*   /1* /2* lcurve.Bootstrap(trainset.inputs_,trainset.labels_,1.,1.); *2/ *1/ */
/*   /1* lcurve.Split(trainset,testset,2); *1/ */
/*   /1* lcurve.test_errors_.save("ldc.csv",arma::csv_ascii); *1/ */



/*   return 0; */
/* } */

using LSVC  = algo::classification::SVM<mlpack::LinearKernel>; 
using GSVC  = algo::classification::SVM<mlpack::GaussianKernel>; 
using ESVC  = algo::classification::SVM<mlpack::EpanechnikovKernel>; 
using LDC   = algo::classification::LDC<>; 
using Dataset = data::classification::oml::Dataset<>;
using Metric = mlpack::Accuracy;

int main()
{
  arma::irowvec ids = {11};
  Dataset dataset(ids[0]);
  Metric metric;
  for (int i=0; i<1000; i++)
  {
    arma::uvec idx = arma::randi<arma::uvec>(60,arma::distr_param(0,627));
    arma::Mat<DTYPE> inp = dataset.inputs_.cols(idx);
    arma::Row<size_t> lab = dataset.labels_.cols(idx);
    GSVC gsvc(inp,lab,2,DTYPE(1.),DTYPE(0.01));
    PRINT(metric.Evaluate(gsvc,dataset.inputs_, dataset.labels_));
  }
  /* LSVC lsvc(dataset.inputs_,dataset.labels_,dataset.num_class_,DTYPE(1.)); */
  /* PRINT(metric.Evaluate(lsvc,dataset.inputs_, dataset.labels_)); */
  return 0;
}
