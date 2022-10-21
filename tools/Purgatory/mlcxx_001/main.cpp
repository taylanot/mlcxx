/**
 * @file main.cpp
 * @author Ozgur Taylan Turan
 *
 * TODO: Combine taking input from a file with data specification 
 *
 *
 */

#include <jem/base/String.h>
#include <jem/base/System.h>
#include <jem/util/Timer.h>
#include <jem/util/Properties.h>
//#include <torch/torch.h>
//#include <ATen/ATen.h>
#include <iostream>
#include <typeinfo>
#include <map>
#include <string>
#include <tuple>
#include <math.h>
//#include <armadillo>
//#include <mlpack/core/kernels/gaussian_kernel.hpp>

#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/core/cv/simple_cv.hpp>
#include <mlpack/core/cv/k_fold_cv.hpp>
#include <mlpack/core/cv/k_fold_cv.hpp>
#include <mlpack/core/cv/metrics/mse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "utils/convert.h"
#include "utils/datagen.h"
#include "utils/covmat.h"
#include "mlpack_models/kernelridge.h"

//#include "mlpack_models/trial.h"
//using namespace arma;
//using namespace mlpack;
//using namespace mlpack::regression;

//arma::Mat<float> toMatrix ( at::Tensor data )
//{
//  float *dataptr = data.data_ptr<float>();
//  arma::Mat<float> data_converted(dataptr,
//                                  data.sizes()[1],
//                                  data.sizes()[0],false,false);
//  return arma::trans(data_converted);
//}
// static std::map<at::kInt, int> dict;
//
//template <class T>
//class Data {
//   private:
//    // Variable of type T
//    T data;
//
//   public:
//    Data(T n) : data(n) {}   // constructor
//
//    T getData() {
//        return data;
//    }
//};

//torch::Tensor distance(torch::Tensor a, torch::Tensor b)
//{
//  torch::Tensor dist = (a * a).sum(1).reshape({a.sizes()[0],1}) + (b * b).sum(1) - 2.0 * torch::matmul(a, b.transpose(0,1));
//  return torch::nn::functional::relu(dist);
//}
//
//torch::Tensor rbf(torch::Tensor dist, float l=0.1)
//{
//  return torch::exp(-0.5 * dist/(l*l));
//} 

//mat LinearRegression(mat& X, rowvec& y)
//{
// const size_t nCols = X.n_cols;
//
//  arma::mat p = X;
//  arma::rowvec r = y;
//
//  // Here we add the row of ones to the predictors.
//  // The intercept is not penalized. Add an "all ones" row to design and set
//  // intercept = false to get a penalized intercept.
//  p.insert_rows(0, arma::ones<arma::mat>(1, nCols));
//
//  // Convert to this form:
//  // a * (X X^T) = y X^T.
//  // Then we'll use Armadillo to solve it.
//  // The total runtime of this should be O(d^2 N) + O(d^3) + O(dN).
//  // (assuming the SVD is used to solve it)
//  double lambda = 0.;
//  arma::mat cov = p * p.t() +
//      lambda * arma::eye<arma::mat>(p.n_rows, p.n_rows);
//  
// return arma::solve(cov, p * r.t());
//} 

//double ComputeError(mat& X, rowvec& y)
//{
//  const size_t nCols = X.n_cols;
//  const size_t nRows = X.n_rows;
//
//  // Calculate the differences between actual responses and predicted responses.
//  // We must also add the intercept (parameters(0)) to the predictions.
//  arma::rowvec temp;
//  arma::vec parameters = LinearRegression(X, y);
//  temp = y - (parameters(0) +
//        arma::trans(parameters.subvec(1, parameters.n_elem - 1)) * X);
//
//  const double cost = arma::dot(temp, temp) / nCols;
//
//  return cost;
//}
//
//

//template<class T, typename... Types>
// K(arma::mat X1, arma::mat X2, Types... var2)
//{
//  arma::mat cov(X1.n_rows, X2.n_rows);
//  T kernel(var2...);
//  for (int i = 0; i < int(X1.n_rows); i++)
//  {
//    for (int j = 0; j < int(X2.n_rows); j++)
//      {
//        cov(i,j) = kernel.Evaluate(X1.row(i),X2.row(j));
//      }
//  }
//  return cov;
//}

template<class T>
struct foo
{
  template<class... Args>
    foo(arma::mat X, Args&&... args)
    {
      T k(args...);
      std::cout << k.Evaluate(X.row(0), X.row(1)) << std::endl;
    }
};



int run ()
{

  

  




  //torch::Tensor a=torch::linspace(-1,1,20).reshape({10,2});
  ////torch::Tensor dist = distance(a,a);
  ////std::cout << rbf(dist,1) << std::endl;
  ////std::cout << rbf(distance(a,a),1) << std::endl;
  //for(int e=0;e<10000;e++)
  //{
  //for(int i=0;i<10000;i++)
  //{
  //  torch::Tensor result = rbf(distance(a,a),1);
  //}
  //}
  //torch::Tensor b = torch::mul(a,a);
  //std::cout << b  << std::endl;
  //arma::rowvec  datay = arma::linspace<arma::rowvec>(1,2,5);
  //arma::mat     datax = arma::linspace<arma::rowvec>(1,2,5);
  //std::cout << datay << std::endl;
  //datay = datax;
  //std::cout << datay << std::endl;


  mlpack::kernel::GaussianKernel kernel(0.1);
  //double l = 0.1;
  arma::mat X1(2,2,arma::fill::eye);
  std::cout << utils::covar(kernel, X1) << std::endl;
  //utils::covmat<mlpack::kernel::GaussianKernel> cov(X1,100);
  //std::cout << cov.matrix <<  std::endl;

  //utils::covmat<mlpack::kernel::GaussianKernel> cov(X1);
  //std::cout << cov.matrix << std::endl;
  //utils::covmat<mlpack::kernel::GaussianKernel> cov(int 1);
  //covmat<mlpack::kernel::GaussianKernel> cov;
  //
  //mlpack::regression::KernelRidge<mlpack::kernel::GaussianKernel> model;
  //model.Train(datax.t(),datay, arma::rowvec(),  false);
  //arma::rowvec predictions;
  //model.Predict(datax.t(),predictions);
  //std::cout << model.ComputeError(datax.t(),predictions) << std::endl; 
  // 
  //utils::covmat<mlpack::kernel::GaussianKernel> cov(X1);
  //std::cout << cov.matrix << std::endl;
  //
    //  std::cout << cov << std::endl;
  //  //std::cout << arma::norm(arma::dot((X-X),(X-X).t()),2)/(0.1*0.1*2) << std::endl;
  //  std::cout << arma::dot((X-X),(X-X).t())/(0.1*0.1*2) << std::endl;

  //int D, N; D=2; N=100;
  //arma::mat covmat   = arma::eye(D,D)*0.1;
  //arma::vec meanvec  = arma::vec(D,arma::fill::ones)*0.;
  //arma::mat noise = arma::mvnrnd(meanvec,covmat,N);
  //std::cout << noise << std::endl;
  
  //output = output + noise;
  //arma::mat input = arma::mat(1,2, arma::fill::randu);
  //arma::rowvec output = arma::rowvec(2,arma::fill::randu);
  //output(0) = input(0,0);
  //output(1) = input(0,1);
  ////std::cout << ComputeError(input, output) << std::endl;
  //arma::rowvec weights = arma::rowvec(2,arma::fill::ones);
  ////std::cout << LinearRegression(input, output) << std::endl;
  //mlpack::regression::LinearRegression model;
  //std::cout << model.Train(input, output) << std::endl;
  ////std::cout << ComputeError(input, output) << std::endl;
  //arma::arma_version ver;
  //std::cout << "ARMA version: "<< ver.as_string() << std::endl;
  //arma::mat parameters = model.Parameters();

  //std::cout << parameters << std::endl;

  //jem::String ali = "Taylan";
  //const char* aliptr = ali.addr();
  //std::cout << std::string(aliptr) << std::endl;
//  size_t N = 10000;
//  double slope, noise_std; slope = 1; noise_std = 0.;
//  datagen::regression::linear::dataset train(N, slope, noise_std);

//    int D, N; D=2; N=100;
//    utils::datagen::regression::nonlinear::dataset train(N,1.,0.,0.);
////    std::cout << train.inputs << std::endl;
////    train.inputs.save("input.csv",arma::file_type::csvascii);
////    train.labels.save("label.csv",arma::file_type::csv_ascii);
//
//  arma::mat Xtrn = train.inputs.t();
//  arma::rowvec ytrn = arma::vectorise(train.labels,1);
//  for(size_t i=0; i<100; i++) {
//  //mlpack::regression::KernelRidge<mlpack::kernel::GaussianKernel> model(Xtrn,ytrn,0.0,false,0.);
//  mlpack::regression::KernelRidge<mlpack::kernel::GaussianKernel> model(1.0, 2.0);
  //mlpack::regression::KernelRidge<mlpack::kernel::GaussianKernel> model;
  //mlpack::regression::KernelRidge<mlpack::kernel::GaussianKernel> model;
  //model.Train(Xtrn,ytrn,arma::rowvec(), false);
  //std::cout << model.ComputeError(Xtrn, ytrn) << std::endl;
//  //mlpack::regression::LinearRegression model(Xtrn, ytrn, 0.);

//  //arma::rowvec y_pred = arma::rowvec(N);
//  //model.Predict(Xtrn, y_pred);
//  //std::cout << model.ComputeError(Xtrn, ytrn) << std::endl;
//  //std::cout << y_pred << std::endl;
//
//  mlpack::cv::KFoldCV<mlpack::regression::LinearRegression, mlpack::cv::MSE> 
//    cv(100, Xtrn, ytrn);
//
  //mlpack::cv::KFoldCV<mlpack::regression::LinearRegression, mlpack::cv::MSE> cv;
  //mlpack::regression::LinearRegression2 myObj;
  //for(int i=0; i < 100; i++)
  //{
  //  for(int j=0; j < 1; j++)
  //  {
  //  double lrlambda = 0.5;
  //  arma::vec lrMSE = cv.Evaluate_mod(lrlambda);
  //  //std::cout << "Mean      : " << arma::mean(lrMSE)    << std::endl;
  //  //std::cout << "Variance  : " << arma::stddev(lrMSE)  << std::endl;
  //  }
  //  //kasdjf
  //}

  //for(int i=0; i < 100; i++)
  //{
  //  double lrlambda = 0.5;
  //  double lrMSE = cv.Evaluate(lrlambda);
  //  std::cout << lrMSE << std::endl;
  //} 
  //double lrMSE = cv.Evaluate(lrlambda);
  //std::cout << lrMSE << std::endl;
  //std::cout << arma::mean(train.input,1) << std::endl;
  //std::cout << arma::mean(test.input,1) << std::endl;
  //std::cout << arma::var(arma::vectorise(train.label)) << std::endl;
  //std::cout << arma::var(arma::vectorise(test.label)) << std::endl;
  //datagen::regression::linear::dataset test;
  //std::cout << test.input << std::endl;
  return 0;
}

int main ()
{
  jem::util::Timer timer;
  timer.start();
  jem::System::exec ( run );
  std::cout << "CPU Time Spent : " << timer.toDouble() << std::endl;
}
