/**
 * @file xval.cpp
 * @author Ozgur Taylan Turan
 *
 * Let's check xval results
 */

#include <headers.h>


using LDC   = algo::classification::LDC<>; 
using OpenML = data::classification::oml::Dataset<>;
using Dataset = data::classification::Dataset<>;
using Metric = utils::LogLossRaw;
using Metric2 = utils::LogLoss;
using Loss = mlpack::Accuracy;

/* int main ( int argc, char** argv ) */
/* { */
/*   arma::Mat<DTYPE> X = {-0.9,-1.1,0.9,1.1}; */
/*   arma::Row<size_t> Y = {0,0,1,1}; */

/*   arma::Mat<DTYPE> x = arma::linspace<arma::Mat<DTYPE>>(-10,10,1000).t(); */
/*   /1* arma::Row<size_t> y = {1,0}; *1/ */

/*   /1* Dataset(x,y); *1/ */
/*   LDC model(X,Y,2,0.); */
/*   Metric metric; */
/*   Metric2 metric2; */
/*   arma::Row<size_t> t; */
/*   arma::Mat<DTYPE> p; */
/*   model.Classify(x,t,p); */
/*   p.row(0).save("hey.csv",arma::csv_ascii); */
/*   /1* PRINT_VAR(t); *1/ */
/*   /1* PRINT_VAR(p); *1/ */
/*   /1* PRINT_VAR(metric.Evaluate(model,x,y)); *1/ */
/*   /1* PRINT_VAR(metric2.Evaluate(model,x,y)); *1/ */
/* } */

/* int main ( int argc, char** argv ) */
/* { */
/*   arma::Mat<DTYPE> X = {-0.9,-1.1,0.9,1.1}; */
/*   arma::Row<size_t> Y = {0,0,1,1}; */

/*   arma::Mat<DTYPE> x = arma::linspace<arma::Mat<DTYPE>>(-10,10,1000).t(); */
/*   /1* arma::Row<size_t> y = {1,0}; *1/ */

/*   /1* Dataset(x,y); *1/ */
/*   LDC model(X,Y,2,0.); */
/*   Metric metric; */
/*   Metric2 metric2; */
/*   arma::Row<size_t> t; */
/*   arma::Mat<DTYPE> p; */
/*   model.Classify(x,t,p); */
/*   p.row(0).save("hey.csv",arma::csv_ascii); */
/*   /1* PRINT_VAR(t); *1/ */
/*   /1* PRINT_VAR(p); *1/ */
/*   /1* PRINT_VAR(metric.Evaluate(model,x,y)); *1/ */
/*   /1* PRINT_VAR(metric2.Evaluate(model,x,y)); *1/ */
/* } */

int main ( int argc, char** argv )
{
  
  Dataset dataset(2,10000,2);
  dataset.Generate("Simple");
  PRINT_VAR(arma::size(dataset.labels_));
  PRINT_VAR(arma::size(dataset.inputs_));

  arma::wall_clock timer;
  timer.tic();
  arma::irowvec Ns = arma::regspace<arma::irowvec>(1000,1,2000);
  #pragma omp parallel for
  for (size_t i = 0; i<Ns.n_elem;i++)
  {
    xval::KFoldCV<LDC, mlpack::Accuracy> xv(100, dataset.inputs_.cols(0,Ns[i]), dataset.labels_.cols(0,Ns[i]),true);
    xv.Evaluate(2);
  }
  
  PRINT_TIME(timer.toc());

  /* PRINT_VAR(t); */
  /* PRINT_VAR(p); */
  /* PRINT_VAR(metric.Evaluate(model,x,y)); */
  /* PRINT_VAR(metric2.Evaluate(model,x,y)); */
}
