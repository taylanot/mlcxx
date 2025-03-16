/*
 * @file xval.cpp
 * @author Ozgur Taylan Turan
 *
 * Let's see what is going wrong with some classifier xvals
 *
 */
#define DTYPE double

#include <headers.h>


/* using MODEL = mlpack::AdaBoost<>; */ 
/* using MODEL = mlpack::RandomForest<>; */ 
/* using MODEL = mlpack::DecisionTree<>; */ 
using MODEL = mlpack::DecisionTree<>; 
using METRIC = mlpack::Accuracy;
using OpenML = data::oml::Dataset<size_t>;

int main ( int argc, char** argv )
{

  arma::wall_clock timer;
  timer.tic();
  OpenML dataset(11);

  mlpack::KFoldCV<MODEL,METRIC,arma::Mat<DTYPE>,arma::Row<size_t>> 
                                    xv(5,dataset.inputs_,dataset.labels_,dataset.num_class_);
  data::report(dataset);
  xv.Evaluate(8);

  PRINT_TIME(timer.toc());

  
  return 0;
}
