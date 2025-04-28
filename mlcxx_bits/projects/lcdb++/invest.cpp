/**
 * @file invest.cpp
 * @author Ozgur Taylan Turan
 *
 * Investigate the effect of multi-class and dataset imbalance on the problem
 */

#include <headers.h>

using OpenML = data::classification::oml::Dataset<>;

template<class Dataset,class O=DTYPE>
void report( const Dataset& dataset )
{
    PRINT("### DATASET INFORMATION ###");
    PRINT("features : " << dataset.dimension_ );
    PRINT("size : " << dataset.size_ );

    PRINT("### FEATURE INFORMATION ###");
    PRINT("Mean :  \n" << arma::mean(dataset.inputs_,1) );
    PRINT("Median :  \n" << arma::median(dataset.inputs_,1) );
    PRINT("Variance :  \n" << arma::var(dataset.inputs_.t()) );
    PRINT("Min :  \n" << arma::min(dataset.inputs_,1) );
    PRINT("Max :  \n" << arma::max(dataset.inputs_,1) );
    PRINT("Covariance : \n" << arma::cov(dataset.inputs_.t()) );

    PRINT("### LABEL INFORMATION ###");
    PRINT("Unique :  \n" << arma::unique(dataset.labels_) );
    PRINT("Counts :  \n" << arma::hist(dataset.labels_,arma::unique(dataset.labels_)) );
}

// multiple-cross validation procedure for one given model
int main ( int argc, char** argv )
{
  size_t did = 61;
  OpenML dataset(did);
  report (dataset);
  data::classification::Transformer scale(dataset);
  dataset = scale.Trans(dataset);
  report (dataset);
  /* report (dataset.labels_); */

}


