/**
 * @file data.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef DATA_H
#define DATA_H


#include "dataset.h"
#include "collect.h"
#include "manip.h"
#include "sample.h"
#include "transform.h"

namespace data {

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

};
#endif
