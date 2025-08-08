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
#include "gram.h"

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

//-----------------------------------------------------------------------
// Load
//-----------------------------------------------------------------------
template<class T, class O=DTYPE>
arma::Mat<O> Load ( const T& filename,
                    const bool& transpose,
                    const bool& count = false )
{
  arma::Mat<O> matrix;
  mlpack::data::DatasetInfo info;
  if ( count )
  {
    mlpack::data::Load(filename,matrix,info,true,transpose);
  }
  else
    mlpack::data::Load(filename,matrix,true,transpose);
  return matrix;
}
//-----------------------------------------------------------------------------
// Save : Save a data container to a file
//-----------------------------------------------------------------------------
template<class T>
void Save ( const std::filesystem::path& filename,
            const T& data,
            const bool transpose=true )
{
  T temp;

  if (transpose)
    temp = data.t();
  else
    temp = data;

  std::string ext = filename.extension();

  std::filesystem::create_directories(filename.parent_path());

  if (ext == "csv")
  {
    temp.save(filename,arma::csv_ascii);
  }
  else if (ext == "bin")
  {
    temp.save(filename,arma::arma_binary);
  }
  else if (ext == "arma")
  {
    temp.save(filename,arma::arma_ascii);
  }
  else if (ext == "txt")
  {
    temp.save(filename,arma::raw_ascii);
  }
  else
    throw std::runtime_error("Not Implemented save extension!");

}

};
#endif
