/**
 * @file datagen_impl.h
 * @author Ozgur Taylan Turan
 *
 * A simple toy data generation interface
 *
 * TODO: Increase the dimensionality, add some classification datasets banana??
 * TODO: Maybe add a base class for datagen
 *
 *
 */
#ifndef DATAGEN_IMPL_H
#define DATAGEN_IMPL_H
// standard
#include <tuple>
#include <string>
// armadillo
#include <armadillo>
// local
#include "datagen.h"
#include "save.h"

namespace utils {
namespace data {
namespace regression {

  Dataset::Dataset(size_t D, size_t N, double noise_std)
  {
    this -> size = N;
    this -> dimension = D;
    this -> noise_std = noise_std;
    
  }

  void Dataset::Generate(double scale, double phase, std::string type)
  {
    if (type == "Linear")
    {
      this -> inputs = (arma::randn(dimension,size) + phase);
      this -> labels = (inputs*scale + arma::randn(dimension,size)*noise_std);
    }
    else if (type == "Sine")
    {
      this -> inputs = (arma::randn(dimension,size) + phase);
      this -> labels = (scale * arma::sin(inputs)
                                        +arma::randn(dimension,size)*noise_std);
    }
  }
  void Dataset::Save(std::string filename)
  {
    arma::mat data = arma::join_cols(inputs, labels);
    utils::Save(filename, data, true);
  }

} // namespace regression
} // namespace data
} // namespace utils

#endif
