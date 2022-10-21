/**
 * @file datagen.h
 * @author Ozgur Taylan Turan
 *
 * A simple toy data generation interface
 *
 * TODO: Increase the dimensionality, add some classification datasets banana??
 *
 *
 */

#ifndef DATAGEN_H
#define DATAGEN_H
// standard
#include <tuple>
#include <string>
// armadillo
#include <armadillo>
// locall
#include "save.h"

namespace utils {
namespace data {
namespace regression {
struct Dataset 
{
  size_t size;
  size_t dimension;
  double noise_std;

  arma::mat inputs;
  arma::mat labels;

  Dataset ( size_t D, size_t N, double noise_std );

  void Generate ( double scale, double phase, std::string type );

  void Save( std::string filename );
};
} // regression 
} // namespace data
} // namespace utils


#include "datagen_impl.h"

#endif
