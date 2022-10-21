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

namespace utils {
namespace datagen {
namespace regression {
namespace linear {

struct dataset 
{
  size_t size;
  size_t dimension;

  arma::mat inputs;
  arma::mat labels;

  dataset( size_t N );
  dataset( size_t N, double slope );
  dataset( size_t N, double slope, double noise_std );

  dataset( size_t D, size_t N);
  dataset( size_t D, size_t N, double noise_std );

};
      } // namespace linear
    } // namespace regression
  } // namespace datagen
} // namespace utils

namespace utils {
namespace datagen {
namespace regression {
namespace nonlinear{

struct dataset 
{
  size_t size;
  size_t dimension;

  arma::mat inputs;
  arma::mat labels;

  dataset( size_t N );
  dataset( size_t N, double amplitude, double phase);
  dataset( size_t N, double amplitude, double phase, double noise_std );

  dataset( size_t D, size_t N);
  dataset( size_t D, size_t N, double noise_std );

};
      } // namespace linear
    } // namespace regression
  } // namespace datagen
} // namespace utils


#endif
