/**
 * @file datagen.cpp
 * @author Ozgur Taylan Turan
 *
 * A simple toy data generation interface
 *
 * TODO: Increase the dimensionality, add some classification datasets banana??
 *
 *
 */
// standard
#include <tuple>
#include <string>
// armadillo
#include <armadillo>
// local
#include "datagen.h"

namespace utils {
namespace datagen {
namespace regression {
namespace linear {

  dataset::dataset(size_t N, double slope, double noise_std)
  {
    size = N;
    dimension = 1;
    inputs = arma::randn(dimension,size);
    labels = inputs*slope + arma::randn(dimension,N)*noise_std;
  }

  dataset::dataset(size_t N, double slope): 
    dataset::dataset(N, slope, double(1.)) { }

  dataset::dataset(size_t N): 
    dataset::dataset(N, double(1.), double(1.)) { }
      } // namespace linear
    } // namespace regression
  } // namespace datagen
} // namespace utils

namespace utils {
namespace datagen {
namespace regression {
namespace nonlinear {

  dataset::dataset(size_t N, double amplitude, double phase,  double noise_std)
  {
    size = N;
    dimension = 1;
    inputs = arma::randn(dimension,size);
    labels = amplitude * arma::sin(inputs + phase)
                                        +arma::randn(dimension,N)*noise_std;
  }

  dataset::dataset(size_t N, double amplitude, double phase): 
    dataset::dataset(N, amplitude, phase, double(1.)) { }

  dataset::dataset(size_t N): 
    dataset::dataset(N, double(1.), double(0.)) { }
      } // namespace linear
    } // namespace regression
  } // namespace datagen
}
