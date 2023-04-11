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

  Dataset ( );
  Dataset ( size_t D, size_t N, double noise_std );

  void Generate ( double scale, double phase, std::string type );
  void Generate ( size_t M, std::string type );

  void Save( std::string filename );
};

struct SineGen
{
  size_t size;
  size_t dimension;

  arma::rowvec a;
  arma::rowvec p;

  SineGen ( );
  SineGen ( size_t M );

  arma::mat Predict ( arma::mat inputs, std::string type ) const;

};

} // regression 
} // namespace data
} // namespace utils


#include "datagen_impl.h"

#endif
