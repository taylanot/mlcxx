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
  size_t size_;
  size_t dimension_;

  arma::mat inputs_;
  arma::mat labels_;

  Dataset ( );
  Dataset ( const size_t& D,
            const size_t& N );

  void Set ( const size_t& D,
             const size_t& N );

  void Generate ( const double& scale,
                  const double& phase,
                  const std::string& type ) ;

  void Generate ( const double& scale,
                  const double& phase,
                  const std::string& type,
                  const double& noise_std );

  void Generate ( const size_t& M,
                  const std::string& type );

  void Generate ( const std::string& type,
                  const double& noise_std );

  void Generate ( const size_t& M,
                  const std::string& type,
                  const arma::rowvec& noise_std );

  void Generate ( const size_t& M,
                  const std::string& type,
                  const double& noise_std );

  void Noise ( const double& noise_std );

  void Noise ( const size_t& M,
               const double& noise_std );

  void Noise ( const size_t& M,
               const arma::rowvec& noise_std );

  void Save ( const std::string& filename );
};

struct SineGen
{
  size_t size_;
  size_t dimension_;

  arma::rowvec a_;
  arma::rowvec p_;

  SineGen ( );
  SineGen ( const size_t& M );

  arma::mat Predict ( const arma::mat& inputs,
                      const std::string& type ) const;

  arma::mat Predict ( const arma::mat& inputs,
                      const std::string& type,
                      const double& eps ) const;

};

} // regression 

namespace classification{

struct Dataset 
{
  size_t size_;
  size_t dimension_;
  size_t num_class_;

  arma::mat inputs_;
  arma::mat labels_;

  Dataset ( );
  Dataset ( const size_t& D,
            const size_t& N,
            const size_t& Nc );

  void Set ( const size_t& D,
             const size_t& N,
             const size_t& Nc );

  void Generate ( const std::string& type );

  void _dipping ( const double& r,
                  const double& noise_std );

  void _2classgauss ( const arma::vec& mean1,
                      const arma::vec& means2,
                      const double& eps);

  template<class... Ts> 
  void _dipping ( const Ts... args );

  void Save ( const std::string& filename );

};


} // namespace classification

} // namespace data

} // namespace utils


#include "datagen_impl.h"

#endif
