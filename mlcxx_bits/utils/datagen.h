/**
 * @file datagen.h
 * @author Ozgur Taylan Turan
 *
 * A simple toy data generation interface
 *
 *
 */

#ifndef DATAGEN_H
#define DATAGEN_H

namespace utils {
namespace data {
namespace regression {

//=============================================================================
// Dataset
//=============================================================================

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

  void Generate ( const std::string& type,
                  const double& noise_std );

  void Generate ( const std::string& type );

  void Noise ( const double& noise_std );

  void Save ( const std::string& filename );

  void Load ( const std::string& filename,
              const size_t& Din,
              const size_t& Dout,
              const bool& transpose,
              const bool& count );

  void Load ( const std::string& filename,
              const arma::uvec& ins,
              const arma::uvec& outs, 
              const bool& transpose,
              const bool& count );

};

} // regression 
  
namespace functional
{

struct Dataset 
{
  size_t size_;
  size_t dimension_;
  size_t nfuncs_;
  arma::mat weights_;

  arma::mat inputs_;
  arma::mat labels_;

  Dataset ( );
  Dataset ( const size_t& D,
            const size_t& N,
            const size_t& M );

  void Generate ( const std::string& type );

  void Generate ( const std::string& type,
                  const double& noise_std );

  void Generate ( const std::string& type,
                  const arma::rowvec& noise_std );

  void Noise ( const double& noise_std );

  void Noise ( const arma::rowvec& noise_std );

  void Save ( const std::string& filename );

  void Load ( const std::string& filename,
              const size_t& Din,
              const size_t& Dout, 
              const bool& transpose,
              const bool& count );

  void Normalize ( );

  void UnNormalize ( );

};

//=============================================================================
// SineGen
//=============================================================================

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

  arma::mat Mean ( const arma::mat& inputs,
                   const std::string& type ) const;

  arma::mat Mean ( const arma::mat& inputs,
                   const std::string& type,
                   const double& eps ) const;

  size_t GetM ( );

};

} //functional
namespace classification{

//=============================================================================
// Dataset
//=============================================================================

struct Dataset 
{
  size_t size_;
  size_t dimension_;
  size_t num_class_;

  arma::mat inputs_;
  arma::Row<size_t> labels_;

  Dataset ( );
  Dataset ( const size_t& D,
            const size_t& N,
            const size_t& Nc );

  void Set ( const size_t& D,
             const size_t& N,
             const size_t& Nc );

  void Generate ( const std::string& type );

  void _banana ( const double& delta );

  void _dipping ( const double& r,
                  const double& noise_std );

  void _2classgauss ( const arma::vec& mean1,
                      const arma::vec& mean2,
                      const double& eps,
                      const double& delta );

  void _imbalance2classgauss ( const double& perc );

  void Save ( const std::string& filename );

  void Load ( const std::string& filename,
              const bool& transpose,
              const bool& count  );

};


} // namespace classification

} // namespace data
} // namespace utils


#include "datagen_impl.h"

#endif
