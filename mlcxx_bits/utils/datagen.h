/**
 * @file datagen.h
 * @author Ozgur Taylan Turan
 *
 * A simple toy data generation interface
 * Code a Simple DataLoader Class 
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
template<class T=DTYPE>
struct Dataset 
{
  size_t size_;
  size_t dimension_;

  arma::Mat<T> inputs_;
  arma::Mat<T> labels_;

  Dataset ( );
  Dataset ( const size_t& D,
            const size_t& N );

  void Set ( const size_t& D,
             const size_t& N );

  void Outlier_ ( const size_t& Nout );

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
              const bool transpose = true,
              const bool count = false );

  void Load ( const std::string& filename,
              const arma::uvec& ins,
              const arma::uvec& outs, 
              const bool transpose = true,
              const bool count = false );

  arma::Row<T> GetLabels( const size_t id );

};

} // regression 
  
namespace functional
{

template<class T=DTYPE>
struct Dataset 
{
  size_t size_;
  size_t dimension_;
  size_t nfuncs_;
  arma::Mat<T> weights_;

  arma::Mat<T> inputs_;
  arma::Mat<T> labels_;

  Dataset ( );
  Dataset ( const size_t& D,
            const size_t& N,
            const size_t& M );

  void Generate ( const std::string& type );

  void Generate ( const std::string& type,
                  const double& noise_std );

  void Generate ( const std::string& type,
                  const arma::Row<T>& noise_std );

  void Noise ( const double& noise_std );

  void Noise ( const arma::Row<T>& noise_std );

  void Save ( const std::string& filename );

  void Load ( const std::string& filename,
              const size_t& Din,
              const size_t& Dout, 
              const bool& transpose = true,
              const bool& count = false );

  void Normalize ( );

  void UnNormalize ( );

};

//=============================================================================
// SineGen
//=============================================================================
template<class T=DTYPE>
struct SineGen
{
  size_t size_;
  size_t dimension_;

  arma::Row<T> a_;
  arma::Row<T> p_;

  SineGen ( );
  SineGen ( const size_t& M );

  arma::Mat<T> Predict ( const arma::Mat<T>& inputs,
                         const std::string& type  = "Phase" ) const;

  arma::Mat<T> Predict ( const arma::Mat<T>& inputs,
                         const std::string& type,
                         const double& eps ) const;

  arma::Mat<T> Mean ( const arma::Mat<T>& inputs,
                      const std::string& type = "Phase" ) const;

  arma::Mat<T> Mean ( const arma::Mat<T>& inputs,
                      const std::string& type,
                      const double& eps ) const;

  size_t GetM ( );

};

} //functional
namespace classification{

//=============================================================================
// Dataset
//=============================================================================

template<class T=DTYPE>
struct Dataset 
{
  size_t size_;
  size_t dimension_;
  size_t num_class_;

  arma::Mat<T> inputs_;
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

  void _2classgauss ( const arma::Col<T>& mean1,
                      const arma::Col<T>& mean2,
                      const double& eps,
                      const double& delta );

  void _imbalance2classgauss ( const double& perc );

  void Save ( const std::string& filename );

  void Load ( const std::string& filename,
              const bool& transpose = true ,
              const bool& count = false );

};


} // namespace classification

} // namespace data
} // namespace utils


#include "datagen_impl.h"

#endif
