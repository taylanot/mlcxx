/**
 * @file optim.h
 * @author Ozgur Taylan Turan
 * 
 * Optimization Module 
 */
#ifndef OPTIM_H 
#define OPTIM_H

namespace opt {

//=============================================================================
//    LM
//    * Levenberg-Marquardt nonlinear least squares
//    INPUTS:
//    ------
//    x   : inputs 
//    y   : outputs -> f(x,p)
//    p   : parameters 
//    dp  : perterbation
//=============================================================================

class LM
{
public:
  LM( );
  LM( std::string lambda_update );
  LM( size_t maxIteration,
      double epsilon1, double epsilon2, double epsilon4,
      double lambda0, double lambda_up, double lambda_down,
      std::string lambda_update ) ;

  template<class Functor>
  void Optimize( Functor f,  const arma::mat& x, const arma::rowvec& y,
            arma::rowvec& p, const double& dp );

  template<class Functor, class Janctor>
  void Optimize( Functor f, Janctor J,  const arma::mat& x, const arma::rowvec& y,
              arma::rowvec& p, const double& dp );

  template<class Functor>
  std::tuple<arma::mat, arma::mat, double> 
    _matx ( Functor f, const arma::mat& x, const arma::rowvec& p,
            const arma::rowvec& y, const double dp, const arma::rowvec w );

private:
  size_t maxIteration_, iter_;
  double epsilon1_, epsilon2_, epsilon4_; 
  double lambda0_, lambda_up_, lambda_down_;

  size_t N_;
  size_t M_;

  double lambda_;
  double weight_;

  arma::rowvec p_old_;
  arma::rowvec y_old_;

  arma::mat J_;

  double dX2_;
  double dof_;

  std::string lambda_update_;
};

} // namespace opt

#include "optim_impl.h"

#endif
