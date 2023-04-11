/**
 * @file optim_impl.h
 * @author Ozgur Taylan Turan
 * 
 * Optimization Module Implementation
 */
#ifndef OPTIM_IMPL_H 
#define OPTIM_IMPL_H

#include "optim.h"

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
//
LM::LM ( ) : maxIteration_(5000), iter_(0),  epsilon1_(1e-3), epsilon2_(1e-3),
epsilon4_(1e-2), lambda0_(1e-2), lambda_up_(11),
lambda_down_(9), dX2_(1000.), lambda_update_("LM") { }

LM::LM ( std::string lambda_update ) :
      maxIteration_(5000), iter_(0),  epsilon1_(1e-3), epsilon2_(1e-3),
      epsilon4_(1e-2), lambda0_(1e-2), lambda_up_(11),
      lambda_down_(9), dX2_(1000.), lambda_update_(lambda_update) { }

LM::LM ( size_t maxIteration,
        double epsilon1, double epsilon2, double epsilon4 ,
        double lambda0, double lambda_up, double lambda_down,
        std::string lambda_update) :
maxIteration_(maxIteration), iter_(0), epsilon1_(epsilon1), epsilon2_(epsilon2),
epsilon4_(epsilon4), lambda0_(lambda0), lambda_up_(lambda_up),
lambda_down_(lambda_down), dX2_(1000.), lambda_update_(lambda_update) { }

template<class Functor>
std::tuple<arma::mat, arma::mat, double> 
    LM::_matx ( Functor f, const arma::mat& x, const arma::rowvec& y,
                const arma::rowvec& p, const double dp, const arma::rowvec w )
{
  arma::rowvec y_new = f(x,p);
  arma::mat JWJ, JWdy;

  double error;

  if ((iter_ % (2*N_)) || (dX2_>0))
    J_ = diff::FD_dfdp(f, x, y, p,dp); // finite-difference
  else
    J_ = diff::J_update(J_, p_old_, p, y_old_, y_new); // rank-1 update
  arma::rowvec dy = y-y_new;

  error = arma::dot(dy, (dy%w));
  JWJ = J_ * (J_ % (arma::ones(1,M_).t() * w)).t();
  JWdy = J_ * (  w % dy ).t();

  return {JWJ, JWdy, error};
}

template<class Functor>
void LM::LM::Optimize( Functor f,  const arma::mat& x, const arma::rowvec& y,
                       arma::rowvec& p, const double& dp )
{
  //PRINT_VAR(x);
  //PRINT_VAR(y);
  N_ = x.n_cols; M_ = p.n_cols; dof_ = N_-M_+1;
  double p_min, p_max;
  p_min = -100; p_max = 100; 

  //X2_ = 1e10; double X2_old = 1e10; dX2_ = 1.;

  //arma::uvec idx = arma::regspace<arma::uvec>(0,1,p.n_cols-1); // id
  
  weight_ = 1. / arma::dot(y,y); // calculate the weights from the data

  arma::rowvec w = arma::ones<arma::rowvec>(N_) * weight_; // weight vector

  arma::mat JWJ, JWdy; // relavent quantities

  double X2, X2_old, X2_try; // error  measures

  arma::rowvec y_new = f(x,p); // accepted y values 
  // first get the relevant quantities 
  std::tie(JWJ, JWdy, X2) = this->_matx(f, x, y, p, dp, w); 

  if ( lambda_update_ == "LM")
    lambda_ = lambda0_;
  else
    lambda_ = lambda0_ * arma::max(arma::max(arma::diagmat(JWJ)));

  bool stop = false;  // main loop stoppers 

  arma::vec hvec;        // change in parameters 
  arma::rowvec h, p_try; // trial parameters 

  std::string status; // status of the optimization 

  double rho;         // metric for acceptance
  while ((!stop) && (iter_ <= maxIteration_))
  {
    iter_++; 
    if ( lambda_update_ == "LM")
      hvec = arma::solve((JWJ + lambda_ * arma::diagmat(JWJ)), JWdy);
    else if ( lambda_update_ == "Q" ||  lambda_update_ == "N") 
      hvec = arma::solve((JWJ + lambda_ * arma::eye(arma::size(JWJ))), JWdy);
    else
      throw std::runtime_error("Not an Option!");

    h = arma::conv_to<arma::rowvec>::from(hvec);

    p_try = p + h;
    p_try.clamp(p_min, p_max);

    X2_old = X2;
    arma::rowvec dy = y - f(x,p_try);

    X2_try = arma::dot(dy, (dy%w));

    double alpha = 1e-12;
    double nu = 2.;

    if ( lambda_update_ == "Q" )
    {
      alpha = arma::dot(JWdy,h) / (0.5 * (X2_try-X2) 
                                          + 2.*arma::dot(JWdy,h));
      h = alpha * h;

      p_try = p + h;
      p_try.clamp(p_min, p_max);

      dy = y - f(x,p_try);
      X2_try = arma::dot(dy, (dy%w));
    }

    if (dy.has_nan())
    {
      status = "FAILED!";
      break;
    }

    if ( lambda_update_ == "Q" )
      rho = (X2-X2_try) / arma::dot(h, (lambda_*arma::diagmat(JWJ)*h.t()+JWdy));
    else
      rho = (X2-X2_try) / arma::dot(h, (lambda_*h+JWdy.t()));

    if ( rho > epsilon4_ )
    {
      dX2_ = X2-X2_old;
      X2_old = X2;
      p_old_ = p;
      y_old_ = y_new;
      p = p_try;
      std::tie(JWJ, JWdy, X2) = this->_matx(f, x, y, p, dp, w); 

      if ( lambda_update_ == "LM")
        lambda_ = std::max(lambda_/lambda_down_, 1e-7);
      else if ( lambda_update_ == "Q")
        lambda_ = std::max(lambda_/(1+alpha), 1e-7);
      else if ( lambda_update_ == "N")
        lambda_ = lambda_*std::max(1./3., std::pow(1.-(2.*rho-1.),3));

    }
    else
    {
      X2 = X2_old;
      if ( !iter_%(2*M_) )
        std::tie(JWJ, JWdy, X2) = this->_matx(f, x, p, y, dp, w); 

      if ( lambda_update_ == "LM")
        lambda_ = std::min(lambda_/lambda_up_, 1e7);
      else if ( lambda_update_ == "Q")
        lambda_ += std::abs(0.5*(X2_try-X2)/alpha);
      else if ( lambda_update_ == "N")
      {
        lambda_ *= nu; nu *= 2.;
      }

    }
    // Check convergence
    double cond1 = arma::abs(JWdy).max();
    double cond2 = (arma::abs(h)/arma::abs(p)).max()+1e-12;
    bool check_iter = ( iter_ > 2 );
    if ( (cond1 < epsilon1_) &&  check_iter )
      stop = true;
    else if ( (cond2 < epsilon2_) && check_iter ) 
      stop = true;
  }

}



} // namespace opt
#endif
