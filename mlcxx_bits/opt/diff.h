/**
 * @file differentiation.h
 * @author Ozgur Taylan Turan
 * 
 * Differentiation Module
 */
#ifndef FINITEDIFF_H 
#define FINITEDIFF_H

namespace diff {

//-----------------------------------------------------------------------------
//    FD_dfdp 
//    * Finite difference for df(x,p)/dp
//    INPUTS:
//    ------
//    x   : inputs 
//    y   : outputs -> f(x,p)
//    p   : parameters 
//    dp  : perterbation
//-----------------------------------------------------------------------------
template<class Functor>
arma::mat FD_dfdp ( Functor f,  const arma::mat& x, const arma::rowvec& y,
                    const arma::rowvec& p, const double& dp )
{
  size_t n, m; n = x.n_cols;  m = p.n_cols;

  arma::mat J(m,n);

  arma::rowvec del_(m);
  arma::rowvec yp(n);
  arma::rowvec yn(n);

  arma::rowvec pcc = p;

  for ( size_t i=0; i < m; i++ )
  {
    del_(i) = dp * (1+std::abs(p(i)));

    pcc(i) = p(i) + del_(i);
    if ( del_(i) != 0. )
    {
      yp = f(x,pcc);
      if ( dp < 0 )
        // Backwards diff
        J.row(i) = (yp - y) / del_(i);
      else
      {
        // Central diff
        pcc(i) = p(i) - del_(i);
        yn = f(x, pcc);
        J.row(i) = (yp - yn) / (2 * del_(i));
      }
    }
    pcc(i) = p(i);
  }
  return J;
}

//-----------------------------------------------------------------------------
//    J_update
//    * Broyden Rank-1 Update for the Jacobian 
//    * (Only if you don't have the exact Jacobian)
//    INPUTS:
//    ------
//    J       : current Jacobian 
//    p_old   : old_parameters 
//    p       : parameters 
//    y_old   : old_outputs -> f(x,p)
//    y       : outputs -> f(x,p)
//-----------------------------------------------------------------------------
arma::mat J_update ( const arma::mat J, const arma::rowvec& p_old,
                     const arma::rowvec& p, const arma::rowvec& y_old,
                     const arma::rowvec& y )
{
  arma::mat J_new;
  arma::rowvec dp = p_old - p;
  arma::rowvec dy = y_old - y;
  
  arma::mat nom = dp.t()*( dy - dp*J );
  arma::mat denom = dp * dp.t();
  J_new = J + ( nom / denom(0,0) );
  return J_new;
}


} // namspace diff

#endif
