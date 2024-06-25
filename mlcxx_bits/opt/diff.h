/**
 * @file differentiation.h
 * @author Ozgur Taylan Turan
 * 
 * Differentiation Module
 */
#ifndef FINITEDIFF_H 
#define FINITEDIFF_H

namespace opt {

template<class FUNC, class T=arma::Col<DTYPE>>
arma::Mat<DTYPE> fdiff(FUNC& f, const T& x,
                const std::string& type="central",
                const double& h = 1e-5, const size_t order = 1 ) 
{
    BOOST_ASSERT_MSG ( (type == "central" || type == "forward" ||
          type == "backward"), "Not a valid type for central difference!");

    BOOST_ASSERT_MSG( order == 1 || order == 2, "Check the order!");

    size_t n = x.n_elem;

    // Create matrices for forward and backward perturbed points
    arma::Mat<DTYPE> E = arma::eye<arma::Mat<DTYPE>>(n, n);
    arma::Mat<DTYPE> xcp = arma::repmat(x, 1, n);
    arma::Mat<DTYPE> x_ = xcp + h * E;
    arma::Mat<DTYPE> x__ = xcp + 2 * h * E;
    arma::Mat<DTYPE> _x = xcp - h * E;
    arma::Mat<DTYPE> __x = xcp - 2 * h * E;

    // Compute the gradient using central difference formula
    arma::Mat<DTYPE> grad;
    if (type == "central")
    {
      if (order == 1)
        grad = (f.Evaluate(x_) - f.Evaluate(_x)) / (2 * h);
      else if (order == 2)
        grad = (f.Evaluate(x_) - 2*f.Evaluate(_x) + f.Evaluate(_x)) / h;
    }
    else if (type == "forward")
    {
      if (order == 1)
        grad = (f.Evaluate(x_) - f.Evaluate(xcp)) / h;
      else if (order == 2)
        grad = (f.Evaluate(xcp) - 2*f.Evaluate(x_) + f.Evaluate(x__)) / h;
    }
    else
    {
      if (order == 1)
        grad = (f.Evaluate(xcp) - f.Evaluate(_x)) / h;
      else if (order == 2)
        grad = (1.5*f.Evaluate(xcp) - 2*f.Evaluate(_x) - 0.5*f.Evaluate(__x)) 
                                                                            / h;
    }
    return grad;
}

} // namespace opt
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
