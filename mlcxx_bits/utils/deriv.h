/**
 * @file deriv.h
 * @author Ozgur Taylan Turan
 *
 * Derivative related stuff
 *
 * TODO: 
 *
 *
 */

#ifndef DERIV_H
#define DERIV_H

namespace utils {

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

} // namespace utils
#endif
