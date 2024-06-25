/**
 * @file root.h
 * @author Ozgur Taylan Turan
 *
 * Root finding related stuff
 *
 * TODO: 
 *
 *
 */

#ifndef ROOT_H
#define ROOT_H

namespace opt {


template<class OBJ,class T=arma::Col<DTYPE>>
void broyden(OBJ& obj, T& mu, const double tol = 1e-6,
                              const size_t max_iter = 100)
{
  arma::Mat<DTYPE> J;
  obj.Gradient(mu, J);

  arma::Mat<DTYPE> jitter = arma::eye<arma::Mat<DTYPE>>(arma::size(J));
  jitter = jitter * 1e-6;

  arma::Col<DTYPE> fx = obj.Evaluate(mu);
  size_t iter = 0;

  arma::Col<DTYPE> dx;
  while ( iter < max_iter )
  {
    try
    {
      dx = arma::solve(J+jitter, -fx);
    }
    catch (...) 
    {
      PRINT("Trying pinv in broyden!");
      dx = arma::pinv(J)*-fx;
    }

    // Update the solution
    mu += dx;
    // Update the solution 
    arma::Col<DTYPE> df = obj.Evaluate(mu) - fx;
    fx += df;
    J = J + ((df - J * dx) * dx.t()) / arma::dot(dx, dx);
    if (arma::norm(fx, "inf") < tol)
      break;
    iter++;
  }
  BOOST_ASSERT_MSG( iter < max_iter, "Solution not found!");
  
}

template<class OBJ,class T=arma::Col<DTYPE>, class... Ts>
void fsolve(OBJ& obj, T& mu, std::string type="broyden", const Ts&... args)
{
  BOOST_ASSERT_MSG( type == "broyden",
      "Only available option is broyden for now..."); 
  broyden(obj, mu, args...);
}

} // namespace utils
#endif
