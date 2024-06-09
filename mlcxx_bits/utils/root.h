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

namespace utils {


template<class OBJ, class T=arma::vec>
void broyden(OBJ& obj, T& mu, const double tol = 1e-6,
                              const size_t max_iter = 100)
{
  arma::mat J;
  obj.Gradient(mu, J);
  arma::vec fx = obj.Evaluate(mu);
  size_t iter = 0;
  while ( iter < max_iter )
  {
    arma::vec dx = arma::solve(J, -fx);
    // Update the solution
    mu += dx;
    // Update the solution 
    arma::vec df = obj.Evaluate(mu) - fx;
    fx += df;
    J = J + ((df - J * dx) * dx.t()) / arma::dot(dx, dx);
    if (arma::norm(fx, "inf") < tol)
      break;
    iter++;
  }
  BOOST_ASSERT_MSG( iter < max_iter, "Solution not found!");
  
}

template<class OBJ, class T=arma::vec, class... Ts>
void fsolve(OBJ& obj, T& mu, std::string type="broyden", const Ts&... args)
{
  BOOST_ASSERT_MSG( type == "broyden",
      "Only available option is broyden for now..."); 
  broyden(obj, mu, args...);
}

} // namespace utils
#endif
