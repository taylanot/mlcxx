/**
 * @file mlcxx.cpp
 * @author Ozgur Taylan Turan
 *
 * Main file of mlcxx where you do not have to do anything...
 */

//#define ARMA_DONT_USE_LAPACK
//#define ARMA_DONT_USE_BLAS
//#define ARMA_DONT_USE_ARPACK
//#define ARMA_DONT_USE_OPENMP
#include <headers.h>
/* template<class OBJ, class T=arma::vec> */
/* void fsolve(OBJ& obj, T& mu, double tol = 1e-8, size_t max_iter = 100) */
/* { */
/*   arma::mat J; */
/*   obj.Gradient(mu, J); */
/*   arma::vec fx = obj.Evaluate(mu); */
/*   size_t iter = 0; */
/*   while ( iter < max_iter && arma::norm(fx, "inf") > tol) */
/*   { */
/*     arma::vec dx = arma::solve(J, -fx); */
/*     // Update the solution */
/*     mu += dx; */
/*     // Update the solution */ 
/*     arma::vec df = obj.Evaluate(mu) - fx; */
/*     fx += df; */
/*     J = J + ((df - J * dx) * dx.t()) / arma::dot(dx, dx); */
/*     iter++; */
/*   } */
/* } */

/* template<class FUNC, class T=arma::vec> */
/* arma::mat fdiff(FUNC& f, const T& x, */
/*                 std::string type="central", double h = 1e-5) */ 
/* { */
/*     BOOST_ASSERT_MSG ( (type == "central" || type == "forward" || */
/*           type == "backward"), "Not a valid type for central difference!"); */

/*     size_t n = x.n_elem; */

/*     // Create matrices for forward and backward perturbed points */
/*     arma::mat E = arma::eye(n, n); */
/*     arma::mat xcp = arma::repmat(x, 1, n); */
/*     arma::mat x_ = xcp + h * E; */
/*     arma::mat _x = xcp - h * E; */

/*     // Compute the gradient using central difference formula */
/*     arma::mat grad; */
/*     if (type == "central") */
/*       grad = (f.Evaluate(x_) - f.Evaluate(_x)) / (2 * h); */
/*     else if (type == "forward") */
/*       grad = (f.Evaluate(x_) - f.Evaluate(xcp)) / h; */
/*     else */
/*       grad = (f.Evaluate(xcp) - f.Evaluate(_x)) / h; */
/*     return grad; */
/* } */

/* //============================================================================= */
/* // X^2: Just a function */
/* //============================================================================= */
class X2
{
  public:

  X2 ( ) { } 

  
  arma::vec Evaluate (const arma::vec& mu)
  { 
    return arma::pow(mu,2);
  }

  void Gradient ( const arma::vec& mu,
                  arma::mat& gradient )
  {
    gradient = 2*mu;
  }

  void Gradient_ ( const arma::vec& mu,
                   arma::mat& gradient )
  {
    gradient = utils::fdiff(*this, mu);
  }

};




/* //============================================================================= */
/* // EMPRICAL: Emprical Mean Objective */
/* //============================================================================= */
/* class EMPIRICAL */
/* { */
/*   public: */

/*   EMPIRICAL ( ) { } */ 

/*   EMPIRICAL ( const arma::mat& X ) : X_(X) { } */
  
/*   arma::vec Evaluate (const arma::vec& mu) */
/*   { */ 
/*     return arma::sum(X_.each_col()-mu,1); */
/*   } */

/*   void Gradient ( const arma::vec& mu, */
/*                   arma::mat& gradient ) */
/*   { */
/*     gradient = -double(X_.n_cols); */
/*   } */

/*   void Gradient_ ( const arma::vec& mu, */
/*                   arma::mat& gradient ) */
/*   { */
/*     gradient = utils::fdiff(*this, mu); */
/*   } */

/*   private: */
  
/*   arma::mat X_; */

/* }; */



/* vec broyden(vec x0, double tol = 1e-8, int max_iter = 100) { */
/*     int n = x0.n_elem; */
/*     vec x = x0; */
/*     vec fx = f(x); */
/*     mat B = eye(n, n);  // Initial approximation of the Jacobian */

/*     for (int iter = 0; iter < max_iter; ++iter) { */
/*         // Solve B * dx = -fx */
/*         vec dx = solve(B, -fx); */

/*         // Update the solution */
/*         vec x_new = x + dx; */
/*         vec fx_new = f(x_new); */

/*         // Check for convergence */
/*         if (norm(fx_new, "inf") < tol) { */
/*             cout << "Converged in " << iter + 1 << " iterations." << endl; */
/*             return x_new; */
/*         } */

/*         // Broyden's update */
/*         vec delta_x = x_new - x; */
/*         vec delta_f = fx_new - fx; */
/*         B = B + ((delta_f - B * delta_x) * delta_x.t()) / dot(delta_x, delta_x); */

/*         // Update for next iteration */
/*         x = x_new; */
/*         fx = fx_new; */
/*     } */

/*     cout << "Maximum iterations reached without convergence." << endl; */
/*     return x; */
/* } */

/* void fsolve(arma::mat& X, double& mu) */
/* { */
/*   double obj, grad; */
/*   for (int i=0; i<200;i++) */
/*   { */
/*     obj = arma::mean(X-mu,1).eval()(0,0); */
/*     grad = -1/X.n_cols; */
/*     PRINT_VAR(obj) */
/*     PRINT_VAR(mu) */
/*     mu += 0.001; */
/*   } */

/* } */


//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{
  /* X1 obj; */
  /* double x = 10; */
  /* utils::fsolve1(obj,x); */
  /* arma::mat data = arma::randn(1,10); */
  /* PRINT(utils::mediomean(data)); */
  /* X2 obj; */
  /* arma::vec x = arma::ones(1); */
  /* utils::fsolve(obj,x); */
  /* PRINT_VAR(x); */




  /* algo::approx::TAYLOR foo(1, 1e-6, "central"); */
  /* arma::mat x0(1,1); */
  /* arma::mat x = arma::linspace<arma::mat>(-0.1,0.1,50); */
  /* arma::inplace_trans(x); */
  /* X2 func; */
  /* foo.Train(func, x0); */
  /* PRINT(foo.ComputeError(x,arma::conv_to<arma::rowvec>::from(func.Evaluate(x.t())))); */
  /* arma::vec x = arma::linspace(0, 10, 100); */
  /* arma::vec y = arma::sin(x); */
  /* arma::mat xy = arma::join_horiz(x,y); */
  /* xy.save("xy.csv",arma::csv_ascii); */

  /* mlpack::RandomSeed(SEED); */

  /* /1* utils::Sampler<boost::random::beta_distribution<>> dist; *1/ */
  /* /1* arma::mat ali = dist.Sample(5,5); *1/ */
  /* arma::mat ali = utils::dists::pareto(10000); */

  /* arma::wall_clock timer; */
  /* timer.tic(); */

  arma::mat x = arma::randn<arma::mat>(1,5);
  /* arma::vec kappa = utils::mediomean(x); */
  /* double delta = 0.01; */
  /* utils::Lee obj(x, kappa ,delta ); */
  /* arma::vec alpha(1); */
  /* alpha(0)=-1; */
  /* arma::mat J; */
  /* /1* obj.Gradient(alpha,J); *1/ */
  /* /1* PRINT_VAR(J); *1/ */
  /* PRINT_VAR(obj.Evaluate(alpha-10)) */
  /* PRINT_VAR(obj.Evaluate(alpha+10)) */
  
  /* PRINT(arma::mean(x,1)); */
  /* PRINT(utils::emean(x)); */
  /* utils::catoni(x); */
  /* PRINT(utils::tmean(x)); */
  PRINT(utils::lee(x));
  /* /1* ens::L_BFGS opt; *1/ */
  /* /1* EMPIRICAL obj(x); *1/ */
  /* /1* arma::vec mu = arma::ones(1,1); *1/ */
  /* /1* mu *= 10; *1/ */
  /* /1* mu.print("mu"); *1/ */
  /* /1* opt.Optimize(obj,mu); *1/ */
  /* /1* mu.print("mu"); *1/ */

  /* PRINT_SEED(SEED); */

  /* PRINT_GOOD(); */ 

  /* PRINT_TIME(timer.toc()); */
  
  return 0; 
}
