/**
 * @file basis_.cpp
 * @author Ozgur Taylan Turan
 *
 * Basis Pursuit algorithm
 */

#include <headers.h>

using namespace arma;
using namespace ens;


/* template<class T=double> */
/* class LinearProgFunc */
/* { */
/*  public: */
/*   LinearProgFunc( arma::Col<T> c, */
/*                   arma::Mat<T> G, */
/*                   arma::Col<T> h, */
/*                   arma::Mat<T> A, */
/*                   arma::Col<T> b) : c_(c), G_(G), h_(h), A_(A), b_(b) { }; */

/*   // Return the objective function f(x) for the given x. */
/*   double Evaluate( const arma::Mat<T>& x ) */
/*   { */
/*     return arma::dot(c_,x); */ 
/*   } */

/*   // Compute the gradient of f(x) for the given x and store the result in g. */
/*   void Gradient( const arma::Mat<T>& x, arma::Mat<T>& g ) */
/*   { */
/*     g = c_; */
/*   } */

/*   // get the number of constraints on the objective function. */
/*   size_t NumConstraints( ) const {return A_.n_rows + G_.n_rows;} */

/*   // Evaluate constraint i at the parameters x.  If the constraint is */
/*   // unsatisfied, a value greater than 0 should be returned.  If the constraint */
/*   // is satisfied, 0 should be returned.  The optimizer will add this value to */
/*   // its overall objective that it is trying to minimize. */
/*   double EvaluateConstraint( const size_t i, const arma::Mat<T>& x ) */
/*   { */ 
/*    // arma::dot(A_.row(i),x) - b_(i) == 0. ? return 0.; : return 1.; ; */
/*     return (arma::dot(A_.row(i), x) - b_(i) == 0.0) ? 0.0 : 1.0; */
/*   } */

/*   // Evaluate the gradient of constraint i at the parameters x, storing the */
/*   // result in the given matrix g.  If the constraint is not satisfied, the */
/*   // gradient should be set in such a way that the gradient points in the */
/*   // direction where the constraint would be satisfied. */
/*   void GradientConstraint( const size_t i, const arma::Mat<T>& x, */
/*                                            arma::Mat<T>& g ) */
/*   { */
/*     g = ((arma::dot(A_.row(i),x) - b_(i)) >= 0.) ? */
/*                                     A_.row(i).t().eval(): -A_.row(i).t().eval(); */
/*   } */

/*   arma::Col<T> c_; */
/*   arma::Mat<T> G_; */
/*   arma::Col<T> h_; */
/*   arma::Mat<T> A_; */
/*   arma::Col<T> b_; */
/* }; */

template<class T=double>
class LinearProgFunc
{
 public:
  LinearProgFunc( arma::Col<T> c,
                  arma::Mat<T> G,
                  arma::Col<T> h,
                  arma::Mat<T> A,
                  arma::Col<T> b) : c_(c), G_(G), h_(h), A_(A), b_(b) { };

  // Return the objective function f(x) for the given x.
  double Evaluate( const arma::Mat<T>& x )
  {
    return arma::dot(c_,x); 
  }

  // Compute the gradient of f(x) for the given x and store the result in g.
  void Gradient( const arma::Mat<T>& x, arma::Mat<T>& g )
  {
    g = c_;
  }

  // get the number of constraints on the objective function.
  size_t NumConstraints( ) const {return A_.n_rows + G_.n_rows;}

  // Evaluate constraint i at the parameters x.  If the constraint is
  // unsatisfied, a value greater than 0 should be returned.  If the constraint
  // is satisfied, 0 should be returned.  The optimizer will add this value to
  // its overall objective that it is trying to minimize.
  double EvaluateConstraint( const size_t i, const arma::Mat<T>& x )
  { 
    int j;
    if (i < A_.n_rows)
      return (arma::dot(A_.row(i), x) - b_(i) == 0.0) ? 0.0 : 1.0;
    else
    {
      j = i - A_.n_rows;
      return (arma::dot(G_.row(j), x) - h_(j) < 0.0) ? 0.0 : 1.0;
    }
  }

  // Evaluate the gradient of constraint i at the parameters x, storing the
  // result in the given matrix g.  If the constraint is not satisfied, the
  // gradient should be set in such a way that the gradient points in the
  // direction where the constraint would be satisfied.
  void GradientConstraint( const size_t i, const arma::Mat<T>& x,
                                           arma::Mat<T>& g )
  {
    int j;
    if (i < A_.n_rows)
      g = ((arma::dot(A_.row(i),x) - b_(i)) >= 0.) ?
                                    A_.row(i).t().eval(): -A_.row(i).t().eval();
    else
    {
      j = i - A_.n_rows;
      g = ((arma::dot(G_.row(j),x) - h_(j)) > 0.) ?
                G_.row(j).t().eval(): -G_.row(j).t().eval();
    }
  }

  arma::Col<T> c_;
  arma::Mat<T> G_;
  arma::Col<T> h_;
  arma::Mat<T> A_;
  arma::Col<T> b_;
};

template<class T=double>
class QuadProgFunc
{
 public:
  QuadProgFunc ( arma::Mat<T> Q,
                 arma::Col<T> c,
                 arma::Mat<T> G,
                 arma::Col<T> h,
                 arma::Mat<T> A,
                 arma::Col<T> b) : Q_(Q), c_(c), G_(G), h_(h), A_(A), b_(b) { };

  // Return the objective function f(x) for the given x.
  double Evaluate( const arma::Mat<T>& x )
  {
    return(arma::dot(x, Q_*x)) + arma::dot(c_,x); 
  }

  // Compute the gradient of f(x) for the given x and store the result in g.
  void Gradient( const arma::Mat<T>& x, arma::Mat<T>& g )
  {
    g =  2 * Q_ * x + c_;
  }

  // get the number of constraints on the objective function.
  size_t NumConstraints( ) const {return A_.n_rows + G_.n_rows;}

  // Evaluate constraint i at the parameters x.  If the constraint is
  // unsatisfied, a value greater than 0 should be returned.  If the constraint
  // is satisfied, 0 should be returned.  The optimizer will add this value to
  // its overall objective that it is trying to minimize.
  double EvaluateConstraint( const size_t i, const arma::Mat<T>& x )
  { 
    int j;
    if (i < A_.n_rows)
      return (arma::dot(A_.row(i), x) - b_(i) == 0.0) ? 0.0 : 1.0;
    else
    {
      j = i - A_.n_rows;
      return (arma::dot(G_.row(j), x) - h_(j) < 0.0) ? 0.0 : 1.0;
    }
  }

  // Evaluate the gradient of constraint i at the parameters x, storing the
  // result in the given matrix g.  If the constraint is not satisfied, the
  // gradient should be set in such a way that the gradient points in the
  // direction where the constraint would be satisfied.
  void GradientConstraint( const size_t i, const arma::Mat<T>& x,
                                           arma::Mat<T>& g )
  {
    int j;
    if (i < A_.n_rows)
      g = ((arma::dot(A_.row(i),x) - b_(i)) >= 0.) ?
                                    A_.row(i).t().eval(): -A_.row(i).t().eval();
    else
    {
      j = i - A_.n_rows;
      g = ((arma::dot(G_.row(j),x) - h_(j)) > 0.) ?
                G_.row(j).t().eval(): -G_.row(j).t().eval();
    }
  }

  arma::Mat<T> Q_;
  arma::Col<T> c_;
  arma::Mat<T> G_;
  arma::Col<T> h_;
  arma::Mat<T> A_;
  arma::Col<T> b_;
};

/* // FOR LINER PROGRAMMING // */
/* int main() */
/* { */
/*   arma::wall_clock timer; */
/*   timer.tic(); */

/*   arma::rowvec c = {-3, -4}; */
/*   arma::vec ccol = {-3, -4}; */

/*   arma::mat G = {{2, -1}, */
/*                 {1, 2}, */
/*                 {-1, 0}, */
/*                 {0, -1}}; */

/*   arma::rowvec h = {8,6,0,0}; */
/*   arma::vec hcol = {8,6,0,0}; */

/*   arma::mat A= {{1, 1}}; */
/*   arma::rowvec b= {1}; */
/*   arma::vec bcol= {1}; */

/*   arma::rowvec x0; */
/*   arma::vec x0col; */
/*   arma::mat G_; */
/*   arma::rowvec h_; */

/*   /1* opt::linprog(x0,c,G,h,A,b); *1/ */
/*   opt::linprog(x0,c,G,h,A,b,true,true); */
/*   PRINT(x0); */


/*   LinearProgFunc f(ccol,G,hcol,A,bcol); */
/*   ens::AugLagrangian optimizer; */
/*   x0col.zeros(c.n_elem); */
/*   optimizer.Optimize(f, x0col); */ 
/*   PRINT_VAR(x0col); */
/*   PRINT_TIME(timer.toc()); */


/*     return 0; */
/* } */

// FOR QUADRATIC PROGRAMMING //
int main() 
{
  arma::rowvec c = {-3, -4};
  arma::vec ccol = {-3, -4};

  arma::mat Q = {{2, 0},
                 {0, 2}};

  arma::mat G = {{2, -1},
                 {1, 2},
                 {-1, 0},
                 {0, -1}};

  arma::rowvec h = {8,6,0,0};
  arma::vec hcol = {8,6,0,0};

  arma::mat A= {{1, 1}};
  arma::rowvec b= {1};
  arma::vec bcol= {1};

  arma::rowvec x0(2) ;
  arma::vec x0_(2) ;
  
  opt::quadprog(x0,Q,c,G,h,A,b);
  PRINT_VAR(x0);

  opt::quadprog(x0,Q,c,G,h,A,b);
  QuadProgFunc f(Q,ccol,G,hcol,A,bcol);
  ens::AugLagrangian optimizer;
  optimizer.Optimize(f, x0_); 
  PRINT_VAR(x0_);
  return 0;
}
