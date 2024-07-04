/**
 * @file basis_.cpp
 * @author Ozgur Taylan Turan
 *
 * Basis Pursuit algorithm
 */

#include <headers.h>

using namespace arma;
using namespace ens;

// Define the objective function
template<class T=DTYPE>
class L1NormObjective
{
private:
    const arma::Mat<T>& A;
    const arma::Row<T>& y;

public:
    L1NormObjective(const arma::Mat<T>& A, const arma::Row<T>& y) : A(A), y(y) {}

    // Objective function: f(x) = ||x||_1
    double Evaluate(const arma::Mat<T>& x) const
    {
        return norm(x, 1);
    }

    // Gradient of the objective function
    void Gradient(const arma::Mat<T>& x, arma::Mat<T>& grad) const
    {
        grad = sign(x);
    }

    // Number of constraints
    size_t NumConstraints() const { return A.n_rows; }

    // Constraint evaluation
    double EvaluateConstraint(const size_t i, const mat& x) 
    {

        return (A(1,i) * x - y(i)).eval()(0,0) ;
    }

    // Gradient of the i-th constraint
    void GradientConstraint(const size_t i,const mat& x, mat& grad) const
    {
        grad = trans(A.row(i));
    }
};

template<class T=DTYPE>
arma::Col<T> mysolve(arma::Mat<T> A, arma::Row<T>b)
{
  // Initial guess
  arma::Col<T> x = zeros<arma::Col<T>>(A.n_cols);

  // Instantiate the objective function
  L1NormObjective obj(A, b);

  // Optimization
  AugLagrangian opt;
  opt.Optimize(obj,x);

  return x;
}

template<class T=DTYPE>
class LinearRegression
{
public:

  LinearRegression ( ) { }

  LinearRegression ( const arma::Mat<T>& X, const arma::Mat<T>& y )
  {
    /* arma::Mat<T> X_ = arma::join_vert(X,arma::ones<arma::Mat<T>>(1,X.n_cols)); */
    /* Train(X_,y); */
    Train(X,y);
  }

  void Train( const arma::Mat<T>& X, const arma::Mat<T>& y)
  {
  auto b = arma::conv_to<arma::Row<T>>::
                      from(X * y.t()) ;
    parameters_ = mysolve<T>(X*X.t(),b);
  }

  void Predict( const arma::Mat<T>& X, arma::Mat<T>& preds)
  {
    /* arma::Mat<T> X_ = arma::join_vert(X,arma::ones<arma::Mat<T>>(1,X.n_cols)); */
    /* preds = parameters_.t() * X_; */
    preds = parameters_.t() * X;
  }
  
private:
  arma::Mat<T> parameters_;
};

int main()
{
  arma::wall_clock timer;
  timer.tic();

  size_t D = 3;
  utils::data::regression::Dataset dataset(D,10000);
  dataset.Generate("Linear",0.1);

  /* arma::fmat A = dataset.inputs_ * dataset.inputs_.t(); */
  /* auto b = arma::conv_to<arma::frowvec>:: */
  /*                   from(dataset.inputs_ * dataset.labels_.t()) ; */

  /* PRINT(arma::solve(A,b.t())); */
  /* PRINT(mysolve(A,b)); */

  arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,20);

  /* src::LCurve<LinearRegression<DTYPE>,mlpack::MSE> lc(Ns,1000,true); */
  src::LCurve<mlpack::LinearRegression<arma::fmat>,mlpack::MSE> lc(Ns,1000,true);
  lc.Bootstrap(dataset.inputs_,dataset.labels_);
  /* lc.Bootstrap(dataset.inputs_,dataset.labels_,0,true); */
  arma::Col<DTYPE> error = arma::conv_to<arma::Col<DTYPE>>::from(std::get<1>(lc.stats_).row(0));
  error.save("another.csv",arma::csv_ascii);
  

  PRINT_TIME(timer.toc());


    return 0;
}
