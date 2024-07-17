/**
 * @file kaczmarz.cpp
 * @author Ozgur Taylan Turan
 *
 * Developing kaczmarz solver
 */
#define DTYPE float
#include <headers.h>

template<class T=DTYPE>
arma::Col<T> mysolve(arma::Mat<T> A,arma::Row<T>b,
                   size_t maxproj= 100000, T tol=1e-12)
{
  arma::Mat<T> x(arma::size(b),arma::fill::ones);
  size_t i;  
  size_t iter = 0;
  arma::Row<T> dx;
  while (iter++ < maxproj)
  {
    i = arma::randi(arma::distr_param(0,b.n_elem-1)); 
    dx = (b[i]-(arma::dot(A.col(i),x))) * A.col(i).t()/std::pow(arma::norm(A.col(i),2),2);
    x += dx;
    if (arma::norm(dx,2) < tol)
      break;
  }
  return x.t();
}

template<class T=DTYPE>
class LinearRegression
{
public:

  LinearRegression ( ) { }

  LinearRegression ( const arma::Mat<T>& X, const arma::Mat<T>& y )
  {
    Train(X,y);
  }

  void Train( const arma::Mat<T>& X, const arma::Mat<T>& y)
  {
    arma::Mat<T> X_ = arma::join_vert(X,arma::ones<arma::Mat<T>>(1,X.n_cols));
    auto b = arma::conv_to<arma::Row<T>>::
                      from(X_ * y.t()) ;
    parameters_ = mysolve<T>(X_*X_.t(),b);
  }

  void Predict( const arma::Mat<T>& X, arma::Mat<T>& preds)
  {
    arma::Mat<T> X_ = arma::join_vert(X,arma::ones<arma::Mat<T>>(1,X.n_cols));
    preds = parameters_.t() * X_;
  }
  T ComputeError( const arma::Mat<T>& X, const arma::Mat<T>& preds)
  {
    arma::Mat<T> temp;
    Predict(X,temp);
    return arma::dot(temp,temp)/temp.n_elem;
    
  }
  
private:
  arma::Mat<T> parameters_;
};

int main ( int argc, char** argv )
{
  std::filesystem::path path = ".09_07_23";
  std::filesystem::create_directories(path);
  std::filesystem::current_path(path);

  arma::wall_clock timer;
  timer.tic();

  size_t D = 10;
  utils::data::regression::Dataset dataset(D,10000);
  dataset.Generate("Linear",0.1);

  arma::irowvec Ns = arma::regspace<arma::irowvec>(1,10,200);

  src::LCurve<LinearRegression<DTYPE>,mlpack::MSE> lc(Ns,1000,true,false,true);
  src::LCurve<mlpack::LinearRegression<arma::fmat>,mlpack::MSE> lc_(Ns,1000,true,false,true);
  lc.Bootstrap(dataset.inputs_,dataset.labels_);
  lc_.Bootstrap(dataset.inputs_,dataset.labels_,0,true);
  arma::fmat error = lc.test_errors_;
  error.save("kaczmarz_"+std::to_string(D)+".csv",arma::csv_ascii);
  arma::fmat error_ = lc_.test_errors_;
  error_.save("pinv_"+std::to_string(D)+".csv",arma::csv_ascii);
  

  PRINT_TIME(timer.toc());

  return 0;
}
