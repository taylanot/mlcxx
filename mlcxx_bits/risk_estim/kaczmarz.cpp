/**
 * @file logistic .cpp
 * @author Ozgur Taylan Turan
 *
 * Checking the mnist loading 
 */

#include <headers.h>

template<class T=DTYPE>
arma::Col<T> mysolve(arma::Mat<T> A,arma::Row<T>b,
                   size_t maxproj= 100000, T tol=1e-12)
{
  arma::Mat<T> x(arma::size(b));
  size_t i;  
  size_t iter = 0;
  arma::Row<T> dx;
  while (iter++ < maxproj)
  {
    i = arma::randi(arma::distr_param(0,b.n_elem-1)); 
    dx = (b[i]-(A.row(i)*x.t())) * A.row(i)/std::pow(arma::norm(A.row(i),2),2);
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

template<class T=DTYPE>
arma::Mat<T> rff(const arma::Mat<T>& Xtrn, size_t basis=100)
{
  arma::Mat<T> w(Xtrn.n_rows,basis);
  w.randn()*0.2;
  arma::Mat<T> res = w.t()*Xtrn;
  return arma::join_vert(arma::cos(res),arma::sin(res));
  return res;
}

template<class T=DTYPE>
std::tuple<arma::Mat<T>,
           arma::Row<T>,
           arma::Mat<T>,
           arma::Row<T>> extract_mnist (size_t N= 10000)
{
  arma::fmat mnist, Xtrn,ytrn,Xtst,ytst; 
  mlpack::data::Load("datasets/mnist-dataset/mnist_train.csv", mnist,true);
  ytrn = (mnist.row(0)).cols(0,N); Xtrn=(mnist.rows(1,mnist.n_rows-1)).cols(0,N) / 255;
  mnist.reset();
  mlpack::data::Load("datasets/mnist-dataset/mnist_test.csv", mnist,true);
  ytst = mnist.row(0); Xtst=mnist.rows(1,mnist.n_rows-1) / 255;
  return std::make_tuple(Xtrn,ytrn,Xtst,ytst);  
}
//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{
  /* std::filesystem::path path = "build/risk_estim/outputs"; */
  /* std::filesystem::create_directories(path); */
  /* std::filesystem::current_path(path); */

  arma::wall_clock timer;
  timer.tic();

  auto data = extract_mnist(1000);

  auto Xtrn = std::get<0>(data);
  auto ytrn = std::get<1>(data);
  auto Xtst = std::get<2>(data);
  auto ytst = std::get<3>(data);

  size_t ndig = 10;

  arma::frowvec nfeat = {1,100,200,500,600,700,800,1000};
  /* arma::frowvec nfeat = {50}; */
  arma::irowvec Ns = arma::regspace<arma::irowvec>(1,10,20);

  for (size_t k=0;k<1;k++)
  {
    for (size_t j=0;j<nfeat.n_elem;j++)
    {
      /* std::vector<LinearRegression<>> models(ndig); */
      std::vector<mlpack::LinearRegression<arma::fmat>> models(ndig);
      arma::frowvec y1(ytrn.n_elem);
      arma::frowvec y2(ytst.n_elem);
      arma::frowvec err(ndig);

      for (size_t i=0;i<ndig;i++)
      {
        y1.elem( find(ytrn == i) ).ones();
        models[i].Train(rff(Xtrn,nfeat[j]), y1);
        y2.elem( find(ytst == i) ).ones();
        err[i] = models[i].ComputeError(rff(Xtst,nfeat[j]),y2);
        y1.zeros();y2.zeros();
      }
      PRINT(arma::mean(err))
    }
  }
  /* mlpack::LinearRegression<arma::fmat> model(rff(Xtrn), ytrn); */
  /* PRINT(model.ComputeError(rff(Xtst),ytst)); */
   
  PRINT_TIME(timer.toc());

  return 0;
}


/* int main ( int argc, char** argv ) */
/* { */
/*   /1* std::filesystem::path path = "build/risk_estim/outputs"; *1/ */
/*   /1* std::filesystem::create_directories(path); *1/ */
/*   /1* std::filesystem::current_path(path); *1/ */

/*   arma::wall_clock timer; */
/*   timer.tic(); */

/*   size_t D = 3; */
/*   utils::data::regression::Dataset dataset(D,10000); */
/*   dataset.Generate("Linear",0.1); */

/*   /1* arma::fmat A = dataset.inputs_ * dataset.inputs_.t(); *1/ */
/*   /1* auto b = arma::conv_to<arma::frowvec>:: *1/ */
/*   /1*                   from(dataset.inputs_ * dataset.labels_.t()) ; *1/ */

/*   /1* PRINT(arma::solve(A,b.t())); *1/ */
/*   /1* PRINT(mysolve(A,b)); *1/ */

/*   arma::irowvec Ns = arma::regspace<arma::irowvec>(1,10,200); */

/*   /1* src::LCurve<LinearRegression<DTYPE>,mlpack::MSE> lc(Ns,1000,true); *1/ */
/*   src::LCurve<mlpack::LinearRegression<arma::fmat>,mlpack::MSE> lc(Ns,1000,true); */
/*   lc.Bootstrap(dataset.inputs_,dataset.labels_); */
/*   /1* lc.Bootstrap(dataset.inputs_,dataset.labels_,0,true); *1/ */
/*   arma::Col<DTYPE> error = arma::conv_to<arma::Col<DTYPE>>::from(std::get<1>(lc.stats_).row(0)); */
/*   error.save("larger_.csv",arma::csv_ascii); */
  

/*   PRINT_TIME(timer.toc()); */

/*   return 0; */
/* } */
