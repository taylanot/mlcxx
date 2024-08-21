/**
 * @file mnist.cpp
 * @author Ozgur Taylan Turan
 *
 * Checking the mnist experiments of u-turn on double descend paper
 */

#define DTYPE float
#include <headers.h>

template<class T=DTYPE>
std::tuple<arma::Mat<T>,
           arma::Row<T>,
           arma::Mat<T>,
           arma::Row<T>> extract_mnist (size_t N= 10000)
{
  arma::fmat mnist, Xtrn,ytrn,Xtst,ytst; 
  mlpack::data::Load(DATASET_PATH/"mnist-dataset/mnist_train.csv", mnist,true);
  ytrn = (mnist.row(0)).cols(0,N); Xtrn=(mnist.rows(1,mnist.n_rows-1)).cols(0,N) / 255;
  mnist.reset();
  mlpack::data::Load(DATASET_PATH/"mnist-dataset/mnist_test.csv", mnist,true);
  ytst = mnist.row(0); Xtst=mnist.rows(1,mnist.n_rows-1) / 255;
  return std::make_tuple(Xtrn,ytrn,Xtst,ytst);  
}

template<class T=DTYPE>
arma::Mat<T> rff(const arma::Mat<T>& Xtrn, size_t basis=100)
{
  arma::Mat<T> w(Xtrn.n_rows,basis);
  w.randn()*0.2;
  arma::Mat<T> res = w.t()*Xtrn;
  return arma::join_vert(arma::cos(res),arma::sin(res));
  /* return res; */
}

struct hightol
{
  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b) 
  {
    return (b*arma::pinv(A,10.)).t() ;
  }
};

struct pinv
{
  template<class T=DTYPE>
  arma::Col<T> operator()(arma::Mat<T> A,arma::Row<T>b) 
  {
    return arma::pinv(A)*b.t();
  }
};

template<class SOLVER=pinv,class T=DTYPE>
class LinearRegression
{
public:

  LinearRegression ( bool bias = true) : bias_(bias) { }

  template<class... Args>
  LinearRegression ( const arma::Mat<T>& X, const arma::Mat<T>& y, Args&... args)
  {
    Train(X,y);
  }

  void Train( const arma::Mat<T>& X, const arma::Mat<T>& y)
  {
    arma::Mat<T> X_;
    if (bias_)
      X_ = arma::join_vert(X,arma::ones<arma::Mat<T>>(1,X.n_cols));
    else 
      X_ = X;

    auto b = arma::conv_to<arma::Row<T>>::from(X_ * y.t()) ;
    parameters_ = solver_((X_*X_.t()).eval(),b);
  }

  void Predict( const arma::Mat<T>& X, arma::Mat<T>& preds)
  {
    arma::Mat<T> X_;
    if (bias_)
      X_ = arma::join_vert(X,arma::ones<arma::Mat<T>>(1,X.n_cols));
    else
      X_ = X;
    preds = parameters_.t() * X_;
  }
  arma::Mat<T> Parameters( ){return parameters_.t();}

  T ComputeError( const arma::Mat<T>& X, const arma::Mat<T>& preds)
  {
    arma::Mat<T> temp;
    Predict(X,temp);
    return arma::dot(temp,temp)/temp.n_elem;
    
  }
  
private:
  arma::Mat<T> parameters_;
  SOLVER solver_;
  bool bias_ = false;
};

template<class solver=pinv,class T=DTYPE>
T onevsrest ( size_t Nsamp, size_t Nfeat, size_t ndig,
                                 arma::Mat<T> Xtrn, arma::Row<T> ytrn,
                                 arma::Mat<T> Xtst, arma::Row<T> ytst )
{
  std::vector<LinearRegression<solver>> models(ndig);
  arma::frowvec y1(Nsamp);
  arma::frowvec y2(ytst.n_elem);
  arma::frowvec err(ndig);

  /* auto trainset = data::StratifiedSplit(Xtrn,ytrn,Nsamp); */
  auto trainset = data::Split(Xtrn,ytrn,Nsamp);

  auto Xtrn_ = std::get<0>(trainset);
  auto ytrn_ = std::get<2>(trainset);

  for (size_t i=0;i<ndig;i++)
  {
    y1.elem( arma::find(ytrn_ == i) ).ones();
    models[i].Train(rff(Xtrn_,Nfeat), y1);
    y2.elem( arma::find(ytst == i) ).ones();
    err[i] = models[i].ComputeError(rff(Xtst,Nfeat),y2);
    y1.zeros();y2.zeros();
  }
  return arma::accu(err)/err.n_elem;
}

//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{
  std::filesystem::path path = EXP_PATH/"09_07_23/mnist";
  std::filesystem::create_directories(path);
  std::filesystem::current_path(path);


  arma::wall_clock timer;
  timer.tic();

  auto data = extract_mnist(100);
  auto Xtrn = std::get<0>(data);
  auto ytrn = std::get<1>(data);
  auto Xtst = std::get<2>(data);
  auto ytst = std::get<3>(data);

  size_t ndig = 10;
  size_t rep = 100;

  /* arma::frowvec nfeat = {1,100,200,500,600,700,800,1000}; */
  arma::frowvec nfeat = {10};
  arma::irowvec Ns = arma::regspace<arma::irowvec>(10,1,30);

    

  arma::Mat<DTYPE> error(Ns.n_elem,rep);

  ProgressBar bar1(Ns.n_elem*rep);
  #pragma omp parallel for collapse(2)
  for (size_t j=0;j<Ns.n_elem;j++)
    for (size_t k=0;k<rep;k++)
    {
      error(j,k) = onevsrest<pinv,DTYPE>(Ns(j),nfeat[0],ndig,Xtrn,ytrn,Xtst,ytst);
      bar1.Update();
    }

  error.save("pinv.csv",arma::csv_ascii);

  ProgressBar bar2(Ns.n_elem*rep);
  #pragma omp parallel for collapse(2)
  for (size_t j=0;j<Ns.n_elem;j++)
    for (size_t k=0;k<rep;k++)
    {
      error(j,k) = onevsrest<hightol,DTYPE>(Ns(j),nfeat[0],ndig,Xtrn,ytrn,Xtst,ytst);
      bar2.Update();
    }

  error.save("hightol.csv",arma::csv_ascii);


  PRINT_TIME(timer.toc());

  return 0;
}


