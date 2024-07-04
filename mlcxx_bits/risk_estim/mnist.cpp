/**
 * @file mnist.cpp
 * @author Ozgur Taylan Turan
 *
 * Checking the mnist experiments of u-turn on double descend paper
 */

#include <headers.h>
/* class RandomFourier */
/* { */
/* public: */
/*   RandomFourier() : n_(10) */ 
/*   {BOOST_ASSERT_MSG(n_>=1, "RandomFourier: number of basis is ill-defined!");} */
/*   RandomFourier(const size_t num_feat) : n_(num_feat) */
/*   {BOOST_ASSERT_MSG(n_>=1, "RandomFourier: number of basis is ill-defined!");} */

/*   template<class T=DTYPE> */
/*   double Evaluate(const arma::Mat<T>& a, const arma::Mat<T>& b) const */
/*   { */
/*     arma::Mat<T> w = */ 
/*       arma::mvnrnd(arma::ones<arma::Mat<T>>(a.n_elem), */
/*                    arma::eye<arma::Mat<T>>(a.n_elem,a.n_elem),n_); */

/*     arma::Row<T> p = */ 
/*       arma::randu<arma::Row<T>>(n_, arma::distr_param(0.,2*arma::datum::pi)); */
/*     return 1./n_*arma::dot( */
/*         std::sqrt(2)*arma::cos(a.t()*w+p), */
/*         std::sqrt(2)*arma::cos(b.t()*w+p)); */
/*   } */

/*   template<class T=DTYPE> */
/*   double Evaluate(const arma::Mat<T>& a, const arma::Mat<T>& b) const */
/*   { */
/*     arma::Mat<T> w = */ 
/*       arma::mvnrnd(arma::ones<arma::Mat<T>>(a.n_elem), */
/*                    arma::eye<arma::Mat<T>>(a.n_elem,a.n_elem),n_); */

/*     return 1./n_*arma::sum(arma::cos((a.t()-b.t())*w),1).eval()(0,0); */
/*   } */

/*   template<class T=DTYPE> */
/*   arma::Mat<T> GetMatrix(const arma::Mat<T>& a, const arma::Mat<T>& b) const */
/*   { */
/*     arma::Mat<T> w = arma::randn<arma::Mat<T>>(a.n_rows,n_); */

/*     return arma::cos(a*w); */ 
/*   } */

/* private: */
/*   size_t n_; */

/* }; */
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

template<class T=DTYPE>
arma::Mat<T> rff(const arma::Mat<T>& Xtrn, size_t basis=100)
{
  arma::Mat<T> w(Xtrn.n_rows,basis);
  w.randn()*0.2;
  arma::Mat<T> res = w.t()*Xtrn;
  return arma::join_vert(arma::cos(res),arma::sin(res));
  return res;
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

  auto data = extract_mnist(100);
  auto Xtrn = std::get<0>(data);
  auto ytrn = std::get<1>(data);
  auto Xtst = std::get<2>(data);
  auto ytst = std::get<3>(data);

  size_t ndig = 10;

  /* arma::frowvec nfeat = {1,100,200,500,600,700,800,1000}; */
  arma::frowvec nfeat = {50};
  arma::irowvec Ns = arma::regspace<arma::irowvec>(1,10,20);

  for (size_t k=0;k<100;k++)
  {
    for (size_t j=0;j<nfeat.n_elem;j++)
    {
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


