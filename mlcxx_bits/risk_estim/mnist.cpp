/**
 * @file mnist.cpp
 * @author Ozgur Taylan Turan
 *
 * Checking the mnist loading 
 */

#include <headers.h>

class RandomFourier
{
public:
  RandomFourier() : n_(10) 
  {BOOST_ASSERT_MSG(n_>=1, "RandomFourier: number of basis is ill-defined!");}
  RandomFourier(const size_t num_feat) : n_(num_feat)
  {BOOST_ASSERT_MSG(n_>=1, "RandomFourier: number of basis is ill-defined!");}

  template<typename VecTypeA, typename VecTypeB>
  double Evaluate(const VecTypeA& a, const VecTypeB& b) const
  {
    arma::mat w = 
      arma::mvnrnd(arma::ones(a.n_elem),arma::eye(a.n_elem,a.n_elem),n_);

    arma::rowvec p = 
      arma::randu<arma::rowvec>(n_, arma::distr_param(0.,2*arma::datum::pi));
    return 1./n_*arma::dot(
        std::sqrt(2)*arma::cos(a.t()*w+p),
        std::sqrt(2)*arma::cos(b.t()*w+p));
  }

private:
  size_t n_;

};


//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{
  arma::wall_clock timer;
  timer.tic();

  /* algo::classification::LDC ldc; */
  /* arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,10); */
  /* src::LCurve<algo::classification::LDC<>,mlpack::Accuracy> lc(Ns,100); */

  /* arma::fmat mnist; */ 
  /* mlpack::data::Load("datasets/mnist-dataset/mnist_train.csv", mnist,true); */
  /* PRINT(arma::size(mnist)); */

  /* std::filesystem::path path = "build/risk_estim/outputs/"; */
  /* std::filesystem::create_directories(path); */
  /* std::filesystem::current_path(path); */

  /* arma::fmat a = arma::zeros<arma::fmat>(42000,42000); */
  /* arma::mat a = arma::randn<arma::mat>(785,10000); */
  /* arma::mat b = arma::randn<arma::mat>(785,10000); */

  /* utils::covmat<RandomFourier> Kr(10); */
  /* PRINT(arma::size(Kr.GetMatrixT(mnist,mnist))); */

  /* PRINT_TIME(timer.toc()); */

  return 0;
}


