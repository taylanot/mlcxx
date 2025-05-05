/**
 * @file hpt.cpp
 * @author Ozgur Taylan Turan
 *
 */


#include <headers.h>

using LOSS = mlpack::MSE;
using SAMPLE = data::RandomSelect<>;
using DATASET = data::Dataset<arma::Mat<DTYPE>>;

int main ( int argc, char** argv )
{
  arma::wall_clock timer;
  timer.tic();

  DATASET dataset(2);
  dataset.Linear(5000);

  auto lrs = arma::logspace<arma::Row<DTYPE>>(-5,-2,10);
 
  /* typedef mlpack::FFN<mlpack::CrossEntropyError> NetworkType; */
  typedef mlpack::FFN<mlpack::MeanSquaredError> NetworkType;
  NetworkType network, network2;
  network.Add<mlpack::Linear>(1);
  network2.Add<mlpack::Linear>(2);
  network2.Add<mlpack::Linear>(1);
  std::vector<NetworkType> nets = {network,network2};




  mlpack::HyperParameterTuner<algo::ANN<NetworkType>, mlpack::MSE, mlpack::SimpleCV> 
    hpt(0.2,dataset.inputs_,dataset.labels_);
  /* std::vector<bool> early = {true,false}; */
  std::vector<NetworkType> neti = {network};
  std::vector<bool> early = {true};
  /* auto a = hpt.Optimize(mlpack::Fixed(network),early,lrs); */
  auto a = hpt.Optimize(neti,early,lrs);
  /* PRINT(std::get<0>(a)); */
  /* PRINT(std::get<1>(a)); */

  /* ens::Adam opt; */
/* // Get the unique labels */
  /* this -> ulab_ = arma::unique(labels); */
  /* // OneHotEncode the labels for the classifier network */
  /* auto convlabels = _OneHotEncode(labels,ulab_); */

  /* network.Train(dataset.inputs_,,opt); */

/*   auto Ns = arma::regspace<arma::Row<size_t>>(10,10,100); */ 

/*   lcurve::LCurve< algo::ANN<NetworkType>, */
/*                   DATASET, */
/*                   SAMPLE, */
/*                   LOSS>  curve(dataset,Ns,1,true,true); */

/*   curve.GenerateFix ( 0.2, mlpack::Fixed(network), mlpack::Fixed(true),lrs); */

  /* lcurve::LCurve< algo::ANN<NetworkType>, */
  /*                 DATASET, */
  /*                 SAMPLE, */
  /*                 LOSS>  curve2(dataset,Ns,1,true,true); */

  /* curve.Generate( 0.2, mlpack::Fixed(network2) ); */

  /* curve.Generate<NetworkType>( network ); */

  /* curve2.Generate<NetworkType>( network2 ); */



  /* std::vector<bool> early = {true,false}; */
  /* curve.Generate<mlpack::SimpleCV,ens::GridSearch,DTYPE,NetworkType>( 0.2, mlpack::Fixed(network),early ); */
  /* curve.GenerateHpt( 0.2, mlpack::Fixed(&network), early ); */



  PRINT_TIME(timer.toc())
  return 0;
}




