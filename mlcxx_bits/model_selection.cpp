/**
 * @file model_selection.cpp
 * @author Ozgur Taylan Turan
 *
 * Main file of mlcxx where you do not have to do anything...
 */

#include <headers.h>

const int  SEED = 8 ; // KOBEEEE

//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{
  std::filesystem::path dir = ".ms";
  std::filesystem::create_directories(dir);
  
  int D, Ntrn;  D=1; Ntrn=1000; 
  double a, p, eps; a = 1.0; p = 0.1; eps = 1.;
  utils::data::regression::Dataset dataset(D, Ntrn);

  dataset.Generate(a, p, "Linear", eps);
  auto inputs = dataset.inputs_;
  auto labels = arma::conv_to<arma::rowvec>::from(dataset.labels_);

  std::vector<int> reps = {5,10,50,100,1000};
  for ( int repeat : reps)
  {
    arma::irowvec Ns = arma::regspace<arma::irowvec>(20,1,500);

    src::regression::LCurve<mlpack::LinearRegression,
               mlpack::MSE> lcurve(Ns,repeat);
    std::filesystem::path nobias, bias;

    nobias= dir/("nobias_"+std::to_string(repeat)+".csv");
    bias= dir/("wbias_"+std::to_string(repeat)+".csv");

    lcurve.Generate(inputs, labels, 0., false); 
    utils::Save(nobias,lcurve.test_errors_);
    lcurve.Generate(inputs, labels, 0., true); 
    utils::Save(bias,lcurve.test_errors_);
  }
}
