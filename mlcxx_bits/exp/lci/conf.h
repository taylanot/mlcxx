/**
 *
 * @file conf.h
 * @author Ozgur Taylan Turan
 *
 * Configuration variables for lci experiments
 */

#ifndef LCI_CONF_H
#define LCI_CONF_H

namespace experiments {
namespace lci {
namespace conf {

  std::filesystem::path dir_lci= ".lci"; 


  // lin
  // dataset
  const int D = 1;
  const int N = 5000;
  const int Nc = 2;
  
  const arma::rowvec stds = arma::regspace<arma::rowvec>(0.,3.);

  const std::string type = "Linear";
  const std::string classtype = "Banana";
  const std::string noise_type = "Gaussian";

  // related to learning curves 
  const std::filesystem::path dataset_dir = "datasets" ;
  //const std::filesystem::path filename = "expsal.csv";
  const std::filesystem::path filename = "fuel_cpi-44.csv";
  const std::filesystem::path filename_var = "fuel_cpi-44-varrep.csv";
  const arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,20);
  const arma::irowvec repeats = arma::regspace<arma::irowvec>(9999999,-500000,1);
  
  // real world datasets
  const std::filesystem::path class_filename = "datasets/winequality-white.csv";
  const std::filesystem::path reg_filename = "datasets/housing.csv";
  const size_t repeat = 1e5;

  
} // namespace conf
} // namespace lci
} // namespace experiments

#endif

