/**
 *
 * @file conf.h
 * @author Ozgur Taylan Turan
 *
 * Configuration variables for yawlc experiments
 */

#ifndef YAWLC_CONF_H
#define YAWLC_CONF_H

namespace experiments {
namespace yawlc {
namespace conf {

  std::filesystem::path dir_yawlc = ".yawlc"; 


  // dataset
  const int D = 1;
  const int N = 100000;
  const double a = 1.;
  const double b = 0.;
  const std::string type = "Linear";

  // related to learning curve linhes
  const arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,30);
  const size_t replinhes = 1e3;
  const size_t replearncurve = 1e3;
  const bool clip_y = false;
  const bool clip_x = true;
  
  // related to learning curve gd 
  const size_t repeat = 100;
  const size_t archtype = 1;
  const size_t nonlintype = 0;

  
} // namespace conf
} // namespace yawlc
} // namespace experiments

#endif

