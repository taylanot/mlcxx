/**
 * @file base.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef LLC_BASE_H 
#define LLC_BASE_H

namespace experiments {
namespace llc {

class EXP3
{
  public:
  
  EXP3(arma::mat& X, arma::vec& y) : X(X), y(y) { }
  
  double Evaluate (const arma::mat& theta)
  {
    const arma::vec res = Residual(theta);
    return arma::dot(res,res);
  }
  void Gradient ( const arma::mat& theta, arma::mat& gradient )
  {
    const arma::vec res = Residual(theta);
    const arma::vec dfdp1 = arma::exp(-X.t() * theta(1,0));
    const arma::mat dfdp2 = -theta(0,0)*arma::exp(-X.t() * theta(1,0))%X.t();
    const arma::vec dfdp3(X.n_cols, arma::fill::ones);
    gradient = arma::join_cols(dfdp1.t()*res, dfdp2.t()*res,dfdp3.t()*res);
    gradient *= 2;
  }

  arma::vec Residual(const arma::mat& theta)
  {
    return (theta(0,0)*arma::exp(-X.t() * theta(1,0))+theta(2,0)) - y;
  }
  
  private:
  arma::mat& X;
  const arma::vec& y;
};

//class POW3
//{
//  public:
//  
//  POW3(arma::mat& X, arma::vec& y) : X(X), y(y) { }
//  
//  double Evaluate (const arma::mat& theta)
//  {
//    const arma::vec res = Residual(theta);
//    return arma::dot(res,res);
//  }
//  void Gradient ( const arma::mat& theta, arma::mat& gradient )
//  {
//    const arma::vec res = Residual(theta);
//    const arma::vec dfdp1 = arma::exp(-X.t() * theta(1,0));
//    const arma::mat dfdp2 = -theta(0,0)*arma::exp(-X.t() * theta(1,0))%X.t();
//    const arma::vec dfdp3(X.n_cols, arma::fill::ones);
//    gradient = arma::join_cols(dfdp1.t()*res, dfdp2.t()*res,dfdp3.t()*res);
//    gradient *= 2;
//  }
//
//  arma::vec Residual(const arma::mat& theta)
//  {
//    return (theta(0,0)*arma::exp(-X.t() * theta(1,0))+theta(2,0)) - y;
//  }
//  
//  private:
//  arma::mat& X;
//  const arma::vec& y;
//}
//
//class MMF4
//{
//  public:
//  
//  MMF4(arma::mat& X, arma::vec& y) : X(X), y(y) { }
//  
//  double Evaluate (const arma::mat& theta)
//  {
//    const arma::vec res = Residual(theta);
//    return arma::dot(res,res);
//  }
//  void Gradient ( const arma::mat& theta, arma::mat& gradient )
//  {
//    const arma::vec res = Residual(theta);
//    const arma::vec dfdp1 = arma::exp(-X.t() * theta(1,0));
//    const arma::mat dfdp2 = -theta(0,0)*arma::exp(-X.t() * theta(1,0))%X.t();
//    const arma::vec dfdp3(X.n_cols, arma::fill::ones);
//    gradient = arma::join_cols(dfdp1.t()*res, dfdp2.t()*res,dfdp3.t()*res);
//    gradient *= 2;
//  }
//
//  arma::vec Residual(const arma::mat& theta)
//  {
//    return (theta(0,0)*arma::exp(-X.t() * theta(1,0))+theta(2,0)) - y;
//  }
//  
//  private:
//  arma::mat& X;
//  const arma::vec& y;
//}
//
//class WBL4
//{
//  public:
//  
//  WBL4(arma::mat& X, arma::vec& y) : X(X), y(y) { }
//  
//  double Evaluate (const arma::mat& theta)
//  {
//    const arma::vec res = Residual(theta);
//    return arma::dot(res,res);
//  }
//  void Gradient ( const arma::mat& theta, arma::mat& gradient )
//  {
//    const arma::vec res = Residual(theta);
//    const arma::vec dfdp1 = arma::exp(-X.t() * theta(1,0));
//    const arma::mat dfdp2 = -theta(0,0)*arma::exp(-X.t() * theta(1,0))%X.t();
//    const arma::vec dfdp3(X.n_cols, arma::fill::ones);
//    gradient = arma::join_cols(dfdp1.t()*res, dfdp2.t()*res,dfdp3.t()*res);
//    gradient *= 2;
//  }
//
//  arma::vec Residual(const arma::mat& theta)
//  {
//    return (theta(0,0)*arma::exp(-X.t() * theta(1,0))+theta(2,0)) - y;
//  }
//  
//  private:
//  arma::mat& X;
//  const arma::vec& y;
//}
} // namespace llc
} // namespace experiments

///////////////////////////////////////////////////////////////////////////////
//  GENERATE
///////////////////////////////////////////////////////////////////////////////


