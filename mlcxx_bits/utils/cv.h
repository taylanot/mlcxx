/**
 * @file xval.h
 * @author Ozgur Taylan Turan
 *
 * I will try to expend the functionalities of the cross-validation of mlpack.
 * Doin this to get the error distribution out of it!
 */
namespace xval {

class SimpleCV : mlpack::SimpleCV
{
  arma::vec restuls;

}
