/**
 * @file transform.h
 * @author Ozgur Taylan Turan
 *
 * A simple transform wrapper, where you can apply mlpack transformations to the 
 * Dataset objects of our own. Here, you can also choose to apply those to 
 * inputs or the labels.
 * -- Some of the transforming classes in mlpack does not have the float option
 *  yet, beware.
 */

#ifndef TRANSFORM_H
#define TRANSFORM_H

namespace data {
namespace regression {
//-----------------------------------------------------------------------------
// Transformer
//-----------------------------------------------------------------------------
template<class T = mlpack::data::StandardScaler,
         class D = data::Dataset<arma::Row<DTYPE>>>
class Transformer
{
public:
  /**
   * @brief Construct and fit transformers for both inputs and labels.
   * @param data Dataset to fit the transformers on.
   */
  Transformer(const D& data);

  /**
   * @brief Transform only the inputs of a dataset.
   * @param data Dataset to transform.
   * @return Transformed dataset (inputs only).
   */
  D TransInp(const D& data);

  /**
   * @brief Transform only the labels of a dataset.
   * @param data Dataset to transform.
   * @return Transformed dataset (labels only).
   */
  D TransLab(const D& data);

  /**
   * @brief Transform both inputs and labels of a dataset.
   * @param data Dataset to transform.
   * @return Fully transformed dataset.
   */
  D Trans(const D& data);

  /**
   * @brief Inverse transform both inputs and labels.
   * @param data Dataset to inverse transform.
   * @return Original-scale dataset.
   */
  D InvTrans(const D& data);

  /**
   * @brief Inverse transform only the inputs.
   * @param data Dataset to inverse transform.
   * @return Dataset with original-scale inputs.
   */
  D InvTransInp(const D& data);

  /**
   * @brief Inverse transform only the labels.
   * @param data Dataset to inverse transform.
   * @return Dataset with original-scale labels.
   */
  D InvTransLab(const D& data);

private:
  T inp_; // Transformer for inputs
  T lab_; // Transformer for labels
}; 
} // regression namespace
  
namespace classification {

//-----------------------------------------------------------------------------
// Transformer
//-----------------------------------------------------------------------------
template<class T = mlpack::data::StandardScaler,
         class D = data::Dataset<arma::Row<size_t>>>
class Transformer
{
public:
  /**
   * @brief Construct and fit the transformer.
   * @param data Dataset to fit the transformer on.
   */
  Transformer(const D& data);

  /**
   * @brief Transform dataset features.
   * @param data Dataset to transform.
   * @return Transformed dataset.
   */
  D Trans(const D& data);

  /**
   * @brief Inverse transform dataset features.
   * @param data Dataset to inverse transform.
   * @return Dataset with original-scale features.
   */
  D InvTrans(const D& data);

private:
  T trans_; // Transformer for features
};

} // classification namespace 

} // data namespace
#include "transform_impl.h"  

#endif
