/**
 * @file transform.h
 * @author Ozgur Taylan Turan
 *
 * A simple transform wrapper
 *
 */

#ifndef TRANSFORM_H
#define TRANSFORM_H

namespace utils {
namespace data {
namespace regression {
//=============================================================================
// Transformer
//=============================================================================

template<class T = mlpack::data::StandardScaler,
         class D = utils::data::regression::Dataset>
class Transformer
{
  public:
  Transformer ( const D& data );

  D TransInp ( const D& data );
  D TransLab ( const D& data );
  D Trans    ( const D& data );

  D InvTrans    ( const D& data );
  D InvTransInp ( const D& data );
  D InvTransLab ( const D& data );



  private:
  T inp_;
  T lab_;
};

} 
namespace classification{
//=============================================================================
// Transformer
//=============================================================================

template<class T = mlpack::data::StandardScaler,
         class D = utils::data::classification::Dataset>
class Transformer
{
  public:
  Transformer ( const D& data );

  D Trans ( const D& data );

  D InvTrans( const D& data );

  private:
  T trans_;

};

} // regressionnamespace //
} // data namespace
} // utils namespace
  //
#include "transform_impl.h"  

#endif
