/**
 * @file transform.h
 * @author Ozgur Taylan Turan
 *
 * A simple transform wrapper
 *
 */

#ifndef TRANSFORM_H
#define TRANSFORM_H

namespace data {
namespace regression {
//=============================================================================
// Transformer
//=============================================================================

template<class T = mlpack::data::StandardScaler,
         class D = data::regression::Dataset<>>
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
         class D = data::classification::Dataset<>>
class Transformer
{
  public:
  Transformer ( const D& data );

  D Trans ( const D& data );

  D InvTrans( const D& data );

  private:
  T trans_;

};

} // classification namespace 
} // data namespace
#include "transform_impl.h"  

#endif
