
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
 *
 *  Commercial License Usage
 *
 *  This file may be used under the terms of a commercial license
 *  provided with the software, or under the terms contained in a written
 *  agreement between you and DRG. For more information contact DRG at
 *  http://www.dynaflow.com.
 *
 *  GNU Lesser General Public License Usage
 *
 *  Alternatively, this file may be used under the terms of the GNU
 *  Lesser General Public License version 2.1 or version 3 as published
 *  by the Free Software Foundation and appearing in the file
 *  LICENSE.LGPLv21 and LICENSE.LGPLv3 included in the packaging of this
 *  file. Please review the following information to ensure the GNU
 *  Lesser General Public License requirements will be met:
 *  https://www.gnu.org/licenses/lgpl.html and
 *  http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */


#include <jem/base/rtti.h>
#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/None.h>
#include <jive/util/FuncUtils.h>
#include <jive/util/ObjectConverter.h>


JEM_DEFINE_CLASS( jive::util::ObjectConverter );


JIVE_BEGIN_PACKAGE( util )


using jem::minOf;
using jem::maxOf;
using jem::toValue;
using jem::isInstance;
using jem::newInstance;
using jem::IllegalInputException;
using jem::util::isNone;


//=======================================================================
//   class ObjectConverter
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ObjectConverter::ObjectConverter

  ( const Properties&  globdat ) :

    globdat_ ( globdat )

{}


ObjectConverter::~ObjectConverter ()
{}


//-----------------------------------------------------------------------
//   convert (bool)
//-----------------------------------------------------------------------


void ObjectConverter::convert

  ( bool&               out,
    const Ref<Object>&  obj ) const

{
  if ( toValue( out, obj ) )
  {
    return;
  }

  if ( isInstance<String>( obj ) )
  {
    out = toBool_ ( toValue<String>( obj ) );

    return;
  }

  if ( isNone( obj ) )
  {
    out = false;

    return;
  }

  typeError_ ( "a boolean, an expression or none", obj );
}


//-----------------------------------------------------------------------
//   convert (char)
//-----------------------------------------------------------------------


void ObjectConverter::convert

  ( char&               out,
    const Ref<Object>&  obj ) const

{
  if ( toValue( out, obj ) )
  {
    return;
  }

  if ( isNone( obj ) )
  {
    out = '\0';

    return;
  }

  typeError_ ( "a character or none", obj );
}


//-----------------------------------------------------------------------
//   convert (byte)
//-----------------------------------------------------------------------


void ObjectConverter::convert

  ( byte&               out,
    const Ref<Object>&  obj ) const

{
  if ( toValue( out, obj ) )
  {
    return;
  }

  if ( isInstance<String>( obj ) )
  {
    out = (byte)

      toInt_ ( toValue<String>( obj ), 0, maxOf( out ) );

    return;
  }

  if ( isNone( obj ) )
  {
    out = 0;

    return;
  }

  typeError_ ( "a byte, an expression or none", obj );
}


//-----------------------------------------------------------------------
//   convert (short)
//-----------------------------------------------------------------------


void ObjectConverter::convert

  ( short&              out,
    const Ref<Object>&  obj ) const

{
  if ( toValue( out, obj ) )
  {
    return;
  }

  if ( isInstance<String>( obj ) )
  {
    out = (short)

      toInt_ ( toValue<String>( obj ), minOf( out ), maxOf( out ) );

    return;
  }

  if ( isNone( obj ) )
  {
    out = 0;

    return;
  }

  typeError_ ( "a short integer, an expression or none", obj );
}


//-----------------------------------------------------------------------
//   convert (int)
//-----------------------------------------------------------------------


void ObjectConverter::convert

  ( int&                out,
    const Ref<Object>&  obj ) const

{
  if ( toValue( out, obj ) )
  {
    return;
  }

  if ( isInstance<String>( obj ) )
  {
    out = (int)

      toInt_ ( toValue<String>( obj ), minOf( out ), maxOf( out ) );

    return;
  }

  if ( isNone( obj ) )
  {
    out = 0;

    return;
  }

  typeError_ ( "an integer, an expression or none", obj );
}


//-----------------------------------------------------------------------
//   convert (long)
//-----------------------------------------------------------------------


void ObjectConverter::convert

  ( long&               out,
    const Ref<Object>&  obj ) const

{
  if ( toValue( out, obj ) )
  {
    return;
  }

  if ( isInstance<String>( obj ) )
  {
    out = (long)

      toInt_ ( toValue<String>( obj ), minOf( out ), maxOf( out ) );

    return;
  }

  if ( isNone( obj ) )
  {
    out = 0;

    return;
  }

  typeError_ ( "a long integer, an expression or none", obj );
}


//-----------------------------------------------------------------------
//   convert (lint)
//-----------------------------------------------------------------------


void ObjectConverter::convert

  ( lint&               out,
    const Ref<Object>&  obj ) const

{
  if ( toValue( out, obj ) )
  {
    return;
  }

  if ( isInstance<String>( obj ) )
  {
    out =

      toInt_ ( toValue<String>( obj ), minOf( out ), maxOf( out ) );

    return;
  }

  if ( isNone( obj ) )
  {
    out = 0;

    return;
  }

  typeError_ ( "a long integer, an expression or none", obj );
}


//-----------------------------------------------------------------------
//   convert (idx_t)
//-----------------------------------------------------------------------


void ObjectConverter::convert

  ( idx_t&              out,
    const Ref<Object>&  obj ) const

{
  if ( toValue( out, obj ) )
  {
    return;
  }

  if ( isInstance<String>( obj ) )
  {
    out = (idx_t)

      toInt_ ( toValue<String>( obj ),
               (lint) minOf( out ),
               (lint) maxOf( out ) );

    return;
  }

  if ( isNone( obj ) )
  {
    out = 0;

    return;
  }

  typeError_ ( "an index (integer), an expression or none", obj );
}

//-----------------------------------------------------------------------
//   convert (float)
//-----------------------------------------------------------------------


void ObjectConverter::convert

  ( float&              out,
    const Ref<Object>&  obj ) const

{
  if ( toValue( out, obj ) )
  {
    return;
  }

  if ( isInstance<String>( obj ) )
  {
    out = (float)

      toFloat_ ( toValue<String>( obj ), maxOf( out ) );

    return;
  }

  if ( isNone( obj ) )
  {
    out = 0.0;

    return;
  }

  typeError_ ( "a single precision number, "
               "an expression or none", obj );
}


//-----------------------------------------------------------------------
//   convert (double)
//-----------------------------------------------------------------------


void ObjectConverter::convert

  ( double&             out,
    const Ref<Object>&  obj ) const

{
  if ( toValue( out, obj ) )
  {
    return;
  }

  if ( isInstance<String>( obj ) )
  {
    out = FuncUtils::evalExpr

      ( toValue<String>( obj ), globdat_ );

    return;
  }

  if ( isNone( obj ) )
  {
    out = 0.0;

    return;
  }

  typeError_ ( "a double precision number, "
               "an expression or none", obj );
}


//-----------------------------------------------------------------------
//   convert (String)
//-----------------------------------------------------------------------


void ObjectConverter::convert

  ( String&             out,
    const Ref<Object>&  obj ) const

{
  if ( ! toValue( out, obj ) )
  {
    if ( isNone( obj ) )
    {
      out = "";
    }
    else
    {
      typeError_ ( "a string or none", obj );
    }
  }
}


//-----------------------------------------------------------------------
//   convert (BoolVector)
//-----------------------------------------------------------------------


void ObjectConverter::convert

  ( BoolVector&         out,
    const Ref<Object>&  obj ) const

{
  bool  b;

  if ( isInstance<BoolVector>( obj ) )
  {
    out.ref ( toValue<const BoolVector&>( obj ) );

    return;
  }

  if ( toValue( b, obj ) )
  {
    out.resize ( 1 );

    out[0] = b;

    return;
  }

  if ( isInstance<String>( obj ) )
  {
    out.resize ( 1 );

    out[0] = toBool_ ( toValue<String>( obj ) );

    return;
  }

  if ( isInstance<StringVector>( obj ) )
  {
    const StringVector&  v = toValue<const StringVector&> ( obj );

    out.resize ( v.size() );

    for ( idx_t i = 0; i < v.size(); i++ )
    {
      out[i] = toBool_ ( v[i] );
    }

    return;
  }

  if ( isNone( obj ) )
  {
    out.resize ( 0 );

    return;
  }

  typeError_ ( "a boolean (array), "
               "an expression (array) or none", obj );
}


//-----------------------------------------------------------------------
//   convert (IntVector)
//-----------------------------------------------------------------------


void ObjectConverter::convert

  ( IntVector&          out,
    const Ref<Object>&  obj ) const

{
  int  i;

  if ( isInstance<IntVector>( obj ) )
  {
    out.ref ( toValue<const IntVector&>( obj ) );

    return;
  }

  if ( toValue( i, obj ) )
  {
    out.resize ( 1 );

    out[0] = i;

    return;
  }

  if ( isInstance<String>( obj ) )
  {
    out.resize ( 1 );

    out[0] = (int) toInt_

      ( toValue<String>( obj ), minOf( i ), maxOf( i ) );

    return;
  }

  if ( isInstance<StringVector>( obj ) )
  {
    const StringVector&  v = toValue<const StringVector&> ( obj );

    out.resize ( v.size() );

    for ( idx_t j = 0; j < v.size(); j++ )
    {
      out[i] = (int) toInt_ ( v[j], minOf( i ), maxOf( i )  );
    }

    return;
  }

  if ( isNone( obj ) )
  {
    out.resize ( 0 );

    return;
  }

  typeError_ ( "an integer (array), "
               "an expression (array) or none", obj );
}


//-----------------------------------------------------------------------
//   convert (IdxVector)
//-----------------------------------------------------------------------


void ObjectConverter::convert

  ( IdxVector&          out,
    const Ref<Object>&  obj ) const

{
  idx_t  i;

  if ( isInstance<IdxVector>( obj ) )
  {
    out.ref ( toValue<const IdxVector&>( obj ) );

    return;
  }

  if ( isInstance<IntVector>( obj ) )
  {
    IntVector  tmp = toValue<const IntVector&> ( obj );

    out.resize ( tmp.size() );

    for ( idx_t i = 0; i < tmp.size(); i++ )
    {
      out[i] = (idx_t) tmp[i];
    }

    return;
  }

  if ( toValue( i, obj ) )
  {
    out.resize ( 1 );

    out[0] = i;

    return;
  }

  if ( isInstance<String>( obj ) )
  {
    out.resize ( 1 );

    out[0] = (idx_t) toInt_

      ( toValue<String>( obj ), minOf( i ), maxOf( i ) );

    return;
  }

  if ( isInstance<StringVector>( obj ) )
  {
    const StringVector&  v = toValue<const StringVector&> ( obj );

    out.resize ( v.size() );

    for ( idx_t j = 0; j < v.size(); j++ )
    {
      out[i] = (idx_t) toInt_ ( v[j], minOf( i ), maxOf( i )  );
    }

    return;
  }

  if ( isNone( obj ) )
  {
    out.resize ( 0 );

    return;
  }

  typeError_ ( "an integer (array), "
               "an expression (array) or none", obj );
}


//-----------------------------------------------------------------------
//   convert (Vector)
//-----------------------------------------------------------------------


void ObjectConverter::convert

  ( Vector&             out,
    const Ref<Object>&  obj ) const

{
  double  x;

  if ( isInstance<Vector>( obj ) )
  {
    out.ref ( toValue<const Vector&>( obj ) );

    return;
  }

  if ( toValue( x, obj ) )
  {
    out.resize ( 1 );

    out[0] = x;

    return;
  }

  if ( isInstance<String>( obj ) )
  {
    out.resize ( 1 );

    out[0] = FuncUtils::evalExpr

      ( toValue<String>( obj ), globdat_ );

    return;
  }

  if ( isInstance<StringVector>( obj ) )
  {
    const StringVector&  v = toValue<const StringVector&> ( obj );

    out.resize ( v.size() );

    for ( idx_t i = 0; i < v.size(); i++ )
    {
      out[i] = FuncUtils::evalExpr ( v[i], globdat_ );
    }

    return;
  }

  if ( isNone( obj ) )
  {
    out.resize ( 0 );

    return;
  }

  typeError_ ( "a double precision number (array), "
               "an expression (array) or none", obj );
}


//-----------------------------------------------------------------------
//   convert (StringVector)
//-----------------------------------------------------------------------


void ObjectConverter::convert

  ( StringVector&       out,
    const Ref<Object>&  obj ) const

{
  if ( isInstance<StringVector>( obj ) )
  {
    out.ref ( toValue<const StringVector&>( obj ) );

    return;
  }

  if ( isInstance<String>( obj ) )
  {
    out.resize ( 1 );

    out[0] = toValue<String> ( obj );

    return;
  }

  if ( isNone( obj ) )
  {
    out.resize ( 0 );

    return;
  }

  typeError_ ( "a string (array) or none", obj );
}


//-----------------------------------------------------------------------
//   toBool_
//-----------------------------------------------------------------------


bool ObjectConverter::toBool_ ( const String& str ) const
{
  double  x = FuncUtils::evalExpr ( str, globdat_ );

  return (x * x > 0.25);
}


//-----------------------------------------------------------------------
//   toInt_
//-----------------------------------------------------------------------


lint ObjectConverter::toInt_

  ( const String&  str,
    lint           minval,
    lint           maxval ) const

{
  double  x = FuncUtils::evalExpr ( str, globdat_ );

  if ( x < (double) minval || x > (double) maxval )
  {
    throw IllegalInputException (
      CLASS_NAME,
      String::format (
        "integer overflow; valid range is [%d,%d]",
        minval,
        maxval
      )
    );
  }

  return (lint) x;
}


//-----------------------------------------------------------------------
//   toFloat_
//-----------------------------------------------------------------------


double ObjectConverter::toFloat_

  ( const String&  str,
    double         maxval ) const

{
  double  x = FuncUtils::evalExpr ( str, globdat_ );

  if ( x < -maxval || x > maxval )
  {
    throw IllegalInputException (
      CLASS_NAME,
      String::format (
        "floating point overflow; valid range is [%e,%e]",
        -maxval,
        maxval
      )
    );
  }

  return x;
}


//-----------------------------------------------------------------------
//   typeError_
//-----------------------------------------------------------------------


void ObjectConverter::typeError_

  ( const String&       expected,
    const Ref<Object>&  obj )

{
  String  whatis;

  if ( ! obj )
  {
    whatis = "nullptr";
  }
  else
  {
    whatis = obj->getClass()->getWhatis ();
  }

  throw IllegalInputException (
    CLASS_NAME,
    String::format (
      "type mismatch; expected %s, but got %s",
      expected,
      whatis
    )
  );
}


JIVE_END_PACKAGE( util )
