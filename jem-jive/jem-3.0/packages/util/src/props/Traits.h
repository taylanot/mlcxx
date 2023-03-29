
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
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
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */

#ifndef JEM_UTIL_PROPS_TRAITS_H
#define JEM_UTIL_PROPS_TRAITS_H

#include <jem/base/rtti.h>
#include <jem/base/limits.h>
#include <jem/base/array/Array.h>
#include <jem/io/PrintWriter.h>
#include <jem/util/None.h>
#include <jem/util/Tokenizer.h>
#include <jem/util/ArrayBuffer.h>
#include "Errors.h"


JEM_BEGIN_PACKAGE( util )
JEM_BEGIN_NAMESPACE( props )


using jem::io::PrintWriter;


//-----------------------------------------------------------------------
//   class Traits (primary template)
//-----------------------------------------------------------------------


template <class T> class Traits {};


//-----------------------------------------------------------------------
//   class Traits<bool>
//-----------------------------------------------------------------------


template <>

  class Traits<bool>

{
 public:

  static inline bool      convert

    ( bool&                 val,
      const Ref<Object>&    obj );

  static inline bool      parse

    ( Tokenizer&            in );

};


//-----------------------------------------------------------------------
//   class Traits<char>
//-----------------------------------------------------------------------


template <>

  class Traits<char>

{
 public:

  static inline bool      convert

    ( char&                 val,
      const Ref<Object>&    obj );

  static inline char      parse

    ( Tokenizer&            in );

};


//-----------------------------------------------------------------------
//   class IntTraits
//-----------------------------------------------------------------------


template <class T>

  class IntTraits

{
 public:

  static inline bool      convert

    ( T&                    val,
      const Ref<Object>&    obj );

  static inline T         parse

    ( Tokenizer&            in );

};


//-----------------------------------------------------------------------
//   class Traits<byte>
//-----------------------------------------------------------------------


template <>

  class Traits<byte> : public IntTraits<byte>

{};


//-----------------------------------------------------------------------
//   class Traits<short>
//-----------------------------------------------------------------------


template <>

  class Traits<short> : public IntTraits<short>

{};


//-----------------------------------------------------------------------
//   class Traits<int>
//-----------------------------------------------------------------------


template <>

  class Traits<int> : public IntTraits<int>

{};


//-----------------------------------------------------------------------
//   class Traits<long>
//-----------------------------------------------------------------------


template <>

  class Traits<long> : public IntTraits<long>

{};


//-----------------------------------------------------------------------
//   class Traits<lint>
//-----------------------------------------------------------------------


template <>

  class Traits<lint>

{
 public:

  static inline bool      convert

    ( lint&                 val,
      const Ref<Object>&    obj );

  static inline lint      parse

    ( Tokenizer&            in );

};


//-----------------------------------------------------------------------
//   class Traits<idx_t>
//-----------------------------------------------------------------------


template <>

  class Traits<idx_t> : public IntTraits<idx_t>

{};


//-----------------------------------------------------------------------
//   class FloatTraits
//-----------------------------------------------------------------------


template <class T>

  class FloatTraits

{
 public:

  static inline bool      convert

    ( T&                    val,
      const Ref<Object>&    obj );

  static inline T         parse

    ( Tokenizer&            in );

};


//-----------------------------------------------------------------------
//   class Traits<float>
//-----------------------------------------------------------------------


template <>

  class Traits<float> : public FloatTraits<float>

{};


//-----------------------------------------------------------------------
//   class Traits<double>
//-----------------------------------------------------------------------


template <>

  class Traits<double>

{
 public:

  static inline bool    convert

    ( double&             val,
      const Ref<Object>&  obj );

  static inline double  parse

    ( Tokenizer&          in );

};


//-----------------------------------------------------------------------
//   class Traits<String>
//-----------------------------------------------------------------------


template <>

  class Traits<String>

{
 public:

  static inline bool      convert

    ( String&               val,
      const Ref<Object>&    obj );

  static inline String    parse

    ( Tokenizer&            in );

};


//-----------------------------------------------------------------------
//   class Traits< Array<T> >
//-----------------------------------------------------------------------


template <class T>

  class Traits< Array<T> >

{
 public:

  static inline bool      convert

    ( Array<T>&             val,
      const Ref<Object>&    obj );

  static Array<T>         parse

    ( Tokenizer&            in );

};


//-----------------------------------------------------------------------
//   class Traits< Array<idx_t> >
//-----------------------------------------------------------------------


template <>

  class Traits< Array<idx_t> >

{
 public:

  static inline bool      convert

    ( Array<idx_t>&         val,
      const Ref<Object>&    obj );

};




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Traits<bool>
//=======================================================================

//-----------------------------------------------------------------------
//   convert
//-----------------------------------------------------------------------


inline bool Traits<bool>::convert

  ( bool&               val,
    const Ref<Object>&  obj )

{
  if ( toValue( val, obj ) )
  {
    return true;
  }

  if ( isNone( obj ) )
  {
    val = false;

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


inline bool Traits<bool>::parse ( Tokenizer& in )
{
  if ( in.nextToken() == Tokenizer::BOOLEAN_TOKEN )
  {
    return in.getBoolean ();
  }

  in.parseError ( "expected a boolean" );

  return false;
}


//=======================================================================
//   class Traits<char>
//=======================================================================

//-----------------------------------------------------------------------
//   convert
//-----------------------------------------------------------------------


inline bool Traits<char>::convert

  ( char&               val,
    const Ref<Object>&  obj )

{
  if ( toValue( val, obj ) )
  {
    return true;
  }

  if ( isNone( obj ) )
  {
    val = '\0';

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


inline char Traits<char>::parse ( Tokenizer& in )
{
  if ( in.nextToken() == Tokenizer::STRING_TOKEN )
  {
    String  tmp = in.getString ();

    if ( tmp.size() == 1 )
    {
      return tmp[0];
    }
  }

  in.parseError ( "expected a character" );

  return '\0';
}


//=======================================================================
//   class IntTraits
//=======================================================================

//-----------------------------------------------------------------------
//   convert
//-----------------------------------------------------------------------


template <class T>

  inline bool IntTraits<T>::convert

  ( T&                  val,
    const Ref<Object>&  obj )

{
  if ( toValue( val, obj ) )
  {
    return true;
  }

  if ( isNone( obj ) )
  {
    val = T ();

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


template <class T>

  inline T IntTraits<T>::parse ( Tokenizer& in )

{
  lint  i = Traits<lint>::parse ( in );

  if ( i < (lint) minOf<T>() || i > (lint) maxOf<T>() )
  {
    in.rangeError ( i, minOf<T>(), maxOf<T>() );
  }

  return (T) i;
}


//=======================================================================
//   class Traits<lint>
//=======================================================================

//-----------------------------------------------------------------------
//   convert
//-----------------------------------------------------------------------


inline bool Traits<lint>::convert

  ( lint&               val,
    const Ref<Object>&  obj )

{
  if ( toValue( val, obj ) )
  {
    return true;
  }

  if ( isNone( obj ) )
  {
    val = 0_lint;

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


inline lint Traits<lint>::parse ( Tokenizer& in )
{
  if ( in.nextToken() == Tokenizer::INTEGER_TOKEN )
  {
    return in.getInteger ();
  }

  in.parseError ( "expected an integer" );

  return 0_lint;
}


//=======================================================================
//   class FloatTraits
//=======================================================================

//-----------------------------------------------------------------------
//   convert
//-----------------------------------------------------------------------


template <class T>

  inline bool FloatTraits<T>::convert

  ( T&                  val,
    const Ref<Object>&  obj )

{
  if ( toValue( val, obj ) )
  {
    return true;
  }

  if ( isNone( obj ) )
  {
    val = T ();

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


template <class T>

  inline T FloatTraits<T>::parse ( Tokenizer& in )

{
  double  x = Traits<double>::parse ( in );

  if ( x > (double) maxOf<T>() || -x > (double) maxOf<T>() )
  {
    in.rangeError ( x, maxOf<T>() );
  }

  return (T) x;
}


//=======================================================================
//   class Traits<double>
//=======================================================================


//-----------------------------------------------------------------------
//   convert
//-----------------------------------------------------------------------


inline bool Traits<double>::convert

  ( double&             val,
    const Ref<Object>&  obj )

{
  if ( toValue( val, obj ) )
  {
    return true;
  }

  if ( isNone( obj ) )
  {
    val = 0.0;

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


inline double Traits<double>::parse ( Tokenizer& in )
{
  if ( in.nextToken() == Tokenizer::FLOAT_TOKEN )
  {
    return in.getFloat ();
  }

  if ( in.nextToken() == Tokenizer::INTEGER_TOKEN )
  {
    return (double) in.getInteger ();
  }

  in.parseError ( "expected a floating point number" );

  return 0.0;
}


//=======================================================================
//   class Traits<String>
//=======================================================================

//-----------------------------------------------------------------------
//   convert
//-----------------------------------------------------------------------


inline bool Traits<String>::convert

  ( String&             val,
    const Ref<Object>&  obj )

{
  if ( toValue( val, obj ) )
  {
    return true;
  }

  if ( isNone( obj ) )
  {
    val = String ();

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


inline String Traits<String>::parse ( Tokenizer& in )
{
  if ( in.nextToken() == Tokenizer::STRING_TOKEN )
  {
    return in.getString ();
  }

  in.parseError ( "expected a string" );

  return String ();
}


//=======================================================================
//   class Traits< Array<T> >
//=======================================================================

//-----------------------------------------------------------------------
//   convert
//-----------------------------------------------------------------------


template <class T>

  inline bool Traits< Array<T> >::convert

  ( Array<T>&           val,
    const Ref<Object>&  obj )

{
  if ( isInstance< Array<T> >( obj ) )
  {
    val.ref ( toValue< const Array<T>& >( obj ) );

    return true;
  }

  if ( isNone( obj ) )
  {
    val.ref ( Array<T>() );

    return true;
  }

  // Try to convert from a scalar value.

  T  tmp;

  if ( Traits<T>::convert( tmp, obj ) )
  {
    val.ref ( Array<T>( 1 ) );

    val[0] = tmp;

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


template <class T>

  Array<T> Traits< Array<T> >::parse ( Tokenizer& in )

{
  ArrayBuffer<T>  buf;

  buf.reserve ( 3 );

  while ( true )
  {
    if ( buf.size() == maxOf( buf.size() ) )
    {
      Errors::arraySizeOverflow ( in.getLineNumber() );
    }

    buf.pushBack ( Traits<T>::parse( in ) );

    int  token = in.nextToken ();

    if ( token == Tokenizer::CLOSE_BRACKET_TOKEN )
    {
      break;
    }

    if ( token == Tokenizer::COMMA_TOKEN )
    {
      continue;
    }

    in.parseError ( "expected a `]\' or `,\'" );
  }

  return buf.toArray ();
}


//=======================================================================
//   class Traits< Array<idx_t> >
//=======================================================================

//-----------------------------------------------------------------------
//   convert
//-----------------------------------------------------------------------


inline bool Traits< Array<idx_t> >::convert

  ( Array<idx_t>&       val,
    const Ref<Object>&  obj )

{
  if ( isInstance< Array<idx_t> >( obj ) )
  {
    val.ref ( toValue< const Array<idx_t>& >( obj ) );

    return true;
  }

  if ( isInstance< Array<int> >( obj ) )
  {
    Array<int>  ia ( toValue< const Array<int>& >( obj ) );

    val.resize ( ia.size() );

    for ( idx_t i = 0; i < ia.size(); i++ )
    {
      val[i] = (idx_t) ia[i];
    }

    return true;
  }

  if ( isNone( obj ) )
  {
    val.ref ( Array<idx_t>() );

    return true;
  }

  // Try to convert from a scalar value.

  idx_t  tmp;

  if ( Traits<idx_t>::convert( tmp, obj ) )
  {
    val.ref ( Array<idx_t>( 1 ) );

    val[0] = tmp;

    return true;
  }

  return false;
}



JEM_END_NAMESPACE( props )
JEM_END_PACKAGE( util )

#endif
