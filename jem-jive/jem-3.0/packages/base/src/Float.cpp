
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


#include <cmath>
#include <cstdlib>
#include <cctype>
#include <cfloat>
#include <cstdio>
#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/Float.h>
#include <jem/base/ParseException.h>
#include <jem/base/TypeTraits.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>


JEM_DEFINE_CUSTOM_CLASS( jem::Float );


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class Float
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const double  Float::MIN_VALUE = DBL_MIN;
const double  Float::MAX_VALUE = DBL_MAX;
const double  Float::EPSILON   = DBL_EPSILON;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Float::Float ( const String& s ) : value ( parse( s ) )
{}


Float::~Float ()
{}


//-----------------------------------------------------------------------
//   makeClass
//-----------------------------------------------------------------------


Class* Float::makeClass ()
{
  ClassTemplate<Self>*  myClass = new ClassTemplate<Self> ();

  myClass->setWhatis      ( "a floating point number" );
  myClass->addDefaultCtor ();

  return myClass;
}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void Float::readFrom ( io::ObjectInput& in )
{
  decode ( in, value );
}


void Float::writeTo ( io::ObjectOutput& out ) const
{
  encode ( out, value );
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String Float::toString () const
{
  return String ( value );
}


//-----------------------------------------------------------------------
//   hashValue
//-----------------------------------------------------------------------


lint Float::hashValue () const
{
  return TypeTraits<double>::hashValue ( value );
}


//-----------------------------------------------------------------------
//   equals
//-----------------------------------------------------------------------


bool Float::equals ( const Ref<Object>& rhs ) const
{
  if ( rhs.getType() == Ref<Object>::FLOAT_TYPE )
  {
    return isTiny ( value - rhs.getFloat() );
  }

  if ( rhs.getType() == Ref<Object>::OBJECT_TYPE )
  {
    Float*  f = dynamicCast ( rhs.getObject() );

    if ( f )
    {
      return isTiny ( value - f->value );
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> Float::clone () const
{
  return newInstance<Self> ( value );
}


//-----------------------------------------------------------------------
//   toInteger & toFloat
//-----------------------------------------------------------------------


lint Float::toInteger () const
{
  return (lint) value;
}


double Float::toFloat () const
{
  return value;
}


//-----------------------------------------------------------------------
//   isFloat
//-----------------------------------------------------------------------


bool Float::isFloat () const
{
  return true;
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


double Float::parse ( const String& s )
{
  double  val = 0.0;

  if ( ! parse( val, s ) )
  {
    throw ParseException (
      JEM_FUNC,
      String::format ( "invalid floating point number: `%s\'", s )
    );
  }

  return val;
}


bool Float::parse

  ( float&         value,
    const String&  s )

{
  double  xtmp;

  if ( ! parse( xtmp, s ) )
  {
    return false;
  }

  if ( std::fabs( xtmp ) > (double) FLT_MAX )
  {
    return false;
  }

  value = (float) xtmp;

  return true;
}


bool Float::parse

  ( double&        value,
    const String&  s )

{
  const idx_t  len     = s.size ();
  const int    BUFSIZE = 64;

  char         buf[BUFSIZE + 1];
  char*        end;
  double       tmp;
  idx_t        i;
  int          j;

  value = 0;
  i = j = 0;

  while ( i < len && std::isspace(s[i]) )
  {
    i++;
  }
  while ( i < len && j < BUFSIZE && ! std::isspace(s[i]) )
  {
    buf[j++] = s[i++];
  }

  buf[j] = '\0';

  tmp = strtod ( buf, & end );

  if ( *buf == '\0' || *end != '\0' )
  {
    return false;
  }
  else
  {
    value = tmp;
    return true;
  }
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


idx_t Float::print ( char* dest, idx_t n, double value )
{
  const int  BUFSIZE = 64;

  char       buffer[BUFSIZE];

  if ( n < BUFSIZE )
  {
    int  k = std::sprintf ( buffer, "%#.4g", value );

    if ( k > n )
    {
      return 0;
    }
    else
    {
      std::memcpy ( dest, buffer, (size_t) k );
      return k;
    }
  }
  else
  {
    return std::sprintf ( dest, "%#.4g", value );
  }
}


//-----------------------------------------------------------------------
//   isNaN
//-----------------------------------------------------------------------


bool Float::isNaN ( float val ) noexcept
{
  // Use a temporary, volatile variable to make sure that the
  // comparison below is not optimized away.

  volatile float  tmp = val;

  return (bool) (tmp != tmp);
}


bool Float::isNaN ( double val ) noexcept
{
  // Use a temporary, volatile variable to make sure that the
  // comparison below is not optimized away.

  volatile double  tmp = val;

  return (bool) (tmp != tmp);
}


//-----------------------------------------------------------------------
//   isTiny
//-----------------------------------------------------------------------


bool Float::isTiny ( float val ) noexcept
{
  return jem::isTiny ( val );
}


bool Float::isTiny ( double val ) noexcept
{
  return jem::isTiny ( val );
}


//-----------------------------------------------------------------------
//   dynamicCast
//-----------------------------------------------------------------------


Float* Float::dynamicCast ( Object& rhs )
{
  return ClassTemplate<Self>::dynamicCast ( rhs );
}


JEM_END_PACKAGE_BASE
