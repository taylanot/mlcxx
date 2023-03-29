
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


#include <cstdlib>
#include <cctype>
#include <climits>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/Character.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/Integer.h>
#include <jem/base/ParseException.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>


JEM_DEFINE_CUSTOM_CLASS( jem::Integer );


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class Integer
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const lint  Integer::MIN_VALUE = LLONG_MIN;
const lint  Integer::MAX_VALUE = LLONG_MAX;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Integer::Integer ( const String& s ) : value ( parse( s ) )
{}


Integer::~Integer ()
{}


//-----------------------------------------------------------------------
//   makeClass
//-----------------------------------------------------------------------


Class* Integer::makeClass ()
{
  ClassTemplate<Self>*  myClass = new ClassTemplate<Self> ();

  myClass->setWhatis      ( "an integer" );
  myClass->addDefaultCtor ();

  return myClass;
}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void Integer::readFrom ( io::ObjectInput& in )
{
  decode ( in, value );
}


void Integer::writeTo ( io::ObjectOutput& out ) const
{
  encode ( out, value );
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String Integer::toString () const
{
  return String ( value );
}


//-----------------------------------------------------------------------
//   hashValue
//-----------------------------------------------------------------------


lint Integer::hashValue () const
{
  return value;
}


//-----------------------------------------------------------------------
//   equals
//-----------------------------------------------------------------------


bool Integer::equals ( const Ref<Object>& rhs ) const
{
  if ( rhs.getType() == Ref<Object>::INT_TYPE )
  {
    return (value == rhs.getInt());
  }

  if ( rhs.getType() == Ref<Object>::OBJECT_TYPE )
  {
    Integer*  i = dynamicCast ( rhs.getObject() );

    if ( i )
    {
      return (value == i->value);
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> Integer::clone () const
{
  return newInstance<Self> ( value );
}


//-----------------------------------------------------------------------
//   toInteger & toFloat
//-----------------------------------------------------------------------


lint Integer::toInteger () const
{
  return value;
}


double Integer::toFloat () const
{
  return (double) value;
}


//-----------------------------------------------------------------------
//   isFloat
//-----------------------------------------------------------------------


bool Integer::isFloat () const
{
  return false;
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


lint Integer::parse ( const String& s )
{
  lint  val = 0;

  if ( ! parse( val, s ) )
  {
    throw ParseException (
      JEM_FUNC,
      String::format ( "invalid integer: `%s\'", s )
    );
  }

  return val;
}


bool Integer::parse

  ( int&           value,
    const String&  s )

{
  lint  itmp;

  if ( ! parse( itmp, s ) )
  {
    return false;
  }

  if ( (itmp < (lint) minOf( value )) ||
       (itmp > (lint) maxOf( value )) )
  {
    return false;
  }

  value = (int) itmp;

  return true;
}


bool Integer::parse

  ( long&          value,
    const String&  s )

{
  lint  itmp;

  if ( ! parse( itmp, s ) )
  {
    return false;
  }

  if ( (itmp < (lint) minOf( value )) ||
       (itmp > (lint) maxOf( value )) )
  {
    return false;
  }

  value = (long) itmp;

  return true;
}


bool Integer::parse

  ( lint&          value,
    const String&  s )

{
  const idx_t  len     = s.size ();
  const int    BUFSIZE = 64;

  char         buf[BUFSIZE + 1];
  char*        end;
  lint         tmp;
  idx_t        i;
  int          j;

  value = 0;
  i = j = 0;

  while ( i < len && std::isspace(s[i]) )
  {
    i++;
  }

  while ( i < len && j < BUFSIZE && ! std::isspace( s[i] ) )
  {
    buf[j++] = s[i++];
  }

  buf[j] = '\0';

  tmp = std::strtol ( buf, &end, 0 );

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


bool Integer::parse

  ( idx_t&         value,
    const String&  s )

{
  lint  itmp;

  if ( ! parse( itmp, s ) )
  {
    return false;
  }

  if ( (itmp < (lint) minOf( value )) ||
       (itmp > (lint) maxOf( value )) )
  {
    return false;
  }

  value = (idx_t) itmp;

  return true;
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


idx_t Integer::print

  ( char*  buf,
    idx_t  bufsize,
    lint   value,
    int    base )

{
  JEM_PRECHECK2 ( base > 0 && base <= 16, "invalid base" );

  const char*  digits = Character::LOWERCASE_DIGITS;

  lint   quot;
  char   tmp;
  idx_t  i, j, n;


  i = 0;

  if ( value < 0 )
  {
    if ( i >= bufsize )
    {
      return 0;
    }

    buf[i++] = '-';
  }

  if ( base == 16 )
  {
    if ( i - 1 >= bufsize )
    {
      return 0;
    }

    buf[i++] = '0';
    buf[i++] = 'x';
  }

  j = i;

  if ( j >= bufsize )
  {
    return 0;
  }

  do
  {
    quot     = value / base;
    buf[j++] = digits[ ::labs( (long) (value - quot * base) ) ];
    value    = quot;
  }
  while ( j < bufsize && value != 0 );

  if ( value )
  {
    return 0;
  }

  n = j;

  for ( j--; i < j; i++, j-- )
  {
    tmp    = buf[i];
    buf[i] = buf[j];
    buf[j] = tmp;
  }

  return n;
}


//-----------------------------------------------------------------------
//   dynamicCast
//-----------------------------------------------------------------------


Integer* Integer::dynamicCast ( Object& rhs )
{
  return ClassTemplate<Self>::dynamicCast ( rhs );
}


JEM_END_PACKAGE_BASE
