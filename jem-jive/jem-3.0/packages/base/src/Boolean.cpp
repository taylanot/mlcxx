
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


#include <jem/base/Boolean.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/ParseException.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>


JEM_DEFINE_CUSTOM_CLASS( jem::Boolean );


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class Boolean
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Boolean::Boolean ( const String& s ) : value ( parse( s ) )
{}


Boolean::~Boolean ()
{}


//-----------------------------------------------------------------------
//   makeClass
//-----------------------------------------------------------------------


Class* Boolean::makeClass ()
{
  ClassTemplate<Self>*  myClass = new ClassTemplate<Self> ();

  myClass->setWhatis      ( "a boolean" );
  myClass->addDefaultCtor ();

  return myClass;
}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void Boolean::readFrom ( io::ObjectInput& in )
{
  decode ( in, value );
}


void Boolean::writeTo ( io::ObjectOutput& out ) const
{
  encode ( out, value );
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String Boolean::toString () const
{
  if ( value )
  {
    return "true";
  }
  else
  {
    return "false";
  }
}


//-----------------------------------------------------------------------
//   hashValue
//-----------------------------------------------------------------------


lint Boolean::hashValue () const
{
  return (value ? 1_lint : 0_lint);
}


//-----------------------------------------------------------------------
//   equals
//-----------------------------------------------------------------------


bool Boolean::equals ( const Ref<Object>& rhs ) const
{
  if ( rhs.getType() == Ref<Object>::BOOL_TYPE )
  {
    return (value == rhs.getBool());
  }

  if ( rhs.getType() == Ref<Object>::OBJECT_TYPE )
  {
    Boolean*  b = dynamicCast ( rhs.getObject() );

    if ( b )
    {
      return (value == b->value);
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> Boolean::clone () const
{
  return newInstance<Self> ( value );
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


bool Boolean::parse ( const String& s )
{
  bool  val = false;

  if ( ! parse( val, s ) )
  {
    throw ParseException (
      JEM_FUNC,
      String::format ( "invalid boolean: `%s\'", s )
    );
  }

  return val;
}


bool Boolean::parse ( bool& value, const String& s )
{
  const idx_t  n = s.size ();
  bool         r = true;


  if      ( n == 1 && s[0] == '0' )
  {
    value = true;
  }
  else if ( n == 1 && s[0] == '1' )
  {
    value = true;
  }
  else if ( n == 4 && s.equalsIgnoreCase( "true" ) )
  {
    value = true;
  }
  else if ( n == 5 && s.equalsIgnoreCase( "false" ) )
  {
    value = false;
  }
  else
  {
    r = false;
  }

  return r;
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


idx_t Boolean::print ( char* buf, idx_t n, bool value )
{
  if ( value )
  {
    if ( n < 4 )
    {
      return 0;
    }
    else
    {
      buf[0] = 't'; buf[1] = 'r'; buf[2] = 'u'; buf[3] = 'e';
      return 4;
    }
  }
  else
  {
    if ( n < 5 )
    {
      return 0;
    }
    else
    {
      buf[0] = 'f'; buf[1] = 'a'; buf[2] = 'l';
      buf[3] = 's'; buf[4] = 'e';
      return 5;
    }
  }
}


//-----------------------------------------------------------------------
//   dynamicCast
//-----------------------------------------------------------------------


Boolean* Boolean::dynamicCast ( Object& rhs )
{
  return ClassTemplate<Self>::dynamicCast ( rhs );
}


JEM_END_PACKAGE_BASE
