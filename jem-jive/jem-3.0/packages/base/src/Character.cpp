
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


#include <climits>
#include <jem/base/Character.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>


JEM_DEFINE_CUSTOM_CLASS( jem::Character );


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class Character
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Character::LOWERCASE_DIGITS = "0123456789abcdef";
const char*  Character::UPPERCASE_DIGITS = "0123456789ABCDEF";
const char   Character::MIN_VALUE        = CHAR_MIN;
const char   Character::MAX_VALUE        = CHAR_MAX;


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


Character::~Character ()
{}


//-----------------------------------------------------------------------
//   makeClass
//-----------------------------------------------------------------------


Class* Character::makeClass ()
{
  ClassTemplate<Self>*  myClass = new ClassTemplate<Self> ();

  myClass->setWhatis      ( "a character" );
  myClass->addDefaultCtor ();

  return myClass;
}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void Character::readFrom ( io::ObjectInput& in )
{
  decode ( in, value );
}


void Character::writeTo ( io::ObjectOutput& out ) const
{
  encode ( out, value );
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String Character::toString () const
{
  return String ( value );
}


//-----------------------------------------------------------------------
//   hashValue
//-----------------------------------------------------------------------


lint Character::hashValue () const
{
  return value;
}


//-----------------------------------------------------------------------
//   equals
//-----------------------------------------------------------------------


bool Character::equals ( const Ref<Object>& rhs ) const
{
  if ( rhs.getType() == Ref<Object>::CHAR_TYPE )
  {
    return (value == rhs.getChar());
  }

  if ( rhs.getType() == Ref<Object>::OBJECT_TYPE )
  {
    Character*  c = dynamicCast ( rhs.getObject() );

    if ( c )
    {
      return (value == c->value);
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> Character::clone () const
{
  return newInstance<Self> ( value );
}


//-----------------------------------------------------------------------
//   dynamicCast
//-----------------------------------------------------------------------


Character* Character::dynamicCast ( Object& rhs )
{
  return ClassTemplate<Self>::dynamicCast ( rhs );
}


JEM_END_PACKAGE_BASE
