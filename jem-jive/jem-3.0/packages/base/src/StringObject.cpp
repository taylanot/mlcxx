
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


#include <jem/base/ClassTemplate.h>
#include <jem/base/StringObject.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>


JEM_DEFINE_CUSTOM_CLASS( jem::StringObject );


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class StringObject
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


StringObject::~StringObject ()
{}


//-----------------------------------------------------------------------
//   makeClass
//-----------------------------------------------------------------------


Class* StringObject::makeClass ()
{
  ClassTemplate<Self>*  myClass = new ClassTemplate<Self> ();

  myClass->setWhatis      ( "a string" );
  myClass->addDefaultCtor ();

  return myClass;
}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StringObject::readFrom ( io::ObjectInput& in )
{
  decode ( in, value );
}


void StringObject::writeTo ( io::ObjectOutput& out ) const
{
  encode ( out, value );
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String StringObject::toString () const
{
  return value;
}


//-----------------------------------------------------------------------
//   hashValue
//-----------------------------------------------------------------------


lint StringObject::hashValue () const
{
  return value.hashValue ();
}


//-----------------------------------------------------------------------
//   equals
//-----------------------------------------------------------------------


bool StringObject::equals ( const Ref<Object>& rhs ) const
{
  if ( rhs.getType() == Ref<Object>::STRING_TYPE )
  {
    return (value == rhs.getString());
  }

  if ( rhs.getType() == Ref<Object>::OBJECT_TYPE )
  {
    StringObject*  s = dynamicCast ( rhs.getObject() );

    if ( s )
    {
      return (value == s->value);
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> StringObject::clone () const
{
  return newInstance<Self> ( value );
}


//-----------------------------------------------------------------------
//   dynamicCast
//-----------------------------------------------------------------------


StringObject* StringObject::dynamicCast ( Object& rhs )
{
  return ClassTemplate<Self>::dynamicCast ( rhs );
}


JEM_END_PACKAGE_BASE
