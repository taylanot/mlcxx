
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


#include <jem/base/Error.h>
#include <jem/base/Float.h>
#include <jem/base/Integer.h>
#include <jem/base/Boolean.h>
#include <jem/base/Character.h>
#include <jem/base/StringObject.h>
#include <jem/base/Clonable.h>
#include <jem/base/TypeTraits.h>
#include <jem/base/NullPointerException.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/base/TypeTraits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/Object.h>


template class jem::ClassRegisterAgent<jem::Object>;


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class Object
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Object::CLASS_NAME = "jem::Object";


//-----------------------------------------------------------------------
//   makeClass
//-----------------------------------------------------------------------


Class* Object::makeClass ()
{
  return new ClassTemplate<Self> ( CLASS_NAME );
}


//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


Class* Object::getType ()
{
  return ClassTemplate<Self>::getInstance ();
}


//-----------------------------------------------------------------------
//   getClass
//-----------------------------------------------------------------------


Class* Object::getClass () const
{
  return ClassTemplate<Self>::getInstance ();
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String Object::toString () const
{
  return ( getClass()->getName() + "@" + String( hashValue() ) );
}


//-----------------------------------------------------------------------
//   hashValue
//-----------------------------------------------------------------------


lint Object::hashValue () const
{
  return TypeTraits<void*>::hashValue ( (void*) this );
}


//-----------------------------------------------------------------------
//   equals
//-----------------------------------------------------------------------


bool Object::equals ( const Ref<Object>& obj ) const
{
  if ( obj.getType() == Ref<Object>::OBJECT_TYPE )
  {
    return (this == &(obj.getObject()));
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> Object::clone () const
{
  const Clonable*  c = dynamic_cast<const Clonable*> ( this );

  if ( c )
  {
    return c->clone ();
  }

  String  className = getClass()->getName ();

  throw IllegalOperationException (
    JEM_FUNC,
    String::format ( "class `%s\' is not clonable", className )
  );
}


//=======================================================================
//   class Ref<Object>
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


Ref<Object>::Ref ( const Ref& rhs ) noexcept
{
  type_ = rhs.type_;
  data_ = rhs.data_;

  if ( type_ == OBJECT_TYPE )
  {
    Collectable::addRef ( *data_.object );
  }
}


Ref<Object>::Ref ( Ref&& rhs ) noexcept
{
  type_            = rhs.type_;
  data_            = rhs.data_;
  rhs.type_        = NIL_TYPE;
  rhs.data_.object = nullptr;
}


Ref<Object>::Ref

  ( const String&  rhs,
    const RawRef&  raw )

{
  const idx_t  len = rhs.size ();

  if ( len <= TinyString::MAXLEN )
  {
    const char*  str = rhs.addr ();

    data_.strptr = nullptr;

    for ( idx_t i = 0; i < len; i++ )
    {
      data_.string[i] = str[i];
    }

    type_ = STRING_TYPE;
  }
  else
  {
    Ref<Object>  tmp = newInstance<StringObject> ( rhs );

    type_            = OBJECT_TYPE;
    data_.object     = tmp.data_.object;
    tmp.type_        = NIL_TYPE;
    tmp.data_.object = nullptr;
  }
}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


Ref<Object>& Ref<Object>::operator = ( const Ref& rhs ) noexcept
{
  if ( rhs.type_ == OBJECT_TYPE )
  {
    Collectable::addRef ( *rhs.data_.object );
  }

  if ( type_    == OBJECT_TYPE )
  {
    Collectable::delRef ( *data_.object );
  }

  type_ = rhs.type_;
  data_ = rhs.data_;

  return *this;
}


Ref<Object>& Ref<Object>::operator = ( Object* rhs ) noexcept
{
  TypeID  rhsType;

  if ( rhs )
  {
    rhsType = OBJECT_TYPE;

    Collectable::addRef ( *rhs );
  }
  else
  {
    rhsType = NIL_TYPE;
  }

  if ( type_ == OBJECT_TYPE )
  {
    Collectable::delRef ( *data_.object );
  }

  type_        = rhsType;
  data_.object = rhs;

  return *this;
}


Ref<Object>& Ref<Object>::operator = ( std::nullptr_t ) noexcept
{
  if ( type_ == OBJECT_TYPE )
  {
    Collectable::delRef ( *data_.object );
  }

  type_        = NIL_TYPE;
  data_.object = nullptr;

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


void Ref<Object>::swap ( Ref& rhs ) noexcept
{
  const TypeID  rhsType = rhs.type_;
  const Data_   rhsData = rhs.data_;

  rhs.type_ = type_;
  rhs.data_ = data_;
  type_     = rhsType;
  data_     = rhsData;
}


//-----------------------------------------------------------------------
//   getClass
//-----------------------------------------------------------------------


Class* Ref<Object>::getClass () const noexcept
{
  switch ( type_ )
  {
  case NIL_TYPE:

    return nullptr;

  case BOOL_TYPE:

    return Boolean      :: getType  ();

  case CHAR_TYPE:

    return Character    :: getType  ();

  case STRING_TYPE:

    return StringObject :: getType  ();

  case INT_TYPE:

    return Integer      :: getType  ();

  case FLOAT_TYPE:

    return Float        :: getType  ();

  case OBJECT_TYPE:

    return data_.object -> getClass ();
  }

  return nullptr;
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String Ref<Object>::toString () const
{
  switch ( type_ )
  {
  case NIL_TYPE:

    return String ( "NIL" );

  case BOOL_TYPE:

    return String ( getBool() );

  case CHAR_TYPE:

    return String ( getChar() );

  case STRING_TYPE:

    return getString ();

  case INT_TYPE:

    return String ( getInt() );

  case FLOAT_TYPE:

    return String ( getFloat() );

  case OBJECT_TYPE:

    return data_.object->toString ();
  }

  return String ();
}


//-----------------------------------------------------------------------
//   hashValue
//-----------------------------------------------------------------------


lint Ref<Object>::hashValue () const
{
  switch ( type_ )
  {
  case NIL_TYPE:

    return 0_lint;

  case BOOL_TYPE:
  case CHAR_TYPE:

    return data_.ivalue;

  case STRING_TYPE:

    return getString().hashValue ();

  case INT_TYPE:

    return data_.ivalue;

  case FLOAT_TYPE:

    return TypeTraits<double>::hashValue ( data_.xvalue );

  case OBJECT_TYPE:

    return data_.object->hashValue ();
  }

  return 0_lint;
}


//-----------------------------------------------------------------------
//   equals
//-----------------------------------------------------------------------


bool Ref<Object>::equals ( const Ref& rhs ) const
{
  if ( type_ == rhs.type_ )
  {
    switch ( type_ )
    {
    case NIL_TYPE:

      return true;

    case BOOL_TYPE:
    case CHAR_TYPE:

      return (data_.ivalue == rhs.data_.ivalue);

    case STRING_TYPE:

      return (data_.strptr == rhs.data_.strptr);

    case INT_TYPE:

      return (data_.ivalue == rhs.data_.ivalue);

    case FLOAT_TYPE:

      return (data_.xvalue == rhs.data_.xvalue);

    case OBJECT_TYPE:

      return data_.object->equals ( rhs );
    }
  }

  if ( type_ == OBJECT_TYPE )
  {
    return data_.object->equals ( rhs );
  }

  if ( rhs.type_ == OBJECT_TYPE )
  {
    return rhs.data_.object->equals ( *this );
  }

  return false;
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> Ref<Object>::clone () const
{
  if ( type_ == OBJECT_TYPE )
  {
    return data_.object->clone ();
  }
  else
  {
    return *this;
  }
}


//-----------------------------------------------------------------------
//   dynamicCast
//-----------------------------------------------------------------------


Object* Ref<Object>::dynamicCast

  ( Object&  rhs,
    Class*   target )

{
  Class*  type = rhs.getClass ();

  if ( type == target || type->getSuperClass() == target )
  {
    return & rhs;
  }

  type = type->getSuperClass ();

  while ( type )
  {
    if ( type == target )
    {
      return & rhs;
    }

    type = type->getSuperClass ();
  }

  return (Object*) nullptr;
}


//-----------------------------------------------------------------------
//   toObject_
//-----------------------------------------------------------------------


void Ref<Object>::toObject_ () const
{
  Ref<Object>  tmp;

  switch ( type_ )
  {
  case BOOL_TYPE:

    tmp = newInstance<Boolean>      ( getBool() );

    break;

  case CHAR_TYPE:

    tmp = newInstance<Character>    ( getChar() );

    break;

  case STRING_TYPE:

    tmp = newInstance<StringObject> ( getString() );

    break;

  case INT_TYPE:

    tmp = newInstance<Integer>      ( getInt() );

    break;

  case FLOAT_TYPE:

    tmp = newInstance<Float>        ( getFloat() );

    break;

  default:

    throw Error (
      JEM_FUNC,
      String::format ( "unexpected object type: %d", (int) type_ )
    );
  }

  Ref&  self        = const_cast<Ref&> ( *this );

  self.type_        = OBJECT_TYPE;
  self.data_.object = tmp.data_.object;
  tmp .type_        = NIL_TYPE;
  tmp .data_.object = nullptr;
}


//-----------------------------------------------------------------------
//   nullError_
//-----------------------------------------------------------------------


void Ref<Object>::nullError_ ()
{
  throw NullPointerException (
    JEM_FUNC,
    "attempt to dereference a NULL Object pointer"
  );
}


JEM_END_PACKAGE_BASE
