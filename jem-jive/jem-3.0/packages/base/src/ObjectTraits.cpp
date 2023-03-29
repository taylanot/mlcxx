
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
#include <jem/base/limits.h>
#include <jem/base/Boolean.h>
#include <jem/base/Character.h>
#include <jem/base/Integer.h>
#include <jem/base/Float.h>
#include <jem/base/StringObject.h>
#include <jem/base/NullPointerException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/Once.h>
#include <jem/base/ObjectTraits.h>


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class ObjectTraitsUtils
//=======================================================================

//-----------------------------------------------------------------------
//   typeError
//-----------------------------------------------------------------------


void ObjectTraitsUtils::typeError

  ( const ObjRef&  obj,
    Class*         target )

{
  String  targetKind = target->getWhatis ();

  if ( obj )
  {
    String  sourceKind = obj.getClass()->getWhatis ();

    throw IllegalArgumentException (
      JEM_FUNC,
      String::format(
        "argument type mismatch: expected %s but got %s",
        targetKind,
        sourceKind
      )
    );
  }
  else
  {
    throw NullPointerException (
      JEM_FUNC,
      String::format ( "null argument; expected %s", targetKind )
    );
  }
}


//-----------------------------------------------------------------------
//   rangeError
//-----------------------------------------------------------------------


void ObjectTraitsUtils::rangeError

  ( lint  value,
    lint  lbound,
    lint  ubound )

{
  throw IllegalArgumentException (
    JEM_FUNC,
    String::format (
      "integer value (%d) out of range; valid range is "
      "[%d,%d]",
      value,
      lbound,
      ubound
    )
  );
}


void ObjectTraitsUtils::rangeError

  ( double  value,
    double  ubound )

{
  throw IllegalArgumentException (
    JEM_FUNC,
    String::format (
      "float value (%g) out of range; valid range is "
      "[%g,%g]",
      value,
      -ubound,
      ubound
    )
  );
}


//=======================================================================
//   class Void
//=======================================================================


class Void : public Class
{
 public:

                          Void          ();

  virtual bool            isInstance

    ( const Ref<Object>&    obj )          const;

  virtual bool            equalType

    ( const Ref<Object>&    obj )          const override;

  static void             makeInstance  ();


 public:

  static Class*           instance;

};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


Class*  Void::instance = nullptr;


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


Void::Void () : Class ( "jem::Void" )
{}


//-----------------------------------------------------------------------
//   isInstance
//-----------------------------------------------------------------------


bool Void::isInstance ( const Ref<Object>& obj ) const
{
  return false;
}


//-----------------------------------------------------------------------
//   equalType
//-----------------------------------------------------------------------


bool Void::equalType ( const Ref<Object>& obj ) const
{
  return false;
}


//-----------------------------------------------------------------------
//   makeInstance
//-----------------------------------------------------------------------


void Void::makeInstance ()
{
  instance = new Void ();
}


//=======================================================================
//   class ObjectTraits<void>
//=======================================================================

//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


Class* ObjectTraits<void>::getType ()
{
  static Once once = JEM_ONCE_INITIALIZER;

  runOnce ( once, & Void::makeInstance );

  return Void::instance;
}


//-----------------------------------------------------------------------
//   isInstance
//-----------------------------------------------------------------------


bool ObjectTraits<void>::isInstance ( const ObjRef& obj )
{
  return false;
}


//=======================================================================
//   class ObjectTraits<bool>
//=======================================================================

//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


Class* ObjectTraits<bool>::getType ()
{
  return Boolean::getType ();
}


//-----------------------------------------------------------------------
//   toValue_
//-----------------------------------------------------------------------


bool ObjectTraits<bool>::toValue_ ( const ObjRef& obj )
{
  if ( obj.getType() == ObjRef::OBJECT_TYPE )
  {
    Boolean*  b = Boolean::dynamicCast ( obj.getObject() );

    if ( b )
    {
      return b->value;
    }
  }

  typeError ( obj, getType() );

  return false;
}


bool ObjectTraits<bool>::toValue_

  ( bool&          value,
    const ObjRef&  obj )

{
  if ( obj.getType() == ObjRef::OBJECT_TYPE )
  {
    Boolean*  b = Boolean::dynamicCast ( obj.getObject() );

    if ( b )
    {
      value = b->value;

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   isInstance_
//-----------------------------------------------------------------------


bool ObjectTraits<bool>::isInstance_ ( const ObjRef& obj )
{
  if ( obj.getType() == ObjRef::OBJECT_TYPE )
  {
    return (Boolean::dynamicCast( obj.getObject() ) != nullptr);
  }
  else
  {
    return false;
  }
}


//=======================================================================
//   class ObjectTraits<char>
//=======================================================================

//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


Class* ObjectTraits<char>::getType ()
{
  return Character::getType ();
}


//-----------------------------------------------------------------------
//   toValue_
//-----------------------------------------------------------------------


char ObjectTraits<char>::toValue_ ( const ObjRef& obj )
{
  do
  {
    if ( obj.getType() == ObjRef::STRING_TYPE )
    {
      String  s = obj.getString ();

      if ( s.size() == 1_lint )
      {
        return s[0];
      }
      else
      {
        break;
      }
    }

    if ( obj.getType() != ObjRef::OBJECT_TYPE )
    {
      break;
    }

    Character*  c = Character::dynamicCast ( obj.getObject() );

    if ( c )
    {
      return c->value;
    }

    StringObject*  s = StringObject::dynamicCast ( obj.getObject() );

    if ( s )
    {
      if ( s->value.size() == 1_lint )
      {
        return s->value[0];
      }
    }
  }
  while ( false );

  typeError ( obj, getType() );

  return '\0';
}


bool ObjectTraits<char>::toValue_

  ( char&          value,
    const ObjRef&  obj )

{
  if ( obj.getType() == ObjRef::STRING_TYPE )
  {
    String  s = obj.getString ();

    if ( s.size() == 1_lint )
    {
      value = s[0];

      return true;
    }
    else
    {
      return false;
    }
  }

  if ( obj.getType() != ObjRef::OBJECT_TYPE )
  {
    return false;
  }

  Character*  c = Character::dynamicCast ( obj.getObject() );

  if ( c )
  {
    value = c->value;

    return true;
  }

  StringObject*  s = StringObject::dynamicCast ( obj.getObject() );

  if ( s )
  {
    if ( s->value.size() == 1_lint )
    {
      value = s->value[0];

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   isInstance_
//-----------------------------------------------------------------------


bool ObjectTraits<char>::isInstance_ ( const ObjRef& obj )
{
  if ( obj.getType() == ObjRef::STRING_TYPE )
  {
    return (obj.getString().size() == 1_lint);
  }

  if ( obj.getType() != ObjRef::OBJECT_TYPE )
  {
    return false;
  }

  Character*  c = Character::dynamicCast ( obj.getObject() );

  if ( c )
  {
    return true;
  }

  StringObject*  s = StringObject::dynamicCast ( obj.getObject() );

  if ( s )
  {
    return (s->value.size() == 1_lint);
  }

  return false;
}


//=======================================================================
//   class ObjectTraits<byte>
//=======================================================================

//-----------------------------------------------------------------------
//   toValue
//-----------------------------------------------------------------------


byte ObjectTraits<byte>::toValue ( const ObjRef& obj )
{
  lint  i = ObjectTraits<lint>::toValue ( obj );

  if ( i < 0_lint || i > (lint) maxOf<byte>() )
  {
    rangeError ( i, 0_lint, (lint) maxOf<byte>() );
  }

  return (byte) i;
}


bool ObjectTraits<byte>::toValue

  ( byte&          value,
    const ObjRef&  obj )

{
  lint  i;

  if ( ObjectTraits<lint>::toValue( i, obj ) )
  {
    if ( i >= 0_lint && i <= (lint) maxOf<byte>() )
    {
      value = (byte) i;

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   isInstance
//-----------------------------------------------------------------------


bool ObjectTraits<byte>::isInstance ( const ObjRef& obj )
{
  lint  i;

  if ( ObjectTraits<lint>::toValue( i, obj ) )
  {
    return (i >= 0_lint && i <= (lint) maxOf<byte>());
  }
  else
  {
    return false;
  }
}


//=======================================================================
//   class ObjectTraits<short>
//=======================================================================

//-----------------------------------------------------------------------
//   toValue
//-----------------------------------------------------------------------


short ObjectTraits<short>::toValue ( const ObjRef& obj )
{
  lint  i = ObjectTraits<lint>::toValue ( obj );

  if ( i < (lint) minOf<short>() || i > (lint) maxOf<short>() )
  {
    rangeError ( i, (lint) minOf<short>(), (lint) maxOf<short>() );
  }

  return (short) i;
}


bool ObjectTraits<short>::toValue

  ( short&         value,
    const ObjRef&  obj )

{
  lint  i;

  if ( ObjectTraits<lint>::toValue( i, obj ) )
  {
    if ( i >= (lint) minOf<short>() && i <= (lint) maxOf<short>() )
    {
      value = (short) i;

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   isInstance
//-----------------------------------------------------------------------


bool ObjectTraits<short>::isInstance ( const ObjRef& obj )
{
  lint  i;

  if ( ObjectTraits<lint>::toValue( i, obj ) )
  {
    return (i >= (lint) minOf<short>() &&
            i <= (lint) maxOf<short>());
  }
  else
  {
    return false;
  }
}


//=======================================================================
//   class ObjectTraits<int>
//=======================================================================

//-----------------------------------------------------------------------
//   toValue
//-----------------------------------------------------------------------


int ObjectTraits<int>::toValue ( const ObjRef& obj )
{
  lint  i = ObjectTraits<lint>::toValue ( obj );

  if ( i < (lint) minOf<int>() || i > (lint) maxOf<int>() )
  {
    rangeError ( i, (lint) minOf<int>(), (lint) maxOf<int>() );
  }

  return (int) i;
}


bool ObjectTraits<int>::toValue

  ( int&           value,
    const ObjRef&  obj )

{
  lint  i;

  if ( ObjectTraits<lint>::toValue( i, obj ) )
  {
    if ( i >= (lint) minOf<int>() && i <= (lint) maxOf<int>() )
    {
      value = (int) i;

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   isInstance
//-----------------------------------------------------------------------


bool ObjectTraits<int>::isInstance ( const ObjRef& obj )
{
  lint  i;

  if ( ObjectTraits<lint>::toValue( i, obj ) )
  {
    return (i >= (lint) minOf<int>() &&
            i <= (lint) maxOf<int>());
  }
  else
  {
    return false;
  }
}


//=======================================================================
//   class ObjectTraits<long>
//=======================================================================

//-----------------------------------------------------------------------
//   toValue
//-----------------------------------------------------------------------


long ObjectTraits<long>::toValue ( const ObjRef& obj )
{
  lint  i = ObjectTraits<lint>::toValue ( obj );

  if ( i < (lint) minOf<long>() || i > (lint) maxOf<long>() )
  {
    rangeError ( i, (lint) minOf<long>(), (lint) maxOf<long>() );
  }

  return (long) i;
}


bool ObjectTraits<long>::toValue

  ( long&          value,
    const ObjRef&  obj )

{
  lint  i;

  if ( ObjectTraits<lint>::toValue( i, obj ) )
  {
    if ( i >= (lint) minOf<long>() && i <= (lint) maxOf<long>() )
    {
      value = (long) i;

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   isInstance
//-----------------------------------------------------------------------


bool ObjectTraits<long>::isInstance ( const ObjRef& obj )
{
  lint  i;

  if ( ObjectTraits<lint>::toValue( i, obj ) )
  {
    return (i >= (lint) minOf<long>() &&
            i <= (lint) maxOf<long>());
  }
  else
  {
    return false;
  }
}


//=======================================================================
//   class ObjectTraits<lint>
//=======================================================================

//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


Class* ObjectTraits<lint>::getType ()
{
  return Integer::getType ();
}


//-----------------------------------------------------------------------
//   toValue_
//-----------------------------------------------------------------------


lint ObjectTraits<lint>::toValue_ ( const ObjRef& obj )
{
  if ( obj.getType() == ObjRef::OBJECT_TYPE )
  {
    Integer*  i = Integer::dynamicCast ( obj.getObject() );

    if ( i )
    {
      return i->value;
    }

    Number*  n = Number::dynamicCast ( obj.getObject() );

    if ( n && n->isInteger() )
    {
      return n->toInteger ();
    }
  }

  typeError ( obj, getType() );

  return 0_lint;
}


bool ObjectTraits<lint>::toValue_

  ( lint&          value,
    const ObjRef&  obj )

{
  if ( obj.getType() == ObjRef::OBJECT_TYPE )
  {
    Integer*  i = Integer::dynamicCast ( obj.getObject() );

    if ( i )
    {
      value = i->value;

      return true;
    }

    Number*  n = Number::dynamicCast ( obj.getObject() );

    if ( n && n->isInteger() )
    {
      value = n->toInteger ();

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   isInstance_
//-----------------------------------------------------------------------


bool ObjectTraits<lint>::isInstance_ ( const ObjRef& obj )
{
  if ( obj.getType() == ObjRef::OBJECT_TYPE )
  {
    Integer*  i = Integer::dynamicCast ( obj.getObject() );

    if ( i )
    {
      return true;
    }

    Number*  n = Number::dynamicCast ( obj.getObject() );

    if ( n )
    {
      return n->isInteger ();
    }
  }

  return false;
}


//=======================================================================
//   class ObjectTraits<idx_t>
//=======================================================================

//-----------------------------------------------------------------------
//   toValue
//-----------------------------------------------------------------------


idx_t ObjectTraits<idx_t>::toValue ( const ObjRef& obj )
{
  lint  i = ObjectTraits<lint>::toValue ( obj );

  if ( i < (lint) minOf<idx_t>() || i > (lint) maxOf<idx_t>() )
  {
    rangeError ( i, (lint) minOf<idx_t>(), (lint) maxOf<idx_t>() );
  }

  return (idx_t) i;
}


bool ObjectTraits<idx_t>::toValue

  ( idx_t&         value,
    const ObjRef&  obj )

{
  lint  i;

  if ( ObjectTraits<lint>::toValue( i, obj ) )
  {
    if ( i >= (lint) minOf<idx_t>() && i <= (lint) maxOf<idx_t>() )
    {
      value = (idx_t) i;

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   isInstance
//-----------------------------------------------------------------------


bool ObjectTraits<idx_t>::isInstance ( const ObjRef& obj )
{
  lint  i;

  if ( ObjectTraits<lint>::toValue( i, obj ) )
  {
    return (i >= (lint) minOf<idx_t>() &&
            i <= (lint) maxOf<idx_t>());
  }
  else
  {
    return false;
  }
}


//=======================================================================
//   class ObjectTraits<float>
//=======================================================================

//-----------------------------------------------------------------------
//   toValue
//-----------------------------------------------------------------------


float ObjectTraits<float>::toValue ( const ObjRef& obj )
{
  double  x = ObjectTraits<double>::toValue ( obj );

  if ( std::fabs( x ) > (double) maxOf<float>() )
  {
    rangeError ( x, (double) maxOf<float>() );
  }

  return (float) x;
}


bool ObjectTraits<float>::toValue

  ( float&         value,
    const ObjRef&  obj )

{
  double  x;

  if ( ObjectTraits<double>::toValue ( x, obj ) )
  {
    if ( std::fabs( x ) <= (double) maxOf<float>() )
    {
      value = (float) x;

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   isInstance
//-----------------------------------------------------------------------


bool ObjectTraits<float>::isInstance ( const ObjRef& obj )
{
  double  x;

  if ( ObjectTraits<double>::toValue ( x, obj ) )
  {
    return (std::fabs( x ) <= (double) maxOf<float>());
  }
  else
  {
    return false;
  }
}


//=======================================================================
//   class ObjectTraits<double>
//=======================================================================

//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


Class* ObjectTraits<double>::getType ()
{
  return Float::getType ();
}


//-----------------------------------------------------------------------
//   toValue_
//-----------------------------------------------------------------------


double ObjectTraits<double>::toValue_ ( const ObjRef& obj )
{
  if ( obj.getType() == ObjRef::OBJECT_TYPE )
  {
    Float*  f = Float::dynamicCast ( obj.getObject() );

    if ( f )
    {
      return f->value;
    }

    Number*  n = Number::dynamicCast ( obj.getObject() );

    if ( n && n->isFloat() )
    {
      return n->toFloat ();
    }
  }

  typeError ( obj, getType() );

  return 0.0;
}


bool ObjectTraits<double>::toValue_

  ( double&        value,
    const ObjRef&  obj )

{
  if ( obj.getType() == ObjRef::OBJECT_TYPE )
  {
    Float*  f = Float::dynamicCast ( obj.getObject() );

    if ( f )
    {
      value = f->value;

      return true;
    }

    Number*  n = Number::dynamicCast ( obj.getObject() );

    if ( n && n->isFloat() )
    {
      value = n->toFloat ();

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   isInstance_
//-----------------------------------------------------------------------


bool ObjectTraits<double>::isInstance_ ( const ObjRef& obj )
{
  if ( obj.getType() == ObjRef::OBJECT_TYPE )
  {
    Float*  f = Float::dynamicCast ( obj.getObject() );

    if ( f )
    {
      return true;
    }

    Number*  n = Number::dynamicCast ( obj.getObject() );

    if ( n )
    {
      return n->isFloat ();
    }
  }

  return false;
}


//=======================================================================
//   class ObjectTraits<String>
//=======================================================================

//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


Class* ObjectTraits<String>::getType ()
{
  return StringObject::getType ();
}


//-----------------------------------------------------------------------
//   toValue_
//-----------------------------------------------------------------------


String ObjectTraits<String>::toValue_ ( const ObjRef& obj )
{
  if ( obj.getType() == ObjRef::CHAR_TYPE )
  {
    return String ( obj.getChar() );
  }

  if ( obj.getType() == ObjRef::OBJECT_TYPE )
  {
    StringObject*  s = StringObject::dynamicCast ( obj.getObject() );

    if ( s )
    {
      return s->value;
    }

    Character*  c = Character::dynamicCast ( obj.getObject() );

    if ( c )
    {
      return String ( c->value );
    }
  }

  typeError ( obj, getType() );

  return String ();
}


bool ObjectTraits<String>::toValue_

  ( String&        value,
    const ObjRef&  obj )

{
  if ( obj.getType() == ObjRef::CHAR_TYPE )
  {
    value = String ( obj.getChar() );

    return true;
  }

  if ( obj.getType() == ObjRef::OBJECT_TYPE )
  {
    StringObject*  s = StringObject::dynamicCast ( obj.getObject() );

    if ( s )
    {
      value = s->value;

      return true;
    }

    Character*  c = Character::dynamicCast ( obj.getObject() );

    if ( c )
    {
      value = String ( c->value );

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   isInstance_
//-----------------------------------------------------------------------


bool ObjectTraits<String>::isInstance_ ( const ObjRef&  obj )
{
  if ( obj.getType() == ObjRef::CHAR_TYPE )
  {
    return true;
  }

  if ( obj.getType() == ObjRef::OBJECT_TYPE )
  {
    if ( StringObject::dynamicCast( obj.getObject() ) )
    {
      return true;
    }

    if ( Character::dynamicCast( obj.getObject() ) )
    {
      return true;
    }
  }

  return false;
}


JEM_END_PACKAGE_BASE
