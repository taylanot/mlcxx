
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


#include <jem/base/reflect/Constructor.h>
#include <jem/base/reflect/AttributeTable.h>
#include <jem/base/Class.h>
#include <jem/base/ClassTable.h>
#include <jem/base/InstantiationException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/Once.h>
#include <jem/base/ReadWriteMutex.h>
#include <jem/base/utilities.h>
#include <jem/util/ArrayBuffer.h>


JEM_BEGIN_PACKAGE_BASE


using jem::util::ArrayBuffer;
using jem::util::Dictionary;


//=======================================================================
//   class Class::StaticData_
//=======================================================================


class Class::StaticData_
{
 public:

  static inline
    StaticData_&          getInstance     ();


  ClassTable              classTable;


 protected:

  inline                  StaticData_     ();
  inline                 ~StaticData_     ();


 private:

  static void             makeInstance_   ();


 private:

  static StaticData_*     instance_;

};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


Class::StaticData_* Class::StaticData_::instance_ = nullptr;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline Class::StaticData_::StaticData_ ()
{}


inline Class::StaticData_::~StaticData_ ()
{}


//-----------------------------------------------------------------------
//   getInstance
//-----------------------------------------------------------------------


inline Class::StaticData_& Class::StaticData_::getInstance ()
{
  static Once once = JEM_ONCE_INITIALIZER;

  runOnce ( once, makeInstance_ );

  return *instance_;
}


//-----------------------------------------------------------------------
//   makeInstance_
//-----------------------------------------------------------------------


void Class::StaticData_::makeInstance_ ()
{
  instance_ = new StaticData_ ();
}


//=======================================================================
//   class Class::Data_
//=======================================================================


class Class::Data_
{
 public:

  typedef ArrayBuffer
    < Ref<Ctor> >         CtorList;


  ReadWriteMutex          mutex;
  String                  whatis;
  AttributeTable          attribs;
  CtorList                ctors;

};


//=======================================================================
//   class Class
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Class::Class ( const String& name ) :

  super_ ( nullptr ),
  name_  ( name ),
  data_  ( nullptr )

{
  data_ = new Data_ ();
}


Class::Class ( Class* super, const String& name ) :

  super_ ( super ),
  name_  ( name ),
  data_  ( nullptr )

{
  data_ = new Data_ ();
}


Class::~Class ()
{
  delete data_;
  data_ = nullptr;
}


//-----------------------------------------------------------------------
//   getName
//-----------------------------------------------------------------------


String Class::getName () const noexcept
{
  return name_;
}


//-----------------------------------------------------------------------
//   getWhatis
//-----------------------------------------------------------------------


String Class::getWhatis () const
{
  String  whatis;

  {
    ReadLock lock ( data_->mutex );

    whatis = data_->whatis;
  }

  if ( whatis.size() == 0 )
  {
    whatis = String::format ( "an object of type `%s\'", name_ );
  }

  return whatis;
}


//-----------------------------------------------------------------------
//   setWhatis
//-----------------------------------------------------------------------


void Class::setWhatis ( const String& whatis )
{
  WriteLock lock ( data_->mutex );

  data_->whatis = whatis;
}


//-----------------------------------------------------------------------
//   isBaseOf
//-----------------------------------------------------------------------


bool Class::isBaseOf ( Class* type ) const
{
  while ( type )
  {
    if ( type == this )
    {
      return true;
    }

    type = type->super_;
  }

  return false;
}


//-----------------------------------------------------------------------
//   isInstance
//-----------------------------------------------------------------------


bool Class::isInstance ( const Ref<Object>& obj ) const
{
  Class*  type = obj.getClass ();

  while ( type )
  {
    if ( type == this )
    {
      return true;
    }

    type = type->super_;
  }

  return false;
}


//-----------------------------------------------------------------------
//   newInstance
//-----------------------------------------------------------------------


Ref<Object> Class::newInstance () const
{
  const Data_::CtorList&  list = data_->ctors;

  Ref<Ctor>  ctor;

  {
    ReadLock  lock ( data_->mutex );

    for ( idx_t i = 0; i < list.size(); i++ )
    {
      if ( list[i]->argCount() == 0 )
      {
        ctor = list[i];
        break;
      }
    }
  }

  if ( ! ctor )
  {
    throw InstantiationException (
      JEM_FUNC,
      String ( "class `" ) + name_ +
      String ( "\' does not have a default constructor" )
    );
  }

  return ctor->call ( nullptr );
}


//-----------------------------------------------------------------------
//   addConstructor
//-----------------------------------------------------------------------


void Class::addConstructor ( const Ref<Ctor>& ctor )
{
  if ( ctor )
  {
    WriteLock lock ( data_->mutex );

    staticCast<Member*> ( ctor ) -> myClass_ = this;

    data_->ctors.pushBack ( ctor );
  }
}


//-----------------------------------------------------------------------
//   getConstructors
//-----------------------------------------------------------------------


Class::CtorArray Class::getConstructors () const
{
  CtorArray  list;

  {
    ReadLock lock ( data_->mutex );

    list.ref ( data_->ctors.toArray().clone() );
  }

  return list;
}


//-----------------------------------------------------------------------
//   getConstructor
//-----------------------------------------------------------------------


Ref<Constructor> Class::getConstructor

  ( Class**  argTypes,
    idx_t    argCount )

{
  const int  MAX_ARG_COUNT = 16;

  const Data_::CtorList&  list = data_->ctors;

  Class*     formalTypes [ MAX_ARG_COUNT ];
  ReadLock   lock        ( data_->mutex  );
  Ref<Ctor>  ctor;
  idx_t      j;


  if ( argCount > MAX_ARG_COUNT )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      String::format (
        "too many constructor arguments (%d) specified",
        argCount
      )
    );
  }

  for ( idx_t i = 0; i < list.size(); i++ )
  {
    if ( list[i]->argCount() != argCount )
    {
      continue;
    }

    list[i]->getArgTypes ( formalTypes );

    for ( j = 0; j < argCount; j++ )
    {
      if ( formalTypes[j] == argTypes[j] )
      {
        continue;
      }

      if ( formalTypes[j]->isBaseOf ( argTypes[j] ) )
      {
        continue;
      }

      break;
    }

    if ( j == argCount )
    {
      ctor = list[i];
    }
  }

  return ctor;
}


//-----------------------------------------------------------------------
//   addAttribute
//-----------------------------------------------------------------------


void Class::addAttribute

  ( const String&       name,
    const Ref<Object>&  attrib )

{
  WriteLock lock ( data_->mutex );

  data_->attribs.addAttribute ( name, attrib );
}


//-----------------------------------------------------------------------
//   getAttributes
//-----------------------------------------------------------------------


Ref<Dictionary> Class::getAttributes () const
{
  AttributeTable attribs ( 16 );

  getAttributes_ ( attribs );

  return attribs.getAttributes ();
}


//-----------------------------------------------------------------------
//   getDeclaredAttribs
//-----------------------------------------------------------------------


Ref<Dictionary> Class::getDeclaredAttribs () const
{
  Ref<Dictionary>  attribs;

  {
    ReadLock lock ( data_->mutex );

    attribs = data_->attribs.getAttributes ();
  }

  return attribs;
}


//-----------------------------------------------------------------------
//   getAttribute
//-----------------------------------------------------------------------


Ref<Object> Class::getAttribute ( const String& name ) const
{
  const Class*  cls = this;
  Ref<Object>   attrib;

  while ( cls )
  {
    ReadLock lock ( cls->data_->mutex );

    attrib = cls->data_->attribs.getAttribute ( name );

    if ( attrib )
    {
      break;
    }

    cls = cls->super_;
  }

  return attrib;
}


//-----------------------------------------------------------------------
//   getDeclaredAttrib
//-----------------------------------------------------------------------


Ref<Object> Class::getDeclaredAttrib ( const String& name ) const
{
  Ref<Object> attrib;

  {
    ReadLock lock ( data_->mutex );

    attrib = data_->attribs.getAttribute ( name );
  }

  return attrib;
}


//-----------------------------------------------------------------------
//   addField
//-----------------------------------------------------------------------


void Class::addField ( const Ref<Field>& field )
{
  if ( field )
  {
    WriteLock lock ( data_->mutex );

    staticCast<Member*> ( field ) -> myClass_ = this;

    data_->attribs.addField ( field );
  }
}


//-----------------------------------------------------------------------
//   getFields
//-----------------------------------------------------------------------


Class::FieldArray Class::getFields () const
{
  AttributeTable attribs ( 16 );

  getAttributes_ ( attribs );

  return attribs.getFields ();
}


//-----------------------------------------------------------------------
//   getDeclaredFields
//-----------------------------------------------------------------------


Class::FieldArray Class::getDeclaredFields () const
{
  FieldArray fields;

  {
    ReadLock lock ( data_->mutex );

    fields.ref ( data_->attribs.getFields() );
  }

  return fields;
}


//-----------------------------------------------------------------------
//   getField
//-----------------------------------------------------------------------


Ref<Field> Class::getField ( const String& name ) const
{
  return dynamicCast<Field> ( getAttribute( name ) );
}


//-----------------------------------------------------------------------
//   getDeclaredField
//-----------------------------------------------------------------------


Ref<Field> Class::getDeclaredField ( const String& name ) const
{
  return dynamicCast<Field> ( getDeclaredAttrib( name ) );
}


//-----------------------------------------------------------------------
//   addMethod
//-----------------------------------------------------------------------


void Class::addMethod ( const Ref<Method>& method )
{
  if ( method )
  {
    WriteLock lock ( data_->mutex );

    staticCast<Member*> ( method ) -> myClass_ = this;

    data_->attribs.addMethod ( method );
  }
}


//-----------------------------------------------------------------------
//   getMethods
//-----------------------------------------------------------------------


Class::MethodArray Class::getMethods () const
{
  AttributeTable  attribs ( 16 );

  getAttributes_ ( attribs );

  return attribs.getMethods ();
}



//-----------------------------------------------------------------------
//   getDeclaredMethods
//-----------------------------------------------------------------------


Class::MethodArray Class::getDeclaredMethods () const
{
  MethodArray  methods;

  {
    ReadLock lock ( data_->mutex );

    methods.ref ( data_->attribs.getMethods() );
  }

  return methods;
}


//-----------------------------------------------------------------------
//   getMethod
//-----------------------------------------------------------------------


Ref<Method> Class::getMethod ( const String& name ) const
{
  return dynamicCast<Method> ( getAttribute( name ) );
}


//-----------------------------------------------------------------------
//   getDeclaredMethod
//-----------------------------------------------------------------------


Ref<Method> Class::getDeclaredMethod ( const String& name ) const
{
  return dynamicCast<Method> ( getDeclaredAttrib( name ) );
}


//-----------------------------------------------------------------------
//   registerClass
//-----------------------------------------------------------------------


void Class::registerClass ( Class* c )
{
  try
  {
    StaticData_::getInstance().classTable.insert( c );
  }
  catch ( ... )
  {
    abort ( JEM_FUNC, "caught an unexpected exception while "
                      "registering a class" );
  }
}


//-----------------------------------------------------------------------
//   findClass
//-----------------------------------------------------------------------


Class* Class::findClass ( const String& name )
{
  return StaticData_::getInstance().classTable.find ( name );
}


//-----------------------------------------------------------------------
//   getAttributes_
//-----------------------------------------------------------------------


void Class::getAttributes_ ( AttributeTable& attribs ) const
{
  const Class*  type = this;

  while ( type )
  {
    ReadLock lock ( type->data_->mutex );

    attribs.mergeWith ( type->data_->attribs );

    type = type->super_;
  }
}


JEM_END_PACKAGE_BASE
