
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

#ifndef JEM_BASE_REFLECT_ATTRIBUTETABLE_H
#define JEM_BASE_REFLECT_ATTRIBUTETABLE_H

#include <jem/base/Object.h>
#include <jem/base/array/Array.h>
#include <jem/base/reflect/Field.h>
#include <jem/base/reflect/Method.h>
#include <jem/util/Dictionary.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class AttributeTable
//-----------------------------------------------------------------------


class AttributeTable
{
 public:

  typedef util::Dictionary      Dict;
  typedef AttributeTable        Self;
  typedef Array< Ref<Method> >  MethodArray;
  typedef Array< Ref<Field> >   FieldArray;


                        AttributeTable  ();

  explicit              AttributeTable

    ( idx_t               cap );

  Ref<Dict>             getAttributes   () const;
  MethodArray           getMethods      () const;
  FieldArray            getFields       () const;

  inline Ref<Object>    getAttribute

    ( const String&       name )           const;

  inline Ref<Method>    getMethod

    ( const String&       name )           const;

  inline Ref<Field>     getField

    ( const String&       name )           const;

  void                  addAttribute

    ( const String&       name,
      const Ref<Object>&  attrib );

  void                  addMethod

    ( const Ref<Method>&  method );

  void                  addField

    ( const Ref<Field>&   field );

  void                  mergeWith

    ( const Self&         rhs );


 private:

  static Ref<Dict>      newDictionary_  ();


 private:

  Ref<Dict>             attributes_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getAttribute
//-----------------------------------------------------------------------


inline Ref<Object>

  AttributeTable::getAttribute ( const String& name ) const

{
  if ( attributes_ )
  {
    Ref<Object>  attrib;

    attributes_->find ( attrib, name );

    return attrib;
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   getMethod
//-----------------------------------------------------------------------


inline Ref<Method>

  AttributeTable::getMethod ( const String& name ) const

{
  return dynamicCast<Method> ( getAttribute( name ) );
}


//-----------------------------------------------------------------------
//   getField
//-----------------------------------------------------------------------


inline Ref<Field>

  AttributeTable::getField ( const String& name ) const

{
  return dynamicCast<Field> ( getAttribute( name ) );
}


JEM_END_PACKAGE_BASE

#endif
