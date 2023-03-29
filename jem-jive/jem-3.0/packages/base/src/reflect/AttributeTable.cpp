
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


#include <jem/util/HashDictionary.h>
#include <jem/base/reflect/AttributeTable.h>


JEM_BEGIN_PACKAGE_BASE


using util::Dictionary;
using util::HashDict;
using util::DictEnum;


//=======================================================================
//   class AttributeTable
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


AttributeTable::AttributeTable ()
{}


AttributeTable::AttributeTable ( idx_t cap )
{
  attributes_ = newDictionary_ ();

  attributes_->reserve ( cap );
}


//-----------------------------------------------------------------------
//   getAttributes
//-----------------------------------------------------------------------


Ref<Dictionary> AttributeTable::getAttributes() const
{
  if ( ! attributes_ )
  {
    return newDictionary_ ();
  }
  else
  {
    return checkedCast<Dictionary> ( attributes_->clone() );
  }
}


//-----------------------------------------------------------------------
//   getMethods
//-----------------------------------------------------------------------


Array< Ref<Method> > AttributeTable::getMethods () const
{
  if ( ! attributes_ )
  {
    return MethodArray ();
  }
  else
  {
    MethodArray    methods ( attributes_->size() );
    Ref<Method>    method;
    Ref<DictEnum>  e;
    idx_t          n;

    e = attributes_->enumerate ();

    for ( n = 0; ! e->atEnd(); e->toNext() )
    {
      method = dynamicCast<Method> ( e->getValue() );

      if ( method )
      {
        methods[n++] = method;
      }
    }

    return methods[slice(BEGIN,n)];
  }
}


//-----------------------------------------------------------------------
//   getFields
//-----------------------------------------------------------------------


Array< Ref<Field> > AttributeTable::getFields () const
{
  if ( ! attributes_ )
  {
    return FieldArray ();
  }
  else
  {
    FieldArray     fields ( attributes_->size() );
    Ref<Field>     field;
    Ref<DictEnum>  e;
    idx_t          n;

    e = attributes_->enumerate ();

    for ( n = 0; ! e->atEnd(); e->toNext() )
    {
      field = dynamicCast<Field> ( e->getValue() );

      if ( field )
      {
        fields[n++] = field;
      }
    }

    return fields[slice(BEGIN,n)];
  }
}


//-----------------------------------------------------------------------
//   addAttribute
//-----------------------------------------------------------------------


void AttributeTable::addAttribute

  ( const String&       name,
    const Ref<Object>&  attrib )

{
  if ( ! attrib )
  {
    return;
  }

  if ( ! attributes_ )
  {
    attributes_ = newDictionary_ ();
  }

  attributes_->insert ( name, attrib );
}


//-----------------------------------------------------------------------
//   addMethod
//-----------------------------------------------------------------------


void AttributeTable::addMethod ( const Ref<Method>& method )
{
  addAttribute ( method->getName(), method );
}


//-----------------------------------------------------------------------
//   addField
//-----------------------------------------------------------------------


void AttributeTable::addField ( const Ref<Field>& field )
{
  addAttribute ( field->getName(), field );
}


//-----------------------------------------------------------------------
//   mergeWith
//-----------------------------------------------------------------------


void AttributeTable::mergeWith ( const Self& rhs )
{
  if ( rhs.attributes_ )
  {
    Ref<Object>  dummy;

    if ( ! attributes_ )
    {
      attributes_ = newDictionary_ ();
    }

    Ref<DictEnum>  e = rhs.attributes_->enumerate ();

    for ( ; ! e->atEnd(); e->toNext() )
    {
      String  name = e->getKey ();

      if ( ! attributes_->find( dummy, name ) )
      {
        attributes_->insert ( name, e->getValue() );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   newDictionary_
//-----------------------------------------------------------------------


Ref<Dictionary> AttributeTable::newDictionary_ ()
{
  return newInstance<HashDict> ();
}


JEM_END_PACKAGE_BASE
