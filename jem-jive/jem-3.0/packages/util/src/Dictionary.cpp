
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


#include <jem/base/rtti.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/ObjectPair.h>
#include <jem/util/Dictionary.h>


JEM_DEFINE_CUSTOM_CLASS( jem::util::Dictionary );


JEM_BEGIN_PACKAGE( util )


using jem::io::ObjectInput;
using jem::io::ObjectOutput;


//=======================================================================
//   class Dictionary
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


Dictionary::~Dictionary ()
{}


//-----------------------------------------------------------------------
//   makeClass
//-----------------------------------------------------------------------


Class* Dictionary::makeClass ()
{
  ClassTemplate<Self>*  myClass = new ClassTemplate<Self> ();

  myClass->setWhatis ( "a dictionary" );

  return myClass;
}


//-----------------------------------------------------------------------
//   enumerate
//-----------------------------------------------------------------------


Ref<DictionaryEnumerator> Dictionary::enumerate () const
{
  return getDictEnum ();
}


//-----------------------------------------------------------------------
//   getEnumerator
//-----------------------------------------------------------------------


Ref<Enumerator> Dictionary::getEnumerator () const
{
  return getDictEnum ();
}


//-----------------------------------------------------------------------
//   contains
//-----------------------------------------------------------------------


bool Dictionary::contains ( const String& key ) const
{
  Ref<Object>  dummy;

  return find ( dummy, key );
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


void Dictionary::insert

  ( String&&       key,
    Ref<Object>&&  value )

{
  insert ( key, value );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<Object> Dictionary::get ( const String& key ) const
{
  Ref<Object>  obj;

  find ( obj, key );

  return obj;
}


//-----------------------------------------------------------------------
//   add
//-----------------------------------------------------------------------


bool Dictionary::add ( const Ref<Object>& obj )
{
  if ( obj )
  {
    Ref<ObjPair>  p = checkedCast<ObjPair> ( obj );

    insert ( toValue<String>( p->getFirst() ), p->getSecond() );

    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   readFrom_
//-----------------------------------------------------------------------


void Dictionary::readFrom_ ( ObjectInput& in )
{
  uidx_t  sz;

  decode ( in, sz );

  idx_t  n = (idx_t) sz;

  clear   ();
  reserve ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    Ref<Object>  value;
    String       key;

    decode ( in, key, value );
    insert ( key, value );
  }
}


//-----------------------------------------------------------------------
//   writeTo_
//-----------------------------------------------------------------------


void Dictionary::writeTo_ ( ObjectOutput& out ) const
{
  Ref<DictEnum>  e = getDictEnum ();

  encode ( out, size() );

  for ( ; ! e->atEnd(); e->toNext() )
  {
    encode ( out, e->getKey(), e->getValue() );
  }
}


//-----------------------------------------------------------------------
//   clone_
//-----------------------------------------------------------------------


void Dictionary::clone_ ( const Dictionary& rhs )
{
  Ref<DictEnum>  e = rhs.getDictEnum ();

  clear   ();
  reserve ( rhs.size() );

  for ( ; ! e->atEnd(); e->toNext() )
  {
    Ref<Object>  obj = e->getValue ();

    insert ( e->getKey(), obj.clone() );
  }
}


JEM_END_PACKAGE( util )
