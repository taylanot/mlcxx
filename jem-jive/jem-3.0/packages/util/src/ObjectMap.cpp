
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
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/ObjectPair.h>
#include <jem/util/ObjectMap.h>


JEM_DEFINE_CLASS( jem::util::ObjectMap );


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class ObjectMap
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


ObjectMap::~ObjectMap ()
{}


//-----------------------------------------------------------------------
//   enumerate
//-----------------------------------------------------------------------


Ref<MapEnum> ObjectMap::enumerate () const
{
  return getMapEnum ();
}


//-----------------------------------------------------------------------
//   getEnumerator
//-----------------------------------------------------------------------


Ref<Enumerator> ObjectMap::getEnumerator () const
{
  return getMapEnum ();
}


//-----------------------------------------------------------------------
//   contains
//-----------------------------------------------------------------------


bool ObjectMap::contains ( const Ref<Object>& key ) const
{
  Ref<Object>  dummy;

  return find ( dummy, key );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<Object> ObjectMap::get ( const Ref<Object>& key ) const
{
  Ref<Object>  value;

  find ( value, key );

  return value;
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


void ObjectMap::insert

  ( Ref<Object>&&  key,
    Ref<Object>&&  value )

{
  insert ( key, value );
}


//-----------------------------------------------------------------------
//   add
//-----------------------------------------------------------------------


bool ObjectMap::add ( const Ref<Object>& obj )
{
  if ( obj )
  {
    Ref<ObjPair>  p = checkedCast<ObjPair> ( obj );

    insert ( p->getFirst(), p->getSecond() );

    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   clone_
//-----------------------------------------------------------------------


void ObjectMap::clone_ ( const ObjectMap& rhs )
{
  Ref<MapEnum>  e = rhs.enumerate();

  clear   ();
  reserve ( rhs.size() );

  for ( ; ! e->atEnd(); e->toNext() )
  {
    Ref<Object>  key   = e->getKey   ();
    Ref<Object>  value = e->getValue ();

    insert ( key.clone(), value.clone() );
  }
}


//-----------------------------------------------------------------------
//   readFrom_
//-----------------------------------------------------------------------


void ObjectMap::readFrom_ ( io::ObjectInput& in )
{
  uidx_t  sz;

  decode ( in, sz );

  idx_t  n = (idx_t) sz;

  clear   ();
  reserve ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    Ref<Object>  key   = in.decodeObject ();
    Ref<Object>  value = in.decodeObject ();

    insert ( key, value );
  }
}


//-----------------------------------------------------------------------
//   writeTo_
//-----------------------------------------------------------------------


void ObjectMap::writeTo_ ( io::ObjectOutput& out ) const
{
  Ref<MapEnum>  e = enumerate ();

  encode ( out, size() );

  for ( ; ! e->atEnd(); e->toNext() )
  {
    out.encodeObject ( e->getKey()   );
    out.encodeObject ( e->getValue() );
  }
}


JEM_END_PACKAGE( util )
