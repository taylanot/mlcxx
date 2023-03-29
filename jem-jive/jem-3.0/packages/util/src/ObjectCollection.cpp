
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


#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/error.h>
#include <jem/util/ObjectCollection.h>


JEM_DEFINE_CLASS( jem::util::ObjectCollection );


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class ObjectCollection
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


ObjectCollection::~ObjectCollection ()
{}


//-----------------------------------------------------------------------
//   enumerate
//-----------------------------------------------------------------------


Ref<Enumerator> ObjectCollection::enumerate () const
{
  return getEnumerator ();
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void ObjectCollection::reserve ( idx_t count )
{}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void ObjectCollection::trimToSize ()
{}


//-----------------------------------------------------------------------
//   capacity
//-----------------------------------------------------------------------


idx_t ObjectCollection::capacity () const
{
  return size ();
}


//-----------------------------------------------------------------------
//   clone_
//-----------------------------------------------------------------------


void ObjectCollection::clone_ ( const Self& rhs )
{
  JEM_PRECHECK ( &rhs != this );

  Ref<Enumerator>  e = rhs.getEnumerator ();

  clear   ();
  reserve ( rhs.size() );

  for ( ; ! e->atEnd(); e->toNext() )
  {
    Ref<Object>  obj = e->get ();

    add ( obj.clone() );
  }
}


//-----------------------------------------------------------------------
//   readFrom_
//-----------------------------------------------------------------------


void ObjectCollection::readFrom_ ( io::ObjectInput& in )
{
  uidx_t  sz;

  decode ( in, sz );

  idx_t  n = (idx_t) sz;

  clear   ();
  reserve ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    add ( in.decodeObject() );
  }
}


//-----------------------------------------------------------------------
//   writeTo_
//-----------------------------------------------------------------------


void ObjectCollection::writeTo_ ( io::ObjectOutput& out ) const
{
  Ref<Enumerator>  e = enumerate ();

  encode ( out, size() );

  for ( ; ! e->atEnd(); e->toNext() )
  {
    out.encodeObject ( e->get() );
  }
}


//-----------------------------------------------------------------------
//   expand_
//-----------------------------------------------------------------------


idx_t ObjectCollection::expand_ ( idx_t cap )
{
  idx_t  n = (idx_t) ((double) maxOf( cap ) / 1.501) - 4;

  if ( cap >= n )
  {
    n = maxOf ( cap );
  }
  else
  {
    n = (idx_t) (1.5 * (double) cap) + 4;
  }

  if ( cap >= n )
  {
    overflowError ( JEM_FUNC, cap );
  }

  return n;
}


//-----------------------------------------------------------------------
//   writeError_
//-----------------------------------------------------------------------


void ObjectCollection::writeError_ () const
{
  Class*  type   = getClass ();
  String  whatis = type->getWhatis ();

  throw IllegalOperationException (
    type->getName  (),
    String::format ( "%s may not be modified while an "
                     "enumerator is active", whatis )
  );
}


JEM_END_PACKAGE( util )
