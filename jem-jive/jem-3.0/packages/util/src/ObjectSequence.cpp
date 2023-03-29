
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
#include <jem/base/IllegalIndexException.h>
#include <jem/util/EmptySequenceException.h>
#include <jem/util/ObjectSequence.h>


JEM_DEFINE_CLASS( jem::util::ObjectSequence );


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class ObjectSequence
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


ObjectSequence::~ObjectSequence ()
{}


//-----------------------------------------------------------------------
//   enumerate
//-----------------------------------------------------------------------


Ref<SeqEnum> ObjectSequence::enumerate () const
{
  return getSeqEnum ();
}


//-----------------------------------------------------------------------
//   getEnumerator
//-----------------------------------------------------------------------


Ref<Enumerator> ObjectSequence::getEnumerator () const
{
  return getSeqEnum ();
}


//-----------------------------------------------------------------------
//   insertSpace
//-----------------------------------------------------------------------


void ObjectSequence::insertSpace

  ( idx_t  idx,
    idx_t  count )

{
  if ( idx < size() / 2 )
  {
    for ( ; count > 0; count-- )
    {
      insert ( idx, nullptr );
    }
  }
  else
  {
    for ( ; count > 0; count-- )
    {
      insert ( idx++, nullptr );
    }
  }
}


//-----------------------------------------------------------------------
//   eraseRange
//-----------------------------------------------------------------------


void ObjectSequence::eraseRange

  ( idx_t  first,
    idx_t  last )

{
  for ( idx_t i = first; i < last; i++ )
  {
    erase ( first );
  }
}


//-----------------------------------------------------------------------
//   add
//-----------------------------------------------------------------------


bool ObjectSequence::add ( const Ref<Object>& obj )
{
  pushBack ( obj );

  return true;
}


//-----------------------------------------------------------------------
//   front
//-----------------------------------------------------------------------


Ref<Object> ObjectSequence::front () const
{
  if ( size() == 0 )
  {
    emptyError_ ();
  }

  return get ( 0 );
}


//-----------------------------------------------------------------------
//   back
//-----------------------------------------------------------------------


Ref<Object> ObjectSequence::back () const
{
  if ( size() == 0 )
  {
    emptyError_ ();
  }

  return get ( size() - 1 );
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


void ObjectSequence::pushBack ( const Ref<Object>& obj )
{
  insert ( size(), obj );
}


void ObjectSequence::pushBack ( Ref<Object>&& obj )
{
  insert ( size(), std::move( obj ) );
}


//-----------------------------------------------------------------------
//   pushFront
//-----------------------------------------------------------------------


void ObjectSequence::pushFront ( const Ref<Object>& obj )
{
  insert ( 0, obj );
}


void ObjectSequence::pushFront ( Ref<Object>&& obj )
{
  insert ( 0, std::move( obj ) );
}


//-----------------------------------------------------------------------
//   popBack
//-----------------------------------------------------------------------


void ObjectSequence::popBack ()
{
  if ( size() == 0 )
  {
    emptyError_ ();
  }

  erase ( size() - 1 );
}


//-----------------------------------------------------------------------
//   popFront
//-----------------------------------------------------------------------


void ObjectSequence::popFront ()
{
  if ( size() == 0 )
  {
    emptyError_ ();
  }

  erase ( 0 );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void ObjectSequence::clear ()
{
  eraseRange ( 0, size() );
}


//-----------------------------------------------------------------------
//   emptyError_
//-----------------------------------------------------------------------


void ObjectSequence::emptyError_ () const
{
  Class*  type = getClass ();

  throw EmptySequenceException ( type->getName() );
}


//-----------------------------------------------------------------------
//   indexError_
//-----------------------------------------------------------------------


void ObjectSequence::indexError_ ( idx_t idx ) const
{
  indexError_ ( idx, size() );
}


void ObjectSequence::indexError_

  ( idx_t  idx,
    idx_t  ubound ) const

{
  Class*  type = getClass ();

  throw IllegalIndexException (
    type->getName  (),
    String::format (
      "invalid sequence index: %d; valid range is [0,%d)",
      idx,
      ubound
    )
  );
}


JEM_END_PACKAGE( util )
