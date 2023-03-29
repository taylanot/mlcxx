
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


#include <utility>
#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/ObjectPair.h>


JEM_DEFINE_SERIABLE_CLASS( jem::util::ObjectPair );


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class ObjectPair
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ObjectPair::ObjectPair ()
{}


ObjectPair::ObjectPair

  ( const Ref<Object>&  first,
    const Ref<Object>&  second ) :

    first_  ( first  ),
    second_ ( second )

{}


ObjectPair::~ObjectPair()
{}


//-----------------------------------------------------------------------
//   hashValue
//-----------------------------------------------------------------------


lint ObjectPair::hashValue () const
{
  lint  i = first_ .hashValue ();
  lint  j = second_.hashValue ();

  return (i ^ j);
}


//-----------------------------------------------------------------------
//   equals
//-----------------------------------------------------------------------


bool ObjectPair::equals ( const Ref<Object>& rhs ) const
{
  Self*  p = dynamicCast<Self*> ( rhs );

  if ( p )
  {
    return ( first_ .equals( p->first_ ) &&
             second_.equals( p->second_ ) );
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void ObjectPair::readFrom ( io::ObjectInput& in )
{
  first_  = in.decodeObject ();
  second_ = in.decodeObject ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void ObjectPair::writeTo ( io::ObjectOutput& out ) const
{
  out.encodeObject ( first_  );
  out.encodeObject ( second_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> ObjectPair::clone () const
{
  return newInstance<Self> ( first_.clone(), second_.clone() );
}


//-----------------------------------------------------------------------
//   getFirst
//-----------------------------------------------------------------------


Ref<Object> ObjectPair::getFirst () const
{
  return first_;
}


//-----------------------------------------------------------------------
//   getSecond
//-----------------------------------------------------------------------


Ref<Object> ObjectPair::getSecond () const
{
  return second_;
}


//-----------------------------------------------------------------------
//   setFirst
//-----------------------------------------------------------------------


void ObjectPair::setFirst ( const Ref<Object>& obj )
{
  first_ = obj;
}


void ObjectPair::setFirst ( Ref<Object>&& obj )
{
  first_ = std::move ( obj );
}


//-----------------------------------------------------------------------
//   setSecond
//-----------------------------------------------------------------------


void ObjectPair::setSecond ( const Ref<Object>& obj )
{
  second_ = obj;
}


void ObjectPair::setSecond ( Ref<Object>&& obj )
{
  second_ = std::move ( obj );
}


JEM_END_PACKAGE( util )
