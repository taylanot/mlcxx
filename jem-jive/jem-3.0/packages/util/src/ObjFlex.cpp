
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
#include <jem/base/MemCache.h>
#include <jem/base/RangeException.h>
#include <jem/base/IllegalIndexException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/error.h>
#include <jem/util/ObjFlex.h>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class ObjFlex
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ObjFlex::ObjFlex () noexcept
{
  size_     = 0;
  capacity_ = 0;
  objects_  = 0;
}


ObjFlex::ObjFlex ( idx_t n )
{
  if ( n < 0 )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      "invalid size: " + String ( n )
    );
  }

  size_     = n;
  capacity_ = n;
  objects_  = 0;

  if ( n )
  {
    objects_ = (Object**)

      MemCache::alloc ( (size_t) n * sizeof(*objects_) );

    for ( idx_t i = 0; i < n; i++ )
    {
      objects_[i] = 0;
    }
  }
}


ObjFlex::ObjFlex ( const ObjFlex& rhs )
{
  ObjFlex  tmp ( rhs.size() );

  size_     = 0;
  capacity_ = 0;
  objects_  = 0;

  tmp       = rhs;

  swap ( tmp );
}


ObjFlex::ObjFlex ( ObjFlex&& rhs ) noexcept
{
  size_     = 0;
  capacity_ = 0;
  objects_  = 0;

  swap ( rhs );
}


ObjFlex::~ObjFlex ()
{
  if ( objects_ )
  {
    for ( idx_t i = size_ - 1; i >= 0; i-- )
    {
      Collectable::delRef ( objects_[i] );
    }

    MemCache::dealloc ( objects_, (size_t) capacity_ *
                                  sizeof(*objects_) );

    objects_ = 0;
  }
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


void ObjFlex::swap ( ObjFlex& rhs ) noexcept
{
  jem::swap ( size_,     rhs.size_     );
  jem::swap ( capacity_, rhs.capacity_ );
  jem::swap ( objects_,  rhs.objects_  );
}


//-----------------------------------------------------------------------
//   assignment operator
//-----------------------------------------------------------------------


ObjFlex& ObjFlex::operator = ( const ObjFlex& rhs )
{
  const idx_t  n = rhs.size_;

  if ( this != & rhs )
  {
    if ( size_ != n )
    {
      resize ( n );
    }

    for ( idx_t i = 0; i < n; i++ )
    {
      Object*  tmp = rhs.objects_[i];

      Collectable::addRef ( tmp );
      Collectable::delRef ( objects_[i] );

      objects_[i] = tmp;
    }
  }

  return *this;
}


ObjFlex& ObjFlex::operator = ( ObjFlex&& rhs ) noexcept
{
  swap ( rhs );

  return *this;
}


//-----------------------------------------------------------------------
//   set
//-----------------------------------------------------------------------


void ObjFlex::set

  ( idx_t    idx,
    Object*  obj )

{
  JEM_ASSERT2 ( idx >= 0 && idx < size(), "invalid ObjFlex index" );

  Collectable::addRef ( obj );
  Collectable::delRef ( objects_[idx] );

  objects_[idx] = obj;
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


void ObjFlex::pushBack ( Object* obj )
{
  if ( size_ >= capacity_ )
  {
    expand_ ( size_ + 1 );
  }

  Collectable::addRef ( obj );

  objects_[size_++] = obj;
}


//-----------------------------------------------------------------------
//   popBack
//-----------------------------------------------------------------------


void ObjFlex::popBack ()
{
  JEM_ASSERT2 ( size() > 0, "empty ObjFlex" );

  --size_;

  Collectable::delRef ( objects_[size_] );

  objects_[size_] = 0;
}


void ObjFlex::popBack ( idx_t n )
{
  if      ( n < 0 )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      String::format ( "invalid number of objects to pop: %d", n )
    );
  }
  else if ( n > size_ )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      String::format (
        "number of objects to pop (%d) "
        "is greater than the size (%d)", n, size_
      )
    );
  }

  n = size_ - n;

  while ( size_ > n )
  {
    size_--;

    Collectable::delRef ( objects_[size_] );

    objects_[size_] = 0;
  }
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


void ObjFlex::insert

  ( idx_t    idx,
    Object*  obj )

{
  if ( idx < 0 || idx > size_ )
  {
    throw IllegalIndexException ( JEM_FUNC, idx, size_ + 1 );
  }

  if ( size_ >= capacity_ )
  {
    expand_ ( size_ + 1 );
  }

  for ( idx_t j = size_; j > idx; j-- )
  {
    objects_[j] = objects_[j - 1];
  }

  size_++;

  Collectable::addRef ( obj );

  objects_[idx] = obj;
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


void ObjFlex::erase ( idx_t idx )
{
  if ( idx < 0 || idx >= size_ )
  {
    throw IllegalIndexException ( JEM_FUNC, idx, size_ );
  }

  Collectable::delRef ( objects_[idx] );

  for ( idx_t j = idx + 1; j < size_; j++ )
  {
    objects_[j - 1] = objects_[j];
  }

  size_--;

  objects_[size_] = 0;
}


void ObjFlex::erase ( idx_t idx, idx_t jdx )
{
  if ( idx < 0 || jdx < idx || jdx > size_ )
  {
    throw RangeException (
      JEM_FUNC,
      String::format (
        "invalid index range specified: [%d,%d)", idx, jdx
      )
    );
  }

  idx_t  k;

  for ( k = idx; k < jdx; k++ )
  {
    Collectable::delRef ( objects_[k] );
  }

  k = jdx - idx;

  for ( jdx++; jdx < size_; jdx++ )
  {
    objects_[jdx - k] = objects_[jdx];
  }

  for ( idx = size_ - k; idx < size_; idx++ )
  {
    objects_[idx] = 0;
  }

  size_ -= k;
}


idx_t ObjFlex::erase ( Object* obj )
{
  idx_t  i, j;

  for ( i = j = 0; i < size_; i++ )
  {
    if ( objects_[i] == obj )
    {
      Collectable::delRef ( obj );
    }
    else
    {
      objects_[j++] = objects_[i];
    }
  }

  for ( i = j; i < size_; i++ )
  {
    objects_[i] = 0;
  }

  i     = size_ - j;
  size_ = j;

  return i;
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void ObjFlex::clear ()
{
  while ( size_ > 0 )
  {
    size_--;

    Collectable::delRef ( objects_[size_] );

    objects_[size_] = 0;
  }
}


//-----------------------------------------------------------------------
//   resize
//-----------------------------------------------------------------------


void ObjFlex::resize ( idx_t n )
{
  if ( n < 0 )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      String::format ( "invalid size: %d", n )
    );
  }

  if      ( n < size_ )
  {
    while ( size_ > n )
    {
      size_--;

      Collectable::delRef ( objects_[size_] );

      objects_[size_] = 0;
    }
  }
  else if ( n > size_ )
  {
    if ( n > capacity_ )
    {
      realloc_ ( n );
    }

    while ( size_ < n )
    {
      objects_[size_++] = 0;
    }
  }
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void ObjFlex::reserve ( idx_t cap )
{
  if ( cap < 0 )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      String::format ( "invalid capacity: %d", cap )
    );
  }

  if ( cap > capacity_ )
  {
    realloc_ ( cap );
  }
}


//-----------------------------------------------------------------------
//   expand_
//-----------------------------------------------------------------------


void ObjFlex::expand_ ( idx_t cap )
{
  idx_t  n = (idx_t) ((double) maxOf( capacity_ ) / 1.501) - 4;

  if ( capacity_ >= n )
  {
    n = maxOf ( capacity_ );
  }
  else
  {
    n = jem::max ( (idx_t) (1.5 * (double) capacity_) + 4, cap );
  }

  if ( cap < 0 || n < cap )
  {
    overflowError ( JEM_FUNC, capacity_ );
  }

  realloc_ ( n );
}


//-----------------------------------------------------------------------
//   realloc_
//-----------------------------------------------------------------------


void ObjFlex::realloc_ ( idx_t cap )
{
  if ( cap < size_ )
  {
    cap = size_;
  }

  objects_ = (Object**)

    MemCache::realloc ( objects_,
                        (size_t) capacity_ * sizeof(*objects_),
                        (size_t) cap       * sizeof(*objects_) );

  capacity_ = cap;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


void                  decode

  ( io::ObjectInput&    in,
    ObjFlex&            ofx )

{
  ObjFlex  tmp;
  uidx_t   sz;

  decode ( in, sz );

  idx_t  n = (idx_t) sz;

  tmp.reserve ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    Ref<Object>  obj = in.decodeObject ();

    tmp.pushBack ( obj.get() );
  }

  ofx.swap ( tmp );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


void                  encode

  ( io::ObjectOutput&   out,
    const ObjFlex&      ofx )

{
  const idx_t  n = ofx.size ();

  encode ( out, n );

  for ( idx_t i = 0; i < n; i++ )
  {
    Ref<Object>  obj = ofx.get ( i );

    out.encodeObject ( obj );
  }
}


JEM_END_PACKAGE( util )
