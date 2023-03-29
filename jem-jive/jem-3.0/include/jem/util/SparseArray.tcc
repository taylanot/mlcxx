
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

#ifndef JEM_UTIL_SPARSEARRAY_TCC
#define JEM_UTIL_SPARSEARRAY_TCC

#include <cmath>
#include <utility>
#include <jem/base/assert.h>
#include <jem/base/utilities.h>
#include <jem/base/MemCache.h>
#include <jem/base/TypeTraits.h>
#include <jem/base/tuple/utilities.h>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   private stuff
//=======================================================================

JEM_BEGIN_NAMESPACE( sparse )

//-----------------------------------------------------------------------
//   typedefs
//-----------------------------------------------------------------------


typedef Tuple<idx_t,1>    Idx1;
typedef Tuple<idx_t,2>    Idx2;
typedef Tuple<idx_t,3>    Idx3;
typedef Tuple<idx_t,4>    Idx4;
typedef Tuple<idx_t,4>    Idx5;
typedef Tuple<idx_t,4>    Idx6;

typedef Tuple<ulint,1>    Ulint1;
typedef Tuple<ulint,2>    Ulint2;
typedef Tuple<ulint,3>    Ulint3;
typedef Tuple<ulint,4>    Ulint4;
typedef Tuple<ulint,4>    Ulint5;
typedef Tuple<ulint,4>    Ulint6;

enum                    { MAX_TABLE_BITS = 31 };


//-----------------------------------------------------------------------
//   class Errors
//-----------------------------------------------------------------------


class Errors
{
 public:

  static const char*      INVALID_SHAPE;
  static const char*      INVALID_INDEX;

};


//-----------------------------------------------------------------------
//   utility functions
//-----------------------------------------------------------------------


ulint                     getTableBits

  ( ulint                   n );

ulint                     largerPrime

  ( idx_t                   n );

ulint                     modmul

  ( ulint                   i,
    ulint                   j,
    ulint                   n );


//-----------------------------------------------------------------------
//   hashValue
//-----------------------------------------------------------------------


lint                      hashValue

  ( const idx_t*            i,
    int                     rank );


template <int N>

  inline lint             hashValue

  ( const Tuple<idx_t,N>&   i )

{
  return hashValue ( i.addr(), N );
}


inline lint               hashValue

  ( const Idx1&             i )

{
  return (lint) i[0];
}


inline lint               hashValue

  ( const Idx2&             i )

{
  ulint  k;

  k  = (ulint) i[0] + 0x9e3779b9_ulint;
  k ^= (ulint) i[1] + 0x9e3779b9_ulint + (k << 6) + (k >> 2);

  return (lint) k;
}


inline lint               hashValue

  ( const Idx3&             i )

{
  ulint  k;

  k  = (ulint) i[0] + 0x9e3779b9_ulint;
  k ^= (ulint) i[1] + 0x9e3779b9_ulint + (k << 6) + (k >> 2);
  k ^= (ulint) i[2] + 0x9e3779b9_ulint + (k << 6) + (k >> 2);

  return (lint) k;
}


inline lint               hashValue

  ( const Idx4&             i )

{
  ulint  k;

  k  = (ulint) i[0] + 0x9e3779b9_ulint;
  k ^= (ulint) i[1] + 0x9e3779b9_ulint + (k << 6) + (k >> 2);
  k ^= (ulint) i[2] + 0x9e3779b9_ulint + (k << 6) + (k >> 2);
  k ^= (ulint) i[3] + 0x9e3779b9_ulint + (k << 6) + (k >> 2);

  return (lint) k;
}


ulint                     hashValue

  ( const idx_t*            i,
    const ulint*            p,
    ulint                   n,
    ulint                   mask,
    int                     rank );

template <int N>

  inline ulint            hashValue

  ( const Tuple<idx_t,N>&   i,
    const Tuple<ulint,N>&   p,
    ulint                   n,
    ulint                   mask )

{
  return hashValue ( i.addr(), p.addr(), n, mask, N );
}


inline ulint              hashValue

  ( const Idx1&             i,
    const Ulint1&           p,
    ulint                   n,
    ulint                   mask )

{
  return ((ulint) i[0] & mask);
}


inline ulint              hashValue

  ( const Idx2&             i,
    const Ulint2&           p,
    ulint                   n,
    ulint                   mask )

{
  ulint  k;

  k = (ulint) i[0] & mask;
  k = (k + modmul( (ulint) i[1], p[1], n )) & mask;

  return k;
}


inline ulint              hashValue

  ( const Idx3&             i,
    const Ulint3&           p,
    ulint                   n,
    ulint                   mask )

{
  ulint  k;

  k = (ulint) i[0] & mask;
  k = (k + modmul( (ulint) i[1], p[1], n )) & mask;
  k = (k + modmul( (ulint) i[2], p[2], n )) & mask;

  return k;
}


inline ulint              hashValue

  ( const Idx4&             i,
    const Ulint4&           p,
    ulint                   n,
    ulint                   mask )

{
  ulint  k;

  k = (ulint) i[0] & mask;
  k = (k + modmul( (ulint) i[1], p[1], n )) & mask;
  k = (k + modmul( (ulint) i[2], p[2], n )) & mask;
  k = (k + modmul( (ulint) i[3], p[3], n )) & mask;

  return k;
}


//-----------------------------------------------------------------------
//   largerEqual
//-----------------------------------------------------------------------


bool                      largerEqual

  ( const idx_t*            lhs,
    const idx_t*            rhs,
    int                     rank );

template <int N>

  inline bool             largerEqual

  ( const Tuple<idx_t,N>&   lhs,
    const Tuple<idx_t,N>&   rhs )

{
  return largerEqual ( lhs.addr(), rhs.addr(), N );
}


inline bool               largerEqual

  ( const Idx1&             lhs,
    const Idx1&             rhs )

{
  return (lhs[0] >= rhs[0]);
}


inline bool               largerEqual

  ( const Idx2&             lhs,
    const Idx2&             rhs )

{
  return (((idx_t) (lhs[0] >= rhs[0]) |
           (idx_t) (lhs[1] >= rhs[1])) != 0);
}


inline bool               largerEqual

  ( const Idx3&             lhs,
    const Idx3&             rhs )

{
  return (((idx_t) (lhs[0] >= rhs[0]) |
           (idx_t) (lhs[1] >= rhs[1]) |
           (idx_t) (lhs[2] >= rhs[2])) != 0);
}


inline bool               largerEqual

  ( const Idx4&             lhs,
    const Idx4&             rhs )

{
  return (((idx_t) (lhs[0] >= rhs[0]) |
           (idx_t) (lhs[1] >= rhs[1]) |
           (idx_t) (lhs[2] >= rhs[2]) |
           (idx_t) (lhs[3] >= rhs[3])) != 0);
}


//-----------------------------------------------------------------------
//   updateShape
//-----------------------------------------------------------------------


void                      updateShape

  ( idx_t*                  sh,
    const idx_t*            i,
    int                     rank );

template <int N>

  inline void             updateShape

  ( Tuple<idx_t,N>&         sh,
    const Tuple<idx_t,N>&   i )

{
  updateShape ( sh.addr(), i.addr(), N );
}


//-----------------------------------------------------------------------
//   updateBounds
//-----------------------------------------------------------------------


void                      updateBounds

  ( idx_t*                  b,
    const idx_t*            sh,
    double                  x,
    int                     rank );

template <int N>

  inline void             updateBounds

  ( Tuple<idx_t,N>&         b,
    const Tuple<idx_t,N>&   sh,
    double                  x )

{
  updateBounds ( b.addr(), sh.addr(), x, N );
}


//-----------------------------------------------------------------------
//   updateHash
//-----------------------------------------------------------------------


void                      updateHash

  ( ulint*                  p,
    const idx_t*            b,
    ulint                   n,
    int                     rank );

template <int N>

  inline void             updateHash

  ( Tuple<ulint,N>&         p,
    const Tuple<idx_t,N>&   b,
    ulint                   n )

{
  updateHash ( p.addr(), b.addr(), n, N );
}


JEM_END_NAMESPACE( sparse )


//=======================================================================
//   class SparseArray::Element
//=======================================================================


template <class T, int N, class A>

  inline SparseArray<T,N,A>::Element::Element

  ( const Index&  i,
    const T&      t ) :

    index ( i ),
    value ( t )

{}


template <class T, int N, class A>

  inline SparseArray<T,N,A>::Element::Element

  ( const Element&  rhs ) :

    index ( rhs.index ),
    value ( rhs.value )

{}


//=======================================================================
//   class SparseArray
//=======================================================================

//-----------------------------------------------------------------------
//   default constructor
//-----------------------------------------------------------------------


template <class T, int N, class A>

  SparseArray<T,N,A>::SparseArray () :

    allocator_ ( sizeof(Node) ),
    table_     ( nullptr ),
    zero_      ( nullptr )

{
  Shape  sh;

  sh = 0;

  try
  {
    init_ ( T(), sh, 10 );
  }
  catch ( ... )
  {
    clear_ ();
    throw;
  }
}


//-----------------------------------------------------------------------
//   zero constructor
//-----------------------------------------------------------------------


template <class T, int N, class A>

  SparseArray<T,N,A>::SparseArray ( const T& zero ) :

    allocator_ ( sizeof(Node) ),
    table_     ( nullptr ),
    zero_      ( nullptr )

{
  Shape  sh;

  sh = 0;

  try
  {
    init_ ( zero, sh, 10 );
  }
  catch ( ... )
  {
    clear_ ();
    throw;
  }
}


//-----------------------------------------------------------------------
//   copy constructor
//-----------------------------------------------------------------------


template <class T, int N, class A>

  SparseArray<T,N,A>::SparseArray

  ( const SparseArray& rhs ) :

    allocator_ ( sizeof(Node) ),
    table_     ( nullptr ),
    zero_      ( nullptr )

{
  try
  {
    init_ ( * rhs.zero_, rhs.shape_, rhs.capacity_ );
    copy_ ( rhs );
  }
  catch ( ... )
  {
    clear_ ();
    throw;
  }
}


template <class T, int N, class A>

  SparseArray<T,N,A>::SparseArray

  ( SparseArray&& rhs ) : SparseArray ()

{
  swap ( rhs );
}


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


template <class T, int N, class A>

  SparseArray<T,N,A>::~SparseArray ()

{
  clear_ ();
}


//-----------------------------------------------------------------------
//   begin
//-----------------------------------------------------------------------


template <class T, int N, class A>

  inline typename SparseArray<T,N,A>::Iterator
  SparseArray<T,N,A>::begin () noexcept

{
  return Iterator ( table_ );
}


template <class T, int N, class A>

  inline typename SparseArray<T,N,A>::ConstIterator
  SparseArray<T,N,A>::begin () const noexcept

{
  return ConstIterator ( table_ );
}


template <class T, int N, class A>

  inline typename SparseArray<T,N,A>::Iterator
  SparseArray<T,N,A>::end () noexcept

{
  const ulint  n = 1_ulint << tableBits_;

  return Iterator ( table_ + n, table_[n] );
}


template <class T, int N, class A>

  inline typename SparseArray<T,N,A>::ConstIterator
  SparseArray<T,N,A>::end () const noexcept

{
  const ulint  n = 1_ulint << tableBits_;

  return ConstIterator ( table_ + n, table_[n] );
}


//-----------------------------------------------------------------------
//   operator =
//-----------------------------------------------------------------------


template <class T, int N, class A>

  SparseArray<T,N,A>&
  SparseArray<T,N,A>::operator = ( const SparseArray& rhs )

{
  if ( this != & rhs )
  {
    clear_ ();

    try
    {
      init_ ( * rhs.zero_, rhs.shape_, rhs.capacity_ );
    }
    catch ( ... )
    {
      clear_ ();
      throw;
    }

    copy_ ( rhs );
  }

  return *this;
}


template <class T, int N, class A>

  SparseArray<T,N,A>&
  SparseArray<T,N,A>::operator = ( SparseArray&& rhs ) noexcept

{
  swap ( rhs );

  return *this;
}


template <class T, int N, class A>

  SparseArray<T,N,A>&
  SparseArray<T,N,A>::operator = ( T rhs )

{
  const ulint  n = 1_ulint << tableBits_;

  Node*  node;

  for ( ulint i = 0; i < n; i++ )
  {
    node = static_cast<Node*> ( table_[i] );

    while ( node )
    {
      node->value.value = rhs;
      node = node->next;
    }
  }

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T, int N, class A>

  void SparseArray<T,N,A>::swap ( SparseArray& rhs ) noexcept

{
  allocator_.swap ( rhs.allocator_ );
  shape_    .swap ( rhs.shape_     );
  bounds_   .swap ( rhs.bounds_    );
  params_   .swap ( rhs.params_    );

  jem::swap ( table_,      rhs.table_     );
  jem::swap ( tableBits_,  rhs.tableBits_ );
  jem::swap ( tableMask_,  rhs.tableMask_ );
  jem::swap ( nodeCount_,  rhs.nodeCount_ );
  jem::swap ( capacity_,   rhs.capacity_  );
  jem::swap ( zero_,       rhs.zero_      );
}


//-----------------------------------------------------------------------
//   operator []
//-----------------------------------------------------------------------


template <class T, int N, class A>

  T& SparseArray<T,N,A>::operator [] ( const Index& i )

{
  JEM_ASSERT2 ( min( i ) >= 0, sparse::Errors::INVALID_INDEX );

  Node*  node;
  lint   hash;
  ulint  j;


  hash = sparse::hashValue ( i );
  j    = sparse::hashValue ( i, params_, tableBits_, tableMask_ );
  node = static_cast<Node*> ( table_[j] );

  while ( node )
  {
    if ( (node->hash == hash) && equal( node->value.index, i ) )
    {
      return node->value.value;
    }

    node = node->next;
  }

  node       = newNode_ ( Element( i, *zero_ ), hash );
  node->next = static_cast<Node*> ( table_[j] );
  table_[j]  = node;

  if ( sparse::largerEqual( i, shape_ ) )
  {
    sparse::updateShape ( shape_, i );

    if ( sparse::largerEqual( i, bounds_ ) )
    {
      sparse::updateBounds ( bounds_, shape_, 1.5 );
      rehash_              ( false );
    }
  }

  if ( nodeCount_ > capacity_ )
  {
    reserve_ ( 2 * capacity_ + 1 );
  }

  return node->value.value;
}


template <class T, int N, class A>

  inline T SparseArray<T,N,A>::operator []

  ( const Index& i ) const

{
  JEM_ASSERT2 ( min( i ) >= 0, sparse::Errors::INVALID_INDEX );

  T*  t = find_ ( i );

  if ( t )
  {
    return *t;
  }
  else
  {
    return *zero_;
  }
}


template <class T, int N, class A>

  inline T& SparseArray<T,N,A>::operator [] ( idx_t i )

{
  static_assert ( RANK == 1, "invalid rank" );

  return operator [] ( Index( i ) );
}


template <class T, int N, class A>

  inline T SparseArray<T,N,A>::operator [] ( idx_t i ) const

{
  static_assert ( RANK == 1, "invalid rank" );

  return operator [] ( Index( i ) );
}


//-----------------------------------------------------------------------
//   operator ()
//-----------------------------------------------------------------------


template <class T, int N, class A>

  inline T& SparseArray<T,N,A>::operator ()

  ( idx_t  i,
    idx_t  j )

{
  static_assert ( RANK == 2, "invalid rank" );

  return operator [] ( Index( i, j ) );
}


template <class T, int N, class A>

  inline T SparseArray<T,N,A>::operator ()

  ( idx_t  i,
    idx_t  j ) const

{
  static_assert ( RANK == 2, "invalid rank" );

  return operator [] ( Index( i, j ) );
}


template <class T, int N, class A>

  inline T& SparseArray<T,N,A>::operator ()

  ( idx_t  i,
    idx_t  j,
    idx_t  k )

{
  static_assert ( RANK == 3, "invalid rank" );

  return operator [] ( Index( i, j, k ) );
}


template <class T, int N, class A>

  inline T SparseArray<T,N,A>::operator ()

  ( idx_t  i,
    idx_t  j,
    idx_t  k ) const

{
  static_assert ( RANK == 3, "invalid rank" );

  return operator [] ( Index( i, j, k ) );
}


template <class T, int N, class A>

  inline T& SparseArray<T,N,A>::operator ()

  ( idx_t  i,
    idx_t  j,
    idx_t  k,
    idx_t  l )

{
  static_assert ( RANK == 4, "invalid rank" );

  return operator [] ( Index( i, j, k, l ) );
}


template <class T, int N, class A>

  inline T SparseArray<T,N,A>::operator ()

  ( idx_t  i,
    idx_t  j,
    idx_t  k,
    idx_t  l ) const

{
  static_assert ( RANK == 4, "invalid rank" );

  return operator [] ( Index( i, j, k, l ) );
}


template <class T, int N, class A>

  inline T& SparseArray<T,N,A>::operator ()

  ( idx_t  i,
    idx_t  j,
    idx_t  k,
    idx_t  l,
    idx_t  p )

{
  static_assert ( RANK == 5, "invalid rank" );

  return operator [] ( Index( i, j, k, l, p ) );
}


template <class T, int N, class A>

  inline T SparseArray<T,N,A>::operator ()

  ( idx_t  i,
    idx_t  j,
    idx_t  k,
    idx_t  l,
    idx_t  p ) const

{
  static_assert ( RANK == 5, "invalid rank" );

  return operator [] ( Index( i, j, k, l, p ) );
}


template <class T, int N, class A>

  inline T& SparseArray<T,N,A>::operator ()

  ( idx_t  i,
    idx_t  j,
    idx_t  k,
    idx_t  l,
    idx_t  p,
    idx_t  q )

{
  static_assert ( RANK == 6, "invalid rank" );

  return operator [] ( Index( i, j, k, l, p, q ) );
}


template <class T, int N, class A>

  inline T SparseArray<T,N,A>::operator ()

  ( idx_t  i,
    idx_t  j,
    idx_t  k,
    idx_t  l,
    idx_t  p,
    idx_t  q ) const

{
  static_assert ( RANK == 6, "invalid rank" );

  return operator [] ( Index( i, j, k, l, p, q ) );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


template <class T, int N, class A>

  inline T* SparseArray<T,N,A>::find ( const Index& i )

{
  JEM_ASSERT2 ( min( i ) >= 0, sparse::Errors::INVALID_INDEX );

  return find_ ( i );
}


template <class T, int N, class A>

  inline const T* SparseArray<T,N,A>::find

  ( const Index& i ) const

{
  JEM_ASSERT2 ( min( i ) >= 0, sparse::Errors::INVALID_INDEX );

  return find_ ( i );
}


template <class T, int N, class A>

  inline T* SparseArray<T,N,A>::find ( idx_t i )

{
  static_assert ( RANK == 1, "invalid rank" );

  return find_ ( Index( i ) );
}


template <class T, int N, class A>

  inline const T* SparseArray<T,N,A>::find ( idx_t i ) const

{
  static_assert ( RANK == 1, "invalid rank" );

  return find_ ( Index( i ) );
}


template <class T, int N, class A>

  inline T* SparseArray<T,N,A>::find

  ( idx_t  i,
    idx_t  j )

{
  static_assert ( RANK == 2, "invalid rank" );

  return find_ ( Index( i, j ) );
}


template <class T, int N, class A>

  inline const T* SparseArray<T,N,A>::find

  ( idx_t  i,
    idx_t  j ) const

{
  static_assert ( RANK == 2, "invalid rank" );

  return find_ ( Index( i, j ) );
}


template <class T, int N, class A>

  inline T* SparseArray<T,N,A>::find

  ( idx_t  i,
    idx_t  j,
    idx_t  k )

{
  static_assert ( RANK == 3, "invalid rank" );

  return find_ ( Index( i, j, k ) );
}


template <class T, int N, class A>

  inline const T* SparseArray<T,N,A>::find

  ( idx_t  i,
    idx_t  j,
    idx_t  k ) const

{
  static_assert ( RANK == 3, "invalid rank" );

  return find_ ( Index( i, j, k ) );
}


template <class T, int N, class A>

  inline T* SparseArray<T,N,A>::find

  ( idx_t  i,
    idx_t  j,
    idx_t  k,
    idx_t  l )

{
  static_assert ( RANK == 4, "invalid rank" );

  return find_ ( Index( i, j, k, l ) );
}


template <class T, int N, class A>

  inline const T* SparseArray<T,N,A>::find

  ( idx_t  i,
    idx_t  j,
    idx_t  k,
    idx_t  l ) const

{
  static_assert ( RANK == 4, "invalid rank" );

  return find_ ( Index( i, j, k, l ) );
}


template <class T, int N, class A>

  inline T* SparseArray<T,N,A>::find

  ( idx_t  i,
    idx_t  j,
    idx_t  k,
    idx_t  l,
    idx_t  p )

{
  static_assert ( RANK == 5, "invalid rank" );

  return find_ ( Index( i, j, k, l, p ) );
}


template <class T, int N, class A>

  inline const T* SparseArray<T,N,A>::find

  ( idx_t  i,
    idx_t  j,
    idx_t  k,
    idx_t  l,
    idx_t  p ) const

{
  static_assert ( RANK == 5, "invalid rank" );

  return find_ ( Index( i, j, k, l, p ) );
}


template <class T, int N, class A>

  inline T* SparseArray<T,N,A>::find

  ( idx_t  i,
    idx_t  j,
    idx_t  k,
    idx_t  l,
    idx_t  p,
    idx_t  q )

{
  static_assert ( RANK == 6, "invalid rank" );

  return find_ ( Index( i, j, k, l, p, q ) );
}


template <class T, int N, class A>

  inline const T* SparseArray<T,N,A>::find

  ( idx_t  i,
    idx_t  j,
    idx_t  k,
    idx_t  l,
    idx_t  p,
    idx_t  q ) const

{
  static_assert ( RANK == 6, "invalid rank" );

  return find_ ( Index( i, j, k, l, p, q ) );
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


template <class T, int N, class A>

  bool SparseArray<T,N,A>::erase ( const Index& i )

{
  Node*  node;
  Node*  prev;
  lint   hash;
  ulint  j;


  hash = sparse::hashValue ( i );
  j    = sparse::hashValue ( i, params_, tableBits_, tableMask_ );
  node = static_cast<Node*> ( table_[j] );
  prev = nullptr;

  while ( node )
  {
    if ( (node->hash == hash) && equal( node->value.index, i ) )
    {
      if ( prev )
      {
        prev->next = node->next;
      }
      else
      {
        table_[j] = node->next;
      }

      free_ ( node );
      return true;
    }

    prev = node;
    node = node->next;
  }

  return false;
}


template <class T, int N, class A>

  inline bool SparseArray<T,N,A>::erase ( idx_t i )

{
  static_assert ( RANK == 1, "invalid rank" );

  return erase ( Index( i ) );
}


template <class T, int N, class A>

  inline bool SparseArray<T,N,A>::erase

  ( idx_t  i,
    idx_t  j )

{
  static_assert ( RANK == 2, "invalid rank" );

  return erase ( Index( i, j ) );
}


template <class T, int N, class A>

  inline bool SparseArray<T,N,A>::erase

  ( idx_t  i,
    idx_t  j,
    idx_t  k )

{
  static_assert ( RANK == 3, "invalid rank" );

  return erase ( Index( i, j, k ) );
}


template <class T, int N, class A>

  inline bool SparseArray<T,N,A>::erase

  ( idx_t  i,
    idx_t  j,
    idx_t  k,
    idx_t  l )

{
  static_assert ( RANK == 4, "invalid rank" );

  return erase ( Index( i, j, k, l ) );
}


template <class T, int N, class A>

  inline bool SparseArray<T,N,A>::erase

  ( idx_t  i,
    idx_t  j,
    idx_t  k,
    idx_t  l,
    idx_t  p )

{
  static_assert ( RANK == 5, "invalid rank" );

  return erase ( Index( i, j, k, l, p ) );
}


template <class T, int N, class A>

  inline bool SparseArray<T,N,A>::erase

  ( idx_t  i,
    idx_t  j,
    idx_t  k,
    idx_t  l,
    idx_t  p,
    idx_t  q )

{
  static_assert ( RANK == 6, "invalid rank" );

  return erase ( Index( i, j, k, l, p, q ) );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


template <class T, int N, class A>

  void SparseArray<T,N,A>::clear ()

{
  const ulint  n = 1_ulint << tableBits_;

  Node*  node;

  for ( ulint i = 0; i < n; i++ )
  {
    while ( table_[i] )
    {
      node      = static_cast<Node*> ( table_[i] );
      table_[i] = node->next;

      free_ ( node );
    }
  }

  bounds_ = shape_ = 0;

  sparse::updateHash ( params_, bounds_, tableBits_ );
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


template <class T, int N, class A>

  void SparseArray<T,N,A>::reserve ( idx_t cap )

{
  if ( cap > capacity_ )
  {
    reserve_ ( cap );
  }
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


template <class T, int N, class A>

  void SparseArray<T,N,A>::trimToSize ()

{
  reserve_ ( nodeCount_ );
}


//-----------------------------------------------------------------------
//   capacity
//-----------------------------------------------------------------------


template <class T, int N, class A>

  inline idx_t SparseArray<T,N,A>::capacity () const noexcept

{
  return capacity_;
}


//-----------------------------------------------------------------------
//   reshape
//-----------------------------------------------------------------------


template <class T, int N, class A>

  void SparseArray<T,N,A>::reshape ()

{
  const ulint  n = 1_ulint << tableBits_;

  Shape  sh;
  Node*  node;


  sh = 0;

  for ( ulint i = 0; i < n; i++ )
  {
    node = static_cast<Node*> ( table_[i] );

    while ( node )
    {
      sparse::updateShape ( sh, node->value.index );
      node = node->next;
    }
  }

  if ( ! equal( sh, shape_ ) )
  {
    shape_  = sh;
    bounds_ = sh;
    rehash_ ( false );
  }
}


template <class T, int N, class A>

  void SparseArray<T,N,A>::reshape ( const Shape& sh )

{
  JEM_PRECHECK2 ( min( sh ) >= 0, sparse::Errors::INVALID_SHAPE );

  if ( ! equal( sh, shape_ ) )
  {
    shape_  = sh;
    bounds_ = sh;
    rehash_ ( true );
  }
}


template <class T, int N, class A>

  inline void SparseArray<T,N,A>::reshape ( idx_t m )

{
  static_assert ( RANK == 1, "invalid rank" );

  reshape ( Shape( m ) );
}


template <class T, int N, class A>

  inline void SparseArray<T,N,A>::reshape

  ( idx_t  m,
    idx_t  n )

{
  static_assert ( RANK == 2, "invalid rank" );

  reshape ( Shape( m, n ) );
}


template <class T, int N, class A>

  inline void SparseArray<T,N,A>::reshape

  ( idx_t  m,
    idx_t  n,
    idx_t  p )

{
  static_assert ( RANK == 3, "invalid rank" );

  reshape ( Shape( m, n, p ) );
}


template <class T, int N, class A>

  inline void SparseArray<T,N,A>::reshape

  ( idx_t  m,
    idx_t  n,
    idx_t  p,
    idx_t  q )

{
  static_assert ( RANK == 14, "invalid rank" );

  reshape ( Shape( m, n, p, q ) );
}


template <class T, int N, class A>

  inline void SparseArray<T,N,A>::reshape

  ( idx_t  m,
    idx_t  n,
    idx_t  p,
    idx_t  q,
    idx_t  r )

{
  static_assert ( RANK == 5, "invalid rank" );

  reshape ( Shape( m, n, p, q, r ) );
}


template <class T, int N, class A>

  inline void SparseArray<T,N,A>::reshape

  ( idx_t  m,
    idx_t  n,
    idx_t  p,
    idx_t  q,
    idx_t  r,
    idx_t  s )

{
  static_assert ( RANK == 6, "invalid rank" );

  reshape ( Shape( m, n, p, q, r, s ) );
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


template <class T, int N, class A>

  inline const typename SparseArray<T,N,A>::Shape&
  SparseArray<T,N,A>::shape () const noexcept

{
  return shape_;
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


template <class T, int N, class A>

  inline idx_t SparseArray<T,N,A>::size () const noexcept

{
  static_assert ( RANK == 1, "invalid rank" );

  return shape_[0];
}


template <class T, int N, class A>

  inline idx_t SparseArray<T,N,A>::size ( int idim ) const

{
  return shape_[idim];
}


//-----------------------------------------------------------------------
//   getZero
//-----------------------------------------------------------------------


template <class T, int N, class A>

  inline const T& SparseArray<T,N,A>::getZero () const noexcept

{
  return *zero_;
}


//-----------------------------------------------------------------------
//   setZero
//-----------------------------------------------------------------------


template <class T, int N, class A>

  void SparseArray<T,N,A>::setZero ( const T& zero )

{
  T*  tmp = new T ( zero );

  jem::swap ( zero_, tmp );

  delete tmp;
}


//-----------------------------------------------------------------------
//   nonZeroCount
//-----------------------------------------------------------------------


template <class T, int N, class A>

  inline idx_t SparseArray<T,N,A>::nonZeroCount () const noexcept

{
  return nodeCount_;
}


//-----------------------------------------------------------------------
//   printStats
//-----------------------------------------------------------------------


template <class T, int N, class A>
  template <class Output>

  void SparseArray<T,N,A>::printStats ( Output& out ) const

{
  const ulint  n = 1_ulint << tableBits_;

  Node*   node;
  double  avg;
  double  dev;
  idx_t   chainCount;
  idx_t   minLength;
  idx_t   maxLength;
  idx_t   j;


  chainCount = 0;

  for ( ulint i = 0; i < n; i++ )
  {
    if ( table_[i] )
    {
      chainCount++;
    }
  }

  minLength = 0;
  maxLength = 0;
  dev       = 0.0;
  avg       = 0.0;

  if ( chainCount > 0 )
  {
    avg = (double) nodeCount_ / (double) chainCount;
  }

  for ( ulint i = 0; i < n; i++ )
  {
    node = static_cast<Node*> ( table_[i] );

    if ( ! node )
    {
      continue;
    }

    j = 0;

    while ( node )
    {
      j++;
      node = node->next;
    }

    if ( minLength == 0 || j < minLength )
    {
      minLength = j;
    }
    if ( j > maxLength )
    {
      maxLength = j;
    }

    dev += ((double) j - avg) * ((double) j - avg);
  }

  if ( chainCount > 0 )
  {
    dev = std::sqrt ( dev ) / (double) chainCount;
  }
  else
  {
    dev = 0.0;
  }

  print ( out, "SparseArray statistics:" );

  print ( out, "\n  shape                  : ", shape_     );
  print ( out, "\n  number of non-zeroes   : ", nodeCount_ );
  print ( out, "\n  total number of chains : ", (lint) n   );
  print ( out, "\n  number of used chains  : ", chainCount );
  print ( out, "\n  minimum chain length   : ", minLength  );
  print ( out, "\n  maximum chain length   : ", maxLength  );
  print ( out, "\n  average chain length   : ", avg        );
  print ( out, "\n  chain length deviation : ", dev , '\n' );
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


template <class T, int N, class A>

  void SparseArray<T,N,A>::init_

  ( const T&      zero,
    const Shape&  sh,
    idx_t         cap )

{
  ulint  n;

  shape_      = sh;
  bounds_     = sh;
  tableBits_  = sparse::getTableBits ( (ulint) cap );
  tableMask_  = (1_ulint << tableBits_) - 1_ulint;
  nodeCount_  = 0;
  capacity_   = cap;
  zero_       = new T ( zero );

  n      = 1_ulint << tableBits_;
  table_ = (void**)

    MemCache::alloc ( sizeof(*table_) * (size_t) (n + 1) );

  for ( ulint i = 0; i < n; i++ )
  {
    table_[i] = nullptr;
  }

  table_[n] = table_;

  sparse::updateHash ( params_, bounds_, tableBits_ );
}


//-----------------------------------------------------------------------
//   clear_
//-----------------------------------------------------------------------


template <class T, int N, class A>

  void SparseArray<T,N,A>::clear_ ()

{
  const ulint  n = 1_ulint << tableBits_;

  if ( table_ )
  {
    clear ();
    MemCache::dealloc ( table_, sizeof(*table_) * (size_t) (n + 1) );
    table_ = nullptr;
  }

  if ( zero_ )
  {
    delete zero_;
    zero_  = nullptr;
  }
}


//-----------------------------------------------------------------------
//   copy_
//-----------------------------------------------------------------------


template <class T, int N, class A>

  void SparseArray<T,N,A>::copy_ ( const SparseArray&  rhs )

{
  JEM_ASSERT ( nodeCount_ == 0 );

  const ulint  n = 1_ulint << rhs.tableBits_;

  Node*  node;
  Node*  clone;
  ulint  i, j;


  for ( i = 0; i < n; i++ )
  {
    node = static_cast<Node*> ( rhs.table_[i] );

    while ( node )
    {
      clone = newNode_          ( node->value, node->hash );
      j     = sparse::hashValue ( clone->value.index,
                                  params_,
                                  tableBits_, tableMask_ );

      clone->next = static_cast<Node*> ( table_[j] );
      table_[j]   = clone;
      node        = node->next;
    }
  }
}


//-----------------------------------------------------------------------
//   find_
//-----------------------------------------------------------------------


template <class T, int N, class A>

  inline T* SparseArray<T,N,A>::find_ ( const Index& i ) const

{
  Node*  node;
  lint   hash;
  ulint  j;


  hash = sparse::hashValue ( i );
  j    = sparse::hashValue ( i, params_, tableBits_, tableMask_ );
  node = static_cast<Node*> ( table_[j] );

  while ( node )
  {
    if ( (node->hash == hash) && equal( node->value.index, i ) )
    {
      return & node->value.value;
    }

    node = node->next;
  }

  return nullptr;
}


//-----------------------------------------------------------------------
//   rehash_
//-----------------------------------------------------------------------


template <class T, int N, class A>

  void SparseArray<T,N,A>::rehash_ ( bool prune )

{
  const ulint  n = 1_ulint << tableBits_;

  Node*  list;
  Node*  node;
  Node*  next;
  ulint  i;


  sparse::updateHash ( params_, bounds_, tableBits_ );

  list = nullptr;

  for ( i = 0; i < n; i++ )
  {
    node      = static_cast<Node*> ( table_[i] );
    table_[i] = nullptr;

    while ( node )
    {
      next = node->next;

      if ( prune &&
           sparse::largerEqual( node->value.index, shape_ ) )
      {
        free_ ( node );
      }
      else
      {
        node->next = list;
        list       = node;
      }

      node = next;
    }
  }

  while ( list )
  {
    node = list;
    list = list->next;

    i = sparse::hashValue ( node->value.index,
                            params_,
                            tableBits_, tableMask_ );

    node->next = static_cast<Node*> ( table_[i] );
    table_[i]  = node;
  }
}


//-----------------------------------------------------------------------
//   reserve_
//-----------------------------------------------------------------------


template <class T, int N, class A>

  void SparseArray<T,N,A>::reserve_ ( idx_t cap )

{
  void**  oldTable = table_;

  void**  newTable;
  ulint   newBits;
  ulint   newSize;
  ulint   newMask;
  ulint   oldSize;
  Node*   node;
  ulint   i, j;


  if ( cap < nodeCount_ )
  {
    cap = nodeCount_;
  }

  oldSize  = 1_ulint << tableBits_;
  newBits  = sparse::getTableBits ( (ulint) cap );
  newSize  = 1_ulint << newBits;
  newMask  = newSize - 1_ulint;
  newTable = (void**)

    MemCache::alloc ( sizeof(void*) * (size_t) (newSize + 1) );

  for ( i = 0; i < newSize; i++ )
  {
    newTable[i] = nullptr;
  }

  newTable[newSize] = newTable;

  sparse::updateHash ( params_, bounds_, newBits );

  for ( i = 0; i < oldSize; i++ )
  {
    while ( oldTable[i] )
    {
      node        = static_cast<Node*> ( oldTable[i] );
      oldTable[i] = node->next;

      j = sparse::hashValue ( node->value.index,
                              params_,
                              newBits, newMask );

      node->next  = static_cast<Node*> ( newTable[j] );
      newTable[j] = node;
    }
  }

  table_     = newTable;
  tableBits_ = newBits;
  tableMask_ = newMask;
  capacity_  = cap;

  MemCache::dealloc ( oldTable,
                      sizeof(*oldTable) * (size_t) (oldSize + 1) );
}


//-----------------------------------------------------------------------
//   newNode_
//-----------------------------------------------------------------------


template <class T, int N, class A>

  inline typename SparseArray<T,N,A>::Node*
  SparseArray<T,N,A>::newNode_

  ( const Element&  elem,
    lint            hash )

{
  Node*  node = (Node*) allocator_.alloc ();

  try
  {
    new (node) Node ( elem, hash );
  }
  catch ( ... )
  {
    allocator_.dealloc ( node );
    throw;
  }

  nodeCount_++;
  return node;
}


//-----------------------------------------------------------------------
//   free_
//-----------------------------------------------------------------------


template <class T, int N, class A>

  inline void SparseArray<T,N,A>::free_ ( Node* node )

{
  nodeCount_--;
  node->~Node();
  allocator_.dealloc ( node );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T, int N, class A>

  inline void           swap

  ( SparseArray<T,N,A>&   lhs,
    SparseArray<T,N,A>&   rhs ) noexcept

{
  lhs.swap ( rhs );
}


//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input,
          class T, int N, class A>

  void                        decode

  ( Input&                      in,
    SparseArray<T,N,A>&         spa )

{
  typename SparseArray<T,N,A>::Shape shape;
  typename SparseArray<T,N,A>::Index index;

  uidx_t  sz;
  idx_t   n;

  T  zero;
  T  value;

  for ( idx_t i = 0; i < N; i++ )
  {
    decode ( in, sz );

    shape[i] = (idx_t) sz;
  }

  decode ( in, zero, sz );

  n = (idx_t) sz;

  spa.clear   ();
  spa.setZero ( zero  );
  spa.reshape ( shape );

  for ( idx_t i = 0; i < n; i++ )
  {
    decode ( in, index, value );

    spa[index] = value;
  }
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output,
          class T, int N, class A>

  void                        encode

  ( Output&                     out,
    const SparseArray<T,N,A>&   spa )

{
  typename SparseArray<T,N,A>::ConstIterator  first = spa.begin ();
  typename SparseArray<T,N,A>::ConstIterator  last  = spa.end   ();

  const Tuple<idx_t,N>  sh = spa.shape ();

  for ( int i = 0; i < N; i++ )
  {
    encode ( out, sh[i] );
  }

  encode ( out, spa.getZero(), spa.nonZeroCount() );

  for ( ; first != last; ++first )
  {
    encode ( out, first->index, first->value );
  }
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output,
          class T, int N, class A>

  void                        print

  ( Output&                     out,
    const SparseArray<T,N,A>&   spa )

{
  typename SparseArray<T,N,A>::ConstIterator  first = spa.begin ();
  typename SparseArray<T,N,A>::ConstIterator  last  = spa.end   ();

  while ( first != last )
  {
    print ( out, first->index, " : ", first->value );

    ++first;

    if ( first != last )
    {
      print ( out, '\n' );
    }
  }
}


JEM_END_PACKAGE( util )

JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class TypeTraits< util::SparseArray<T,N,A> >
//=======================================================================


template <class T, int N, class A>

  class TypeTraits< util::SparseArray<T,N,A> > :

    public BasicTypeTraits

{
 public:

  static const bool   IS_SERIALIZABLE = IsSerializable<T>::VALUE;

  static String       whatis   ()
  {
    return ( TypeTraits<T>::whatis()               +
             String ( " sparse array with rank " ) +
             String ( N ) );
  }

  static String       typeName ()
  {
    String  tname = TypeTraits<T>::typeName ();
    String  aname = TypeTraits<A>::typeName ();

    return String::format ( "jem::util::SparseArray<%s,%d,%s>",
                            tname, N, aname );
  }

};


JEM_END_PACKAGE_BASE

#endif
