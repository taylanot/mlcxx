
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

#ifndef JEM_BASE_ARRAY_ARRAY_TCC
#define JEM_BASE_ARRAY_ARRAY_TCC


JEM_BEGIN_PACKAGE_BASE

JEM_BEGIN_NAMESPACE( array )


//=======================================================================
//   class Slicer<R,N>
//=======================================================================


template <class R, int N>

  class Slicer

{
 public:

  template<class T,
           int   K,
           class I>

  static R                  get

    ( const Array<T,K>&       a,
      const I&                i )

  {
    return R ( a, i );
  }


  template<class T,
           class I,
           class J>

  static R                  get

    ( const Array<T,2>&       a,
      const I&                i,
      const J&                j )

  {
    return R ( a, i, j );
  }


  template<class T,
           class I,
           class J,
           class K>

  static R                  get

    ( const Array<T,3>&       a,
      const I&                i,
      const J&                j,
      const K&                k )

  {
    return R ( a, i, j, k );
  }


  template<class T,
           class I,
           class J,
           class K,
           class L>

  static R                  get

    ( const Array<T,4>&       a,
      const I&                i,
      const J&                j,
      const K&                k,
      const L&                l )

  {
    return R ( a, i, j, k, l );
  }


  template<class T,
           class I,
           class J,
           class K,
           class L,
           class P>

  static R                  get

    ( const Array<T,5>&       a,
      const I&                i,
      const J&                j,
      const K&                k,
      const L&                l,
      const P&                p )

  {
    return R ( a, i, j, k, l, p );
  }


  template<class T,
           class I,
           class J,
           class K,
           class L,
           class P,
           class Q>

  static R                  get

    ( const Array<T,6>&       a,
      const I&                i,
      const J&                j,
      const K&                k,
      const L&                l,
      const P&                p,
      const Q&                q )

  {
    return R ( a, i, j, k, l, p, q );
  }

};


//=======================================================================
//   class Slicer<T,0>
//=======================================================================


template <class T>

  class Slicer<T&,0>

{
 public:

  template<class I>

  static T&                 get

    ( const Array<T,1>&       a,
      const I&                i )

  {
    JEM_ASSERT2 ( checkIndex( i, a.size() ), Errors::INVALID_INDEX );

    return a.data_[i * a.stride_];
  }


  template<class I,
           class J>

  static T&                 get

    ( const Array<T,2>&       a,
      const I&                i,
      const J&                j )

  {
    JEM_ASSERT2 ( checkIndex( i, a.size(0) ),
                  Errors::INVALID_INDEX0 );
    JEM_ASSERT2 ( checkIndex( j, a.size(1) ),
                  Errors::INVALID_INDEX1 );

    return a.data_[i * a.stride_[0] +
                   j * a.stride_[1]];
  }


  template<class I,
           class J,
           class K>

  static T&                 get

    ( const Array<T,3>&       a,
      const I&                i,
      const J&                j,
      const K&                k )

  {
    JEM_ASSERT2 ( checkIndex( i, a.size(0) ),
                  Errors::INVALID_INDEX0 );
    JEM_ASSERT2 ( checkIndex( j, a.size(1) ),
                  Errors::INVALID_INDEX1 );
    JEM_ASSERT2 ( checkIndex( k, a.size(2) ),
                  Errors::INVALID_INDEX2 );

    return a.data_[i * a.stride_[0] +
                   j * a.stride_[1] +
                   k * a.stride_[2]];
  }


  template<class I,
           class J,
           class K,
           class L>

  static T&                 get

    ( const Array<T,4>&       a,
      const I&                i,
      const J&                j,
      const K&                k,
      const L&                l )

  {
    JEM_ASSERT2 ( checkIndex( i, a.size(0) ),
                  Errors::INVALID_INDEX0 );
    JEM_ASSERT2 ( checkIndex( j, a.size(1) ),
                  Errors::INVALID_INDEX1 );
    JEM_ASSERT2 ( checkIndex( k, a.size(2) ),
                  Errors::INVALID_INDEX2 );
    JEM_ASSERT2 ( checkIndex( l, a.size(3) ),
                  Errors::INVALID_INDEX3 );

    return a.data_[i * a.stride_[0] +
                   j * a.stride_[1] +
                   k * a.stride_[2] +
                   l * a.stride_[3]];
  }


  template<class I,
           class J,
           class K,
           class L,
           class P>

  static T&                 get

    ( const Array<T,5>&       a,
      const I&                i,
      const J&                j,
      const K&                k,
      const L&                l,
      const P&                p )

  {
    JEM_ASSERT2 ( checkIndex( i, a.size(0) ),
                  Errors::INVALID_INDEX0 );
    JEM_ASSERT2 ( checkIndex( j, a.size(1) ),
                  Errors::INVALID_INDEX1 );
    JEM_ASSERT2 ( checkIndex( k, a.size(2) ),
                  Errors::INVALID_INDEX2 );
    JEM_ASSERT2 ( checkIndex( l, a.size(3) ),
                  Errors::INVALID_INDEX3 );
    JEM_ASSERT2 ( checkIndex( p, a.size(4) ),
                  Errors::INVALID_INDEX4 );

    return a.data_[i * a.stride_[0] +
                   j * a.stride_[1] +
                   k * a.stride_[2] +
                   l * a.stride_[3] +
                   p * a.stride_[4]];
  }


  template<class I,
           class J,
           class K,
           class L,
           class P,
           class Q>

  static T&                 get

    ( const Array<T,6>&       a,
      const I&                i,
      const J&                j,
      const K&                k,
      const L&                l,
      const P&                p,
      const Q&                q )

  {
    JEM_ASSERT2 ( checkIndex( i, a.size(0) ),
                  Errors::INVALID_INDEX0 );
    JEM_ASSERT2 ( checkIndex( j, a.size(1) ),
                  Errors::INVALID_INDEX1 );
    JEM_ASSERT2 ( checkIndex( k, a.size(2) ),
                  Errors::INVALID_INDEX2 );
    JEM_ASSERT2 ( checkIndex( l, a.size(3) ),
                  Errors::INVALID_INDEX3 );
    JEM_ASSERT2 ( checkIndex( p, a.size(4) ),
                  Errors::INVALID_INDEX4 );
    JEM_ASSERT2 ( checkIndex( q, a.size(5) ),
                  Errors::INVALID_INDEX5 );

    return a.data_[i * a.stride_[0] +
                   j * a.stride_[1] +
                   k * a.stride_[2] +
                   l * a.stride_[3] +
                   p * a.stride_[4] +
                   q * a.stride_[5]];
  }

};


JEM_END_NAMESPACE( array )


//=======================================================================
//   class Array<T,N.Nil>
//=======================================================================

//-----------------------------------------------------------------------
//   default constructor
//-----------------------------------------------------------------------


template <class T, int N>

  inline Array<T,N,Nil>::Array () noexcept

{
  block_  = & DataBlock::null;
  data_   = nullptr;
  shape_  = 0_idx;
  stride_ = 1_idx;

  block_->refcount++;
}


//-----------------------------------------------------------------------
//   shape constructors
//-----------------------------------------------------------------------


template <class T, int N>

  inline Array<T,N,Nil>::Array

  ( idx_t m, idx_t n )

{
  static_assert ( RANK == 2, "rank mismatch" );

  init_ ( Shape( m, n ) );
}


template <class T, int N>

  inline Array<T,N,Nil>::Array

  ( idx_t m, idx_t n, idx_t p )

{
  static_assert ( RANK == 3, "rank mismatch" );

  init_ ( Shape( m, n, p ) );
}


template <class T, int N>

  inline Array<T,N,Nil>::Array

  ( idx_t m, idx_t n, idx_t p, idx_t q )

{
  static_assert ( RANK == 4, "rank mismatch" );

  init_ ( Shape( m, n, p, q ) );
}


template <class T, int N>

  inline Array<T,N,Nil>::Array

  ( idx_t m, idx_t n, idx_t p, idx_t q, idx_t r )

{
  static_assert ( RANK == 5, "rank mismatch" );

  init_ ( Shape( m, n, p, q, r ) );
}


template <class T, int N>

  inline Array<T,N,Nil>::Array

  ( idx_t m, idx_t n, idx_t p, idx_t q, idx_t r, idx_t s )

{
  static_assert ( RANK == 6, "rank mismatch" );

  init_ ( Shape( m, n, p, q, r, s ) );
}


template <class T, int N>

  inline Array<T,N,Nil>::Array ( const Shape& sh )

{
  init_ ( sh );
}


//-----------------------------------------------------------------------
//   copy constructor
//-----------------------------------------------------------------------


template <class T, int N>

  inline Array<T,N,Nil>::Array ( const Array& rhs ) noexcept :

    block_  ( rhs.block_  ),
    data_   ( rhs.data_   ),
    shape_  ( rhs.shape_  ),
    stride_ ( rhs.stride_ )

{
  block_->refcount++;
}


//-----------------------------------------------------------------------
//   move constructor
//-----------------------------------------------------------------------


template <class T, int N>

  inline Array<T,N,Nil>::Array ( Array&& rhs ) noexcept :

    block_  ( rhs.block_  ),
    data_   ( rhs.data_   ),
    shape_  ( rhs.shape_  ),
    stride_ ( rhs.stride_ )

{
  rhs.block_ = & DataBlock::null;
  rhs.data_  = nullptr;
}


//-----------------------------------------------------------------------
//   clone constructor
//-----------------------------------------------------------------------


template <class T, int N>

  Array<T,N,Nil>::Array ( const Array& rhs, CloneTag )

{
  // Use a temporary array to make sure that memory is freed if an
  // exception is thrown.

  Array  tmp ( rhs.shape() );

  tmp = rhs;

  grab_ ( tmp );
}


//-----------------------------------------------------------------------
//   expression conversion constructor
//-----------------------------------------------------------------------


template <class T, int N>
  template <class U, class E>

  Array<T,N,Nil>::Array ( const Array<U,N,E>& rhs )

{
  // Use a temporary array to make sure that memory is freed if an
  // exception is thrown.

  Array  tmp ( rhs.shape() );

  tmp = rhs;

  grab_ ( tmp );
}


//-----------------------------------------------------------------------
//   pointer conversion constructor
//-----------------------------------------------------------------------


template <class T, int N>

  inline Array<T,N,Nil>::Array

  ( T*            data,
    const Shape&  sh )

{
  JEM_ASSERT2 ( array::checkShape( sh ),
                array::Errors::INVALID_SHAPE );

  block_     = & DataBlock::null;
  data_      = data;
  shape_     = sh;
  stride_[0] = 1_idx;

  for ( int idim = 1; idim < N; idim++ )
  {
    stride_[idim] = stride_[idim - 1] * sh[idim - 1];
  }
}


//-----------------------------------------------------------------------
//   slice constructors
//-----------------------------------------------------------------------


template <class T, int N>
  template <int K, class I>

  inline Array<T,N,Nil>::Array

  ( const Array<T,K>&  a,
    const I&           i )

{
  static_assert ( RANK <= K, "invalid rank" );

  block_ = a.getDataBlock ();
  data_  = a.addr         ();

  for ( int idim = 0; idim < N; idim++ )
  {
    shape_ [idim] = a.size  (idim);
    stride_[idim] = a.stride(idim);
  }

  slice_ ( i, N - 1, a.size(K - 1), a.stride(K - 1) );

  block_->refcount++;
}


template <class T, int N>
  template <class I, class J>

  inline Array<T,N,Nil>::Array

  ( const Array<T,2>&  a,
    const I&           i,
    const J&           j )

{
  static_assert ( RANK <= 2, "invalid rank" );

  typedef array::SliceTraits<T,I,J>  Traits;

  block_ = a.getDataBlock ();
  data_  = a.addr         ();

  slice_ ( i, Traits::DIM0, a.size(0), a.stride(0) );
  slice_ ( j, Traits::DIM1, a.size(1), a.stride(1) );

  block_->refcount++;
}


template <class T, int N>
  template <class I, class J, class K>

  Array<T,N,Nil>::Array

  ( const Array<T,3>&  a,
    const I&           i,
    const J&           j,
    const K&           k )

{
  static_assert ( RANK <= 3, "invalid rank" );

  typedef array::SliceTraits<T,I,J,K>  Traits;

  block_ = a.getDataBlock ();
  data_  = a.addr         ();

  slice_ ( i, Traits::DIM0, a.size(0), a.stride(0) );
  slice_ ( j, Traits::DIM1, a.size(1), a.stride(1) );
  slice_ ( k, Traits::DIM2, a.size(2), a.stride(2) );

  block_->refcount++;
}


template <class T, int N>
  template <class I, class J, class K, class L>

  Array<T,N,Nil>::Array

  ( const Array<T,4>&  a,
    const I&           i,
    const J&           j,
    const K&           k,
    const L&           l )

{
  static_assert ( RANK <= 4, "invalid rank" );

  typedef array::SliceTraits<T,I,J,K,L>  Traits;

  block_ = a.getDataBlock ();
  data_  = a.addr         ();

  slice_ ( i, Traits::DIM0, a.size(0), a.stride(0) );
  slice_ ( j, Traits::DIM1, a.size(1), a.stride(1) );
  slice_ ( k, Traits::DIM2, a.size(2), a.stride(2) );
  slice_ ( l, Traits::DIM3, a.size(3), a.stride(3) );

  block_->refcount++;
}


template <class T, int N>
  template <class I, class J, class K, class L, class P>

  Array<T,N,Nil>::Array

  ( const Array<T,5>&  a,
    const I&           i,
    const J&           j,
    const K&           k,
    const L&           l,
    const P&           p )

{
  static_assert ( RANK <= 5, "invalid rank" );

  typedef array::SliceTraits<T,I,J,K,L,P>  Traits;

  block_ = a.getDataBlock ();
  data_  = a.addr         ();

  slice_ ( i, Traits::DIM0, a.size(0), a.stride(0) );
  slice_ ( j, Traits::DIM1, a.size(1), a.stride(1) );
  slice_ ( k, Traits::DIM2, a.size(2), a.stride(2) );
  slice_ ( l, Traits::DIM3, a.size(3), a.stride(3) );
  slice_ ( p, Traits::DIM4, a.size(4), a.stride(4) );

  block_->refcount++;
}


template <class T, int N>
  template <class I, class J, class K,
            class L, class P, class Q>

  Array<T,N,Nil>::Array

  ( const Array<T,6>&  a,
    const I&           i,
    const J&           j,
    const K&           k,
    const L&           l,
    const P&           p,
    const Q&           q )

{
  static_assert ( RANK <= 6, "invalid rank" );

  typedef array::SliceTraits<T,I,J,K,L,P,Q>  Traits;

  block_ = a.getDataBlock ();
  data_  = a.addr         ();

  slice_ ( i, Traits::DIM0, a.size(0), a.stride(0) );
  slice_ ( j, Traits::DIM1, a.size(1), a.stride(1) );
  slice_ ( k, Traits::DIM2, a.size(2), a.stride(2) );
  slice_ ( l, Traits::DIM3, a.size(3), a.stride(3) );
  slice_ ( p, Traits::DIM4, a.size(4), a.stride(4) );
  slice_ ( q, Traits::DIM5, a.size(5), a.stride(5) );

  block_->refcount++;
}


//-----------------------------------------------------------------------
//   raw constructor
//-----------------------------------------------------------------------


template <class T, int N>

  inline Array<T,N,Nil>::Array

  ( DataBlock*    block,
    T*            data,
    const Shape&  sh,
    const Shape&  st ) :

    block_  ( block ),
    data_   ( data ),
    shape_  ( sh ),
    stride_ ( st )

{
  block_->refcount++;
}


//-----------------------------------------------------------------------
//   transpose constructor
//-----------------------------------------------------------------------


template <class T, int N>

  Array<T,N,Nil>::Array ( const Array& rhs, TransposeTag_ )

{
  block_ = rhs.block_;
  data_  = rhs.data_;

  for ( int idim = 0; idim < N; idim++ )
  {
    shape_ [idim] = rhs.shape_ [N - 1 - idim];
    stride_[idim] = rhs.stride_[N - 1 - idim];
  }

  block_->refcount++;
}


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


template <class T, int N>

  inline Array<T,N,Nil>::~Array ()

{
  DataBlock::dealloc ( block_ );

#ifndef NDEBUG

  block_ = nullptr;
  data_  = nullptr;

#endif

}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


template <class T, int N>

  const Array<T,N>& Array<T,N,Nil>::operator =

  ( const Array&  rhs ) const

{
  JEM_ASSERT2 ( equal( shape(), rhs.shape() ),
                array::Errors::SHAPE_MISMATCH );

  if ( isContiguous() && rhs.isContiguous() )
  {
    jem::copy ( data_, rhs.data_,
                rhs.data_ + array::size( shape_ ) );
  }
  else
  {
    array::assignArray ( data_, stride_, rhs.data_,
                         rhs.stride_, shape_, array::Pass2nd() );
  }

  return *this;
}


template <class T, int N>
  template <class X>

  inline const Array<T,N>& Array<T,N,Nil>::operator =

  ( const X&  rhs ) const

{
  this->assign ( rhs, array::Pass2nd() );

  return *this;
}


//-----------------------------------------------------------------------
//   assign
//-----------------------------------------------------------------------


template <class T, int N>
  template <class U, class Op>

  inline void Array<T,N,Nil>::assign

  ( const Array<U,N,Nil>&  rhs,
    const Op&              op ) const

{
  JEM_ASSERT2 ( equal( shape(), rhs.shape() ),
                array::Errors::SHAPE_MISMATCH );

  if ( isContiguous() && rhs.isContiguous() )
  {
    array::assignArray ( data_, rhs.data_,
                         array::size( shape_ ), op );
  }
  else
  {
    array::assignArray ( data_, stride_, rhs.data_,
                         rhs.stride_, shape_, op );
  }
}


template <class T, int N>
  template <class U, class E, class Op>

  inline void Array<T,N,Nil>::assign

  ( const Array<U,N,E>&  rhs,
    const Op&            op ) const

{
  JEM_ASSERT2 ( equal( shape(), rhs.shape() ),
                array::Errors::SHAPE_MISMATCH );

  if ( isContiguous() && rhs.isContiguous() )
  {
    array::assignArray ( data_, rhs,
                         array::size( shape_ ), op );
  }
  else
  {
    array::assignArray ( data_, stride_, rhs, shape_, op );
  }
}


template <class T, int N>
  template <class U, class Op>

  inline void Array<T,N,Nil>::assign

  ( const U&   rhs,
    const Op&  op ) const

{
  if ( isContiguous() )
  {
    array::assignScalar ( data_, rhs,
                          array::size( shape_ ), op );
  }
  else
  {
    array::assignScalar ( data_, stride_, rhs, shape_, op );
  }
}


//-----------------------------------------------------------------------
//   begin & end
//-----------------------------------------------------------------------


template <class T, int N>

  inline typename Array<T,N,Nil>::Iterator
  Array<T,N,Nil>::begin () const noexcept

{
  return Iterator ( data_, shape_, stride_ );
}


template <class T, int N>

  inline typename Array<T,N,Nil>::Iterator
  Array<T,N,Nil>::end () const noexcept

{
  return Iterator ( data_, shape_, stride_, Iterator::END_TAG );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


template <class T, int N>

  inline Array<T,N> Array<T,N,Nil>::clone () const

{
  return Array ( *this, CLONE_TAG );
}


//-----------------------------------------------------------------------
//   ref
//-----------------------------------------------------------------------


template <class T, int N>

  void Array<T,N,Nil>::ref ( const Array& rhs )

{
  if ( block_ != rhs.block_ )
  {
    DataBlock::dealloc ( block_ );

    block_ = rhs.block_;

    block_->refcount++;
  }

  data_   = rhs.data_;
  shape_  = rhs.shape_;
  stride_ = rhs.stride_;
}


template <class T, int N>

  inline void Array<T,N,Nil>::ref ( Array&& rhs ) noexcept

{
  swap ( rhs );
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T, int N>

  void Array<T,N,Nil>::swap ( Array& rhs ) noexcept

{
  jem   ::swap ( block_, rhs.block_ );
  jem   ::swap ( data_,  rhs.data_  );
  shape_ .swap ( rhs.shape_  );
  stride_.swap ( rhs.stride_ );
}


//-----------------------------------------------------------------------
//   subscript operators
//-----------------------------------------------------------------------


template <class T, int N>
  template <class I>

  inline T& Array<T,N,Nil>::operator [] ( const Tuple<I,N>& i ) const

{
  JEM_ASSERT2 ( array::checkIndex( i, shape_ ),
                array::Errors::INVALID_INDEX );

  return data_[array::ioffset( i, stride_ )];
}


template <class T, int N>
  template <class I>

  inline typename array::SliceLastTraits<T,N,I>::ResultType

  Array<T,N,Nil>::operator []

  ( const I&  i ) const

{
  using   array::Slicer;
  typedef array::SliceLastTraits<T,N,I>  Traits;

  return Slicer < typename Traits::ResultType, Traits::RANK >
    :: get ( *this, i );
}


template <class T, int N>
  template <class I, class J>

  inline typename array::SliceTraits<T,I,J>::ResultType

  Array<T,N,Nil>::operator ()

  ( const I&  i,
    const J&  j ) const

{
  static_assert ( RANK == 2, "rank mismatch" );

  using   array::Slicer;
  typedef array::SliceTraits<T,I,J>  Traits;

  return Slicer < typename Traits::ResultType, Traits::RANK >
    :: get ( *this, i, j );
}


template <class T, int N>
  template <class I, class J, class K>

  inline typename array::SliceTraits<T,I,J,K>::ResultType

  Array<T,N,Nil>::operator ()

  ( const I&  i,
    const J&  j,
    const K&  k ) const

{
  static_assert ( RANK == 3, "rank mismatch" );

  using   array::Slicer;
  typedef array::SliceTraits<T,I,J,K>  Traits;

  return Slicer < typename Traits::ResultType, Traits::RANK >
    :: get ( *this, i, j, k );
}


template <class T, int N>
  template <class I, class J, class K, class L>

  inline typename array::SliceTraits<T,I,J,K,L>::ResultType

  Array<T,N,Nil>::operator ()

  ( const I&  i,
    const J&  j,
    const K&  k,
    const L&  l ) const

{
  static_assert ( RANK == 4, "rank mismatch" );

  using   array::Slicer;
  typedef array::SliceTraits<T,I,J,K,L>  Traits;

  return Slicer < typename Traits::ResultType, Traits::RANK >
    :: get ( *this, i, j, k, l );
}


template <class T, int N>
  template <class I, class J, class K, class L, class P>

  inline typename array::SliceTraits<T,I,J,K,L,P>::ResultType

  Array<T,N,Nil>::operator ()

  ( const I&  i,
    const J&  j,
    const K&  k,
    const L&  l,
    const P&  p ) const

{
  static_assert ( RANK == 5, "rank mismatch" );

  using   array::Slicer;
  typedef array::SliceTraits<T,I,J,K,L,P>  Traits;

  return Slicer < typename Traits::ResultType, Traits::RANK >
    :: get ( *this, i, j, k, l, p );
}


template <class T, int N>
  template <class I, class J, class K,
            class L, class P, class Q>

  inline typename array::SliceTraits<T,I,J,K,L,P,Q>::ResultType

  Array<T,N,Nil>::operator ()

  ( const I&  i,
    const J&  j,
    const K&  k,
    const L&  l,
    const P&  p,
    const Q&  q ) const

{
  static_assert ( RANK == 6, "rank mismatch" );

  using   array::Slicer;
  typedef array::SliceTraits<T,I,J,K,L,P,Q>  Traits;

  return Slicer < typename Traits::ResultType, Traits::RANK >
    :: get ( *this, i, j, k, l, p, q );
}


//-----------------------------------------------------------------------
//   getFast
//-----------------------------------------------------------------------


template <class T, int N>

  inline T& Array<T,N,Nil>::getFast ( idx_t i ) const noexcept

{
  return data_[i];
}


//-----------------------------------------------------------------------
//   transpose
//-----------------------------------------------------------------------


template <class T, int N>

  inline Array<T,N> Array<T,N,Nil>::transpose () const

{
  return Array ( *this, TRANSPOSE_TAG_ );
}


template <class T, int N>

  Array<T,N> Array<T,N,Nil>::transpose ( const Tuple<int,N>& perm ) const

{
  Shape  sh;
  Shape  st;

  for ( int idim = 0; idim < N; idim++ )
  {
    sh[idim] = shape_ [perm[idim]];
    st[idim] = stride_[perm[idim]];
  }

  return Array ( block_, data_, sh, st );
}


//-----------------------------------------------------------------------
//   isAllocated
//-----------------------------------------------------------------------


template <class T, int N>

  inline bool Array<T,N,Nil>::isAllocated () const noexcept

{
  return (block_ != & DataBlock::null);
}


//-----------------------------------------------------------------------
//   isContiguous
//-----------------------------------------------------------------------


template <class T, int N>

  inline bool Array<T,N,Nil>::isContiguous () const noexcept

{
  return array::isContiguous ( shape_, stride_ );
}


//-----------------------------------------------------------------------
//   resize
//-----------------------------------------------------------------------


template <class T, int N>

  void Array<T,N,Nil>::resize ( idx_t m )

{
  Shape  sh = shape_;

  sh[N - 1] = m;

  resize ( sh );
}


template <class T, int N>

  inline void Array<T,N,Nil>::resize

  ( idx_t m, idx_t n )

{
  static_assert ( RANK == 2, "rank mismatch" );

  resize ( Shape( m, n ) );
}


template <class T, int N>

  inline void Array<T,N,Nil>::resize

  ( idx_t m, idx_t n, idx_t p )

{
  static_assert ( RANK == 3, "rank mismatch" );

  resize ( Shape( m, n, p ) );
}


template <class T, int N>

  inline void Array<T,N,Nil>::resize

  ( idx_t m, idx_t n, idx_t p, idx_t q )

{
  static_assert ( RANK == 4, "rank mismatch" );

  resize ( Shape( m, n, p, q ) );
}


template <class T, int N>

  inline void Array<T,N,Nil>::resize

  ( idx_t m, idx_t n, idx_t p, idx_t q, idx_t r )

{
  static_assert ( RANK == 5, "rank mismatch" );

  resize ( Shape( m, n, p, q, r ) );
}


template <class T, int N>

  inline void Array<T,N,Nil>::resize

  ( idx_t m, idx_t n, idx_t p, idx_t q, idx_t r, idx_t s )

{
  static_assert ( RANK == 6, "rank mismatch" );

  resize ( Shape( m, n, p, q, r, s ) );
}


template <class T, int N>

  void Array<T,N,Nil>::resize ( const Shape& sh )

{
  JEM_ASSERT2 ( array::checkShape( sh ),
                array::Errors::INVALID_SHAPE );

  if ( array::size(sh) > block_->size || block_->refcount > 1 )
  {
    DataBlock::dealloc ( block_ );
    init_              ( sh );
  }
  else
  {
    data_      = block_->getData ();
    shape_     = sh;
    stride_[0] = 1_idx;

    for ( int idim = 1; idim < N; idim++ )
    {
      stride_[idim] = stride_[idim - 1] * shape_[idim - 1];
    }
  }
}


//-----------------------------------------------------------------------
//   reshape
//-----------------------------------------------------------------------


template <class T, int N>

  void Array<T,N,Nil>::reshape ( idx_t m )

{
  Shape  sh = shape_;

  sh[N - 1] = m;

  reshape ( sh );
}


template <class T, int N>

  inline void Array<T,N,Nil>::reshape

  ( idx_t m, idx_t n )

{
  static_assert ( RANK == 2, "rank mismatch" );

  reshape ( Shape( m, n ) );
}


template <class T, int N>

  inline void Array<T,N,Nil>::reshape

  ( idx_t m, idx_t n, idx_t p )

{
  static_assert ( RANK == 3, "rank mismatch" );

  reshape ( Shape( m, n, p ) );
}


template <class T, int N>

  inline void Array<T,N,Nil>::reshape

  ( idx_t m, idx_t n, idx_t p, idx_t q )

{
  static_assert ( RANK == 4, "rank mismatch" );

  reshape ( Shape( m, n, p, q ) );
}


template <class T, int N>

  inline void Array<T,N,Nil>::reshape

  ( idx_t m, idx_t n, idx_t p, idx_t q, idx_t r )

{
  static_assert ( RANK == 5, "rank mismatch" );

  reshape ( Shape( m, n, p, q, r ) );
}


template <class T, int N>

  inline void Array<T,N,Nil>::reshape

  ( idx_t m, idx_t n, idx_t p, idx_t q, idx_t r, idx_t s )

{
  static_assert ( RANK == 6, "rank mismatch" );

  reshape ( Shape( m, n, p, q, r, s ) );
}


template <class T, int N>

  void Array<T,N,Nil>::reshape ( const Shape& sh )

{
  JEM_ASSERT2 ( array::checkShape( sh ),
                array::Errors::INVALID_SHAPE );

  bool  canRealloc = array::isContiguous ( shape_, stride_ );

  for ( int idim = 0; idim < N - 1; idim++ )
  {
    canRealloc = canRealloc && (shape_[idim] == sh[idim]);
  }

  if ( canRealloc              &&
       block_->refcount  == 1  &&
       block_->getData() == data_ )
  {
    block_ = DataBlock::realloc ( block_, array::size(sh) );

    data_         = block_->getData ();
    shape_[N - 1] = sh[N - 1];
  }
  else
  {
    Array  tmp ( sh );
    Shape  s;

    for ( int idim = 0; idim < N; idim++ )
    {
      s[idim] = min ( shape_[idim], sh[idim] );
    }

    array::assignArray ( tmp.data_, tmp.stride_, data_,
                         stride_, s, array::Pass2nd() );

    swap ( tmp );
  }
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


template <class T, int N>

  inline idx_t Array<T,N,Nil>::size () const

{
  return shape_[N - 1];
}


template <class T, int N>

  inline idx_t Array<T,N,Nil>::size ( int idim ) const

{
  return shape_[idim];
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


template <class T, int N>

  inline const typename Array<T,N,Nil>::Shape&
  Array<T,N,Nil>::shape () const noexcept

{
  return shape_;
}


//-----------------------------------------------------------------------
//   stride
//-----------------------------------------------------------------------


template <class T, int N>

  inline idx_t Array<T,N,Nil>::stride () const

{
  return stride_[N - 1];
}


template <class T, int N>

  inline idx_t Array<T,N,Nil>::stride ( int idim ) const

{
  return stride_[idim];
}


//-----------------------------------------------------------------------
//   strides
//-----------------------------------------------------------------------


template <class T, int N>

  inline const typename Array<T,N>::Shape&
  Array<T,N,Nil>::strides () const noexcept

{
  return stride_;
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


template <class T, int N>

  inline T* Array<T,N,Nil>::addr () const noexcept

{
  return data_;
}


template <class T, int N>

  inline T* Array<T,N,Nil>::addr ( idx_t offset ) const

{
  return (data_ + offset * stride_[N - 1]);
}


//-----------------------------------------------------------------------
//   itemCount
//-----------------------------------------------------------------------


template <class T, int N>

  inline idx_t Array<T,N,Nil>::itemCount () const

{
  return array::size ( shape_ );
}


//-----------------------------------------------------------------------
//   getDataBlock
//-----------------------------------------------------------------------


template <class T, int N>

  inline typename Array<T,N,Nil>::DataBlock*
  Array<T,N,Nil>::getDataBlock () const noexcept

{
  return block_;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


template <class T, int N>

  void Array<T,N,Nil>::init_ ( const Shape& sh )

{
  JEM_ASSERT2 ( array::checkShape( sh ),
                array::Errors::INVALID_SHAPE );

  const idx_t  len = array::size ( sh );

  block_     = DataBlock::alloc ( &data_, len );
  shape_     = sh;
  stride_[0] = 1_idx;

  for ( int idim = 1; idim < N; idim++ )
  {
    stride_[idim] = stride_[idim - 1] * shape_[idim - 1];
  }

  init_ ( len, HasTrivialCtor<T>() || IsPOD<T>() );

  block_->size = len;
}


template <class T, int N>

  inline void Array<T,N,Nil>::init_ ( idx_t len, const True& )

{}


template <class T, int N>

  inline void Array<T,N,Nil>::init_ ( idx_t len, const False& )

{
  try
  {
    construct ( data_, len );
  }
  catch ( ... )
  {
    DataBlock::dealloc ( block_ );
    throw;
  }
}


//-----------------------------------------------------------------------
//   grab_
//-----------------------------------------------------------------------


template <class T, int N>

  inline void Array<T,N,Nil>::grab_ ( Array& rhs )

{
  block_     = rhs.block_;
  data_      = rhs.data_;
  shape_     = rhs.shape_;
  stride_    = rhs.stride_;

  rhs.block_ = & DataBlock::null;
  rhs.data_  = nullptr;
}



//-----------------------------------------------------------------------
//   slice_
//-----------------------------------------------------------------------


template <class T, int N>
  template <class I>

  inline void Array<T,N,Nil>::slice_

  ( I      i,
    int    idim,
    idx_t  size,
    idx_t  stride )

{
  JEM_ASSERT2 ( checkIndex( (idx_t) i, size ),
                array::Errors::INVALID_INDEX );

  data_ += i * stride;
}


template <class T, int N>

  inline void Array<T,N,Nil>::slice_

  ( const Slice&  s,
    int           idim,
    idx_t         size,
    idx_t         stride )

{
  JEM_ASSERT2 ( s.first() <= s.last( size ),
                array::Errors::INVALID_SLICE );

  const idx_t  st = s.stride ();

  data_        += stride * s.first();
  stride_[idim] = stride * st;
  shape_ [idim] = s.last( size ) - s.first();

  if ( st > 1_idx )
  {
    shape_[idim] = (shape_[idim] + st - 1_idx) / st;
  }
}


template <class T, int N>

  inline void Array<T,N,Nil>::slice_

  ( const SliceFromTo&  s,
    int                 idim,
    idx_t               size,
    idx_t               stride )

{
  JEM_ASSERT2 ( s.first() <= s.last() && s.last() <= size,
                array::Errors::INVALID_SLICE );

  data_        += stride   * s.first();
  shape_ [idim] = s.last() - s.first();
  stride_[idim] = stride;
}


template <class T, int N>

  inline void Array<T,N,Nil>::slice_

  ( const SliceTo&  s,
    int             idim,
    idx_t           size,
    idx_t           stride )

{
  shape_ [idim] = s.last ( size );
  stride_[idim] = stride;
}


template <class T, int N>

  inline void Array<T,N,Nil>::slice_

  ( const SliceFrom&  s,
    int               idim,
    idx_t             size,
    idx_t             stride )

{
  JEM_ASSERT2 ( s.first() <= size,
                array::Errors::INVALID_SLICE );

  data_        += stride * s.first();
  shape_ [idim] = size   - s.first();
  stride_[idim] = stride;
}


template <class T, int N>

  inline void Array<T,N,Nil>::slice_

  ( const SliceAll&  s,
    int              idim,
    idx_t            size,
    idx_t            stride )

{
  shape_ [idim] = size;
  stride_[idim] = stride;
}


//=======================================================================
//   class Array<T,1,Nil>
//=======================================================================

//-----------------------------------------------------------------------
//   default constructor
//-----------------------------------------------------------------------


template <class T>

  inline Array<T,1,Nil>::Array () noexcept

{
  block_  = & DataBlock::null;
  data_   = nullptr;
  size_   = 0_idx;
  stride_ = 1_idx;

  block_->refcount++;
}


//-----------------------------------------------------------------------
//   shape constructors
//-----------------------------------------------------------------------


template <class T>

  inline Array<T,1,Nil>::Array ( idx_t n )

{
  init_ ( n );
}


template <class T>

  inline Array<T,1,Nil>::Array ( const Shape& sh )

{
  init_ ( sh[0] );
}


//-----------------------------------------------------------------------
//   copy constructor
//-----------------------------------------------------------------------


template <class T>

  inline Array<T,1,Nil>::Array ( const Array& rhs ) noexcept :

    block_  ( rhs.block_  ),
    data_   ( rhs.data_   ),
    size_   ( rhs.size_   ),
    stride_ ( rhs.stride_ )

{
  block_->refcount++;
}


//-----------------------------------------------------------------------
//   move constructor
//-----------------------------------------------------------------------


template <class T>

  inline Array<T,1,Nil>::Array ( Array&& rhs ) noexcept :

    block_  ( rhs.block_  ),
    data_   ( rhs.data_   ),
    size_   ( rhs.size_   ),
    stride_ ( rhs.stride_ )

{
  rhs.block_ = & DataBlock::null;
  rhs.data_  = nullptr;
}


//-----------------------------------------------------------------------
//   clone constructor
//-----------------------------------------------------------------------


template <class T>

  Array<T,1,Nil>::Array ( const Array& rhs, CloneTag )

{
  // Use a temporary array to make sure that memory is freed if an
  // exception is thrown.

  Array  tmp ( rhs.size_ );

  tmp = rhs;

  grab_ ( tmp );
}


//-----------------------------------------------------------------------
//   expression conversion constructor
//-----------------------------------------------------------------------


template <class T>
  template <class U, class E>

  Array<T,1,Nil>::Array ( const Array<U,1,E>& rhs )

{
  // Use a temporary array to make sure that memory is freed if an
  // exception is thrown.

  Array  tmp ( rhs.size() );

  tmp = rhs;

  grab_ ( tmp );
}


//-----------------------------------------------------------------------
//   pointer conversion constructor
//-----------------------------------------------------------------------


template <class T>

  inline Array<T,1,Nil>::Array

  ( T*     data,
    idx_t  size )

{
  JEM_ASSERT2 ( size >= 0, array::Errors::INVALID_SIZE );

  block_  = & DataBlock::null;
  data_   = data;
  size_   = size;
  stride_ = 1_idx;
}


//-----------------------------------------------------------------------
//   input iterator constructor
//-----------------------------------------------------------------------


template <class T>
  template <class InputIterator>

  Array<T,1,Nil>::Array

  ( InputIterator  first,
    InputIterator  last )

{
  size_   = distance         ( first,  last );
  block_  = DataBlock::alloc ( &data_, size_ );
  stride_ = 1_idx;

  try
  {
    construct ( data_, first, last );

    block_->size = size_;
  }
  catch ( ... )
  {
    DataBlock::dealloc ( block_ );
    throw;
  }
}


//-----------------------------------------------------------------------
//   initializer list constructor
//-----------------------------------------------------------------------


template <class T>
  template <class U>

  Array<T,1,Nil>::Array ( std::initializer_list<U> list )

{
  idx_t  i = 0;

  init_ ( (idx_t) list.size() );

  for ( const U& t : list )
  {
    data_[i++] = t;
  }
}


//-----------------------------------------------------------------------
//   slice constructors
//-----------------------------------------------------------------------


template <class T>
  template <class I>

  inline Array<T,1,Nil>::Array

  ( const Array<T,1>&  a,
    const I&           i )

{
  block_ = a.block_;
  data_  = a.data_;

  slice_ ( i, a.size_, a.stride_ );

  block_->refcount++;
}


template <class T>
  template <class I>

  inline Array<T,1,Nil>::Array

  ( const Array<T,2>&  a,
    I                  i )

{
  block_  = a.getDataBlock ();
  data_   = a.addr         ();
  size_   = a.size         (0);
  stride_ = a.stride       (0);

  slice_ ( i, a.size(1), a.stride(1) );

  block_->refcount++;
}


template <class T>
  template <class I, class J>

  inline Array<T,1,Nil>::Array

  ( const Array<T,2>&  a,
    const I&           i,
    const J&           j )

{
  block_ = a.getDataBlock ();
  data_  = a.addr         ();

  slice_ ( i, a.size(0), a.stride(0) );
  slice_ ( j, a.size(1), a.stride(1) );

  block_->refcount++;
}


template <class T>
  template <class I, class J, class K>

  Array<T,1,Nil>::Array

  ( const Array<T,3>&  a,
    const I&           i,
    const J&           j,
    const K&           k )

{
  block_ = a.getDataBlock ();
  data_  = a.addr         ();

  slice_ ( i, a.size(0), a.stride(0) );
  slice_ ( j, a.size(1), a.stride(1) );
  slice_ ( k, a.size(2), a.stride(2) );

  block_->refcount++;
}


template <class T>
  template <class I, class J, class K, class L>

  Array<T,1,Nil>::Array

  ( const Array<T,4>&  a,
    const I&           i,
    const J&           j,
    const K&           k,
    const L&           l )

{
  block_ = a.getDataBlock ();
  data_  = a.addr         ();

  slice_ ( i, a.size(0), a.stride(0) );
  slice_ ( j, a.size(1), a.stride(1) );
  slice_ ( k, a.size(2), a.stride(2) );
  slice_ ( l, a.size(3), a.stride(3) );

  block_->refcount++;
}


template <class T>
  template <class I, class J, class K, class L, class P>

  Array<T,1,Nil>::Array

  ( const Array<T,5>&  a,
    const I&           i,
    const J&           j,
    const K&           k,
    const L&           l,
    const P&           p )

{
  block_ = a.getDataBlock ();
  data_  = a.addr         ();

  slice_ ( i, a.size(0), a.stride(0) );
  slice_ ( j, a.size(1), a.stride(1) );
  slice_ ( k, a.size(2), a.stride(2) );
  slice_ ( l, a.size(3), a.stride(3) );
  slice_ ( p, a.size(4), a.stride(4) );

  block_->refcount++;
}


template <class T>
  template <class I, class J, class K,
            class L, class P, class Q>

  Array<T,1,Nil>::Array

  ( const Array<T,6>&  a,
    const I&           i,
    const J&           j,
    const K&           k,
    const L&           l,
    const P&           p,
    const Q&           q )

{
  block_ = a.getDataBlock ();
  data_  = a.addr         ();

  slice_ ( i, a.size(0), a.stride(0) );
  slice_ ( j, a.size(1), a.stride(1) );
  slice_ ( k, a.size(2), a.stride(2) );
  slice_ ( l, a.size(3), a.stride(3) );
  slice_ ( p, a.size(4), a.stride(4) );
  slice_ ( q, a.size(5), a.stride(5) );

  block_->refcount++;
}


//-----------------------------------------------------------------------
//   raw constructor
//-----------------------------------------------------------------------


template <class T>

  inline Array<T,1,Nil>::Array

  ( DataBlock*  block,
    T*          data,
    idx_t        size,
    idx_t        stride ) :

    block_  ( block  ),
    data_   ( data   ),
    size_   ( size   ),
    stride_ ( stride )

{
  block_->refcount++;
}


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


template <class T>

  inline Array<T,1,Nil>::~Array ()

{
  DataBlock::dealloc ( block_ );

#ifndef NDEBUG

  block_ = 0;
  data_  = 0;

#endif
}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


template <class T>

  const Array<T,1>& Array<T,1,Nil>::operator =

  ( const Array&  rhs ) const

{
  JEM_ASSERT2 ( size() == rhs.size(),
                array::Errors::SIZE_MISMATCH );

  const idx_t  len = size_;
  const idx_t  lst = stride_;
  const idx_t  rst = rhs.stride_;

  if ( lst + rst == 2_idx )
  {
    jem::copy ( data_, rhs.data_, rhs.data_ + len );
  }
  else
  {
    T*        lpt = data_;
    const T*  rpt = rhs.data_;

    for ( idx_t i = 0; i < len; i++ )
    {
      lpt[i * lst] = rpt[i * rst];
    }
  }

  return *this;
}


template <class T>
  template <class X>

  inline const Array<T,1>& Array<T,1,Nil>::operator =

  ( const X&  rhs ) const

{
  this->assign ( rhs, array::Pass2nd() );

  return *this;
}


//-----------------------------------------------------------------------
//   assign
//-----------------------------------------------------------------------


template <class T>
  template <class U, class Op>

  inline void Array<T,1,Nil>::assign

  ( const Array<U,1,Nil>&  rhs,
    const Op&              op ) const

{
  JEM_ASSERT2 ( size() == rhs.size(),
                array::Errors::SIZE_MISMATCH );

  T*           lpt = data_;
  const idx_t  len = size_;
  const idx_t  lst = stride_;
  const U*     rpt = rhs.addr   ();
  const idx_t  rst = rhs.stride ();

  if ( lst + rst == 2_idx )
  {
    for ( idx_t i = 0; i < len; i++ )
    {
      lpt[i] = op ( lpt[i], rpt[i] );
    }
  }
  else
  {
    for ( idx_t i = 0; i < len; i++ )
    {
      lpt[i * lst] = op ( lpt[i * lst], rpt[i * rst] );
    }
  }
}


template <class T>
  template <class U, class E, class Op>

  inline void Array<T,1,Nil>::assign

  ( const Array<U,1,E>&  rhs,
    const Op&            op ) const

{
  JEM_ASSERT2 ( size() == rhs.size(),
                array::Errors::SIZE_MISMATCH );

  if ( isContiguous() && rhs.isContiguous() )
  {
    array::assignArray ( data_, rhs, size_, op );
  }
  else
  {
    array::assignArray ( data_, strides(), rhs, shape(), op );
  }
}


template <class T>
  template <class U, class Op>

  void Array<T,1,Nil>::assign

  ( const U&   rhs,
    const Op&  op ) const

{
  T*           lpt = data_;
  const idx_t  len = size_;
  const idx_t  lst = stride_;

  if ( lst == 1_idx )
  {
    for ( idx_t i = 0; i < len; i++ )
    {
      lpt[i] = op ( lpt[i], rhs );
    }
  }
  else
  {
    for ( idx_t i = 0; i < len; i++ )
    {
      lpt[i * lst] = op ( lpt[i * lst], rhs );
    }
  }
}


//-----------------------------------------------------------------------
//   begin & end
//-----------------------------------------------------------------------


template <class T>

  inline typename Array<T,1,Nil>::Iterator
  Array<T,1,Nil>::begin () const noexcept

{
  return Iterator ( data_, 0_idx, stride_ );
}


template <class T>

  inline typename Array<T,1,Nil>::Iterator
  Array<T,1,Nil>::end () const noexcept

{
  return Iterator ( data_, size_ * stride_, stride_ );
}


//-----------------------------------------------------------------------
//   front & back
//-----------------------------------------------------------------------


template <class T>

  inline T& Array<T,1,Nil>::front () const

{
  JEM_ASSERT2 ( size() > 0, array::Errors::EMPTY_ARRAY );

  return data_[0];
}


template <class T>

  inline T& Array<T,1,Nil>::back () const

{
  JEM_ASSERT2 ( size() > 0, array::Errors::EMPTY_ARRAY );

  return data_[stride_ * (size_ - 1)];
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


template <class T>

  inline Array<T,1> Array<T,1,Nil>::clone () const

{
  return Array ( *this, CLONE_TAG );
}


//-----------------------------------------------------------------------
//   ref
//-----------------------------------------------------------------------


template <class T>

  void Array<T,1,Nil>::ref ( const Array& rhs )

{
  if ( block_ != rhs.block_ )
  {
    DataBlock::dealloc ( block_ );

    block_ = rhs.block_;

    block_->refcount++;
  }

  data_   = rhs.data_;
  size_   = rhs.size_;
  stride_ = rhs.stride_;
}


template <class T>

  inline void Array<T,1,Nil>::ref ( Array&& rhs ) noexcept

{
  swap ( rhs );
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T>

  void Array<T,1,Nil>::swap ( Array& rhs ) noexcept

{
  jem::swap ( block_,  rhs.block_  );
  jem::swap ( data_,   rhs.data_   );
  jem::swap ( size_,   rhs.size_   );
  jem::swap ( stride_, rhs.stride_ );
}


//-----------------------------------------------------------------------
//   subscript operators
//-----------------------------------------------------------------------


template <class T>
  template <class I>

  inline T& Array<T,1,Nil>::operator [] ( const Tuple<I,1>& i ) const

{
  JEM_ASSERT2 ( checkIndex( i[0], size() ),
                array::Errors::INVALID_INDEX );

  return data_[i[0] * stride_];
}


template <class T>
  template <class I>

  inline typename array::SliceTraits<T,I>::ResultType

  Array<T,1,Nil>::operator [] ( const I& i ) const

{
  using   array::Slicer;
  typedef array::SliceTraits<T,I>  Traits;

  return Slicer < typename Traits::ResultType, Traits::RANK >
    :: get ( *this, i );
}


//-----------------------------------------------------------------------
//   getFast
//-----------------------------------------------------------------------


template <class T>

  inline T& Array<T,1,Nil>::getFast ( idx_t i ) const noexcept

{
  return data_[i];
}


//-----------------------------------------------------------------------
//   isAllocated
//-----------------------------------------------------------------------


template <class T>

  inline bool Array<T,1,Nil>::isAllocated () const noexcept

{
  return (block_ != & DataBlock::null);
}


//-----------------------------------------------------------------------
//   isContiguous
//-----------------------------------------------------------------------


template <class T>

  inline bool Array<T,1,Nil>::isContiguous () const noexcept

{
  return (stride_ == 1_idx);
}


//-----------------------------------------------------------------------
//   resize
//-----------------------------------------------------------------------


template <class T>

  void Array<T,1,Nil>::resize ( idx_t n )

{
  JEM_ASSERT2 ( n >= 0, array::Errors::INVALID_SIZE );

  if ( n > block_->size || block_->refcount > 1 )
  {
    DataBlock::dealloc ( block_ );
    init_              ( n );
  }
  else
  {
    data_   = block_->getData ();
    size_   = n;
    stride_ = 1_idx;
  }
}


template <class T>

  inline void Array<T,1,Nil>::resize ( const Shape& sh )

{
  resize ( sh[0] );
}


//-----------------------------------------------------------------------
//   reshape
//-----------------------------------------------------------------------


template <class T>

  void Array<T,1,Nil>::reshape ( idx_t n )

{
  JEM_ASSERT2 ( n >= 0, array::Errors::INVALID_SIZE );

  if ( this  ->stride_   == 1_idx &&
       block_->refcount  == 1             &&
       block_->getData() == data_ )
  {
    block_ = DataBlock::realloc ( block_, n );
    data_  = block_->getData ();
    size_  = n;
  }
  else
  {
    Array  tmp ( n );

    if ( n > size_ )
    {
      n = size_;
    }

    for ( idx_t i = 0; i < n; i++ )
    {
      tmp.data_[i] = data_[i * stride_];
    }

    swap ( tmp );
  }
}


template <class T>

  inline void Array<T,1,Nil>::reshape ( const Shape& sh )

{
  reshape ( sh[0] );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


template <class T>

  inline idx_t Array<T,1,Nil>::size () const noexcept

{
  return size_;
}


template <class T>

  inline idx_t Array<T,1,Nil>::size ( int idim ) const

{
  JEM_ASSERT2 ( idim == 0, array::Errors::INVALID_DIM );

  return size_;
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


template <class T>

  inline typename Array<T,1,Nil>::Shape
  Array<T,1,Nil>::shape () const

{
  return Shape ( size_ );
}


//-----------------------------------------------------------------------
//   stride
//-----------------------------------------------------------------------


template <class T>

  inline idx_t Array<T,1,Nil>::stride () const noexcept

{
  return stride_;
}


template <class T>

  inline idx_t Array<T,1,Nil>::stride ( int idim ) const

{
  JEM_ASSERT2 ( idim == 0, array::Errors::INVALID_DIM );

  return stride_;
}


//-----------------------------------------------------------------------
//   strides
//-----------------------------------------------------------------------


template <class T>

  inline typename Array<T,1>::Shape
  Array<T,1,Nil>::strides () const

{
  return Shape ( stride_ );
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


template <class T>

  inline T* Array<T,1,Nil>::addr () const noexcept

{
  return data_;
}


template <class T>

  inline T* Array<T,1,Nil>::addr ( idx_t offset ) const

{
  return (data_ + offset * stride_);
}


//-----------------------------------------------------------------------
//   itemCount
//-----------------------------------------------------------------------


template <class T>

  inline idx_t Array<T,1,Nil>::itemCount () const

{
  return size_;
}


//-----------------------------------------------------------------------
//   getDataBlock
//-----------------------------------------------------------------------


template <class T>

  inline typename Array<T,1,Nil>::DataBlock*
  Array<T,1,Nil>::getDataBlock () const

{
  return block_;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


template <class T>

  void Array<T,1,Nil>::init_ ( idx_t len )

{
  JEM_ASSERT2 ( len >= 0, array::Errors::INVALID_SIZE );

  block_  = DataBlock::alloc ( &data_, len );
  size_   = len;
  stride_ = 1_idx;

  init_ ( HasTrivialCtor<T>() || IsPOD<T>() );

  block_->size = len;
}


template <class T>

  inline void Array<T,1,Nil>::init_ ( const True& )

{}


template <class T>

  inline void Array<T,1,Nil>::init_ ( const False& )

{
  try
  {
    construct ( data_, size_ );
  }
  catch ( ... )
  {
    DataBlock::dealloc ( block_ );
    throw;
  }
}


//-----------------------------------------------------------------------
//   grab_
//-----------------------------------------------------------------------


template <class T>

  inline void Array<T,1,Nil>::grab_ ( Array& rhs )

{
  block_     = rhs.block_;
  data_      = rhs.data_;
  size_      = rhs.size_;
  stride_    = rhs.stride_;

  rhs.block_ = & DataBlock::null;
  rhs.data_  = nullptr;
}


//-----------------------------------------------------------------------
//   slice_
//-----------------------------------------------------------------------


template <class T>
  template <class I>

  inline void Array<T,1,Nil>::slice_

  ( I      i,
    idx_t  size,
    idx_t  stride )

{
  JEM_ASSERT2 ( checkIndex( (idx_t) i, size ),
                array::Errors::INVALID_INDEX );

  data_ += i * stride;
}


template <class T>

  inline void Array<T,1,Nil>::slice_

  ( const Slice&  s,
    idx_t         size,
    idx_t         stride )

{
  JEM_ASSERT2 ( s.first() <= s.last( size ),
                array::Errors::INVALID_SLICE );

  const idx_t  st = s.stride ();

  data_  += stride * s.first();
  stride_ = stride * st;
  size_   = s.last( size ) - s.first();

  if ( st > 1_idx )
  {
    size_ = (size_ + st - 1_idx) / st;
  }
}


template <class T>

  inline void Array<T,1,Nil>::slice_

  ( const SliceFromTo&  s,
    idx_t               size,
    idx_t               stride )

{
  JEM_ASSERT2 ( s.first() <= s.last() && s.last() <= size,
                array::Errors::INVALID_SLICE );

  data_  += stride   * s.first();
  size_   = s.last() - s.first();
  stride_ = stride;
}


template <class T>

  inline void Array<T,1,Nil>::slice_

  ( const SliceTo&  s,
    idx_t            size,
    idx_t            stride )

{
  size_   = s.last ( size );
  stride_ = stride;
}


template <class T>

  inline void Array<T,1,Nil>::slice_

  ( const SliceFrom&  s,
    idx_t             size,
    idx_t             stride )

{
  JEM_ASSERT2 ( s.first() <= size, array::Errors::INVALID_SLICE );

  data_  += stride * s.first();
  size_   = size   - s.first();
  stride_ = stride;
}


template <class T>

  inline void Array<T,1,Nil>::slice_

  ( const SliceAll&  s,
    idx_t            size,
    idx_t            stride )

{
  size_   = size;
  stride_ = stride;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input,
          class T, int N>

  void              decode

  ( Input&            in,
    Array<T,N>&       a )

{
  typename Array<T,N>::Shape sh;

  for ( int i = 0; i < N; i++ )
  {
    uidx_t  sz;

    decode ( in, sz );

    sh[i] = (idx_t) sz;
  }

  Array<T,N>  tmp ( sh );

  decodeArray ( in, tmp.addr(), tmp.itemCount() );
  a.ref       ( tmp );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output,
          class T, int N>

  void                encode

  ( Output&             out,
    const Array<T,N>&   a )

{
  const Tuple<idx_t,N>  sh = a.shape ();
  const idx_t           sz = array::size ( sh );


  for ( int i = 0; i < N; i++ )
  {
    encode ( out, sh[i] );
  }

  if ( sz > 0 )
  {
    if ( a.isContiguous() )
    {
      encodeArray ( out, a.addr(), sz );
    }
    else
    {
      const idx_t     n = sh[0];
      Tuple<idx_t,N>  i;

      i = 0_idx;

      do
      {
        for ( ; i[0] < n; i[0]++ )
        {
          encode ( out, a[i] );
        }
      }
      while ( array::incrIndex( i, sh ) );
    }
  }
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output,
          class T, int N, class E>

  inline void             print

  ( Output&                 out,
    const Array<T,N,E>&     a )

{
  array::print ( out, typename
                 ArrayTraits< Array<T,N,E> >::Evaluate( a ) );
}


//=======================================================================
//   class TypeTraits< Array<T,N> >
//=======================================================================


template <class T, int N>

  class TypeTraits< Array<T,N> > : public BasicTypeTraits

{
 public:

  static const bool   IS_SERIALIZABLE = IsSerializable<T>::VALUE;


  static String       whatis    ()
  {
    String  kind;

    if      ( N == 1 )
    {
      kind = " array";
    }
    else if ( N == 2 )
    {
      kind = " matrix";
    }
    else
    {
      kind = String ( " array with rank " ) + String ( N );
    }

    return TypeTraits<T>::whatis() + kind;
  }


  static String       typeName  ()
  {
    return ( String ( "jem::Array<" )    +
             TypeTraits<T>::typeName ()  +
             String ( ',' )              +
             String (   N )              +
             String ( '>' ) );
  }


  static Array<T,N>   clone     ( const Array<T,N>& rhs )
  {
    return Array<T,N> ( rhs, Array<T,N>::CLONE_TAG );
  }

};


JEM_END_PACKAGE_BASE

#endif
