
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

#ifndef JEM_BASE_ARRAY_SELECT_TCC
#define JEM_BASE_ARRAY_SELECT_TCC


JEM_BEGIN_PACKAGE_BASE
JEM_BEGIN_NAMESPACE( array )


//=======================================================================
//   utility functions
//=======================================================================

//-----------------------------------------------------------------------
//   checkSelection
//-----------------------------------------------------------------------


template <class I, class E>

  static bool             checkSelection

  ( const Array<I,1,E>&     index,
    idx_t                   ubound )

{
  const idx_t  n = index.size ();

  bool  result  = true;

  for ( idx_t i = 0; i < n; i++ )
  {
    result = result && checkIndex ( index[i], ubound );
  }

  return result;
}


//-----------------------------------------------------------------------
//   toArray
//-----------------------------------------------------------------------


template <class I, class E>

  inline
  const Array<I,1,E>&     toArray

  ( const Array<I,1,E>&     e,
    idx_t                   n )

{
  JEM_ASSERT2 ( checkSelection( e, n ), Errors::INVALID_SELECT );

  return e;
}


inline Array
  <idx_t,1,Slice>         toArray

  ( const Slice&            s,
    idx_t                   n )

{
  return Array<idx_t,1,Slice> ( s.first(), s.last(n), s.stride() );
}


inline Array
  <idx_t,1,SliceFromTo>   toArray

  ( const SliceFromTo&      s,
    idx_t                   n )

{
  return Array<idx_t,1,SliceFromTo> ( s.first(), s.last() );
}


inline Array
  <idx_t,1,SliceTo>       toArray

  ( const SliceTo&          s,
    idx_t                   n )

{
  return Array<idx_t,1,SliceTo> ( s.last() );
}


inline Array
  <idx_t,1,SliceFromTo>   toArray

  ( const SliceFrom&        s,
    idx_t                   n )

{
  return Array<idx_t,1,SliceFromTo> ( s.first(), s.last(n) );
}


inline Array
  <idx_t,1,SliceTo>       toArray

  ( const SliceAll&         s,
    idx_t                   n )

{
  return Array<idx_t,1,SliceTo> ( n );
}


JEM_END_NAMESPACE( array )


//=======================================================================
//   class Array<T,N,SelectExpr<V> >
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


template <class T, int N, class V>
  template <class I, class J>

  Array<T,N,SelectExpr<V> >::Array

  ( const Array<T,2>&  a,
    const I&           i,
    const J&           j )

{
  static_assert ( N <= 2, "invalid rank" );

  typedef array::SliceTraits<T,I,J>  Traits;

  data_ = a.addr ();

  initIndex_ ( i, Traits::DIM0, a.size(0), a.stride(0) );
  initIndex_ ( j, Traits::DIM1, a.size(1), a.stride(1) );
}


template <class T, int N, class V>
  template <class I, class J, class K>

  Array<T,N,SelectExpr<V> >::Array

  ( const Array<T,3>&  a,
    const I&           i,
    const J&           j,
    const K&           k )

{
  static_assert ( N <= 3, "invalid rank" );

  typedef array::SliceTraits<T,I,J,K>  Traits;

  data_ = a.addr ();

  initIndex_ ( i, Traits::DIM0, a.size(0), a.stride(0) );
  initIndex_ ( j, Traits::DIM1, a.size(1), a.stride(1) );
  initIndex_ ( k, Traits::DIM2, a.size(2), a.stride(2) );
}


template <class T, int N, class V>
  template <class I, class J, class K,
            class L>

  Array<T,N,SelectExpr<V> >::Array

  ( const Array<T,4>&  a,
    const I&           i,
    const J&           j,
    const K&           k,
    const L&           l )

{
  static_assert ( N <= 4, "invalid rank" );

  typedef array::SliceTraits<T,I,J,K,L>  Traits;

  data_ = a.addr ();

  initIndex_ ( i, Traits::DIM0, a.size(0), a.stride(0) );
  initIndex_ ( j, Traits::DIM1, a.size(1), a.stride(1) );
  initIndex_ ( k, Traits::DIM2, a.size(2), a.stride(2) );
  initIndex_ ( l, Traits::DIM3, a.size(3), a.stride(3) );
}


template <class T, int N, class V>
  template <class I, class J, class K,
            class L, class P>

  Array<T,N,SelectExpr<V> >::Array

  ( const Array<T,5>&  a,
    const I&           i,
    const J&           j,
    const K&           k,
    const L&           l,
    const P&           p )

{
  static_assert ( N <= 5, "invalid rank" );

  typedef array::SliceTraits<T,I,J,K,L,P>  Traits;

  data_ = a.addr ();

  initIndex_ ( i, Traits::DIM0, a.size(0), a.stride(0) );
  initIndex_ ( j, Traits::DIM1, a.size(1), a.stride(1) );
  initIndex_ ( k, Traits::DIM2, a.size(2), a.stride(2) );
  initIndex_ ( l, Traits::DIM3, a.size(3), a.stride(3) );
  initIndex_ ( p, Traits::DIM4, a.size(4), a.stride(4) );
}


template <class T, int N, class V>
  template <class I, class J, class K,
            class L, class P, class Q>

  Array<T,N,SelectExpr<V> >::Array

  ( const Array<T,6>&  a,
    const I&           i,
    const J&           j,
    const K&           k,
    const L&           l,
    const P&           p,
    const Q&           q )

{
  static_assert ( N <= 6, "invalid rank" );

  typedef array::SliceTraits<T,I,J,K,L,P,Q>  Traits;

  data_ = a.addr ();

  initIndex_ ( i, Traits::DIM0, a.size(0), a.stride(0) );
  initIndex_ ( j, Traits::DIM1, a.size(1), a.stride(1) );
  initIndex_ ( k, Traits::DIM2, a.size(2), a.stride(2) );
  initIndex_ ( l, Traits::DIM3, a.size(3), a.stride(3) );
  initIndex_ ( p, Traits::DIM4, a.size(4), a.stride(4) );
  initIndex_ ( q, Traits::DIM5, a.size(5), a.stride(5) );
}


template <class T, int N, class V>

  inline Array<T,N,SelectExpr<V> >::Array ( const Array& rhs )

{
  data_    = rhs.data_;
  index_   = rhs.index_;
  shape_   = rhs.shape_;
  dstride_ = rhs.dstride_;
  istride_ = rhs.istride_;
}


//-----------------------------------------------------------------------
//   operator =
//-----------------------------------------------------------------------


template <class T, int N, class V>

  inline const
  Array<T,N,SelectExpr<V> >&
  Array<T,N,SelectExpr<V> >::operator = ( const Array& rhs ) const

{
  this->assign ( rhs, array::Pass2nd() );

  return *this;
}


template <class T, int N, class V>
  template <class X>

  inline const
  Array<T,N,SelectExpr<V> >&
  Array<T,N,SelectExpr<V> >::operator = ( const X& rhs ) const

{
  this->assign ( rhs, array::Pass2nd() );

  return *this;
}


//-----------------------------------------------------------------------
//   assign
//-----------------------------------------------------------------------


template <class T, int N, class V>
  template <class U, class Op>

  void Array<T,N,SelectExpr<V> >::assign

  ( const U&   rhs,
    const Op&  op ) const

{
  if ( ! array::size( shape_ ) )
  {
    return;
  }

  const IndexType*  idx0 = index_  [0];
  const IndexType*  idx1 = index_  [1];
  const idx_t       dst0 = dstride_[0];
  const idx_t       dst1 = dstride_[1];
  const idx_t       ist0 = istride_[0];
  const idx_t       ist1 = istride_[1];
  const idx_t       sh0  = shape_  [0];
  const idx_t       sh1  = shape_  [1];

  Shape             pos;
  T*                ptr;


  pos = 0_idx;

  do
  {
    ptr = data_;

    for ( int i = 2; i < N; i++ )
    {
      ptr += dstride_[i] * index_[i][pos[i] * istride_[i]];
    }

    if ( (dst0 + ist0) == 2L )
    {
      for ( idx_t j = 0; j < sh1; j++ )
      {
        ptr += dst1 * idx1[j * ist1];

        for ( idx_t i = 0; i < sh0; i++ )
        {
          T&  t = ptr[idx0[i]];

          t = op ( t, rhs );
        }

        ptr -= dst1 * idx1[j * ist1];
      }
    }
    else
    {
      for ( idx_t j = 0; j < sh1; j++ )
      {
        ptr += dst1 * idx1[j * ist1];

        for ( idx_t i = 0; i < sh0; i++ )
        {
          T&  t = ptr[dst0 * idx0[i * ist0]];

          t = op ( t, rhs );
        }

        ptr -= dst1 * idx1[j * ist1];
      }
    }
  }
  while ( array::incrIndex2( pos, shape_ ) );
}


template <class T, int N, class V>
  template <class U, class E, class Op>

  void Array<T,N,SelectExpr<V> >::assign

  ( const Array<U,N,E>&  rhs,
    const Op&            op ) const

{
  JEM_ASSERT2 ( equal( shape(), rhs.shape() ),
                array::Errors::SHAPE_MISMATCH );

  if ( ! array::size( shape_ ) )
  {
    return;
  }

  const IndexType*  idx0 = index_  [0];
  const IndexType*  idx1 = index_  [1];
  const idx_t       dst0 = dstride_[0];
  const idx_t       dst1 = dstride_[1];
  const idx_t       ist0 = istride_[0];
  const idx_t       ist1 = istride_[1];
  const idx_t       sh0  = shape_  [0];
  const idx_t       sh1  = shape_  [1];

  Shape             pos;
  T*                ptr;


  pos = 0;

  if ( (ist0 + dst0) == 2_idx && rhs.isContiguous() )
  {
    idx_t  k = 0;

    do
    {
      ptr = data_;

      for ( int i = 2; i < N; i++ )
      {
        ptr += dstride_[i] * index_[i][pos[i] * istride_[i]];
      }

      for ( idx_t j = 0; j < sh1; j++ )
      {
        ptr += dst1 * idx1[j * ist1];

        for ( idx_t i = 0; i < sh0; i++, k++ )
        {
          T&  t = ptr[idx0[i]];

          t = op ( t, rhs.getFast(k) );
        }

        ptr -= dst1 * idx1[j * ist1];
      }
    }
    while ( array::incrIndex2( pos, shape_ ) );
  }
  else
  {
    do
    {
      ptr = data_;

      for ( int i = 2; i < N; i++ )
      {
        ptr += dstride_[i] * index_[i][pos[i] * istride_[i]];
      }

      for ( pos[1] = 0; pos[1] < sh1; pos[1]++ )
      {
        ptr += dst1 * idx1[pos[1] * ist1];

        for ( pos[0] = 0; pos[0] < sh0; pos[0]++ )
        {
          T&  t = ptr[dst0 * idx0[pos[0] * ist0]];

          t = op ( t, rhs[pos] );
        }

        ptr -= dst1 * idx1[pos[1] * ist1];
      }
    }
    while ( array::incrIndex2( pos, shape_ ) );
  }
}


//-----------------------------------------------------------------------
//   operator []
//-----------------------------------------------------------------------


template <class T, int N, class V>
  template <class I>

  inline T& Array<T,N,SelectExpr<V> >::operator []

  ( const Tuple<I,N>& pos ) const

{
  JEM_ASSERT2 ( array::checkIndex( pos, shape_ ),
                array::Errors::INVALID_INDEX );

  idx_t  k = dstride_[0] * (idx_t) index_[0][pos[0] * istride_[0]];

  for ( int i = 1; i < N; i++ )
  {
    k += dstride_[i] * (idx_t) index_[i][pos[i] * istride_[i]];
  }

  return data_[k];
}


//-----------------------------------------------------------------------
//   getFast
//-----------------------------------------------------------------------


template <class T, int N, class V>

  T& Array<T,N,SelectExpr<V> >::getFast ( idx_t i ) const

{
  array::getFastError ();

  return data_[0];
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


template <class T, int N, class V>

  inline const Tuple<idx_t,N>&
  Array<T,N,SelectExpr<V> >::shape () const

{
  return shape_;
}


//-----------------------------------------------------------------------
//   isContiguous
//-----------------------------------------------------------------------


template <class T, int N, class V>

  inline bool Array<T,N,SelectExpr<V> >::isContiguous () const

{
  return false;
}


//-----------------------------------------------------------------------
//   initIndex_
//-----------------------------------------------------------------------


template <class T, int N, class V>
  template <class I>

  inline void Array<T,N,SelectExpr<V> >::initIndex_

  ( I      idx,
    int    idim,
    idx_t  size,
    idx_t  stride )

{
  JEM_ASSERT2 ( checkIndex( (idx_t) idx, size ),
                array::Errors::INVALID_INDEX );

  data_ += idx * stride;
}


template <class T, int N, class V>

  inline void Array<T,N,SelectExpr<V> >::initIndex_

  ( const IndexArray&  idx,
    int                idim,
    idx_t              size,
    idx_t              stride )

{
  JEM_ASSERT2 ( array::checkSelection( idx, size ),
                array::Errors::INVALID_SELECT );

  index_  [idim] = idx.addr ();
  shape_  [idim] = idx.size ();
  dstride_[idim] = stride;
  istride_[idim] = idx.stride ();
}


//=======================================================================
//   class Array<T,1,SelectExpr<V> >
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


template <class T, class V>

  inline Array<T,1,SelectExpr<V> >::Array

  ( const Array<T,1>&  a,
    const IndexArray&  i )

{
  JEM_ASSERT2 ( array::checkSelection( i, a.size() ),
                array::Errors::INVALID_SELECT );

  data_    = a.addr   ();
  index_   = i.addr   ();
  size_    = i.size   ();
  dstride_ = a.stride ();
  istride_ = i.stride ();
}


template <class T, class V>
  template <class I, class J>

  inline Array<T,1,SelectExpr<V> >::Array

  ( const Array<T,2>&  a,
    const I&           i,
    const J&           j )

{
  data_ = a.addr ();

  initIndex_ ( i, a.size(0), a.stride(0) );
  initIndex_ ( j, a.size(1), a.stride(1) );
}


template <class T, class V>
  template <class I, class J, class K>

  Array<T,1,SelectExpr<V> >::Array

  ( const Array<T,3>&  a,
    const I&           i,
    const J&           j,
    const K&           k )

{
  data_ = a.addr ();

  initIndex_ ( i, a.size(0), a.stride(0) );
  initIndex_ ( j, a.size(1), a.stride(1) );
  initIndex_ ( k, a.size(2), a.stride(2) );
}


template <class T, class V>
  template <class I, class J, class K,
            class L>

  Array<T,1,SelectExpr<V> >::Array

  ( const Array<T,4>&  a,
    const I&           i,
    const J&           j,
    const K&           k,
    const L&           l )

{
  data_ = a.addr ();

  initIndex_ ( i, a.size(0), a.stride(0) );
  initIndex_ ( j, a.size(1), a.stride(1) );
  initIndex_ ( k, a.size(2), a.stride(2) );
  initIndex_ ( l, a.size(3), a.stride(3) );
}


template <class T, class V>
  template <class I, class J, class K,
            class L, class P>

  Array<T,1,SelectExpr<V> >::Array

  ( const Array<T,5>&  a,
    const I&           i,
    const J&           j,
    const K&           k,
    const L&           l,
    const P&           p )

{
  data_ = a.addr ();

  initIndex_ ( i, a.size(0), a.stride(0) );
  initIndex_ ( j, a.size(1), a.stride(1) );
  initIndex_ ( k, a.size(2), a.stride(2) );
  initIndex_ ( l, a.size(3), a.stride(3) );
  initIndex_ ( p, a.size(4), a.stride(4) );
}


template <class T, class V>
  template <class I, class J, class K,
            class L, class P, class Q>

  Array<T,1,SelectExpr<V> >::Array

  ( const Array<T,6>&  a,
    const I&           i,
    const J&           j,
    const K&           k,
    const L&           l,
    const P&           p,
    const Q&           q )

{
  data_ = a.addr ();

  initIndex_ ( i, a.size(0), a.stride(0) );
  initIndex_ ( j, a.size(1), a.stride(1) );
  initIndex_ ( k, a.size(2), a.stride(2) );
  initIndex_ ( l, a.size(3), a.stride(3) );
  initIndex_ ( p, a.size(4), a.stride(4) );
  initIndex_ ( q, a.size(5), a.stride(5) );
}


template <class T, class V>

  inline Array<T,1,SelectExpr<V> >::Array ( const Array& rhs )

{
  data_    = rhs.data_;
  index_   = rhs.index_;
  size_    = rhs.size_;
  dstride_ = rhs.dstride_;
  istride_ = rhs.istride_;
}


//-----------------------------------------------------------------------
//   operator =
//-----------------------------------------------------------------------


template <class T, class V>
  template <class U>

  const
  Array<T,1,SelectExpr<V> >&
  Array<T,1,SelectExpr<V> >::operator = ( const U& rhs ) const

{
  const idx_t        n   = size_;
  const IndexType*  idx = index_;
  T*                ptr = data_;

  if ( isContiguous() )
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      ptr[idx[i]] = rhs;
    }
  }
  else
  {
    const idx_t  dst = dstride_;
    const idx_t  ist = istride_;

    for ( idx_t i = 0; i < n; i++ )
    {
      ptr[dst * idx[i * ist]] = rhs;
    }
  }

  return *this;
}


template <class T, class V>

  inline const
  Array<T,1,SelectExpr<V> >&
  Array<T,1,SelectExpr<V> >::operator = ( const Array& rhs ) const

{
  if ( this != &rhs )
  {
    this->assign ( rhs, array::Pass2nd() );
  }

  return *this;
}


template <class T, class V>
  template <class U, class E>

  inline const
  Array<T,1,SelectExpr<V> >&
  Array<T,1,SelectExpr<V> >::operator =

  ( const Array<U,1,E>& rhs ) const

{
  this->assign ( rhs, array::Pass2nd() );

  return *this;
}


//-----------------------------------------------------------------------
//   assign
//-----------------------------------------------------------------------


template <class T, class V>
  template <class U, class Op>

  void Array<T,1,SelectExpr<V> >::assign

  ( const U&   rhs,
    const Op&  op ) const

{
  const idx_t        n   = size_;
  const IndexType*  idx = index_;
  T*                ptr = data_;

  if ( isContiguous() )
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      T&  t = ptr[idx[i]];

      t = op ( t, rhs );
    }
  }
  else
  {
    const idx_t  dst = dstride_;
    const idx_t  ist = istride_;

    for ( idx_t i = 0; i < n; i++ )
    {
      T&  t = ptr[dst * idx[i * ist]];

      t = op ( t, rhs );
    }
  }
}


template <class T, class V>
  template <class Op>

  void Array<T,1,SelectExpr<V> >::assign

  ( const Array&  rhs,
    const Op&     op ) const

{
  JEM_ASSERT2 ( size() == rhs.size(),
                array::Errors::SIZE_MISMATCH );

  const idx_t       n    = size_;
  const IndexType*  lidx = index_;
  T*                lptr = data_;
  const IndexType*  ridx = rhs.index_;
  const T*          rptr = rhs.data_;

  if ( isContiguous() && rhs.isContiguous() )
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      T&  t = lptr[lidx[i]];

      t = op ( t, rptr[ridx[i]] );
    }
  }
  else
  {
    const idx_t  ldst = dstride_;
    const idx_t  list = istride_;
    const idx_t  rdst = rhs.dstride_;
    const idx_t  rist = rhs.istride_;

    for ( idx_t i = 0; i < n; i++ )
    {
      T&  t = lptr[ldst * lidx[i * list]];

      t = op ( t, rptr[rdst * ridx[i * rist]]  );
    }
  }
}


template <class T, class V>
  template <class U, class E, class Op>

  void Array<T,1,SelectExpr<V> >::assign

  ( const Array<U,1,E>&  rhs,
    const Op&            op ) const

{
  JEM_ASSERT2 ( size() == rhs.size(),
                array::Errors::SIZE_MISMATCH );

  const idx_t        n   = size_;
  const IndexType*  idx = index_;
  T*                ptr = data_;

  if ( isContiguous() && rhs.isContiguous() )
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      T&  t = ptr[idx[i]];

      t = op ( t, rhs.getFast(i) );
    }
  }
  else
  {
    const idx_t  dst = dstride_;
    const idx_t  ist = istride_;

    for ( idx_t i = 0; i < n; i++ )
    {
      T&  t = ptr[dst * idx[ist * i]];

      t = op ( t, rhs[i] );
    }
  }
}


//-----------------------------------------------------------------------
//   operator []
//-----------------------------------------------------------------------


template <class T, class V>
  template <class I>

  inline T&
  Array<T,1,SelectExpr<V> >::operator [] ( I i ) const

{
  JEM_ASSERT2 ( checkIndex( i, size() ),
                array::Errors::INVALID_INDEX );

  return data_[dstride_ * index_[i * istride_]];
}


template <class T, class V>
  template <class I>

  inline T&
  Array<T,1,SelectExpr<V> >::operator [] ( const Tuple<I,1>& i ) const

{
  JEM_ASSERT2 ( checkIndex( i[0], size() ),
                array::Errors::INVALID_INDEX );

  return data_[dstride_ * index_[i[0] * istride_]];
}


//-----------------------------------------------------------------------
//   getFast
//-----------------------------------------------------------------------


template <class T, class V>

  inline T& Array<T,1,SelectExpr<V> >::getFast ( idx_t i ) const

{
  return data_[index_[i]];
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


template <class T, class V>

  inline idx_t Array<T,1,SelectExpr<V> >::size () const

{
  return size_;
}


template <class T, class V>

  inline Tuple<idx_t,1> Array<T,1,SelectExpr<V> >::shape () const

{
  return Shape ( size_ );
}


//-----------------------------------------------------------------------
//   isContiguous
//-----------------------------------------------------------------------


template <class T, class V>

  inline bool Array<T,1,SelectExpr<V> >::isContiguous () const

{
  return (dstride_ + istride_ == 2_idx);
}


//-----------------------------------------------------------------------
//   initIndex_
//-----------------------------------------------------------------------


template <class T, class V>
  template <class I>

  inline void Array<T,1,SelectExpr<V> >::initIndex_

  ( I      idx,
    idx_t  size,
    idx_t  stride )

{
  JEM_ASSERT2 ( checkIndex( (idx_t) idx, size ),
                array::Errors::INVALID_INDEX );

  data_ += idx * stride;
}


template <class T, class V>

  inline void Array<T,1,SelectExpr<V> >::initIndex_

  ( const IndexArray&  idx,
    idx_t              size,
    idx_t              stride )

{
  JEM_ASSERT2 ( array::checkSelection( idx, size ),
                array::Errors::INVALID_SELECT );

  index_   = idx.addr ();
  size_    = idx.size ();
  dstride_ = stride;
  istride_ = idx.stride ();
}


//=======================================================================
//   class Array<T,N,SelectLastExpr<V> >
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


template <class T, int N, class V>

  Array<T,N,SelectLastExpr<V> >::Array

  ( const Array<T,N>&  a,
    const IndexArray&  i )

{
  static_assert ( N >= 2, "invalid rank" );

  JEM_ASSERT2 ( array::checkSelection( i, a.size(N - 1) ),
                array::Errors::INVALID_SELECT );

  data_    = a.addr    ();
  index_   = i.addr    ();
  shape_   = a.shape   ();
  dstride_ = a.strides ();
  istride_ = i.stride  ();

  shape_[N - 1] = i.size ();
}


template <class T, int N, class V>

  inline Array<T,N,SelectLastExpr<V> >::Array ( const Array& rhs )

{
  data_    = rhs.data_;
  index_   = rhs.index_;
  shape_   = rhs.shape_;
  dstride_ = rhs.dstride_;
  istride_ = rhs.istride_;
}


//-----------------------------------------------------------------------
//   operator =
//-----------------------------------------------------------------------


template <class T, int N, class V>

  inline const
  Array<T,N,SelectLastExpr<V> >&
  Array<T,N,SelectLastExpr<V> >::operator = ( const Array& rhs ) const

{
  this->assign ( rhs, array::Pass2nd() );

  return *this;
}


template <class T, int N, class V>
  template <class X>

  inline const
  Array<T,N,SelectLastExpr<V> >&
  Array<T,N,SelectLastExpr<V> >::operator = ( const X& rhs ) const

{
  this->assign ( rhs, array::Pass2nd() );

  return *this;
}


//-----------------------------------------------------------------------
//   assign
//-----------------------------------------------------------------------


template <class T, int N, class V>
  template <class U, class Op>

  void Array<T,N,SelectLastExpr<V> >::assign

  ( const U&   rhs,
    const Op&  op ) const

{
  if ( ! array::size( shape_ ) )
  {
    return;
  }

  T*  ptr;

  if ( array::isContiguous( shape_, dstride_ ) )
  {
    const idx_t  st1 = dstride_[N - 1];
    const idx_t  sh1 = shape_  [N - 1];

    for ( idx_t i = 0; i < sh1; i++ )
    {
      ptr = data_ + (st1 * index_[i * istride_]);

      for ( idx_t j = 0; j < st1; j++ )
      {
        ptr[j] = op ( ptr[j], rhs );
      }
    }
  }
  else
  {
    const idx_t  st0 = dstride_[0];
    const idx_t  sh0 = shape_  [0];

    Shape       pos;

    pos = 0_idx;

    do
    {
      ptr = data_;

      for ( int i = 1; i < N - 1; i++ )
      {
        ptr += dstride_[i] * pos[i];
      }

      ptr += dstride_[N - 1] * index_[pos[N - 1] * istride_];

      for ( idx_t i = 0; i < sh0; i++ )
      {
        ptr[i * st0] = op ( ptr[i * st0], rhs );
      }

      pos[0] = sh0;
    }
    while ( array::incrIndex( pos, shape_ ) );
  }
}


template <class T, int N, class V>
  template <class U, class E, class Op>

  void Array<T,N,SelectLastExpr<V> >::assign

  ( const Array<U,N,E>&  rhs,
    const Op&            op ) const

{
  JEM_ASSERT2 ( equal( shape(), rhs.shape() ),
                array::Errors::SHAPE_MISMATCH );

  if ( ! array::size( shape_ ) )
  {
    return;
  }

  T*  ptr;

  if ( rhs.isContiguous() &&
       array::isContiguous( shape_, dstride_ ) )
  {
    const idx_t  st1 = dstride_[N - 1];
    const idx_t  sh1 = shape_  [N - 1];

    idx_t        k   = 0;

    for ( idx_t i = 0; i < sh1; i++ )
    {
      ptr = data_ + (st1 * index_[i * istride_]);

      for ( idx_t j = 0; j < st1; j++, k++ )
      {
        ptr[j] = op ( ptr[j], rhs.getFast(k) );
      }
    }
  }
  else
  {
    const idx_t  st0 = dstride_[0];
    const idx_t  sh0 = shape_  [0];

    Shape       pos;

    pos = 0_idx;

    do
    {
      ptr = data_;

      for ( int i = 1; i < N - 1; i++ )
      {
        ptr += dstride_[i] * pos[i];
      }

      ptr += dstride_[N - 1] * index_[pos[N - 1] * istride_];

      for ( pos[0] = 0; pos[0] < sh0; pos[0]++ )
      {
        ptr[pos[0] * st0] = op ( ptr[pos[0] * st0], rhs[pos] );
      }
    }
    while ( array::incrIndex( pos, shape_ ) );
  }
}


//-----------------------------------------------------------------------
//   operator []
//-----------------------------------------------------------------------


template <class T, int N, class V>
  template <class I>

  inline T&
  Array<T,N,SelectLastExpr<V> >::operator []

  ( const Tuple<I,N>& pos ) const

{
  JEM_ASSERT2 ( array::checkIndex( pos, shape_ ),
                array::Errors::INVALID_INDEX );

  idx_t  k = dstride_[0] * pos[0];

  for ( int i = 1; i < N - 1; i++ )
  {
    k += dstride_[i] * pos[i];
  }

  k += dstride_[N - 1] * (idx_t) index_[pos[N - 1] * istride_];

  return data_[k];
}


//-----------------------------------------------------------------------
//   getFast
//-----------------------------------------------------------------------


template <class T, int N, class V>

  T& Array<T,N,SelectLastExpr<V> >::getFast ( idx_t i ) const

{
  array::getFastError ();

  return data_[0];
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


template <class T, int N, class V>

  inline const Tuple<idx_t,N>&
  Array<T,N,SelectLastExpr<V> >::shape () const

{
  return shape_;
}


//-----------------------------------------------------------------------
//   isContiguous
//-----------------------------------------------------------------------


template <class T, int N, class V>

  inline bool Array<T,N,SelectLastExpr<V> >::isContiguous () const

{
  return false;
}


//=======================================================================
//   class Array<T,1,Select1Expr<E,I> >
//=======================================================================


template <class T, class E, class I>

  class Array<T,1,Select1Expr<E,I> >

{
 public:

  typedef Tuple<idx_t,1>        Shape;
  typedef Array<T,1,E>          ArgType;
  typedef typename
    ArgType::Reference          Reference;
  typedef typename
    SelectTraits<I>::ArrayType  IndexType;


                          Array

    ( const ArgType&        arg,
      const IndexType&      idx ) :

      arg_ ( arg ),
      idx_ ( idx )

  {}


                          Array

    ( const Array&          rhs ) :

      arg_ ( rhs.arg_ ),
      idx_ ( rhs.idx_ )

  {}


  const Array&            operator =

    ( const Array&          rhs )          const

  {
    array::assign ( *this, rhs, array::Pass2nd() );

    return *this;
  }


  template <class X>

    const Array&          operator =

    ( const X&              rhs )          const

  {
    array::assign ( *this, rhs, array::Pass2nd() );

    return *this;
  }


  template <class X,
            class Op>

    void                  assign

    ( const X&              rhs,
      const Op&             op )           const

  {
    array::assign ( *this, rhs, op );
  }


  template <class X>

    Reference             operator []

    ( const X&              i )            const

  {
    return arg_[idx_[i]];
  }


  Reference               getFast

    ( idx_t                 i )            const

  {
    return arg_.getFast ( idx_.getFast(i) );
  }


  idx_t                   size          () const
  {
    return idx_.size ();
  }


  Shape                   shape         () const
  {
    return idx_.shape ();
  }


  bool                    isContiguous  () const
  {
    return ( arg_.isContiguous() &&
             idx_.isContiguous() );
  }


 private:

  typename ArrayTraits
    <ArgType>::Reference      arg_;
  typename ArrayTraits
    <IndexType>::Reference    idx_;

};


//=======================================================================
//   class Array<T,2,Select2Expr<E,I,J> >
//=======================================================================


template <class T, class E, class I, class J>

  class Array<T,2,Select2Expr<E,I,J> >

{
 public:

  typedef Tuple<idx_t,2>        Shape;
  typedef Array<T,2,E>          ArgType;
  typedef typename
    ArgType::Reference          Reference;
  typedef typename
    SelectTraits<I>::ArrayType  Index0Type;
  typedef typename
    SelectTraits<J>::ArrayType  Index1Type;


                          Array

    ( const ArgType&        arg,
      const Index0Type&     idx0,
      const Index1Type&     idx1 ) :

      arg_  ( arg  ),
      idx0_ ( idx0 ),
      idx1_ ( idx1 )

  {}


                          Array

    ( const Array&          rhs ) :

      arg_  ( rhs.arg_  ),
      idx0_ ( rhs.idx0_ ),
      idx1_ ( rhs.idx1_ )

  {}


  const Array&            operator =

    ( const Array&          rhs )          const

  {
    array::assign ( *this, rhs, array::Pass2nd() );

    return *this;
  }


  template <class X>

    const Array&          operator =

    ( const X&              rhs )          const

  {
    array::assign ( *this, rhs, array::Pass2nd() );

    return *this;
  }


  template <class X,
            class Op>

    void                  assign

    ( const X&              rhs,
      const Op&             op )           const

  {
    array::assign ( *this, rhs, op );
  }


  template <class X>

    Reference             operator []

    ( const Tuple<X,2>&     i )            const

  {
    return arg_[Tuple<X,2>( idx0_[i[0]],
                            idx1_[i[1]] )];
  }


  Reference               getFast

    ( idx_t                 i )            const

  {
    array::getFastError ();

    return arg_.getFast ( 0 );
  }


  Shape                   shape         () const
  {
    return Shape ( idx0_.size(),
                   idx1_.size() );
  }


  bool                    isContiguous  () const
  {
    return false;
  }


 private:

  typename ArrayTraits
    <ArgType>::Reference      arg_;
  typename ArrayTraits
    <Index0Type>::Reference   idx0_;
  typename ArrayTraits
    <Index1Type>::Reference   idx1_;

};


//=======================================================================
//   class Array<T,3,Select3Expr<E,I,J,K>
//=======================================================================


template <class T, class E, class I, class J, class K>

  class Array<T,3,Select3Expr<E,I,J,K> >

{
 public:

  typedef Tuple<idx_t,3>        Shape;
  typedef Array<T,3,E>          ArgType;
  typedef typename
    ArgType::Reference          Reference;
  typedef typename
    SelectTraits<I>::ArrayType  Index0Type;
  typedef typename
    SelectTraits<J>::ArrayType  Index1Type;
  typedef typename
    SelectTraits<K>::ArrayType  Index2Type;


                        Array

    ( const ArgType&      arg,
      const Index0Type&   idx0,
      const Index1Type&   idx1,
      const Index2Type&   idx2 ) :

      arg_  ( arg  ),
      idx0_ ( idx0 ),
      idx1_ ( idx1 ),
      idx2_ ( idx2 )

  {}


                          Array

    ( const Array&          rhs ) :

      arg_  ( rhs.arg_  ),
      idx0_ ( rhs.idx0_ ),
      idx1_ ( rhs.idx1_ ),
      idx2_ ( rhs.idx2_ )

  {}


  const Array&            operator =

    ( const Array&          rhs )          const

  {
    array::assign ( *this, rhs, array::Pass2nd() );

    return *this;
  }


  template <class X>

    const Array&          operator =

    ( const X&              rhs )          const

  {
    array::assign ( *this, rhs, array::Pass2nd() );

    return *this;
  }


  template <class X,
            class Op>

    void                  assign

    ( const X&              rhs,
      const Op&             op )           const

  {
    array::assign ( *this, rhs, op );
  }


  template <class X>

    Reference             operator []

    ( const Tuple<X,3>&     i )            const

  {
    return arg_[Tuple<X,3>( idx0_[i[0]],
                            idx1_[i[1]],
                            idx2_[i[2]] )];
  }


  Reference               getFast

    ( idx_t                 i )            const

  {
    array::getFastError ();

    return arg_.getFast ( 0 );
  }


  Shape                   shape         () const
  {
    return Shape ( idx0_.size(),
                   idx1_.size(),
                   idx2_.size() );
  }


  bool                    isContiguous  () const
  {
    return false;
  }


 private:

  typename ArrayTraits
    <ArgType>::Reference      arg_;
  typename ArrayTraits
    <Index0Type>::Reference   idx0_;
  typename ArrayTraits
    <Index1Type>::Reference   idx1_;
  typename ArrayTraits
    <Index2Type>::Reference   idx2_;

};


//=======================================================================
//   class Array<T,4,Select4Expr<E,I,J,K,L> >
//=======================================================================


template <class T, class E, class I,
          class J, class K, class L>

  class Array<T,4,Select4Expr<E,I,J,K,L> >

{
 public:

  typedef Tuple<idx_t,4>        Shape;
  typedef Array<T,4,E>          ArgType;
  typedef typename
    ArgType::Reference          Reference;
  typedef typename
    SelectTraits<I>::ArrayType  Index0Type;
  typedef typename
    SelectTraits<J>::ArrayType  Index1Type;
  typedef typename
    SelectTraits<K>::ArrayType  Index2Type;
  typedef typename
    SelectTraits<L>::ArrayType  Index3Type;


                          Array

    ( const ArgType&        arg,
      const Index0Type&     idx0,
      const Index1Type&     idx1,
      const Index2Type&     idx2,
      const Index3Type&     idx3 ) :

      arg_  ( arg  ),
      idx0_ ( idx0 ),
      idx1_ ( idx1 ),
      idx2_ ( idx2 ),
      idx3_ ( idx3 )

  {}


                          Array

    ( const Array&          rhs ) :

      arg_  ( rhs.arg_  ),
      idx0_ ( rhs.idx0_ ),
      idx1_ ( rhs.idx1_ ),
      idx2_ ( rhs.idx2_ ),
      idx3_ ( rhs.idx3_ )

  {}


  const Array&            operator =

    ( const Array&          rhs )          const

  {
    array::assign ( *this, rhs, array::Pass2nd() );

    return *this;
  }


  template <class X>

    const Array&          operator =

    ( const X&              rhs )          const

  {
    array::assign ( *this, rhs, array::Pass2nd() );

    return *this;
  }


  template <class X,
            class Op>

    void                  assign

    ( const X&              rhs,
      const Op&             op )           const

  {
    array::assign ( *this, rhs, op );
  }


  template <class X>

    Reference             operator []

    ( const Tuple<X,4>&     i )            const

  {
    return arg_[Tuple<X,4>( idx0_[i[0]],
                            idx1_[i[1]],
                            idx2_[i[2]],
                            idx3_[i[3]] )];
  }


  Reference               getFast

    ( idx_t                 i )            const

  {
    array::getFastError ();

    return arg_.getFast ( 0 );
  }


  Shape                   shape         () const
  {
    return Shape ( idx0_.size(),
                   idx1_.size(),
                   idx2_.size(),
                   idx3_.size() );
  }


  bool                    isContiguous  () const
  {
    return false;
  }


 private:

  typename ArrayTraits
    <ArgType>::Reference      arg_;
  typename ArrayTraits
    <Index0Type>::Reference   idx0_;
  typename ArrayTraits
    <Index1Type>::Reference   idx1_;
  typename ArrayTraits
    <Index2Type>::Reference   idx2_;
  typename ArrayTraits
    <Index3Type>::Reference   idx3_;

};


//=======================================================================
//   class Array<T,5,Select5Expr<E,I,J,K,L,P> >
//=======================================================================


template <class T, class E, class I,
          class J, class K, class L, class P>

  class Array<T,5,Select5Expr<E,I,J,K,L,P> >

{
 public:

  typedef Tuple<idx_t,5>        Shape;
  typedef Array<T,5,E>          ArgType;
  typedef typename
    ArgType::Reference          Reference;
  typedef typename
    SelectTraits<I>::ArrayType  Index0Type;
  typedef typename
    SelectTraits<J>::ArrayType  Index1Type;
  typedef typename
    SelectTraits<K>::ArrayType  Index2Type;
  typedef typename
    SelectTraits<L>::ArrayType  Index3Type;
  typedef typename
    SelectTraits<P>::ArrayType  Index4Type;


                          Array

    ( const ArgType&        arg,
      const Index0Type&     idx0,
      const Index1Type&     idx1,
      const Index2Type&     idx2,
      const Index3Type&     idx3,
      const Index4Type&     idx4 ) :

      arg_  ( arg  ),
      idx0_ ( idx0 ),
      idx1_ ( idx1 ),
      idx2_ ( idx2 ),
      idx3_ ( idx3 ),
      idx4_ ( idx4 )

  {}


                          Array

    ( const Array&          rhs ) :

      arg_  ( rhs.arg_  ),
      idx0_ ( rhs.idx0_ ),
      idx1_ ( rhs.idx1_ ),
      idx2_ ( rhs.idx2_ ),
      idx3_ ( rhs.idx3_ ),
      idx4_ ( rhs.idx4_ )

  {}


  const Array&            operator =

    ( const Array&          rhs )          const

  {
    array::assign ( *this, rhs, array::Pass2nd() );

    return *this;
  }


  template <class X>

    const Array&          operator =

    ( const X&              rhs )          const

  {
    array::assign ( *this, rhs, array::Pass2nd() );

    return *this;
  }


  template <class X,
            class Op>

    void                  assign

    ( const X&              rhs,
      const Op&             op )           const

  {
    array::assign ( *this, rhs, op );
  }


  template <class X>

    Reference             operator []

    ( const Tuple<X,5>&     i )            const

  {
    return arg_[Tuple<X,5>( idx0_[i[0]],
                            idx1_[i[1]],
                            idx2_[i[2]],
                            idx3_[i[3]],
                            idx4_[i[4]] )];
  }


  Reference               getFast

    ( idx_t                 i )            const

  {
    array::getFastError ();

    return arg_.getFast ( 0 );
  }


  Shape                   shape         () const
  {
    return Shape ( idx0_.size(),
                   idx1_.size(),
                   idx2_.size(),
                   idx3_.size(),
                   idx4_.size() );
  }


  bool                    isContiguous  () const
  {
    return false;
  }


 private:

  typename ArrayTraits
    <ArgType>::Reference      arg_;
  typename ArrayTraits
    <Index0Type>::Reference   idx0_;
  typename ArrayTraits
    <Index1Type>::Reference   idx1_;
  typename ArrayTraits
    <Index2Type>::Reference   idx2_;
  typename ArrayTraits
    <Index3Type>::Reference   idx3_;
  typename ArrayTraits
    <Index4Type>::Reference   idx4_;

};


//=======================================================================
//   class Array<T,6,Select6Expr<E,I,J,K,L,P,Q> >
//=======================================================================


template <class T, class E, class I, class J,
          class K, class L, class P, class Q>

  class Array<T,6,Select6Expr<E,I,J,K,L,P,Q> >

{
 public:

  typedef Tuple<idx_t,6>        Shape;
  typedef Array<T,6,E>          ArgType;
  typedef typename
    ArgType::Reference          Reference;
  typedef typename
    SelectTraits<I>::ArrayType  Index0Type;
  typedef typename
    SelectTraits<J>::ArrayType  Index1Type;
  typedef typename
    SelectTraits<K>::ArrayType  Index2Type;
  typedef typename
    SelectTraits<L>::ArrayType  Index3Type;
  typedef typename
    SelectTraits<P>::ArrayType  Index4Type;
  typedef typename
    SelectTraits<Q>::ArrayType  Index5Type;


                          Array

    ( const ArgType&        arg,
      const Index0Type&     idx0,
      const Index1Type&     idx1,
      const Index2Type&     idx2,
      const Index3Type&     idx3,
      const Index4Type&     idx4,
      const Index5Type&     idx5 ) :

      arg_  ( arg  ),
      idx0_ ( idx0 ),
      idx1_ ( idx1 ),
      idx2_ ( idx2 ),
      idx3_ ( idx3 ),
      idx4_ ( idx4 ),
      idx5_ ( idx5 )

  {}


                          Array

    ( const Array&          rhs ) :

      arg_  ( rhs.arg_  ),
      idx0_ ( rhs.idx0_ ),
      idx1_ ( rhs.idx1_ ),
      idx2_ ( rhs.idx2_ ),
      idx3_ ( rhs.idx3_ ),
      idx4_ ( rhs.idx4_ ),
      idx5_ ( rhs.idx5_ )

  {}


  const Array&            operator =

    ( const Array&          rhs )          const

  {
    array::assign ( *this, rhs, array::Pass2nd() );

    return *this;
  }


  template <class X>

    const Array&          operator =

    ( const X&              rhs )          const

  {
    array::assign ( *this, rhs, array::Pass2nd() );

    return *this;
  }


  template <class X,
            class Op>

    void                  assign

    ( const X&              rhs,
      const Op&             op )           const

  {
    array::assign ( *this, rhs, op );
  }


  template <class X>

    Reference             operator []

    ( const Tuple<X,6>&     i )            const

  {
    return arg_[Tuple<X,6>( idx0_[i[0]],
                            idx1_[i[1]],
                            idx2_[i[2]],
                            idx3_[i[3]],
                            idx4_[i[4]],
                            idx5_[i[5]] )];
  }


  Reference               getFast

    ( idx_t                 i )            const

  {
    array::getFastError ();

    return arg_.getFast ( 0 );
  }


  Shape                   shape         () const
  {
    return Shape ( idx0_.size(),
                   idx1_.size(),
                   idx2_.size(),
                   idx3_.size(),
                   idx4_.size(),
                   idx5_.size() );
  }

  bool                    isContiguous  () const
  {
    return false;
  }


 private:

  typename ArrayTraits
    <ArgType>::Reference      arg_;
  typename ArrayTraits
    <Index0Type>::Reference   idx0_;
  typename ArrayTraits
    <Index1Type>::Reference   idx1_;
  typename ArrayTraits
    <Index2Type>::Reference   idx2_;
  typename ArrayTraits
    <Index3Type>::Reference   idx3_;
  typename ArrayTraits
    <Index4Type>::Reference   idx4_;
  typename ArrayTraits
    <Index5Type>::Reference   idx5_;

};


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   select
//-----------------------------------------------------------------------


template <class T, class E, class I>

  inline Array<T,1,Select1Expr<E,I> >

                        select

  ( const Array<T,1,E>&   arg,
    const I&              i )

{
  using array::toArray;

  return Array<T,1,Select1Expr<E,I> > (
    arg,
    toArray ( i, arg.size() )
  );
}


template <class T, class E,
          class I, class J>

  inline Array<T,2,Select2Expr<E,I,J> >

                        select

  ( const Array<T,2,E>&   arg,
    const I&              i,
    const J&              j )

{
  using array::toArray;

  Tuple<idx_t,2>  sh = arg.shape ();

  return Array<T,2,Select2Expr<E,I,J> > (
    arg,
    toArray ( i, sh[0] ),
    toArray ( j, sh[1] )
  );
}


template <class T, class E,
          class I, class J, class K>

  inline Array<T,3,Select3Expr<E,I,J,K> >

                        select

  ( const Array<T,3,E>&   arg,
    const I&              i,
    const J&              j,
    const K&              k )

{
  using array::toArray;

  Tuple<idx_t,3>  sh = arg.shape ();

  return Array<T,3,Select3Expr<E,I,J,K> > (
    arg,
    toArray ( i, sh[0] ),
    toArray ( j, sh[1] ),
    toArray ( k, sh[2] )
  );
}


template <class T, class E,
          class I, class J, class K, class L>

  inline Array<T,4,Select4Expr<E,I,J,K,L> >

                        select

  ( const Array<T,4,E>&   arg,
    const I&              i,
    const J&              j,
    const K&              k,
    const L&              l )

{
  using array::toArray;

  Tuple<idx_t,4>  sh = arg.shape ();

  return Array<T,4,Select4Expr<E,I,J,K,L> > (
    arg,
    toArray ( i, sh[0] ),
    toArray ( j, sh[1] ),
    toArray ( k, sh[2] ),
    toArray ( l, sh[3] )
  );
}


template <class T, class E,
          class I, class J,
          class K, class L, class P>

  inline Array<T,5,Select5Expr<E,I,J,K,L,P> >

                        select

  ( const Array<T,5,E>&   arg,
    const I&              i,
    const J&              j,
    const K&              k,
    const L&              l,
    const P&              p )

{
  using array::toArray;

  Tuple<idx_t,5>  sh = arg.shape ();

  return Array<T,5,Select5Expr<E,I,J,K,L,P> > (
    arg,
    toArray ( i, sh[0] ),
    toArray ( j, sh[1] ),
    toArray ( k, sh[2] ),
    toArray ( l, sh[3] ),
    toArray ( p, sh[4] )
  );
}


template <class T, class E,
          class I, class J, class K,
          class L, class P, class Q>

  inline Array<T,6,Select6Expr<E,I,J,K,L,P,Q> >

                        select

  ( const Array<T,6,E>&   arg,
    const I&              i,
    const J&              j,
    const K&              k,
    const L&              l,
    const P&              p,
    const Q&              q )

{
  using array::toArray;

  Tuple<idx_t,6>  sh = arg.shape ();

  return Array<T,6,Select6Expr<E,I,J,K,L,P,Q> > (
    arg,
    toArray ( i, sh[0] ),
    toArray ( j, sh[1] ),
    toArray ( k, sh[2] ),
    toArray ( l, sh[3] ),
    toArray ( p, sh[4] ),
    toArray ( q, sh[5] )
  );
}


JEM_END_PACKAGE_BASE

#endif
