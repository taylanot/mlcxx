
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

#ifndef JEM_BASE_ARRAY_TENSOR_TCC
#define JEM_BASE_ARRAY_TENSOR_TCC

#include <jem/base/tuple/utilities.h>
#include <jem/base/array/tensor_axis.h>
#include <jem/base/array/tensor_utils.h>


//=======================================================================
//   helper macros
//=======================================================================


#undef JEM_CHECK_UBOUND
#undef JEM_CHECK_UNIQUE

#ifdef NDEBUG

#  define JEM_CHECK_UBOUND( axis, ix, ub )   ((void) 0)
#  define JEM_CHECK_UNIQUE( idx_t )          ((void) 0)

#else

#  define JEM_CHECK_UBOUND( axis, ix, ub )       \
                                                 \
  if ( (ub) < 0 )                                \
                                                 \
    ::jem::tensor::unboundedError ();            \
                                                 \
  axis.checkUbound ( ix, ub )

#  define JEM_CHECK_UNIQUE( idx_t )              \
                                                 \
  ::jem::tensor::checkUnique ( idx_t )

#endif


JEM_BEGIN_PACKAGE_BASE
JEM_BEGIN_NAMESPACE( tensor )


//=======================================================================
//   class Axis<idx_t,IndexExpr>
//=======================================================================


template <>

  class Axis<idx_t,IndexExpr>

{
 public:

  inline                  Axis

    ( const TensorIndex&    a,
      idx_t*                ix,
      idx_t*                jx );

  inline                  Axis

    ( const Axis&           rhs );

  inline void             reset         ();

  inline void             advance

    ( idx_t*                ix,
      idx_t                 i );

  inline void             advance

    ( idx_t*                ix,
      idx_t*                jx,
      idx_t                 j );

  inline idx_t            operator      []

    ( idx_t                 i )            const;

  inline idx_t            getFast

    ( idx_t                 i )            const;

  inline idx_t            getUbound

    ( idx_t*                ix )           const;

  inline void             checkUbound

    ( idx_t*                ix,
      idx_t                 ub )           const;

  inline bool             isContiguous  () const;


 private:

  Axis&                   operator =

    ( const Axis&           rhs );


 private:

  idx_t*                  addr_;
  idx_t                   value_;
  idx_t                   offset_;
  idx_t                   stride_;
  idx_t                   stride0_;
  idx_t                   stride1_;

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline Axis<idx_t,IndexExpr>::Axis

  ( const TensorIndex&  a,
    idx_t*              ix,
    idx_t*              jx )

{
  addr_    = a.addr   ();
  offset_  = a.first  ();
  stride_  = a.stride ();
  value_   = 0_idx;
  stride0_ = (ix == addr_) ? stride_ : 0_idx;
  stride1_ = (jx == addr_) ? stride_ : 0_idx;
}


inline Axis<idx_t,IndexExpr>::Axis ( const Axis& rhs )
{
  addr_    = rhs.addr_;
  value_   = rhs.value_;
  offset_  = rhs.offset_;
  stride_  = rhs.stride_;
  stride0_ = rhs.stride0_;
  stride1_ = rhs.stride1_;
}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


inline void Axis<idx_t,IndexExpr>::reset ()
{
  value_ = stride_ * (*addr_) + offset_;
}


//-----------------------------------------------------------------------
//   advance
//-----------------------------------------------------------------------


inline void Axis<idx_t,IndexExpr>::advance

  ( idx_t*  ix,
    idx_t   i )

{}


inline void Axis<idx_t,IndexExpr>::advance

  ( idx_t*  ix,
    idx_t*  jx,
    idx_t   j )

{
  value_ += stride1_;
}


//-----------------------------------------------------------------------
//   operator []
//-----------------------------------------------------------------------


inline idx_t Axis<idx_t,IndexExpr>::operator [] ( idx_t i ) const
{
  return (value_ + i * stride0_);
}


//-----------------------------------------------------------------------
//   getFast
//-----------------------------------------------------------------------


inline idx_t Axis<idx_t,IndexExpr>::getFast ( idx_t i ) const
{
  return (value_ + i * stride0_);
}


//-----------------------------------------------------------------------
//   getUbound
//-----------------------------------------------------------------------


inline idx_t Axis<idx_t,IndexExpr>::getUbound ( idx_t* ix ) const
{
  return -1_idx;
}


//-----------------------------------------------------------------------
//   checkUbound
//-----------------------------------------------------------------------


inline void Axis<idx_t,IndexExpr>::checkUbound

  ( idx_t*  ix,
    idx_t   ub ) const

{}


//-----------------------------------------------------------------------
//   isContiguous
//-----------------------------------------------------------------------


inline bool Axis<idx_t,IndexExpr>::isContiguous () const
{
  return true;
}


//=======================================================================
//   class Axis<T,TensorExpr<N> >
//=======================================================================


template <class T, int N>

  class Axis<T,TensorExpr<N> >

{
 public:

  typedef Array
    <T,0,TensorExpr<N> >  Tensor;


  inline                  Axis

    ( const Tensor&         t,
      idx_t*                ix,
      idx_t*                jx );

  inline                  Axis

    ( const Axis&           rhs );

  inline void             reset         ();

  inline void             advance

    ( idx_t*                jx,
      idx_t                 j );

  inline void             advance

    ( idx_t*                ix,
      idx_t*                jx,
      idx_t                 j );

  inline const T&         operator      []

    ( idx_t                 i )            const;

  inline const T&         getFast

    ( idx_t                 i )            const;

  inline idx_t            getUbound

    ( idx_t*                ix )           const;

  inline void             checkUbound

    ( idx_t*                ix,
      idx_t                 ub )           const;

  inline bool             isContiguous  () const;


 private:

  Axis&                   operator =

    ( const Axis&           rhs );


 private:

  const Tensor&           tensor_;
  const T*                addr_;
  const idx_t             stride0_;
  const idx_t             stride1_;

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


template <class T, int N>

  inline Axis<T,TensorExpr<N> >::Axis

  ( const Tensor&  t,
    idx_t*         ix,
    idx_t*         jx ) :

    tensor_  ( t ),
    addr_    ( 0 ),
    stride0_ ( t.getStride( ix ) ),
    stride1_ ( t.getStride( jx ) )

{}


template <class T, int N>

  inline Axis<T,TensorExpr<N> >::Axis ( const Axis& rhs ) :

    tensor_  ( rhs.tensor_  ),
    addr_    ( rhs.addr_    ),
    stride0_ ( rhs.stride0_ ),
    stride1_ ( rhs.stride1_ )

{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


template <class T, int N>

  inline void Axis<T,TensorExpr<N> >::reset ()

{
  addr_ = tensor_.getPointer ();
}


//-----------------------------------------------------------------------
//   advance
//-----------------------------------------------------------------------


template <class T, int N>

  inline void Axis<T,TensorExpr<N> >::advance

  ( idx_t*  ix,
    idx_t   i )

{}


template <class T, int N>

  inline void Axis<T,TensorExpr<N> >::advance

  ( idx_t*  ix,
    idx_t*  jx,
    idx_t   j )

{
  addr_ += stride1_;
}


//-----------------------------------------------------------------------
//   operator []
//-----------------------------------------------------------------------


template <class T, int N>

  inline const T&
  Axis<T,TensorExpr<N> >::operator [] ( idx_t i ) const

{
  return addr_[i * stride0_];
}


//-----------------------------------------------------------------------
//   getFast
//-----------------------------------------------------------------------


template <class T, int N>

  inline const T&
  Axis<T,TensorExpr<N> >::getFast ( idx_t i ) const

{
  return addr_[i];
}


//-----------------------------------------------------------------------
//   getUbound
//-----------------------------------------------------------------------


template <class T, int N>

  inline idx_t Axis<T,TensorExpr<N> >::getUbound ( idx_t* ix ) const

{
  return tensor_.getUbound ( ix );
}


//-----------------------------------------------------------------------
//   checkUbound
//-----------------------------------------------------------------------


template <class T, int N>

  inline void Axis<T,TensorExpr<N> >::checkUbound

  ( idx_t*  ix,
    idx_t   ub ) const

{
  tensor_.checkUbound ( ix, ub );
}


//-----------------------------------------------------------------------
//   isContiguous
//-----------------------------------------------------------------------


template <class T, int N>

  inline bool Axis<T,TensorExpr<N> >::isContiguous () const

{
  return (stride0_ == 1_idx);
}


//=======================================================================
//   reduction functions
//=======================================================================

//-----------------------------------------------------------------------
//   reduceFast
//-----------------------------------------------------------------------


template <class T, class Op, class E>

  inline T                reduceFast

  ( Axis<T,E>&              ax,
    idx_t                   n ,
    const Op&               op )

{
  T  t = op ( ax.getFast( 0_idx ),
              ax.getFast( 1_idx ) );

  for ( idx_t i = 2_idx; i < n; i++ )
  {
    t = op ( t, ax.getFast( i ) );
  }

  return t;
}


//-----------------------------------------------------------------------
//   reduceSlow
//-----------------------------------------------------------------------


template <class T, class Op, class E>

  inline T                reduceSlow

  ( Axis<T,E>&              ax,
    idx_t*                  ix,
    idx_t                   n ,
    const Op&               op )

{
  T  t = ax[0_idx];

  for ( idx_t i = 1_idx; i < n; i++ )
  {
    ax.advance ( ix, i );

    t = op ( t, ax[i] );
  }

  return t;
}


//-----------------------------------------------------------------------
//   reduce
//-----------------------------------------------------------------------


template <class T, class Op, class E>

  inline T                reduce

  ( Axis<T,E>&              ax,
    idx_t*                  ix,
    idx_t                   n ,
    const Op&               op )

{
  if      ( n < 2 )
  {
    return ax[0_idx];
  }
  else if ( ax.isContiguous() )
  {
    return reduceFast ( ax, n, op );
  }
  else
  {
    return reduceSlow ( ax, ix, n, op );
  }
}


//=======================================================================
//   class Axis<T,ReduceExpr<N,Op,A> >
//=======================================================================


template <class T, int N, class Op, class A>

  class Axis<T,ReduceExpr<N,Op,A> >

{
 public:

  typedef A               ArgType;


  template <class E>
    explicit              Axis

    ( const Array<T,0,
        ReduceExpr
          <N,Op,E> >&       rhs );

  inline                  Axis

    ( const Axis&           rhs );

  inline void             reset         ();

  inline void             advance

    ( idx_t*                ix,
      idx_t                 i );

  inline void             advance

    ( idx_t*                ix,
      idx_t*                jx,
      idx_t                 j );

  T                       operator      []

    ( idx_t                 i );

  T                       getFast

    ( idx_t                 i )            const;

  inline idx_t            getUbound

    ( idx_t*                ix )           const;

  inline void             checkUbound

    ( idx_t*                ix,
      idx_t                 ub )           const;

  inline bool             isContiguous  () const;


 private:

  Axis&                   operator =

    ( const Axis&           rhs );


 private:

  const Op                op_;
  ArgType                 raxis_;
  const Tuple<idx_t*,N>   index_;
  Tuple<idx_t,N>          ubound_;

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


template <class T, int N, class Op, class A>
  template <class E>

  Axis<T,ReduceExpr<N,Op,A> >::Axis

  ( const Array<T,0,ReduceExpr<N,Op,E> >& rhs ) :

    op_    ( rhs.op ),
    raxis_ ( getAxis( rhs.arg, rhs.index[0], rhs.index[1] ) ),
    index_ ( rhs.index )

{
  static_assert ( N >= 2, "invalid dimensions size" );

  JEM_CHECK_UNIQUE ( index_ );

  for ( int k = 0; k < N; k++ )
  {
    ubound_[k] = raxis_.getUbound ( index_[k] );

    JEM_CHECK_UBOUND ( raxis_, index_[k], ubound_[k] );
  }
}


template <class T, int N, class Op, class A>

  inline Axis<T,ReduceExpr<N,Op,A> >::Axis

  ( const Axis&  rhs ) :

    op_     ( rhs.op_     ),
    raxis_  ( rhs.raxis_  ),
    index_  ( rhs.index_  ),
    ubound_ ( rhs.ubound_ )

{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


template <class T, int N, class Op, class A>

  inline void Axis<T,ReduceExpr<N,Op,A> >::reset ()

{}


//-----------------------------------------------------------------------
//   advance
//-----------------------------------------------------------------------


template <class T, int N, class Op, class A>

  inline void Axis<T,ReduceExpr<N,Op,A> >::advance

  ( idx_t*  ix,
    idx_t   i )

{
  *ix = i;
}


template <class T, int N, class Op, class A>

  inline void Axis<T,ReduceExpr<N,Op,A> >::advance

  ( idx_t*  ix,
    idx_t*  jx,
    idx_t   j )

{
  *ix = 0_idx;
  *jx = j;
}


//-----------------------------------------------------------------------
//   operator []
//-----------------------------------------------------------------------


template <class T, int N, class Op, class A>

  T Axis<T,ReduceExpr<N,Op,A> >::operator [] ( idx_t i )

{
  if ( ! array::size( ubound_ ) )
  {
    return T ();
  }

  zeroIndex ( index_ );
  raxis_.reset      ();

  const idx_t  m  = ubound_[0];
  const idx_t  n  = ubound_[1];

  idx_t*       ix = index_ [0];
  idx_t*       jx = index_ [1];

  T            t  = reduce ( raxis_, ix, m, op_ );

  do
  {
    for ( idx_t j = 1_idx; j < n; j++ )
    {
      raxis_.advance ( ix, jx, j );

      t = op_ ( t, reduce( raxis_, ix, m, op_ ) );
    }

    if ( ! incrIndex( index_, ubound_ ) )
    {
      break;
    }

    raxis_.reset ();

    t = op_ ( t, reduce( raxis_, ix, m, op_ ) );
  }
  while ( true );

  return t;
}


//-----------------------------------------------------------------------
//   getFast
//-----------------------------------------------------------------------


template <class T, int N, class Op, class A>

  T Axis<T,ReduceExpr<N,Op,A> >::getFast ( idx_t i ) const

{
  array::getFastError ();

  return T ();
}


//-----------------------------------------------------------------------
//   getUbound
//-----------------------------------------------------------------------


template <class T, int N, class Op, class A>

  inline idx_t
  Axis<T,ReduceExpr<N,Op,A> >::getUbound ( idx_t* ix ) const

{
  return raxis_.getUbound ( ix );
}


//-----------------------------------------------------------------------
//   checkUbound
//-----------------------------------------------------------------------


template <class T, int N, class Op, class A>

  inline void Axis<T,ReduceExpr<N,Op,A> >::checkUbound

  ( idx_t*  ix,
    idx_t   ub ) const

{
  raxis_.checkUbound ( ix, ub );

  if ( select( index_, ix, ubound_, -1_idx ) >= 0 )
  {
    usedIndexError ();
  }
}


//-----------------------------------------------------------------------
//   isContiguous
//-----------------------------------------------------------------------


template <class T, int N, class Op, class A>

  inline bool Axis<T,ReduceExpr<N,Op,A> >::isContiguous () const

{
  return false;
}


//=======================================================================
//   class Axis<T,ReduceExpr<1,Op,A> >
//=======================================================================


template <class T, class Op, class A>

  class Axis<T,ReduceExpr<1,Op,A> >

{
 public:

  typedef A               ArgType;


  template <class E>
    inline                Axis

    ( const Array<T,0,
        ReduceExpr
          <1,Op,E> >&       rhs,
      idx_t*                ix );

  inline                  Axis

    ( const Axis&           rhs );

  inline void             reset         ();

  inline void             advance

    ( idx_t*                ix,
      idx_t                 i );

  inline void             advance

    ( idx_t*                ix,
      idx_t*                jx,
      idx_t                 j );

  inline T                operator      []

    ( idx_t                 i );

  T                       getFast

    ( idx_t                 i )            const;

  inline idx_t            getUbound

    ( idx_t*                ix )           const;

  inline void             checkUbound

    ( idx_t*                ix,
      idx_t                 ub )           const;

  inline bool             isContiguous  () const;


 private:

  Axis&                   operator =

    ( const Axis&           rhs );


 private:

  const Op                op_;
  ArgType                 raxis_;
  idx_t*                  index_;
  idx_t                   ubound_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


template <class T, class Op, class A>
  template <class E>

  inline Axis<T,ReduceExpr<1,Op,A> >::Axis

  ( const Array<T,0,ReduceExpr<1,Op,E> >&  rhs,
    idx_t*                                 ix ) :

    op_    ( rhs.op ),
    raxis_ ( getAxis( rhs.arg, rhs.index[0], ix ) ),
    index_ ( rhs.index[0] )

{
  ubound_ = raxis_.getUbound ( rhs.index[0] );

  JEM_CHECK_UBOUND ( raxis_, rhs.index[0], ubound_ );
}


template <class T, class Op, class A>

  inline Axis<T,ReduceExpr<1,Op,A> >::Axis

  ( const Axis&  rhs ) :

    op_     ( rhs.op_     ),
    raxis_  ( rhs.raxis_  ),
    index_  ( rhs.index_  ),
    ubound_ ( rhs.ubound_ )

{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


template <class T, class Op, class A>

  inline void Axis<T,ReduceExpr<1,Op,A> >::reset ()

{
  (*index_) = 0;

  raxis_.reset ();
}


//-----------------------------------------------------------------------
//   advance
//-----------------------------------------------------------------------


template <class T, class Op, class A>

  inline void Axis<T,ReduceExpr<1,Op,A> >::advance

    ( idx_t*  ix,
      idx_t   i )

{
  raxis_.advance ( index_, ix, i );
}


template <class T, class Op, class A>

  inline void Axis<T,ReduceExpr<1,Op,A> >::advance

    ( idx_t*  ix,
      idx_t*  jx,
      idx_t   j )

{
  *ix = 0_idx;
  *jx = j;

  reset ();
}


//-----------------------------------------------------------------------
//   operator []
//-----------------------------------------------------------------------


template <class T, class Op, class A>

  inline T Axis<T,ReduceExpr<1,Op,A> >::operator [] ( idx_t i )

{
  if ( ubound_ == 0_idx )
  {
    return T ();
  }
  else
  {
    return reduce ( raxis_, index_, ubound_, op_ );
  }
}



//-----------------------------------------------------------------------
//   getFast
//-----------------------------------------------------------------------


template <class T, class Op, class A>

  T Axis<T,ReduceExpr<1,Op,A> >::getFast ( idx_t i ) const

{
  array::getFastError ();

  return T ();
}


//-----------------------------------------------------------------------
//   getUbound
//-----------------------------------------------------------------------


template <class T, class Op, class A>

  inline idx_t
  Axis<T,ReduceExpr<1,Op,A> >::getUbound ( idx_t* ix ) const

{
  return raxis_.getUbound ( ix );
}


//-----------------------------------------------------------------------
//   checkUbound
//-----------------------------------------------------------------------


template <class T, class Op, class A>

  inline void Axis<T,ReduceExpr<1,Op,A> >::checkUbound

  ( idx_t*  ix,
    idx_t   ub ) const

{
  raxis_.checkUbound ( ix, ub );
}


//-----------------------------------------------------------------------
//   isContiguous
//-----------------------------------------------------------------------


template <class T, class Op, class A>

  inline bool Axis<T,ReduceExpr<1,Op,A> >::isContiguous () const

{
  return false;
}


//=======================================================================
//   class AxisTraits< Array<T,0,ReduceExpr> >
//=======================================================================


template <class T, int N, class Op, class A>

  class AxisTraits< Array<T,0,ReduceExpr<N,Op,A> > >

{
 public:

  typedef typename AxisTraits<A>::AxisType   ArgType;
  typedef Axis<T,ReduceExpr<N,Op,ArgType> >  AxisType;

  static AxisType         getAxis

    ( const Array<T,0,
      ReduceExpr<N,Op,A> >& a,
      idx_t*                ix,
      idx_t*                jx )

  {
    return AxisType ( a );
  }

};


template <class T, class Op, class A>

  class AxisTraits< Array<T,0,ReduceExpr<1,Op,A> > >

{
 public:

  typedef typename AxisTraits<A>::AxisType   ArgType;
  typedef Axis<T,ReduceExpr<1,Op,ArgType> >  AxisType;

  static AxisType         getAxis

    ( const Array<T,0,
      ReduceExpr<1,Op,A> >& a,
      idx_t*                ix,
      idx_t*                jx )

  {
    return AxisType ( a, ix );
  }

};


JEM_END_NAMESPACE( tensor )


//=======================================================================
//   class Array<T,0,TensorExpr<N> >
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


template <class T, int N>

  Array<T,0,TensorExpr<N> >::Array

  ( const Array<T,1>&   a,
    const TensorIndex&  i )

{
  static_assert ( N == 1, "invalid rank" );

  data_ = a.addr ();

  initIndex_ ( i, 0, a.size(), a.stride() );
}


template <class T, int N>
  template <class I, class J>

  Array<T,0,TensorExpr<N> >::Array

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


template <class T, int N>
  template <class I, class J, class K>

  Array<T,0,TensorExpr<N> >::Array

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


template <class T, int N>
  template <class I, class J, class K, class L>

  Array<T,0,TensorExpr<N> >::Array

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


template <class T, int N>
  template <class I, class J, class K, class L, class P>

  Array<T,0,TensorExpr<N> >::Array

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


template <class T, int N>
  template <class I, class J, class K,
            class L, class P, class Q>

  Array<T,0,TensorExpr<N> >::Array

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


template <class T, int N>

  inline Array<T,0,TensorExpr<N> >::Array ( const Array& rhs )

{
  data_   = rhs.data_;
  index_  = rhs.index_;
  ubound_ = rhs.ubound_;
  stride_ = rhs.stride_;
}


//-----------------------------------------------------------------------
//   operator =
//-----------------------------------------------------------------------


template <class T, int N>

  inline const
  Array<T,0,TensorExpr<N> >&
  Array<T,0,TensorExpr<N> >::operator =

    ( const Array&  rhs ) const

{
  if ( this != &rhs )
  {
    assign_ ( tensor::Is1D<N>(), rhs, array::Pass2nd() );
  }

  return *this;
}


template <class T, int N>
  template <class X>

  inline const
  Array<T,0,TensorExpr<N> >&
  Array<T,0,TensorExpr<N> >::operator =

    ( const X&  rhs ) const

{
  assign_ ( tensor::Is1D<N>(), rhs, array::Pass2nd() );

  return *this;
}


//-----------------------------------------------------------------------
//   assign
//-----------------------------------------------------------------------


template <class T, int N>
  template <class X,
            class Op>

  inline void Array<T,0,TensorExpr<N> >::assign

  ( const X&   rhs,
    const Op&  op ) const

{
  assign_ ( tensor::Is1D<N>(), rhs, op );
}


//-----------------------------------------------------------------------
//   getPointer
//-----------------------------------------------------------------------


template <class T, int N>

  inline T* Array<T,0,TensorExpr<N> >::getPointer () const

{
  return (data_ + tensor::ioffset( index_, stride_ ));
}


//-----------------------------------------------------------------------
//   getUbound
//-----------------------------------------------------------------------


template <class T, int N>

  inline idx_t
  Array<T,0,TensorExpr<N> >::getUbound ( idx_t* i ) const

{
  return tensor::select ( index_, i, ubound_, -1_idx );
}


//-----------------------------------------------------------------------
//   checkUbound
//-----------------------------------------------------------------------


template <class T, int N>

  inline void Array<T,0,TensorExpr<N> >::checkUbound

  ( idx_t*  i,
    idx_t   ub ) const

{
  // Make sure this also works with repeating indices.

  for ( int j = 0; j < N; j++ )
  {
    if ( index_[j] == i )
    {
      tensor::checkUbounds ( ub, ubound_[j] );
    };
  }
}


//-----------------------------------------------------------------------
//   getStride
//-----------------------------------------------------------------------


template <class T, int N>

  inline idx_t Array<T,0,TensorExpr<N> >::getStride ( idx_t* i ) const

{
  // Make sure this also works with repeating indices.

  idx_t  st = 0_idx;

  for ( int j = 0; j < N; j++ )
  {
    if ( index_[j] == i )
    {
      st += stride_[j];
    }
  }

  return st;
}


//-----------------------------------------------------------------------
//   assign_ (one-dimensional, skalar)
//-----------------------------------------------------------------------


template <class T, int N>
  template <class U, class Op>

  void Array<T,0,TensorExpr<N> >::assign_

  ( const True&  is1d,
    const U&     rhs,
    const Op&    op ) const

{
  T* JEM_RESTRICT  p = data_;
  const idx_t      n = ubound_[0];


  if ( stride_[0] == 1_idx )
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      p[i] = op ( p[i], rhs );
    }
  }
  else
  {
    const idx_t  st = stride_[0];

    for ( idx_t i = 0; i < n; i++ )
    {
      p[i * st] = op ( p[i * st], rhs );
    }
  }
}


//-----------------------------------------------------------------------
//   assign_ (one-dimensional, tensor expression)
//-----------------------------------------------------------------------


template <class T, int N>
  template <class U, class E, class Op>

  void Array<T,0,TensorExpr<N> >::assign_

  ( const True&          is1d,
    const Array<U,0,E>&  rhs,
    const Op&            op ) const

{
  typedef typename tensor::
    AxisTraits< Array<U,0,E> >::AxisType  AxisType;

  T* JEM_RESTRICT  p  = data_;
  const idx_t      n  = ubound_[0];
  idx_t*           ix = index_ [0];
  AxisType         ax = getAxis ( rhs, ix, 0 );


  JEM_CHECK_UBOUND ( ax, ix, n );

  *ix = 0_idx;

  ax.reset ();

  if      ( stride_[0] == 1_idx && ax.isContiguous() )
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      p[i] = op ( p[i], ax.getFast( i ) );
    }
  }
  else if ( n > 0_idx )
  {
    const idx_t  st = stride_[0];

    p[0_idx] = op ( p [0_idx],
                            ax[0_idx] );

    for ( idx_t i = 1_idx; i < n; i++ )
    {
      ax.advance ( ix, i );

      p[i * st] = op ( p[i * st], ax[i] );
    }
  }
}


//-----------------------------------------------------------------------
//   assign_ (multi-dimensional, skalar)
//-----------------------------------------------------------------------


template <class T, int N>
  template <class U, class Op>

  void Array<T,0,TensorExpr<N> >::assign_

  ( const False&  is1d,
    const U&      rhs,
    const Op&     op ) const

{
  JEM_CHECK_UNIQUE ( index_ );

  T* JEM_RESTRICT  p;

  const idx_t  sz = array::size ( ubound_ );

  if ( array::isContiguous( ubound_, stride_ ) )
  {
    p = data_;

    for ( idx_t i = 0; i < sz; i++ )
    {
      p[i] = op ( p[i], rhs );
    }
  }
  else if ( sz > 0_idx )
  {
    const idx_t     st = stride_[0];
    const idx_t     m  = ubound_[0];
    const idx_t     n  = ubound_[1];

    Tuple<idx_t,N>  ix;

    ix = 0_idx;

    do
    {
      p = data_ + array::ioffset ( ix, stride_ );

      for ( idx_t j = 0; j < n; j++ )
      {
        for ( idx_t i = 0; i < m; i++ )
        {
          p[i * st] = op ( p[i * st], rhs );
        }
      }

      ix[1] = n;
    }
    while ( tensor::incrIndex( ix, ubound_ ) );
  }
}


//-----------------------------------------------------------------------
//   assign_ (multi-dimensional, tensor expression)
//-----------------------------------------------------------------------


template <class T, int N>
  template <class U, class E, class Op>

  void Array<T,0,TensorExpr<N> >::assign_

  ( const False&         is1d,
    const Array<U,0,E>&  rhs,
    const Op&            op ) const

{
  typedef typename tensor::
    AxisTraits< Array<U,0,E> >::AxisType  AxisType;

  JEM_CHECK_UNIQUE ( index_ );

  if ( ! array::size( ubound_ ) )
  {
    return;
  }

  T* JEM_RESTRICT  p;

  const idx_t  m  = ubound_[0];
  const idx_t  n  = ubound_[1];

  idx_t*       ix = index_ [0];
  idx_t*       jx = index_ [1];

  AxisType     ax = getAxis ( rhs, ix, jx );


  JEM_CHECK_UBOUND ( ax, ix, m );
  JEM_CHECK_UBOUND ( ax, jx, n );

  for ( int i = 2; i < N; i++ )
  {
    JEM_CHECK_UBOUND ( ax, index_[i], ubound_[i] );
  }

  tensor::zeroIndex ( index_ );

  do
  {
    p = this->getPointer ();

    ax.reset ();

    if ( stride_[0] == 1_idx && ax.isContiguous() )
    {
      for ( idx_t j = 0; j < n; j++ )
      {
        for ( idx_t i = 0; i < m; i++ )
        {
          p[i] = op ( p[i], ax.getFast( i ) );
        }

        p += stride_[1];

        ax.advance ( ix, jx, j + 1 );
      }
    }
    else
    {
      const idx_t  st = stride_[0];

      for ( idx_t j = 0; j < n; j++ )
      {
        p[0] = op ( p[0], ax[0] );

        for ( idx_t i = 1; i < m; i++ )
        {
          ax.advance ( ix, i );

          p[i * st] = op ( p[i * st], ax[i] );
        }

        p += stride_[1];

        ax.advance ( ix, jx, j + 1_idx );
      }
    }
  }
  while ( tensor::incrIndex( index_, ubound_ ) );
}


//-----------------------------------------------------------------------
//   assign_ (multi-dimensional, reduce expression)
//-----------------------------------------------------------------------


template <class T, int N>
  template <class U, class A, class Op1, class Op2>

  void Array<T,0,TensorExpr<N> >::assign_

  ( const False&         is1d,
    const Array<U,0,
    tensor::ReduceExpr
    <1,Op1,A> >&         rhs,
    const Op2&           op ) const

{
  using tensor::reduce;

  typedef typename tensor::AxisTraits<A>::AxisType  AxisType;

  JEM_CHECK_UNIQUE ( index_ );

  if ( ! array::size( ubound_ ) )
  {
    return;
  }

  T* JEM_RESTRICT  p;

  const idx_t  st = stride_[0];

  idx_t*       ix = index_[0];
  idx_t*       jx = index_[1];
  idx_t*       rx = rhs.index[0];

  AxisType     ax = getAxis ( rhs.arg, rx, ix );

  const idx_t  m  = ubound_[0];
  const idx_t  n  = ubound_[1];
  const idx_t  r  = ax.getUbound ( rx );


  if ( r == 0_idx )
  {
    assign_ ( False(), U(), op );
    return;
  }

#ifndef NDEBUG

  JEM_CHECK_UBOUND ( ax, rx, r );

  for ( int i = 0; i < N; i++ )
  {
    JEM_CHECK_UBOUND ( ax, index_[i], ubound_[i] );

    if ( rx == index_[i] )
    {
      tensor::usedIndexError ();
    }
  }

#endif

  tensor::zeroIndex ( index_ );

  do
  {
    p = this->getPointer ();

    for ( idx_t j = 0; j < n; j++ )
    {
      *ix = 0_idx;
      *jx = j;
      *rx = 0_idx;

      ax.reset ();

      p[0] = op ( p[0], reduce( ax, rx, r, rhs.op ) );

      for ( idx_t i = 1; i < m; i++ )
      {
        ax.advance ( rx, ix, i );

        p[i * st] = op ( p[i * st], reduce( ax, rx, r, rhs.op ) );
      }

      p += stride_[1];
    }
  }
  while ( tensor::incrIndex( index_, ubound_ ) );
}


//-----------------------------------------------------------------------
//   initIndex_
//-----------------------------------------------------------------------


template <class T, int N>
  template <class I>

  inline void Array<T,0,TensorExpr<N> >::initIndex_

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

  inline void Array<T,0,TensorExpr<N> >::initIndex_

  ( const TensorIndex&  i,
    int                 idim,
    idx_t               size,
    idx_t               stride )

{
  index_[idim] = i.addr ();

  if ( i.isSimple() )
  {
    ubound_[idim] = size;
    stride_[idim] = stride;
  }
  else
  {
    JEM_ASSERT2 ( i.first() >= 0_idx  &&
                  i.first() <= i.last( size ) &&
                  i.stride() > 0_idx,
                  "invalid tensor index" );

    const idx_t  st = i.stride ();

    data_        +=  stride * i.first();
    ubound_[idim] = (i.last( size ) - i.first() + st -
                     1_idx) / st;
    stride_[idim] =  stride * st;
  }
}


//=======================================================================
//   class Array<T,0,tensor::ReduceExpr<N,Op,A> >
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


template <class T, int N, class Op, class A>

  inline Array<T,0,tensor::ReduceExpr<N,Op,A> >::Array

  ( const ArgType&  t,
    TensorIndex&    i ) :

    op    (),
    arg   ( t ),
    index ( i.addr() )

{
  static_assert ( N == 1, "invalid rank" );
}


template <class T, int N, class Op, class A>

  inline Array<T,0,tensor::ReduceExpr<N,Op,A> >::Array

  ( const ArgType&  t,
    TensorIndex&    i,
    TensorIndex&    j ) :

    op    (),
    arg   ( t ),
    index ( i.addr(), j.addr() )

{
  static_assert ( N == 2, "invalid rank" );
}


template <class T, int N, class Op, class A>

  inline Array<T,0,tensor::ReduceExpr<N,Op,A> >::Array

  ( const ArgType&  t,
    TensorIndex&    i,
    TensorIndex&    j,
    TensorIndex&    k ) :

    op    (),
    arg   ( t ),
    index ( i.addr(), j.addr(), k.addr() )

{
  static_assert ( N == 3, "invalid rank" );
}


template <class T, int N, class Op, class A>

  inline Array<T,0,tensor::ReduceExpr<N,Op,A> >::Array

  ( const ArgType&  t,
    TensorIndex&    i,
    TensorIndex&    j,
    TensorIndex&    k,
    TensorIndex&    l ) :

    op    (),
    arg   ( t ),
    index ( i.addr(), j.addr(), k.addr(), l.addr() )

{
  static_assert ( N == 4, "invalid rank" );
}


template <class T, int N, class Op, class A>

  inline Array<T,0,tensor::ReduceExpr<N,Op,A> >::Array

  ( const ArgType&  t,
    TensorIndex&    i,
    TensorIndex&    j,
    TensorIndex&    k,
    TensorIndex&    l,
    TensorIndex&    p ) :

    op    (),
    arg   ( t ),
    index ( i.addr(), j.addr(), k.addr(), l.addr(), p.addr() )

{
  static_assert ( N == 5, "invalid rank" );
}


template <class T, int N, class Op, class A>

  inline Array<T,0,tensor::ReduceExpr<N,Op,A> >::Array

  ( const ArgType&  t,
    TensorIndex&    i,
    TensorIndex&    j,
    TensorIndex&    k,
    TensorIndex&    l,
    TensorIndex&    p,
    TensorIndex&    q ) :

    op    (),
    arg   ( t ),
    index ( i.addr(), j.addr(), k.addr(),
            l.addr(), p.addr(), q.addr() )

{
  static_assert ( N == 6, "invalid rank" );
}


template <class T, int N, class Op, class A>

  inline Array<T,0,tensor::ReduceExpr<N,Op,A> >::Array

  ( const Array&  rhs ) :

    op     ( rhs.op    ),
    arg    ( rhs.arg   ),
    index  ( rhs.index )

{}


//=======================================================================
//   reduction functions
//=======================================================================

//-----------------------------------------------------------------------
//   dot (single tensor argument)
//-----------------------------------------------------------------------


template <class T, class E>

  inline typename tensor::DotTraits<T,T,1,E,E>::ReturnType

                        dot

  ( const Array<T,0,E>&   t,
    TensorIndex&          i )

{
  return dot ( t, t, i );
}


template <class T, class E>

  inline typename tensor::DotTraits<T,T,2,E,E>::ReturnType

                        dot

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j )

{
  return dot ( t, t, i, j );
}


template <class T, class E>

  inline typename tensor::DotTraits<T,T,3,E,E>::ReturnType

                        dot

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k )

{
  return dot ( t, t, i, j, k );
}


template <class T, class E>

  inline typename tensor::DotTraits<T,T,4,E,E>::ReturnType

                        dot

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k,
    TensorIndex&          l )

{
  return dot ( t, t, i, j, k, l );
}


template <class T, class E>

  inline typename tensor::DotTraits<T,T,5,E,E>::ReturnType

                        dot

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k,
    TensorIndex&          l,
    TensorIndex&          p )

{
  return dot ( t, t, i, j, k, l, p );
}


template <class T, class E>

  inline typename tensor::DotTraits<T,T,6,E,E>::ReturnType

                        dot

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k,
    TensorIndex&          l,
    TensorIndex&          p,
    TensorIndex&          q )

{
  return dot ( t, t, i, j, k, l, p, q );
}


//-----------------------------------------------------------------------
//   dot (two tensor arguments)
//-----------------------------------------------------------------------


template <class T, class U, class E1, class E2>

  inline typename tensor::DotTraits<T,U,1,E1,E2>::ReturnType

                          dot

  ( const Array<T,0,E1>&    t,
    const Array<U,0,E2>&    u,
    TensorIndex&            i )

{
  typedef tensor::DotTraits<T,U,1,E1,E2>  Traits;

  return typename Traits::ReturnType

    ( typename Traits::ArrayType( t, u ), i );
}


template <class T, class U, class E1, class E2>

  inline typename tensor::DotTraits<T,U,2,E1,E2>::ReturnType

                          dot

  ( const Array<T,0,E1>&    t,
    const Array<U,0,E2>&    u,
    TensorIndex&            i,
    TensorIndex&            j )

{
  typedef tensor::DotTraits<T,U,2,E1,E2>  Traits;

  return typename Traits::ReturnType

    ( typename Traits::ArrayType( t, u ), i, j );
}


template <class T, class U, class E1, class E2>

  inline typename tensor::DotTraits<T,U,3,E1,E2>::ReturnType

                          dot

  ( const Array<T,0,E1>&    t,
    const Array<U,0,E2>&    u,
    TensorIndex&            i,
    TensorIndex&            j,
    TensorIndex&            k )

{
  typedef tensor::DotTraits<T,U,3,E1,E2>  Traits;

  return typename Traits::ReturnType

    ( typename Traits::ArrayType( t, u ), i, j, k );
}


template <class T, class U, class E1, class E2>

  inline typename tensor::DotTraits<T,U,4,E1,E2>::ReturnType

                          dot

  ( const Array<T,0,E1>&    t,
    const Array<U,0,E2>&    u,
    TensorIndex&            i,
    TensorIndex&            j,
    TensorIndex&            k,
    TensorIndex&            l )

{
  typedef tensor::DotTraits<T,U,4,E1,E2>  Traits;

  return typename Traits::ReturnType

    ( typename Traits::ArrayType( t, u ), i, j, k, l );
}


template <class T, class U, class E1, class E2>

  inline typename tensor::DotTraits<T,U,5,E1,E2>::ReturnType

                          dot

  ( const Array<T,0,E1>&    t,
    const Array<U,0,E2>&    u,
    TensorIndex&            i,
    TensorIndex&            j,
    TensorIndex&            k,
    TensorIndex&            l,
    TensorIndex&            p )

{
  typedef tensor::DotTraits<T,U,5,E1,E2>  Traits;

  return typename Traits::ReturnType

    ( typename Traits::ArrayType( t, u ), i, j, k, l, p );
}


template <class T, class U, class E1, class E2>

  inline typename tensor::DotTraits<T,U,6,E1,E2>::ReturnType

                          dot

  ( const Array<T,0,E1>&    t,
    const Array<U,0,E2>&    u,
    TensorIndex&            i,
    TensorIndex&            j,
    TensorIndex&            k,
    TensorIndex&            l,
    TensorIndex&            p,
    TensorIndex&            q )

{
  typedef tensor::DotTraits<T,U,6,E1,E2>  Traits;

  return typename Traits::ReturnType

    ( typename Traits::ArrayType( t, u ), i, j, k, l, p, q );
}


//-----------------------------------------------------------------------
//   max
//-----------------------------------------------------------------------


template <class T, class E>

  inline typename tensor::ReduceTraits<1,Maximum<T>,E>::ReturnType

                        max

  ( const Array<T,0,E>&   t,
    TensorIndex&          i )

{
  return typename tensor::ReduceTraits<1,Maximum<T>,E>::ReturnType

    ( t, i );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<2,Maximum<T>,E>::ReturnType

                        max

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j )

{
  return typename tensor::ReduceTraits<2,Maximum<T>,E>::ReturnType

    ( t, i, j );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<3,Maximum<T>,E>::ReturnType

                        max

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k )

{
  return typename tensor::ReduceTraits<3,Maximum<T>,E>::ReturnType

    ( t, i, j, k );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<4,Maximum<T>,E>::ReturnType

                        max

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k,
    TensorIndex&          l )

{
  return typename tensor::ReduceTraits<4,Maximum<T>,E>::ReturnType

    ( t, i, j, k, l );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<5,Maximum<T>,E>::ReturnType

                        max

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k,
    TensorIndex&          l,
    TensorIndex&          p )

{
  return typename tensor::ReduceTraits<5,Maximum<T>,E>::ReturnType

    ( t, i, j, k, l, p );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<6,Maximum<T>,E>::ReturnType

                        max

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k,
    TensorIndex&          l,
    TensorIndex&          p,
    TensorIndex&          q )

{
  return typename tensor::ReduceTraits<6,Maximum<T>,E>::ReturnType

    ( t, i, j, k, l, p, q );
}


//-----------------------------------------------------------------------
//   min
//-----------------------------------------------------------------------


template <class T, class E>

  inline typename tensor::ReduceTraits<1,Minimum<T>,E>::ReturnType

                        min

  ( const Array<T,0,E>&   t,
    TensorIndex&          i )

{
  return typename tensor::ReduceTraits<1,Minimum<T>,E>::ReturnType

    ( t, i );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<2,Minimum<T>,E>::ReturnType

                        min

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j )

{
  return typename tensor::ReduceTraits<2,Minimum<T>,E>::ReturnType

    ( t, i, j );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<3,Minimum<T>,E>::ReturnType

                        min

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k )

{
  return typename tensor::ReduceTraits<3,Minimum<T>,E>::ReturnType

    ( t, i, j, k );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<4,Minimum<T>,E>::ReturnType

                        min

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k,
    TensorIndex&          l )

{
  return typename tensor::ReduceTraits<4,Minimum<T>,E>::ReturnType

    ( t, i, j, k, l );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<5,Minimum<T>,E>::ReturnType

                        min

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k,
    TensorIndex&          l,
    TensorIndex&          p )

{
  return typename tensor::ReduceTraits<5,Minimum<T>,E>::ReturnType

    ( t, i, j, k, l, p );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<6,Minimum<T>,E>::ReturnType

                        min

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k,
    TensorIndex&          l,
    TensorIndex&          p,
    TensorIndex&          q )

{
  return typename tensor::ReduceTraits<6,Minimum<T>,E>::ReturnType

    ( t, i, j, k, l, p, q );
}


//-----------------------------------------------------------------------
//   sum
//-----------------------------------------------------------------------


template <class T, class E>

  inline typename tensor::ReduceTraits<1,Plus<T>,E>::ReturnType

                        sum

  ( const Array<T,0,E>&   t,
    TensorIndex&          i )

{
  return typename tensor::ReduceTraits<1,Plus<T>,E>::ReturnType

    ( t, i );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<2,Plus<T>,E>::ReturnType

                        sum

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j )

{
  return typename tensor::ReduceTraits<2,Plus<T>,E>::ReturnType

    ( t, i, j );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<3,Plus<T>,E>::ReturnType

                        sum

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k )

{
  return typename tensor::ReduceTraits<3,Plus<T>,E>::ReturnType

    ( t, i, j, k );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<4,Plus<T>,E>::ReturnType

                        sum

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k,
    TensorIndex&          l )

{
  return typename tensor::ReduceTraits<4,Plus<T>,E>::ReturnType

    ( t, i, j, k, l );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<5,Plus<T>,E>::ReturnType

                        sum

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k,
    TensorIndex&          l,
    TensorIndex&          p )

{
  return typename tensor::ReduceTraits<5,Plus<T>,E>::ReturnType

    ( t, i, j, k, l, p );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<6,Plus<T>,E>::ReturnType

                        sum

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k,
    TensorIndex&          l,
    TensorIndex&          p,
    TensorIndex&          q )

{
  return typename tensor::ReduceTraits<6,Plus<T>,E>::ReturnType

    ( t, i, j, k, l, p, q );
}


//-----------------------------------------------------------------------
//   product
//-----------------------------------------------------------------------


template <class T, class E>

  inline typename tensor::ReduceTraits<1,Multiplies<T>,E>::ReturnType

                        product

  ( const Array<T,0,E>&   t,
    TensorIndex&          i )

{
  return typename tensor::ReduceTraits<1,Multiplies<T>,E>::ReturnType

    ( t, i );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<2,Multiplies<T>,E>::ReturnType

                        product

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j )

{
  return typename tensor::ReduceTraits<2,Multiplies<T>,E>::ReturnType

    ( t, i, j );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<3,Multiplies<T>,E>::ReturnType

                        product

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k )

{
  return typename tensor::ReduceTraits<3,Multiplies<T>,E>::ReturnType

    ( t, i, j, k );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<4,Multiplies<T>,E>::ReturnType

                        product

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k,
    TensorIndex&          l )

{
  return typename tensor::ReduceTraits<4,Multiplies<T>,E>::ReturnType

    ( t, i, j, k, l );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<5,Multiplies<T>,E>::ReturnType

                        product

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k,
    TensorIndex&          l,
    TensorIndex&          p )

{
  return typename tensor::ReduceTraits<5,Multiplies<T>,E>::ReturnType

    ( t, i, j, k, l, p );
}


template <class T, class E>

  inline typename tensor::ReduceTraits<6,Multiplies<T>,E>::ReturnType

                        product

  ( const Array<T,0,E>&   t,
    TensorIndex&          i,
    TensorIndex&          j,
    TensorIndex&          k,
    TensorIndex&          l,
    TensorIndex&          p,
    TensorIndex&          q )

{
  return typename tensor::ReduceTraits<6,Multiplies<T>,E>::ReturnType

    ( t, i, j, k, l, p, q );
}


//-----------------------------------------------------------------------
//   trace
//-----------------------------------------------------------------------


template <class T, int N, class E>

  T                     trace

  ( const Array<T,N,E>&   e )

{
  const idx_t  n = min ( e.shape() );

  if ( n == 0_idx )
  {
    return T ();
  }
  else
  {
    Tuple<idx_t,N>  i;

    i = 0_idx;

    T  t = e[i];

    i[0] = 1_idx;

    while ( i[0] < n )
    {
      for ( int j = 1; j < N; j++ )
      {
        i[j]++;
      }

      t += e[i];

      i[0]++;
    }

    return t;
  }
}


template <class T, int N, class E>

  T                     trace

  ( const Array<T,N>&     e )

{
  const T* JEM_RESTRICT  p = e.addr ();

  const idx_t  n = min ( e.shape() );

  if ( n == 0_idx )
  {
    return T ();
  }
  else
  {
    T      t  = p[0];
    idx_t  st = 0_idx;

    for ( int i = 0; i < N; i++ )
    {
      st += e.stride ( i );
    }

    for ( idx_t i = 1_idx; i < n; i++ )
    {
      t += p[i * st];
    }

    return t;
  }
}


JEM_END_PACKAGE_BASE

#undef JEM_CHECK_UBOUND
#undef JEM_CHECK_UNIQUE

#endif
