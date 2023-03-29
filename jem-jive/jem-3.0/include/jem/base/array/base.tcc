
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

#ifndef JEM_BASE_ARRAY_BASE_TCC
#define JEM_BASE_ARRAY_BASE_TCC


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


inline Tuple<idx_t,1>     shape

  ( idx_t                   m )

{
  return Tuple<idx_t,1> ( m );
}


inline Tuple<idx_t,2>     shape

  ( idx_t                   m,
    idx_t                   n )

{
  return Tuple<idx_t,2> ( m, n );
}


inline Tuple<idx_t,3>     shape

  ( idx_t                   m,
    idx_t                   n,
    idx_t                   p )

{
  return Tuple<idx_t,3> ( m, n, p );
}


inline Tuple<idx_t,4>     shape

  ( idx_t                   m,
    idx_t                   n,
    idx_t                   p,
    idx_t                   q )
{
  return Tuple<idx_t,4> ( m, n, p, q );
}


inline Tuple<idx_t,5>     shape

  ( idx_t                   m,
    idx_t                   n,
    idx_t                   p,
    idx_t                   q,
    idx_t                   r )

{
  return Tuple<idx_t,5> ( m, n, p, q, r );
}


inline Tuple<idx_t,6>     shape

  ( idx_t                   m,
    idx_t                   n,
    idx_t                   p,
    idx_t                   q,
    idx_t                   r,
    idx_t                   s )

{
  return Tuple<idx_t,6> ( m, n, p, q, r, s );
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T, int N>

  inline void             swap

  ( Array<T,N>&             lhs,
    Array<T,N>&             rhs ) noexcept

{
  lhs.swap ( rhs );
}


JEM_BEGIN_NAMESPACE( array )


//=======================================================================
//   private utility functions
//=======================================================================

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t                     size

  ( const idx_t*            shape,
    int                     rank );


template <int N>

  inline idx_t            size

  ( const Tuple<idx_t,N>&   shape )

{
  return size ( shape.addr(), N );
}


inline idx_t              size

  ( const Tuple<idx_t,1>&   shape )

{
  return shape[0];
}


inline idx_t              size

  ( const Tuple<idx_t,2>   shape )

{
  return (shape[0] * shape[1]);
}


inline idx_t              size

  ( const Tuple<idx_t,3>&   shape )

{
  return (shape[0] * shape[1] * shape[2]);
}


//-----------------------------------------------------------------------
//   ioffset
//-----------------------------------------------------------------------


template <class I, int N>

  inline idx_t            ioffset

  ( const Tuple<I,N>&       index,
    const Tuple<idx_t,N>&   stride )

{
  idx_t  k = index[0] * stride[0];

  for ( int i = 1; i < N; i++ )
  {
    k += index[i] * stride[i];
  }

  return k;
}


template <class I>

  inline idx_t            ioffset

  ( const Tuple<I,1>&       index,
    const Tuple<idx_t,1>&   stride )

{
  return (index[0] * stride[0]);
}


template <class I>

  inline idx_t            ioffset

  ( const Tuple<I,2>&       index,
    const Tuple<idx_t,2>&   stride )

{
  return (index[0] * stride[0] +
          index[1] * stride[1]);
}


template <class I>

  inline idx_t            ioffset

  ( const Tuple<I,3>&       index,
    const Tuple<idx_t,3>&   stride )

{
  return (index[0] * stride[0] +
          index[1] * stride[1] +
          index[2] * stride[2]);
}


//-----------------------------------------------------------------------
//   incrIndex
//-----------------------------------------------------------------------


bool                      incrIndex

  ( idx_t*                  index,
    const idx_t*            shape,
    int                     rank );


template <int N>

  inline bool             incrIndex

  ( Tuple<idx_t,N>&         index,
    const Tuple<idx_t,N>&   shape )

{
  static_assert ( N > 2, "incorrect number of dimensions" );

  index[0] = 0;

  if ( (++index[1]) >= shape[1] )
  {
    index[1] = 0;

    if ( (++index[2]) >= shape[2] )
    {
      return incrIndex ( index.addr() + 2,
                         shape.addr() + 2, N - 2 );
    }
  }

  return true;
}


inline bool               incrIndex

  ( Tuple<idx_t,1>&         index,
    const Tuple<idx_t,1>&   shape )

{
  return false;
}


inline bool               incrIndex

  ( Tuple<idx_t,2>&         index,
    const Tuple<idx_t,2>&   shape )

{
  index[0] = 0;
  index[1]++;

  return (index[1] < shape[1]);
}


inline bool               incrIndex

  ( Tuple<idx_t,3>&         index,
    const Tuple<idx_t,3>&   shape )

{
  index[0] = 0;

  if ( (++index[1]) >= shape[1] )
  {
    index[1] = 0;
    index[2]++;

    return (index[2] < shape[2]);
  }

  return true;
}


inline bool               incrIndex

  ( Tuple<idx_t,4>&         index,
    const Tuple<idx_t,4>&   shape )

{
  index[0] = 0;

  if ( (++index[1]) >= shape[1] )
  {
    index[1] = 0;

    if ( (++index[2]) >= shape[2] )
    {
      index[2] = 0;
      index[3]++;

      return (index[3] < shape[3]);
    }
  }

  return true;
}


//-----------------------------------------------------------------------
//   incrIndex2
//-----------------------------------------------------------------------


template <int N>

  inline bool             incrIndex2

  ( Tuple<idx_t,N>&         index,
    const Tuple<idx_t,N>&   shape )

{
  static_assert ( N > 2, "incorrect number of dimensions" );

  index[0] = 0;
  index[1] = 0;

  if ( (++index[2]) >= shape[2] )
  {
    return incrIndex ( index.addr() + 2,
                       shape.addr() + 2, N - 2 );
  }

  return true;
}


inline bool               incrIndex2

  ( Tuple<idx_t,2>&         index,
    const Tuple<idx_t,2>&   shape )

{
  return false;
}


inline bool               incrIndex2

  ( Tuple<idx_t,3>&         index,
    const Tuple<idx_t,3>&   shape )

{
  index[0] = 0;
  index[1] = 0;
  index[2]++;

  return (index[2] < shape[2]);
}


inline bool               incrIndex2

  ( Tuple<idx_t,4>&         index,
    const Tuple<idx_t,4>&   shape )

{
  index[0] = 0;
  index[1] = 0;

  if ( (++index[2]) >= shape[2] )
  {
    index[2] = 0;
    index[3]++;

    return (index[3] < shape[3]);
  }

  return true;
}


//-----------------------------------------------------------------------
//   isContiguous
//-----------------------------------------------------------------------


bool                      isContiguous

  ( const idx_t*            shape,
    const idx_t*            stride,
    int                     rank );


template <int N>

  inline bool             isContiguous

  ( const Tuple<idx_t,N>&   shape,
    const Tuple<idx_t,N>&   stride )

{
  return isContiguous ( shape.addr(), stride.addr(), N );
}


inline bool               isContiguous

  ( const Tuple<idx_t,1>&,
    const Tuple<idx_t,1>&   stride )

{
  return (stride[0] == 1_idx);
}


inline bool               isContiguous

  ( const Tuple<idx_t,2>&   shape,
    const Tuple<idx_t,2>&   stride )

{
  return ( ((stride[0] - 1_idx) |
            (stride[1] - shape[0])) == 0_idx );
}


inline bool               isContiguous

  ( const Tuple<idx_t,3>&   shape,
    const Tuple<idx_t,3>&   stride )

{
  return ( ((stride[0] - 1_idx) |
            (stride[1] - shape[0])     |
            (stride[2] - shape[1] * stride[1])) == 0_idx );
}


inline bool               isContiguous

  ( const Tuple<idx_t,4>&   shape,
    const Tuple<idx_t,4>&   stride )

{
  return ( ((stride[0] - 1_idx)         |
            (stride[1] - shape[0])             |
            (stride[2] - shape[1] * stride[1]) |
            (stride[3] - shape[2] * stride[2])) == 0_idx );
}


//-----------------------------------------------------------------------
//   checkShape
//-----------------------------------------------------------------------


bool                      checkShape

  ( const idx_t*            shape,
    int                     rank );


template <int N>

  inline bool             checkShape

  ( const Tuple<idx_t,N>&   shape )

{
  return checkShape ( shape.addr(), N );
}


inline bool               checkShape

  ( const Tuple<idx_t,1>&   shape )

{
  return (shape[0] >= 0_idx);
}


inline bool               checkShape

  ( const Tuple<idx_t,2>&   shape )

{
  return ( ((int) (shape[0] < 0_idx) |
            (int) (shape[1] < 0_idx)) == 0 );
}


inline bool               checkShape

  ( const Tuple<idx_t,3>&   shape )

{
  return ( ((int) (shape[0] < 0_idx) |
            (int) (shape[1] < 0_idx) |
            (int) (shape[2] < 0_idx)) == 0 );
}


inline bool               checkShape

  ( const Tuple<idx_t,4>&   shape )

{
  return ( ((int) (shape[0] < 0_idx) |
            (int) (shape[1] < 0_idx) |
            (int) (shape[2] < 0_idx) |
            (int) (shape[3] < 0_idx)) == 0 );
}


//-----------------------------------------------------------------------
//   checkIndex
//-----------------------------------------------------------------------


template <class I, int N>

  inline bool             checkIndex

  ( const Tuple<I,N>&       index,
    const Tuple<idx_t,N>&   shape )

{
  bool  result = true;

  for ( int i = 0; i < N; i++ )
  {
    result = result && checkIndex ( index[i], shape[i] );
  }

  return result;
}


//=======================================================================
//  (computed) assignment functions
//=======================================================================

// Only the most essential assignment functions are defined below. See
// the file assign.h for the other assignment functions.

//-----------------------------------------------------------------------
//   assignArray
//-----------------------------------------------------------------------


template <class T, class U, class Op>

  void                    assignArray

  ( T*                      lhs,
    const U*                rhs,
    idx_t                   len,
    const Op&               op )

{
  for ( idx_t i = 0; i < len; i++ )
  {
    lhs[i] = op ( lhs[i], rhs[i] );
  }
}


template <class T, class U, class Op>

  void                    assignArray

  ( T*                      lhs,
    const idx_t*            lst,
    const U*                rhs,
    const idx_t*            rst,
    const idx_t*            sh,
    idx_t*                  idx,
    int                     rank,
    const Op&               op )

// Note: to limit the size of the compiled code, the array rank is not
// a template parameter.

{
  JEM_ASSERT ( rank > 2 );

  const idx_t  lst0 = lst[0];
  const idx_t  rst0 = rst[0];

  const int    r    = rank - 1;
  const idx_t  m    = sh[0];
  const idx_t  n    = sh[1];

  do
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      for ( idx_t i = 0; i < m; i++ )
      {
        lhs[i * lst0] = op ( lhs[i * lst0], rhs[i * rst0] );
      }

      lhs += lst[1];
      rhs += rst[1];
    }

    for ( int i = 1; i < r; i++ )
    {
      lhs -= (sh[i] * lst[i]);
      rhs -= (sh[i] * rst[i]);

      lhs += lst[i + 1];
      rhs += rst[i + 1];

      idx[i] = 0;
      idx[i + 1]++;

      if ( idx[i + 1] < sh[i + 1] )
      {
        break;
      }
    }
  }
  while ( idx[r] < sh[r] );
}


template <class T, class U, int N, class Op>

  inline void             assignArray

  ( T*                      lhs,
    const Tuple<idx_t,N>&   lst,
    const U*                rhs,
    const Tuple<idx_t,N>&   rst,
    const Tuple<idx_t,N>&   sh,
    const Op&               op )

{
  if ( size( sh ) > 0 )
  {
    Tuple<idx_t,N>  i;

    i = 0;

    assignArray ( lhs, lst.addr(), rhs, rst.addr(),
                  sh.addr(), i.addr(), N, op );
  }
}


template <class T, class U, class Op>

  void                    assignArray

  ( T*                      lhs,
    const Tuple<idx_t,1>&   lst,
    const U*                rhs,
    const Tuple<idx_t,1>&   rst,
    const Tuple<idx_t,1>&   sh,
    const Op&               op )

{
  const idx_t  lst0 = lst[0];
  const idx_t  rst0 = rst[0];
  const idx_t  n    = sh [0];

  for ( idx_t i = 0; i < n; i++ )
  {
    lhs[i * lst0] = op ( lhs[i * lst0], rhs[i * rst0] );
  }
}


template <class T, class U, class Op>

  void                    assignArray

  ( T*                      lhs,
    const Tuple<idx_t,2>&   lst,
    const U*                rhs,
    const Tuple<idx_t,2>&   rst,
    const Tuple<idx_t,2>&   sh,
    const Op&               op )

{
  const idx_t  lst0 = lst[0];
  const idx_t  rst0 = rst[0];
  const idx_t  m    = sh [0];
  const idx_t  n    = sh [1];


  for ( idx_t j = 0; j < n; j++ )
  {
    for ( idx_t i = 0; i < m; i++ )
    {
      lhs[i * lst0] = op ( lhs[i * lst0], rhs[i * rst0] );
    }

    lhs += lst[1];
    rhs += rst[1];
  }
}


// Forward declarations; see assign.tcc for the implementation.

template <class T, class U, int N, class E, class Op>

  inline void             assignArray

  ( T*                      lhs,
    const Array<U,N,E>&     rhs,
    idx_t                   len,
    const Op&               op );

template <class T, class U, int N, class E, class Op>

  inline void             assignArray

  ( T*                      lhs,
    const Tuple<idx_t,N>&   lst,
    const Array<U,N,E>&     rhs,
    const Tuple<idx_t,N>&   sh,
    const Op&               op );


//-----------------------------------------------------------------------
//   assignScalar
//-----------------------------------------------------------------------


template <class T, class U, class Op>

  void                    assignScalar

  ( T*                      lhs,
    const U&                rhs,
    idx_t                   len,
    const Op&               op )

{
  for ( idx_t i = 0; i < len; i++ )
  {
    lhs[i] = op ( lhs[i], rhs );
  }
}


template <class T, class U, class Op>

  void                    assignScalar

  ( T*                      lhs,
    const idx_t*            lst,
    const U&                rhs,
    const idx_t*            sh,
    idx_t*                  idx,
    int                     rank,
    const Op&               op )

// Note: to limit the size of the compiled code, the array rank is not
// a template parameter.

{
  JEM_ASSERT ( rank > 2 );

  const idx_t  lst0 = lst[0];

  const int    r    = rank - 1;
  const idx_t  m    = sh[0];
  const idx_t  n    = sh[1];

  do
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      for ( idx_t i = 0; i < m; i++ )
      {
        lhs[i * lst0] = op ( lhs[i * lst0], rhs );
      }

      lhs += lst[1];
    }

    for ( int i = 1; i < r; i++ )
    {
      lhs -= (sh[i] * lst[i]);
      lhs += lst[i + 1];

      idx[i] = 0;
      idx[i + 1]++;

      if ( idx[i + 1] < sh[i + 1] )
      {
        break;
      }
    }
  }
  while ( idx[r] < sh[r] );
}


template <class T, class U, int N, class Op>

  inline void             assignScalar

  ( T*                      lhs,
    const Tuple<idx_t,N>&   lst,
    const U&                rhs,
    const Tuple<idx_t,N>&   sh,
    const Op&               op )

{
  if ( size( sh ) > 0 )
  {
    Tuple<idx_t,N>  i;

    i = 0_idx;

    assignScalar ( lhs, lst.addr(), rhs, sh.addr(), i.addr(), N, op );
  }
}


template <class T, class U, class Op>

  void                    assignScalar

  ( T*                      lhs,
    const Tuple<idx_t,1>&   lst,
    const U&                rhs,
    const Tuple<idx_t,1>&   sh,
    const Op&               op )

{
  const idx_t  lst0 = lst[0];
  const idx_t  n    = sh [0];

  for ( idx_t i = 0; i < n; i++ )
  {
    lhs[i * lst0] = op ( lhs[i * lst0], rhs );
  }
}


template <class T, class U, class Op>

  void                    assignScalar

  ( T*                      lhs,
    const Tuple<idx_t,2>&   lst,
    const U&                rhs,
    const Tuple<idx_t,2>&   sh,
    const Op&               op )

{
  const idx_t  lst0 = lst[0];
  const idx_t  m    = sh [0];
  const idx_t  n    = sh [1];

  for ( idx_t j = 0; j < n; j++ )
  {
    for ( idx_t i = 0; i < m; i++ )
    {
      lhs[i * lst0] = op ( lhs[i * lst0], rhs );
    }

    lhs += lst[1];
  }
}


//=======================================================================
//   print functions
//=======================================================================

//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output, class T>

  void                  print

  ( Output&               out,
    const T*              ptr,
    idx_t*                st,
    idx_t*                sh,
    idx_t*                idx,
    int                   rank )

{
  JEM_ASSERT ( rank > 0 );

  bool  empty;
  bool  done;
  int   k;


  if ( rank > 1 )
  {
    // Swap the first and second dimensions.

    swap ( sh[0], sh[1] );
    swap ( st[0], st[1] );
  }

  done = empty = false;

  for ( k = 0; k < rank; k++ )
  {
    idx[k] = 0;
    empty  = empty || (sh[k] == 0);

    print ( out, '[' );
  }

  if ( empty )
  {
    for ( k = 0; k < rank; k++ )
    {
      print ( out, ']' );
    }

    return;
  }

  print ( out, ' ' );

  do
  {
    print ( out, ptr[0] );

    idx[0]++;
    ptr += st[0];

    if ( idx[0] < sh[0] )
    {
      print ( out, ", " );
    }
    else
    {
      idx[0] = 0;
      ptr   -= (st[0] * sh[0]);

      print ( out, " ]" );

      for ( k = 1; k < rank; k++ )
      {
        idx[k]++;
        ptr += st[k];

        if ( idx[k] < sh[k] )
        {
          break;
        }

        idx[k] = 0;
        ptr   -= (st[k] * sh[k]);

        print ( out, ']' );
      }

      done = (k == rank);

      if ( ! done )
      {
        print ( out, '\n' );

        for ( k = rank - 1; k >= 0; k-- )
        {
          if ( idx[k] == 0 )
          {
            print ( out, '[' );
          }
          else
          {
            print ( out, ' ' );
          }
        }

        print ( out, ' ' );
      }
    }
  }
  while ( ! done );
}


template <class Output, class T, int N>

  void                  print

  ( Output&               out,
    const Array<T,N>&     a )

{
  Tuple<idx_t,N>  sh = a.shape   ();
  Tuple<idx_t,N>  st = a.strides ();

  Tuple<idx_t,N>  i;

  jem::array::print ( out,
                      a.addr(),  st.addr(),
                      sh.addr(), i.addr(), N );
}


template <class Output, class T>

  void                  print

  ( Output&               out,
    const Array<T,1>&     a )

{
  const idx_t  n = a.size ();

  if ( ! n )
  {
    print ( out, "[]" );
  }
  else
  {
    print ( out, "[ ", a[0] );

    for ( idx_t i = 1; i < n; i++ )
    {
      print ( out, ", ", a[i] );
    }

    print ( out, " ]" );
  }
}


JEM_END_NAMESPACE( array )
JEM_END_PACKAGE_BASE

#endif
