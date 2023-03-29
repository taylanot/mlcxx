
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

#ifndef JEM_BASE_ARRAY_ASSIGN_TCC
#define JEM_BASE_ARRAY_ASSIGN_TCC

JEM_BEGIN_PACKAGE_BASE
JEM_BEGIN_NAMESPACE( array )


//-----------------------------------------------------------------------
//   assignFast
//-----------------------------------------------------------------------


template <class T, class U, int N, class E, class Op>

  void                    assignFast

  ( T*                      lhs,
    const Array<U,N,E>&     rhs,
    idx_t                   len,
    const Op&               op )

{
  for ( idx_t i = 0; i < len; i++ )
  {
    lhs[i] = op ( lhs[i], rhs.getFast(i) );
  }
}

// Some special cases for better performance and smaller executables
// (no dependency on array rank).

template <class T, class U, class Op1, class Op2>

  void                    assignFastA

  ( T*                      lhs,
    const U*                rhs,
    idx_t                   len,
    const Op1&              op1,
    const Op2&              op2 )

{
  for ( idx_t i = 0; i < len; i++ )
  {
    lhs[i] = op1 ( lhs[i], op2( rhs[i] ) );
  }
}


template <class T, class U, class V, class Op1, class Op2>

  void                    assignFastAA

  ( T*                      lhs,
    const U*                rh1,
    const V*                rh2,
    idx_t                   len,
    const Op1&              op1,
    const Op2&              op2 )

{
  for ( idx_t i = 0; i < len; i++ )
  {
    lhs[i] = op1 ( lhs[i], op2( rh1[i], rh2[i] ) );
  }
}


template <class T, class U, class V, class Op1, class Op2>

  void                    assignFastAS

  ( T*                      lhs,
    const U*                rh1,
    const V&                rh2,
    idx_t                   len,
    const Op1&              op1,
    const Op2&              op2 )

{
  for ( idx_t i = 0; i < len; i++ )
  {
    lhs[i] = op1 ( lhs[i], op2( rh1[i], rh2 ) );
  }
}


template <class T, class U, class V, class Op1, class Op2>

  void                    assignFastSA

  ( T*                      lhs,
    const U&                rh1,
    const V*                rh2,
    idx_t                   len,
    const Op1&              op1,
    const Op2&              op2 )

{
  for ( idx_t i = 0; i < len; i++ )
  {
    lhs[i] = op1 ( lhs[i], op2( rh1, rh2[i] ) );
  }
}


template <class T, class U, class V, int N, class Op1, class Op2>

  inline void             assignFast

  ( T*                      lhs,
    const Array<U,N,
    UnaryExpr<Op2,
    Array<U,N,Nil> > >&     rhs,
    idx_t                   len,
    const Op1&              op1 )

{
  assignFastA ( lhs, rhs.arg.addr(), len, op1, rhs.op );
}


template <class T, class U,
          class V, class W, int N, class Op1, class Op2>

  inline void             assignFast

  ( T*                      lhs,
    const Array<U,N,
    BinaryExpr<Op2,
    Array<V,N,Nil>,
    Array<W,N,Nil> > >&     rhs,
    idx_t                   len,
    const Op1&              op1 )

{
  assignFastAA ( lhs, rhs.arg1.addr(), rhs.arg2.addr(),
                 len, op1, rhs.op );
}


template <class T, class U,
          class V, class W, int N, class Op1, class Op2>

  inline void             assignFast

  ( T*                      lhs,
    const Array<U,N,
    BinaryExpr<Op2,
    Array<V,N,Nil>,
    ScalarExpr<W> > >&      rhs,
    idx_t                   len,
    const Op1&              op1 )

{
  assignFastAS ( lhs, rhs.arg1.addr(), rhs.arg2,
                 len, op1, rhs.op );
}


template <class T, class U,
          class V, class W, int N, class Op1, class Op2>

  inline void             assignFast

  ( T*                      lhs,
    const Array<U,N,
    BinaryExpr<Op2,
    ScalarExpr<V>,
    Array<W,N,Nil> > >&     rhs,
    idx_t                   len,
    const Op1&              op1 )

{
  assignFastSA ( lhs, rhs.arg1, rhs.arg2.addr(),
                 len, op1, rhs.op );
}


//-----------------------------------------------------------------------
//   assignSlow
//-----------------------------------------------------------------------


template <class T, class U, int N, class E, class Op>

  void                    assignSlow

  ( T*                      lhs,
    const Tuple<idx_t,N>&   lst,
    const Array<U,N,E>&     rhs,
    const Tuple<idx_t,N>&   sh,
    const Op&               op )

{
  if ( size( sh ) > 0 )
  {
    const idx_t     lst0 = lst[0];
    const idx_t     n    = sh [0];

    Tuple<idx_t,N>  i;

    i = 0;

    do
    {
      T*  lhp = lhs + ioffset ( i, lst );

      for ( ; i[0] < n; i[0]++ )
      {
        T&  t = lhp[i[0] * lst0];

        t = op ( t, rhs[i] );
      }
    }
    while ( incrIndex ( i, sh ) );
  }
}


template <class T, class U, class E, class Op>

  void                    assignSlow

   ( T*                     lhs,
    const Tuple<idx_t,1>&   lst,
    const Array<U,1,E>&     rhs,
    const Tuple<idx_t,1>&   sh,
    const Op&               op )

{
  const idx_t  lst0 = lst[0];
  const idx_t  n    = sh [0];

  for ( idx_t i = 0; i < n; i++ )
  {
    lhs[i * lst0] = op ( lhs[i * lst0], rhs[i] );
  }
}

// Some special cases for better performance and smaller executables
// (no dependency on array rank).

template <class T, class U, class Op1, class Op2>

  void                    assignSlowA

  ( T*                      lhs,
    const idx_t*            lst,
    const U*                rhs,
    const idx_t*            rst,
    const idx_t*            sh,
    idx_t*                  idx,
    int                     rank,
    const Op1&              op1,
    const Op2&              op2 )

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
        lhs[i * lst0] = op1 ( lhs[i * lst0], op2( rhs[i * rst0] ) );
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


template <class T, class U, class V, class Op1, class Op2>

  void                    assignSlowAA

  ( T*                      lhs,
    const idx_t*            lst,
    const U*                rh1,
    const idx_t*            st1,
    const V*                rh2,
    const idx_t*            st2,
    const idx_t*            sh,
    idx_t*                  idx,
    int                     rank,
    const Op1&              op1,
    const Op2&              op2 )

{
  JEM_ASSERT ( rank > 2 );

  const idx_t  lst0 = lst[0];
  const idx_t  st10 = st1[0];
  const idx_t  st20 = st2[0];

  const int    r    = rank - 1;
  const idx_t  m    = sh[0];
  const idx_t  n    = sh[1];

  do
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      for ( idx_t i = 0; i < m; i++ )
      {
        lhs[i * lst0] =

          op1 ( lhs[i * lst0], op2( rh1[i * st10], rh2[i * st20] ) );
      }

      lhs += lst[1];
      rh1 += st1[1];
      rh2 += st2[1];
    }

    for ( int i = 1; i < r; i++ )
    {
      lhs -= (sh[i] * lst[i]);
      rh1 -= (sh[i] * st1[i]);
      rh2 -= (sh[i] * st2[i]);

      lhs += lst[i + 1];
      rh1 += st1[i + 1];
      rh2 += st2[i + 1];

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


template <class T, class U, class V, class Op1, class Op2>

  void                    assignSlowAS

  ( T*                      lhs,
    const idx_t*            lst,
    const U*                rh1,
    const idx_t*            st1,
    const V&                rh2,
    const idx_t*            sh,
    idx_t*                  idx,
    int                     rank,
    const Op1&              op1,
    const Op2&              op2 )

{
  JEM_ASSERT ( rank > 2 );

  const idx_t  lst0 = lst[0];
  const idx_t  st10 = st1[0];

  const int    r    = rank - 1;
  const idx_t  m    = sh[0];
  const idx_t  n    = sh[1];

  do
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      for ( idx_t i = 0; i < m; i++ )
      {
        lhs[i * lst0] =

          op1 ( lhs[i * lst0], op2( rh1[i * st10], rh2 ) );
      }

      lhs += lst[1];
      rh1 += st1[1];
    }

    for ( int i = 1; i < r; i++ )
    {
      lhs -= (sh[i] * lst[i]);
      rh1 -= (sh[i] * st1[i]);

      lhs += lst[i + 1];
      rh1 += st1[i + 1];

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


template <class T, class U, class V, class Op1, class Op2>

  void                    assignSlowSA

  ( T*                      lhs,
    const idx_t*            lst,
    const U&                rh1,
    const V*                rh2,
    const idx_t*            st2,
    const idx_t*            sh,
    idx_t*                  idx,
    int                     rank,
    const Op1&              op1,
    const Op2&              op2 )

{
  JEM_ASSERT ( rank > 2 );

  const idx_t  lst0 = lst[0];
  const idx_t  st20 = st2[0];

  const int    r    = rank - 1;
  const idx_t  m    = sh[0];
  const idx_t  n    = sh[1];

  do
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      for ( idx_t i = 0; i < m; i++ )
      {
        lhs[i * lst0] =

          op1 ( lhs[i * lst0], op2( rh1, rh2[i * st20] ) );
      }

      lhs += lst[1];
      rh2 += st2[1];
    }

    for ( int i = 1; i < r; i++ )
    {
      lhs -= (sh[i] * lst[i]);
      rh2 -= (sh[i] * st2[i]);

      lhs += lst[i + 1];
      rh2 += st2[i + 1];

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


template <class T, class U, class V, int N, class Op1, class Op2>

  inline void             assignSlow

  ( T*                      lhs,
    const Tuple<idx_t,N>&   lst,
    const Array<U,N,
    UnaryExpr<Op2,
    Array<V,N,Nil> > >&     rhs,
    const Tuple<idx_t,N>&   sh,
    const Op1&              op1 )

{
  if ( size( sh ) > 0 )
  {
    Tuple<idx_t,N>  i;

    i = 0_idx;

    assignSlowA ( lhs, lst.addr(),
                  rhs.arg.addr(), rhs.arg.strides().addr(),
                  sh.addr(), i.addr(), N, op1, rhs.op );
  }
}


template <class T, class U, class V, class Op1, class Op2>

  void                    assignSlow

  ( T*                      lhs,
    const Tuple<idx_t,1>&   lst,
    const Array<U,1,
    UnaryExpr<Op2,
    Array<V,1,Nil> > >&     rhs,
    const Tuple<idx_t,1>&   sh,
    const Op1&              op1 )

{
  const V*     rhp  = rhs.arg.addr ();

  const idx_t  lst0 = lst[0];
  const idx_t  rst0 = rhs.arg.stride(0);
  const idx_t  n    = sh [0];

  for ( idx_t i = 0; i < n; i++ )
  {
    lhs[i * lst0] = op1 ( lhs[i * lst0], rhs.op( rhp[i * rst0] ) );
  }
}


template <class T, class U, class V, class Op1, class Op2>

  void                    assignSlow

  ( T*                      lhs,
    const Tuple<idx_t,2>&   lst,
    const Array<U,2,
    UnaryExpr<Op2,
    Array<V,2,Nil> > >&     rhs,
    const Tuple<idx_t,2>&   sh,
    const Op1&              op1 )

{
  const V*     rhp  = rhs.arg.addr ();

  const idx_t  lst0 = lst[0];
  const idx_t  rst0 = rhs.arg.stride(0);
  const idx_t   m   = sh [0];
  const idx_t   n   = sh [1];


  for ( idx_t j = 0; j < n; j++ )
  {
    for ( idx_t i = 0; i < m; i++ )
    {
      lhs[i * lst0] = op1 ( lhs[i * lst0], rhs.op( rhp[i * rst0] ) );
    }

    lhs += lst[1];
    rhp += rhs.arg.stride(1);
  }
}



template <class T, class U,
          class V, class W, int N, class Op1, class Op2>

  inline void             assignSlow

  ( T*                      lhs,
    const Tuple<idx_t,N>&   lst,
    const Array<U,N,
    BinaryExpr<Op2,
    Array<V,N,Nil>,
    Array<W,N,Nil> > >&     rhs,
    const Tuple<idx_t,N>&   sh,
    const Op1&              op1 )

{
  if ( size( sh ) > 0 )
  {
    Tuple<idx_t,N>  i;

    i = 0_idx;

    assignSlowAA ( lhs, lst.addr(),
                   rhs.arg1.addr(), rhs.arg1.strides().addr(),
                   rhs.arg2.addr(), rhs.arg2.strides().addr(),
                   sh.addr(), i.addr(), N, op1, rhs.op );
  }
}


template <class T, class U,
          class V, class W, class Op1, class Op2>

  void                    assignSlow

  ( T*                      lhs,
    const Tuple<idx_t,1>&   lst,
    const Array<U,1,
    BinaryExpr<Op2,
    Array<V,1,Nil>,
    Array<W,1,Nil> > >&     rhs,
    const Tuple<idx_t,1>&   sh,
    const Op1&              op1 )

{
  const V*     rh1  = rhs.arg1.addr ();
  const W*     rh2  = rhs.arg2.addr ();

  const idx_t  lst0 = lst[0];
  const idx_t  st10 = rhs.arg1.stride(0);
  const idx_t  st20 = rhs.arg2.stride(0);
  const idx_t  n    = sh [0];

  for ( idx_t i = 0; i < n; i++ )
  {
    lhs[i * lst0] =

      op1 ( lhs[i * lst0], rhs.op( rh1[i * st10], rh2[i * st20] ) );
  }
}


template <class T, class U,
          class V, class W, class Op1, class Op2>

  void                    assignSlow

  ( T*                      lhs,
    const Tuple<idx_t,2>&   lst,
    const Array<U,2,
    BinaryExpr<Op2,
    Array<V,2,Nil>,
    Array<W,2,Nil> > >&     rhs,
    const Tuple<idx_t,2>&   sh,
    const Op1&              op1 )

{
  const U*     rh1  = rhs.arg1.addr ();
  const V*     rh2  = rhs.arg2.addr ();

  const idx_t  lst0 = lst[0];
  const idx_t  st10 = rhs.arg1.stride(0);
  const idx_t  st20 = rhs.arg2.stride(0);
  const idx_t  m    = sh [0];
  const idx_t  n    = sh [1];


  for ( idx_t j = 0; j < n; j++ )
  {
    for ( idx_t i = 0; i < m; i++ )
    {
      lhs[i * lst0] =

        op1 ( lhs[i * lst0], rhs.op( rh1[i * st10], rh2[i * st20] ) );
    }

    lhs += lst[1];
    rh1 += rhs.arg1.stride(1);
    rh2 += rhs.arg2.stride(1);
  }
}


template <class T, class U,
          class V, class W, int N, class Op1, class Op2>

  inline void             assignSlow

  ( T*                      lhs,
    const Tuple<idx_t,N>&   lst,
    const Array<U,N,
    BinaryExpr<Op2,
    Array<V,N,Nil>,
    ScalarExpr<W> > >&     rhs,
    const Tuple<idx_t,N>&  sh,
    const Op1&             op1 )

{
  if ( size( sh ) > 0 )
  {
    Tuple<idx_t,N>  i;

    i = 0_idx;

    assignSlowAS ( lhs, lst.addr(),
                   rhs.arg1.addr(), rhs.arg1.strides().addr(),
                   rhs.arg2, sh.addr(), i.addr(), N, op1, rhs.op );
  }
}


template <class T, class U,
          class V, class W, class Op1, class Op2>

  void                    assignSlow

  ( T*                      lhs,
    const Tuple<idx_t,1>&   lst,
    const Array<U,1,
    BinaryExpr<Op2,
    Array<V,1,Nil>,
    ScalarExpr<W> > >&      rhs,
    const Tuple<idx_t,1>&   sh,
    const Op1&              op1 )

{
  const V*     rh1  = rhs.arg1.addr ();

  const idx_t  lst0 = lst[0];
  const idx_t  st10 = rhs.arg1.stride(0);
  const idx_t   n   = sh [0];

  for ( idx_t i = 0; i < n; i++ )
  {
    lhs[i * lst0] =

      op1 ( lhs[i * lst0], rhs.op( rh1[i * st10], rhs.arg2 ) );
  }
}


template <class T, class U,
          class V, class W, class Op1, class Op2>

  void                    assignSlow

  ( T*                      lhs,
    const Tuple<idx_t,2>&   lst,
    const Array<U,2,
    BinaryExpr<Op2,
    Array<V,2,Nil>,
    ScalarExpr<W> > >&      rhs,
    const Tuple<idx_t,2>&   sh,
    const Op1&              op1 )

{
  const V*     rh1  = rhs.arg1.addr ();

  const idx_t  lst0 = lst[0];
  const idx_t  st10 = rhs.arg1.stride(0);
  const idx_t  m    = sh [0];
  const idx_t  n    = sh [1];


  for ( idx_t j = 0; j < n; j++ )
  {
    for ( idx_t i = 0; i < m; i++ )
    {
      lhs[i * lst0] =

        op1 ( lhs[i * lst0], rhs.op( rh1[i * st10], rhs.arg2 ) );
    }

    lhs += lst[1];
    rh1 += rhs.arg1.stride(1);
  }
}


template <class T, class U,
          class V, class W, int N, class Op1, class Op2>

  inline void             assignSlow

  ( T*                      lhs,
    const Tuple<idx_t,N>&   lst,
    const Array<U,N,
    BinaryExpr<Op2,
    ScalarExpr<V>,
    Array<W,N,Nil> > >&     rhs,
    const Tuple<idx_t,N>&   sh,
    const Op1&              op1 )

{
  if ( size( sh ) > 0 )
  {
    Tuple<idx_t,N>  i;

    i = 0_idx;

    assignSlowSA ( lhs, lst.addr(), rhs.arg1,
                   rhs.arg2.addr(), rhs.arg2.strides().addr(),
                   sh.addr(), i.addr(), N, op1, rhs.op );
  }
}


template <class T, class U,
          class V, class W, class Op1, class Op2>

  void                    assignSlow

  ( T*                      lhs,
    const Tuple<idx_t,1>&   lst,
    const Array<U,1,
    BinaryExpr<Op2,
    ScalarExpr<V>,
    Array<W,1,Nil> > >&     rhs,
    const Tuple<idx_t,1>&   sh,
    const Op1&              op1 )

{
  const W*     rh2  = rhs.arg2.addr ();

  const idx_t  lst0 = lst[0];
  const idx_t  st20 = rhs.arg2.stride(0);
  const idx_t   n   = sh [0];

  for ( idx_t i = 0; i < n; i++ )
  {
    lhs[i * lst0] =

      op1 ( lhs[i * lst0], rhs.op( rhs.arg1, rh2[i * st20] ) );
  }
}


template <class T, class U,
          class V, class W, class Op1, class Op2>

  void                    assignSlow

  ( T*                      lhs,
    const Tuple<idx_t,2>&   lst,
    const Array<U,2,
    BinaryExpr<Op2,
    ScalarExpr<V>,
    Array<W,2,Nil> > >&     rhs,
    const Tuple<idx_t,2>&   sh,
    const Op1&              op1 )

{
  const W*     rh2  = rhs.arg2.addr ();

  const idx_t  lst0 = lst[0];
  const idx_t  st20 = rhs.arg2.stride(0);
  const idx_t  m    = sh [0];
  const idx_t  n    = sh [1];


  for ( idx_t j = 0; j < n; j++ )
  {
    for ( idx_t i = 0; i < m; i++ )
    {
      lhs[i * lst0] =

        op1 ( lhs[i * lst0], rhs.op( rhs.arg1, rh2[i * st20] ) );
    }

    lhs += lst[1];
    rh2 += rhs.arg2.stride(1);
  }
}


//-----------------------------------------------------------------------
//   assign
//-----------------------------------------------------------------------


template <class T, class U, int N, class E1, class E2, class Op>

  inline void             assign

  ( const Array<T,N,E1>&    lhs,
    const Array<U,N,E2>&    rhs,
    const Op&               op )

{
  JEM_ASSERT2 ( equal( lhs.shape(), rhs.shape() ),
                Errors::SHAPE_MISMATCH );

  if ( lhs.isContiguous() && rhs.isContiguous() )
  {
    assignArray ( lhs, rhs, size( lhs.shape() ), op );
  }
  else
  {
    assignArray ( lhs, rhs, lhs.shape(), op );
  }
}


template <class T, class U, int N, class E1, class E2, class Op>

  inline void             assign

  ( const Array<T,N,E1>&    lhs,
    const Array<U,N,E2>&    rhs,
    const Tuple<idx_t,N>&   sh,
    const Op&               op )

{
  assignArray ( lhs, rhs, sh, op );
}


template <class T, class U, int N, class E, class Op>

  inline void             assign

  ( const Array<T,N,E>&     lhs,
    const U&                rhs,
    const Op&               op )

{
  if ( lhs.isContiguous() )
  {
    assignScalar ( lhs, rhs, size( lhs.shape() ), op );
  }
  else
  {
    assignScalar ( lhs, rhs, lhs.shape(), op );
  }
}


template <class T, class U, int N, class E, class Op>

  inline void             assign

  ( const Array<T,N,E>&     lhs,
    const U&                rhs,
    const Tuple<idx_t,N>&   sh,
    const Op&               op )

{
  assignScalar ( lhs, rhs, sh, op );
}


//-----------------------------------------------------------------------
//   assignArray
//-----------------------------------------------------------------------


template <class T, class U, int N, class E, class Op>

  inline void             assignArray

  ( T*                      lhs,
    const Array<U,N,E>&     rhs,
    idx_t                   len,
    const Op&               op )

{
  assignFast ( lhs, rhs, len, op );
}

template <class T, class U, int N, class E, class Op>

  inline void             assignArray

  ( T*                      lhs,
    const Tuple<idx_t,N>&   lst,
    const Array<U,N,E>&     rhs,
    const Tuple<idx_t,N>&   sh,
    const Op&               op )

{
  assignSlow ( lhs, lst, rhs, sh, op );
}


template <class T, class U, int N, class E1, class E2, class Op>

  void                    assignArray

  ( const Array<T,N,E1>&    lhs,
    const Array<U,N,E2>&    rhs,
    idx_t                   len,
    const Op&               op )

{
  for ( idx_t i = 0; i < len; i++ )
  {
    T& t = lhs.getFast(i);

    t = op ( t, rhs.getFast(i) );
  }
}


template <class T, class U, int N, class E1, class E2, class Op>

  void                    assignArray

  ( const Array<T,N,E1>&    lhs,
    const Array<U,N,E2>&    rhs,
    const Tuple<idx_t,N>&   sh,
    const Op&               op )

{
  if ( size( sh ) > 0 )
  {
    const idx_t     n = sh[0];
    Tuple<idx_t,N>  i;

    i = 0;

    do
    {
      for ( ; i[0] < n; i[0]++ )
      {
        T& t = lhs[i];

        t = op ( t, rhs[i] );
      }
    }
    while ( incrIndex( i, sh ) );
  }
}


template <class T, class U, class E1, class E2, class Op>

  void                    assignArray

  ( const Array<T,1,E1>&    lhs,
    const Array<U,1,E2>&    rhs,
    const Tuple<idx_t,1>&   sh,
    const Op&               op )

{
  const idx_t  n = sh[0];

  for ( idx_t i = 0; i < n; i++ )
  {
    T& t = lhs[i];

    t = op ( t, rhs[i] );
  }
}


//-----------------------------------------------------------------------
//   assignScalar
//-----------------------------------------------------------------------


template <class T, class U, int N, class E, class Op>

  void                    assignScalar

  ( const Array<T,N,E>&     lhs,
    const U&                rhs,
    idx_t                   len,
    const Op&               op )

{
  for ( idx_t i = 0; i < len; i++ )
  {
    T& t = lhs.getFast(i);

    t = op ( t, rhs );
  }
}


template <class T, class U, int N, class E, class Op>

  void                    assignScalar

  ( const Array<T,N,E>&     lhs,
    const U&                rhs,
    const Tuple<idx_t,N>&   sh,
    const Op&               op )

{
  if ( size( sh ) > 0 )
  {
    const idx_t     n = sh[0];
    Tuple<idx_t,N>  i;

    i = 0_idx;

    do
    {
      for ( ; i[0] < n; i[0]++ )
      {
        T& t = lhs[i];

        t = op ( t, rhs );
      }
    }
    while ( incrIndex( i, sh ) );
  }
}


template <class T, class U, class E, class Op>

  void                    assignScalar

  ( const Array<T,1,E>&     lhs,
    const U&                rhs,
    const Tuple<idx_t,1>&   sh,
    const Op&               op )

{
  const idx_t  n = sh[0];

  for ( idx_t i = 0; i < n; i++ )
  {
    T& t = lhs[i];

    t = op ( t, rhs );
  }
}


JEM_END_NAMESPACE( array )
JEM_END_PACKAGE_BASE

#endif
