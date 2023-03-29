
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */


#include <jem/base/array/intrinsics.h>
#include <jem/base/IllegalIndexException.h>
#include <jem/io/DataInput.h>
#include <jem/io/DataOutput.h>
#include <jem/util/Properties.h>
#include <jive/geom/Names.h>
#include <jive/geom/BezierExtractor.h>


JIVE_BEGIN_PACKAGE( geom )


//=======================================================================
//   class BezierExtractor
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


BezierExtractor::BezierExtractor ()
{
  shape = 0;
}


BezierExtractor::BezierExtractor ( idx_t m )
{
  JEM_PRECHECK2 ( m >= 0,
                  "invalid Bezier extraction operator size" );

  shape[0] = shape[1] = m;
}


BezierExtractor::BezierExtractor ( idx_t m, idx_t n )
{
  JEM_PRECHECK2 ( m >= 0 && n >= 0,
                  "invalid Bezier extraction operator shape" );

  shape[0] = m;
  shape[1] = n;
}


//-----------------------------------------------------------------------
//   operator =
//-----------------------------------------------------------------------


BezierExtractor& BezierExtractor::operator = ( const Self& rhs )
{
  shape = rhs.shape;

  irows .ref ( rhs.irows );
  jcols .ref ( rhs.jcols );
  values.ref ( rhs.values );

  return *this;
}


//-----------------------------------------------------------------------
//   setIdentity
//-----------------------------------------------------------------------


void BezierExtractor::setIdentity ( idx_t n )
{
  JEM_PRECHECK2 ( n >= 0,
                  "invalid Bezier extraction operator size" );

  shape[0] = shape[1] = n;

  irows .resize ( n );
  jcols .resize ( n );
  values.resize ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    irows [i] = jcols[i] = i;
    values[i] = 1.0;
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void BezierExtractor::configure ( const Properties&  props )
{
  using jem::castTo;

  IntVector  sh;
  IntVector  iv;
  IntVector  jv;
  Vector     vv;
  idx_t      nnz;


  props.get ( sh, PropNames::SHAPE, 1, 1000 );

  if ( sh.size() != 2 )
  {
    props.propertyError (
      PropNames::SHAPE,
      "Bezier extraction operator has an invalid shape; "
      "expected an array of length two"
    );
  }

  props.find ( iv, PropNames::IROWS, 0, sh[0] - 1 );
  props.find ( jv, PropNames::JCOLS, 0, sh[1] - 1 );
  props.find ( vv, PropNames::VALUES );

  nnz = iv.size ();

  if ( jv.size() != nnz )
  {
    props.propertyError (
      PropNames::JCOLS,
      String::format (
        "column index array has wrong length: %d; "
        "expected an array of length %d",
        jv.size (),
        nnz
      )
    );
  }

  if ( vv.size() != nnz )
  {
    props.propertyError (
      PropNames::JCOLS,
      String::format (
        "value array has wrong length: %d; "
        "expected an array of length %d",
        vv.size (),
        nnz
      )
    );
  }

  irows .resize ( nnz );
  jcols .resize ( nnz );
  values.resize ( nnz );

  shape[0] = (idx_t) sh[0];
  shape[1] = (idx_t) sh[1];
  irows    = castTo<idx_t> ( iv );
  jcols    = castTo<idx_t> ( jv );
  values   = vv;
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void BezierExtractor::getConfig ( const Properties& conf ) const
{
  using jem::castTo;

  const idx_t  nnz = nonZeroCount ();

  IntVector    sh ( 2 );
  IntVector    iv ( nnz );
  IntVector    jv ( nnz );

  sh[0] = (int) shape[0];
  sh[1] = (int) shape[1];
  iv    = castTo<int> ( irows );
  jv    = castTo<int> ( jcols );

  conf.set ( PropNames::SHAPE,  sh );
  conf.set ( PropNames::IROWS,  iv );
  conf.set ( PropNames::JCOLS,  jv );
  conf.set ( PropNames::VALUES, values );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   bezierIndexError
//-----------------------------------------------------------------------


#ifndef NDEBUG

static void               bezierIndexError

  ( const String&           where,
    idx_t                   irow,
    idx_t                   jcol )

{
  throw jem::IllegalIndexException (
    JEM_FUNC,
    String::format (
      "invalid Bezier extraction operator; "
      "index pair (%d,%d) out of range",
      irow,
      jcol
    )
  );
}

#endif


//-----------------------------------------------------------------------
//   matmul (matrix * vector)
//-----------------------------------------------------------------------


void                      matmul

  ( const Vector&           lhs,
    const BezierExtractor&  bez,
    const Vector&           rhs )

{
  JEM_PRECHECK2 ( bez.shape[0] == lhs.size() &&
                  bez.shape[1] == rhs.size(),
                  "Array size mismatch" );

  JEM_PRECHECK2 ( bez.irows.size() == bez.jcols.size() &&
                  bez.irows.size() == bez.values.size(),
                  "invalid Bezier extraction operator "
                  "(Array size mismatch)" );

  const idx_t  nnz = bez.nonZeroCount ();

#ifndef NDEBUG
  const idx_t  n   = bez.shape[1];
#endif


  if ( lhs.stride()        +
       rhs.stride()        +
       bez.irows.stride()  +
       bez.jcols.stride()  +
       bez.values.stride() == 5_idx )
  {
    const idx_t*  JEM_RESTRICT  irows  = bez.irows .addr ();
    const idx_t*  JEM_RESTRICT  jcols  = bez.jcols .addr ();
    const double* JEM_RESTRICT  values = bez.values.addr ();
    const double* JEM_RESTRICT  raddr  = rhs       .addr ();
    double*       JEM_RESTRICT  laddr  = lhs       .addr ();

    const idx_t   m = bez.shape[0];

    for ( idx_t i = 0; i < m; i++ )
    {
      laddr[i] = 0.0;
    }

    for ( idx_t i = 0; i < nnz; i++ )
    {
      idx_t  irow = irows[i];
      idx_t  jcol = jcols[i];

#ifndef NDEBUG
      if ( irow < 0 || irow >= m || jcol < 0 || jcol >= n )
      {
        bezierIndexError ( JEM_FUNC, irow, jcol );
      }
#endif

      laddr[irow] += values[i] * raddr[jcol];
    }
  }
  else
  {
#ifndef NDEBUG
    const idx_t   m = bez.shape[0];
#endif

    lhs = 0.0;

    for ( idx_t i = 0; i < nnz; i++ )
    {
      idx_t  irow = bez.irows[i];
      idx_t  jcol = bez.jcols[i];

#ifndef NDEBUG
      if ( irow < 0 || irow >= m || jcol < 0 || jcol >= n )
      {
        bezierIndexError ( JEM_FUNC, irow, jcol );
      }
#endif

      lhs[irow] += bez.values[i] * rhs[jcol];
    }
  }
}


//-----------------------------------------------------------------------
//   matmul (matrix -matrix)
//-----------------------------------------------------------------------


void                      matmul

  ( const Matrix&           lhs,
    const BezierExtractor&  bez,
    const Matrix&           rhs )

{
  JEM_PRECHECK2 ( bez.shape[0] == lhs.size(1) &&
                  bez.shape[1] == rhs.size(1) &&
                  lhs.size (0) == rhs.size(0),
                  "Array shape mismatch" );

  JEM_PRECHECK2 ( bez.irows.size() == bez.jcols.size() &&
                  bez.irows.size() == bez.values.size(),
                  "invalid Bezier extraction operator "
                  "(Array size mismatch)" );

  const double* JEM_RESTRICT  raddr = rhs.addr ();
  double*       JEM_RESTRICT  laddr = lhs.addr ();

  const idx_t   lst1 = lhs.stride (1);
  const idx_t   rst1 = rhs.stride (1);

  const idx_t   nnz  = bez.values.size ();

#ifndef NDEBUG
  const idx_t   m    = bez.shape[0];
  const idx_t   n    = bez.shape[1];
#endif
  const idx_t   p    = rhs.size (0);


  lhs = 0.0;

  if ( lhs.stride(0)       +
       rhs.stride(0)       +
       bez.irows.stride()  +
       bez.jcols.stride()  +
       bez.values.stride() == 5_idx )
  {
    const idx_t*  JEM_RESTRICT  irows  = bez.irows .addr ();
    const idx_t*  JEM_RESTRICT  jcols  = bez.jcols .addr ();
    const double* JEM_RESTRICT  values = bez.values.addr ();

    for ( idx_t i = 0; i < nnz; i++ )
    {
      idx_t   irow  = irows [i];
      idx_t   jcol  = jcols [i];
      double  value = values[i];

#ifndef NDEBUG
      if ( irow < 0 || irow >= m || jcol < 0 || jcol >= n )
      {
        bezierIndexError ( JEM_FUNC, irow, jcol );
      }
#endif

      const double* JEM_RESTRICT  rcol = raddr + jcol * rst1;
      double*       JEM_RESTRICT  lcol = laddr + irow * lst1;

      for ( idx_t j = 0; j < p; j++ )
      {
        lcol[j] += value * rcol[j];
      }
    }
  }
  else
  {
    const idx_t  lst0 = lhs.stride (0);
    const idx_t  rst0 = rhs.stride (0);

    for ( idx_t i = 0; i < nnz; i++ )
    {
      idx_t   irow  = bez.irows [i];
      idx_t   jcol  = bez.jcols [i];
      double  value = bez.values[i];

#ifndef NDEBUG
      if ( irow < 0 || irow >= m || jcol < 0 || jcol >= n )
      {
        bezierIndexError ( JEM_FUNC, irow, jcol );
      }
#endif

      const double* JEM_RESTRICT  rcol = raddr + jcol * rst1;
      double*       JEM_RESTRICT  lcol = laddr + irow * lst1;

      for ( idx_t j = 0; j < p; j++ )
      {
        lcol[j * lst0] += value * rcol[j * rst0];
      }
    }
  }
}


//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


void                        decode

  ( DataInput&                in,
    BezierExtractor&          bez )

{
  for ( int i = 0; i < 2; i++ )
  {
    uidx_t  n;

    decode ( in, n );

    bez.shape[i] = (idx_t) n;
  }

  decode ( in, bez.irows, bez.jcols, bez.values );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


void                        encode

  ( DataOutput&               out,
    const BezierExtractor&    bez )

{
  for ( int i = 0; i < 2; i++ )
  {
    encode ( out, bez.shape[i] );
  }

  encode ( out, bez.irows, bez.jcols, bez.values );
}


JIVE_END_PACKAGE( geom )
