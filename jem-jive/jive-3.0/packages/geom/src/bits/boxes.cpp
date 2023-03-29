
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


#include <cmath>
#include <cfloat>
#include <jem/pragmas.h>
#include <jem/base/assert.h>
#include <jive/geom/boxes.h>


JIVE_BEGIN_PACKAGE( geom )


using jem::IntConstant;


//=======================================================================
//   private data & functions
//=======================================================================

//-----------------------------------------------------------------------
//   constants
//-----------------------------------------------------------------------


static const double  INV_SQRT2_ = 0.7071067811865475;
static const double  INV_SQRT3_ = 0.5773502691896257;


//-----------------------------------------------------------------------
//   getBBox_
//-----------------------------------------------------------------------


template <int R>

  static void             getBBox_

  ( const Vector&           bbox,
    const Matrix&           coords,
    const IntConstant<R>&   rank )

{
  const double* JEM_RESTRICT  cxp = coords.addr ();

  const idx_t  n = coords.size (1);

  double       lb[R];
  double       ub[R];
  double       x;


  for ( int i = 0; i < R; i++ )
  {
    lb[i] = ub[i] = cxp[i];
  }

  cxp += R;

  for ( idx_t j = 1; j < n; j++, cxp += R )
  {
    for ( int i = 0; i < R; i++ )
    {
      x = cxp[i];

      if      ( x < lb[i] )
      {
        lb[i] = x;
      }
      else if ( x > ub[i] )
      {
        ub[i] = x;
      }
    }
  }

  for ( int i = 0; i < R; i++ )
  {
    bbox[i]     = lb[i];
    bbox[i + R] = ub[i];
  }
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   getBoundingBox
//-----------------------------------------------------------------------


void                getBoundingBox

  ( const Vector&     bbox,
    const Matrix&     coords )

{
  JEM_PRECHECK2 ( bbox.size() == 2 * coords.size(0),
                  "bounding box array has wrong size" );

  const idx_t  rank = coords.size (0);


  if ( rank == 0 )
  {
    return;
  }

  if ( coords.size(1) == 0 )
  {
    makeEmptyBox ( bbox );
    return;
  }

  if ( coords.isContiguous() && rank < 4_idx )
  {
    if      ( rank == 1_idx )
    {
      getBBox_ ( bbox, coords, IntConstant<1>() );
    }
    else if ( rank == 2_idx )
    {
      getBBox_ ( bbox, coords, IntConstant<2>() );
    }
    else
    {
      getBBox_ ( bbox, coords, IntConstant<3>() );
    }
  }
  else
  {
    const idx_t  n = coords.size (1);

    for ( idx_t i = 0; i < rank; i++ )
    {
      bbox[i] = bbox[i + rank] = coords(i,0);
    }

    for ( idx_t j = 1; j < n; j++ )
    {
      for ( idx_t i = 0; i < rank; i++ )
      {
        double  x = coords(i,j);

        if      ( x < bbox[i] )
        {
          bbox[i] = x;
        }
        else if ( x > bbox[i + rank] )
        {
          bbox[i + rank] = x;
        }
      }
    }
  }
}


//-----------------------------------------------------------------------
//   findMinMaxBoxes
//-----------------------------------------------------------------------


void                findMinMaxBoxes

  ( idx_t&            jmin,
    idx_t&            jmax,
    const Matrix&     boxes )

{
  JEM_PRECHECK2 ( boxes.size(0) % 2 == 0,
                  "bounding box array has wrong size" );

  const idx_t  rank = boxes.size(0) / 2;
  const idx_t  n    = boxes.size(1);

  double       dmin, dmax;


  jmin = jmax = -1;
  dmin =  DBL_MAX;
  dmax = -dmin;

  if ( rank == 0 )
  {
    return;
  }

  if ( boxes.isContiguous() && rank < 4_idx )
  {
    const double* JEM_RESTRICT  bxp = boxes.addr ();

    double  dd, dx, dy, dz;

    if     ( rank == 1_idx )
    {
      for ( idx_t j = 0; j < n; j++, bxp += 2 )
      {
        if ( bxp[0] > bxp[1] )
        {
          continue;
        }

        dx = bxp[1] - bxp[0];
        dd = dx * dx;

        if ( dd < dmin )
        {
          jmin = j;
          dmin = dd;
        }

        if ( dd > dmax )
        {
          jmax = j;
          dmax = dd;
        }
      }
    }
    else if ( rank == 2_idx )
    {
      for ( idx_t j = 0; j < n; j++, bxp += 4 )
      {
        if ( bxp[0] > bxp[2] )
        {
          continue;
        }

        dx = bxp[2] - bxp[0];
        dy = bxp[3] - bxp[1];

        dd = dx * dx + dy * dy;

        if ( dd < dmin )
        {
          jmin = j;
          dmin = dd;
        }

        if ( dd > dmax )
        {
          jmax = j;
          dmax = dd;
        }
      }
    }
    else if ( rank == 3_idx )
    {
      for ( idx_t j = 0; j < n; j++, bxp += 6 )
      {
        if ( bxp[0] > bxp[3] )
        {
          continue;
        }

        dx = bxp[3] - bxp[0];
        dy = bxp[4] - bxp[1];
        dz = bxp[5] - bxp[2];

        dd = dx * dx + dy * dy + dz * dz;

        if ( dd < dmin )
        {
          jmin = j;
          dmin = dd;
        }

        if ( dd > dmax )
        {
          jmax = j;
          dmax = dd;
        }
      }
    }
  }
  else
  {
    double  dd, dx;

    for ( idx_t j = 0; j < n; j++ )
    {
      if ( boxes(0,j) > boxes(rank,j) )
      {
        continue;
      }

      dd = 0.0;

      for ( idx_t i = 0; i < rank; i++ )
      {
        dx = boxes(i + rank,j) - boxes(i,j);
        dd = dd + dx * dx;
      }

      if ( dd < dmin )
      {
        jmin = j;
        dmin = dd;
      }

      if ( dd > dmax )
      {
        jmax = j;
        dmax = dd;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   sizeofBox
//-----------------------------------------------------------------------


double sizeofBox ( const Vector& box )
{
  JEM_ASSERT2 ( box.size() % 2 == 0,
                "bounding box array has wrong shape" );

  const idx_t  rank = box.size() / 2;

  double       dd, dx, dy, dz;


  if ( rank == 0 )
  {
    return 0.0;
  }

  if ( box[0] > box[rank] )
  {
    return -1.0;
  }

  if      ( rank == 1_idx )
  {
    dx = box[1] - box[0];

    dd = dx * dx;
  }
  else if ( rank == 2_idx )
  {
    dx = box[2] - box[0];
    dy = box[3] - box[1];

    dd = dx * dx + dy * dy;
  }
  else if ( rank == 3_idx )
  {
    dx = box[3] - box[0];
    dy = box[4] - box[1];
    dz = box[5] - box[2];

    dd = dx * dx + dy * dy + dz * dz;
  }
  else
  {
    dd = 0.0;

    for ( idx_t i = 0; i < rank; i++ )
    {
      dx = box[i + rank] - box[i];
      dd = dd + dx * dx;
    }
  }

  return std::sqrt ( dd );
}


//-----------------------------------------------------------------------
//   sizeofSmallestBox
//-----------------------------------------------------------------------


double sizeofSmallestBox ( const Matrix& boxes )
{
  idx_t  jmin, jmax;


  if ( boxes.size(0) == 0 || boxes.size(1) == 0 )
  {
    return 0.0;
  }

  findMinMaxBoxes ( jmin, jmax, boxes );

  if ( jmin < 0 )
  {
    return -1.0;
  }
  else
  {
    return sizeofBox ( boxes[jmin] );
  }
}


//-----------------------------------------------------------------------
//   sizeofLargestBox
//-----------------------------------------------------------------------


double sizeofLargestBox ( const Matrix& boxes )
{
  idx_t  jmin, jmax;


  if ( boxes.size(0) == 0 || boxes.size(1) == 0 )
  {
    return 0.0;
  }

  findMinMaxBoxes ( jmin, jmax, boxes );

  if ( jmax < 0 )
  {
    return -1.0;
  }
  else
  {
    return sizeofBox ( boxes[jmax] );
  }
}


//-----------------------------------------------------------------------
//   getBoxSizes
//-----------------------------------------------------------------------


void                getBoxSizes

  ( const Vector&     sizes,
    const Matrix&     boxes )

{
  JEM_PRECHECK2 ( boxes.size(0) % 2 == 0 &&
                  boxes.size(1)     == sizes.size(),
                  "bounding box array has wrong shape" );

  const idx_t  rank = boxes.size(0) / 2;
  const idx_t  n    = boxes.size(1);


  if ( rank == 0 )
  {
    return;
  }

  if ( boxes.isContiguous() && rank < 4_idx )
  {
    const double* JEM_RESTRICT  bxp = boxes.addr ();

    double  dx, dy, dz, sz;

    if      ( rank == 1_idx )
    {
      for ( idx_t j = 0; j < n; j++, bxp += 2 )
      {
        if ( bxp[0] > bxp[1] )
        {
          sz = -1.0;
        }
        else
        {
          dx = bxp[1] - bxp[0];
          sz = std::sqrt ( dx * dx );
        }

        sizes[j] = sz;
      }
    }
    else if ( rank == 2_idx )
    {
      for ( idx_t j = 0; j < n; j++, bxp += 4 )
      {
        if ( bxp[0] > bxp[2] )
        {
          sz = -1.0;
        }
        else
        {
          dx = bxp[2] - bxp[0];
          dy = bxp[3] - bxp[1];
          sz = std::sqrt ( dx * dx + dy * dy );
        }

        sizes[j] = sz;
      }
    }
    else if ( rank == 3_idx )
    {
      for ( idx_t j = 0; j < n; j++, bxp += 6 )
      {
        if ( bxp[0] > bxp[3] )
        {
          sz = -1.0;
        }
        else
        {
          dx = bxp[3] - bxp[0];
          dy = bxp[4] - bxp[1];
          dz = bxp[5] - bxp[2];
          sz = std::sqrt ( dx * dx + dy * dy + dz * dz );
        }

        sizes[j] = sz;
      }
    }
  }
  else
  {
    double  dx, sz;

    for ( idx_t j = 0; j < n; j++ )
    {
      if ( boxes(0,j) > boxes(rank,j) )
      {
        sz = -1.0;
      }
      else
      {
        sz = 0.0;

        for ( idx_t i = 0; i < rank; i++ )
        {
          dx = boxes(i + rank,j) - boxes(i,j);
          sz = sz + dx * dx;
        }

        sz = std::sqrt ( sz );
      }

      sizes[j] = sz;
    }
  }
}


//-----------------------------------------------------------------------
//   getBoxCenter
//-----------------------------------------------------------------------


void                getBoxCenter

  ( const Vector&     center,
    const Vector&     box )

{
  JEM_ASSERT2 ( 2 * center.size() == box.size(),
                "Array size mismatch" );

  const idx_t  rank = center.size ();


  if      ( rank == 1_idx )
  {
    center[0] = 0.5 * (box[0] + box[1]);
  }
  else if ( rank == 2_idx )
  {
    center[0] = 0.5 * (box[0] + box[2]);
    center[1] = 0.5 * (box[1] + box[3]);
  }
  else if ( rank == 3_idx )
  {
    center[0] = 0.5 * (box[0] + box[3]);
    center[1] = 0.5 * (box[1] + box[4]);
    center[2] = 0.5 * (box[2] + box[5]);
  }
  else if ( rank >= 4_idx )
  {
    for ( idx_t i = 0; i < rank; i++ )
    {
      center[i] = 0.5 * (box[i] + box[i + rank]);
    }
  }
}


//-----------------------------------------------------------------------
//   getBoxCenters
//-----------------------------------------------------------------------


void                getBoxCenters

  ( const Matrix&     centers,
    const Matrix&     boxes )

{
  JEM_PRECHECK2 ( boxes.size(0) == 2 * centers.size(0) &&
                  boxes.size(1) ==     centers.size(1),
                  "bounding box array has wrong shape" );

  const idx_t  rank = centers.size (0);
  const idx_t  n    = centers.size (1);


  if ( rank == 0 )
  {
    return;
  }

  if ( centers.isContiguous() &&
       boxes  .isContiguous() && rank < 4_idx )
  {
    const double* JEM_RESTRICT  bxp = boxes  .addr ();
    double*       JEM_RESTRICT  cxp = centers.addr ();

    if      ( rank == 1_idx )
    {
      for ( idx_t j = 0; j < n; j++, cxp++, bxp += 2 )
      {
        cxp[0] = 0.5 * (bxp[0] + bxp[1]);
      }
    }
    else if ( rank == 2_idx )
    {
      for ( idx_t j = 0; j < n; j++, cxp += 2, bxp += 4 )
      {
        cxp[0] = 0.5 * (bxp[0] + bxp[2]);
        cxp[1] = 0.5 * (bxp[1] + bxp[3]);
      }
    }
    else if ( rank == 3_idx )
    {
      for ( idx_t j = 0; j < n; j++, cxp += 3, bxp += 6 )
      {
        cxp[0] = 0.5 * (bxp[0] + bxp[3]);
        cxp[1] = 0.5 * (bxp[1] + bxp[4]);
        cxp[2] = 0.5 * (bxp[2] + bxp[5]);
      }
    }
  }
  else
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      for ( idx_t i = 0; i < rank; i++ )
      {
        centers(i,j) = 0.5 * (boxes(i,j) + boxes(i + rank,j));
      }
    }
  }
}


//-----------------------------------------------------------------------
//   makeBox
//-----------------------------------------------------------------------


void                makeBox

  ( const Vector&     box,
    const Vector&     center,
    double            size )

{
  JEM_ASSERT2 ( box.size() == 2 * center.size(),
                "Array size mismatch" );

  const idx_t  rank = center.size ();


  size *= 0.5;

  if      ( rank == 1_idx )
  {
    box[0] = center[0] - size;
    box[1] = center[0] + size;
  }
  else if ( rank == 2_idx )
  {
    size  *= INV_SQRT2_;

    box[0] = center[0] - size;
    box[2] = center[0] + size;
    box[1] = center[1] - size;
    box[3] = center[1] + size;
  }
  else if ( rank == 3_idx )
  {
    size  *= INV_SQRT3_;

    box[0] = center[0] - size;
    box[3] = center[0] + size;
    box[1] = center[1] - size;
    box[4] = center[1] + size;
    box[2] = center[2] - size;
    box[5] = center[2] + size;
  }
  else if ( rank >= 4_idx )
  {
    size  *= (1.0 / std::sqrt( (double) rank ));

    for ( idx_t i = 0; i < rank; i++ )
    {
      box[i +    0] = center[i] - size;
      box[i + rank] = center[i] + size;
    }
  }
}


//-----------------------------------------------------------------------
//   makeBoxes (constant size)
//-----------------------------------------------------------------------


void                makeBoxes

  ( const Matrix&     boxes,
    const Matrix&     centers,
    double            size )

{
  JEM_PRECHECK2 ( boxes.size(0) == 2 * centers.size(0) &&
                  boxes.size(1) ==     centers.size(1),
                  "bounding box array has wrong shape" );

  const idx_t  rank = centers.size (0);
  const idx_t  n    = centers.size (1);


  if ( rank == 0 )
  {
    return;
  }

  size *= 0.5;

  if ( centers.isContiguous() &&
       boxes  .isContiguous() && rank < 4_idx )
  {
    double*       JEM_RESTRICT  bxp = boxes  .addr ();
    const double* JEM_RESTRICT  cxp = centers.addr ();

    if      ( rank == 1_idx )
    {
      for ( idx_t j = 0; j < n; j++, cxp++, bxp += 2 )
      {
        bxp[0] = cxp[0] - size;
        bxp[1] = cxp[0] + size;
      }
    }
    else if ( rank == 2_idx )
    {
      size *= INV_SQRT2_;

      for ( idx_t j = 0; j < n; j++, cxp += 2, bxp += 4 )
      {
        bxp[0] = cxp[0] - size;
        bxp[2] = cxp[0] + size;
        bxp[1] = cxp[1] - size;
        bxp[3] = cxp[1] + size;
      }
    }
    else if ( rank == 3_idx )
    {
      size *= INV_SQRT3_;

      for ( idx_t j = 0; j < n; j++, cxp += 3, bxp += 6 )
      {
        bxp[0] = cxp[0] - size;
        bxp[3] = cxp[0] + size;
        bxp[1] = cxp[1] - size;
        bxp[4] = cxp[1] + size;
        bxp[2] = cxp[2] - size;
        bxp[5] = cxp[2] + size;
      }
    }
  }
  else
  {
    size *= (1.0 / std::sqrt( (double) rank ));

    for ( idx_t j = 0; j < n; j++ )
    {
      for ( idx_t i = 0; i < rank; i++ )
      {
        boxes(i +    0,j) = centers(i,j) - size;
        boxes(i + rank,j) = centers(i,j) + size;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   makeBoxes (varying size)
//-----------------------------------------------------------------------


void                makeBoxes

  ( const Matrix&     boxes,
    const Matrix&     centers,
    const Vector&     sizes )

{
  JEM_PRECHECK2 ( boxes.size(0) == 2 * centers.size(0) &&
                  boxes.size(1) ==     centers.size(1) &&
                  sizes.size()  ==     centers.size(1),
                  "Array size mismatch" );

  const idx_t  rank = centers.size (0);
  const idx_t  n    = centers.size (1);


  if ( rank == 0 )
  {
    return;
  }

  if ( centers.isContiguous() &&
       boxes  .isContiguous() && rank < 4_idx )
  {
    double*       JEM_RESTRICT  bxp = boxes  .addr ();
    const double* JEM_RESTRICT  cxp = centers.addr ();

    double  beta, size;

    if      ( rank == 1_idx )
    {
      beta = 0.5;

      for ( idx_t j = 0; j < n; j++, cxp++, bxp += 2 )
      {
        size   = beta * sizes[j];

        bxp[0] = cxp[0] - size;
        bxp[1] = cxp[0] + size;
      }
    }
    else if ( rank == 2_idx )
    {
      beta = 0.5 * INV_SQRT2_;

      for ( idx_t j = 0; j < n; j++, cxp += 2, bxp += 4 )
      {
        size   = beta * sizes[j];

        bxp[0] = cxp[0] - size;
        bxp[2] = cxp[0] + size;
        bxp[1] = cxp[1] - size;
        bxp[3] = cxp[1] + size;
      }
    }
    else if ( rank == 3_idx )
    {
      beta = 0.5 * INV_SQRT3_;

      for ( idx_t j = 0; j < n; j++, cxp += 3, bxp += 6 )
      {
        size   = beta * sizes[j];

        bxp[0] = cxp[0] - size;
        bxp[3] = cxp[0] + size;
        bxp[1] = cxp[1] - size;
        bxp[4] = cxp[1] + size;
        bxp[2] = cxp[2] - size;
        bxp[5] = cxp[2] + size;
      }
    }
  }
  else
  {
    double  beta, size;

    beta = (0.5 / std::sqrt( (double) rank ));

    for ( idx_t j = 0; j < n; j++ )
    {
      size = beta * sizes[j];

      for ( idx_t i = 0; i < rank; i++ )
      {
        boxes(i +    0,j) = centers(i,j) - size;
        boxes(i + rank,j) = centers(i,j) + size;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   makeEmptyBox
//-----------------------------------------------------------------------


void makeEmptyBox ( const Vector& box )
{
  JEM_PRECHECK2 ( box.size() % 2 == 0,
                  "bounding box array has wrong size" );

  const idx_t  rank = box.size () / 2;

  for ( idx_t i = 0; i < rank; i++ )
  {
    box[i + 0]    =  1.0;
    box[i + rank] = -1.0;
  }
}


//-----------------------------------------------------------------------
//   scaleBox
//-----------------------------------------------------------------------


void                scaleBox

  ( const Vector&     box,
    double            scale )

{
  JEM_ASSERT2 ( box.size() % 2 == 0,
                "bounding box array has wrong size" );
  JEM_ASSERT2 ( scale >= 0.0,
                "invalid scale factor" );

  const idx_t  rank = box.size() / 2;

  double       x, dx;

  if ( rank == 0 || box[0] > box[rank] )
  {
    return;
  }

  for ( idx_t i = 0; i < rank; i++ )
  {
    x  = box[i + rank] + box[i];
    dx = box[i + rank] - box[i];
    dx = dx * scale;

    box[i]        = 0.5 * (x - dx);
    box[i + rank] = 0.5 * (x + dx);
  }
}


//-----------------------------------------------------------------------
//   scaleBoxes (constant scale)
//-----------------------------------------------------------------------


void                scaleBoxes

  ( const Matrix&     boxes,
    double            scale )

{
  JEM_PRECHECK2 ( boxes.size(0) % 2 == 0,
                  "bounding box array has wrong size" );
  JEM_PRECHECK2 ( scale >= 0.0,
                  "invalid scale factor" );

  const idx_t  rank = boxes.size(0) / 2;
  const idx_t  n    = boxes.size(1);


  if ( rank == 0 )
  {
    return;
  }

  if ( boxes.isContiguous() && rank < 4_idx )
  {
    double* JEM_RESTRICT  bxp = boxes.addr ();

    double  x, dx;
    double  y, dy;
    double  z, dz;

    if      ( rank == 1_idx )
    {
      for ( idx_t j = 0; j < n; j++, bxp += 2 )
      {
        if ( bxp[0] > bxp[1] )
        {
          continue;
        }

        x  =  bxp[1] + bxp[0];
        dx = (bxp[1] - bxp[0]) * scale;

        bxp[0] = 0.5 * (x - dx);
        bxp[1] = 0.5 * (x + dx);
      }
    }
    else if ( rank == 2_idx )
    {
      for ( idx_t j = 0; j < n; j++, bxp += 4 )
      {
        if ( bxp[0] > bxp[2] )
        {
          continue;
        }

        x  =  bxp[2] + bxp[0];
        dx = (bxp[2] - bxp[0]) * scale;
        y  =  bxp[3] + bxp[1];
        dy = (bxp[3] - bxp[1]) * scale;

        bxp[0] = 0.5 * (x - dx);
        bxp[1] = 0.5 * (y - dy);
        bxp[2] = 0.5 * (x + dx);
        bxp[3] = 0.5 * (y + dy);
      }
    }
    else if ( rank == 3_idx )
    {
      for ( idx_t j = 0; j < n; j++, bxp += 6 )
      {
        if ( bxp[0] > bxp[3] )
        {
          continue;
        }

        x  =  bxp[3] + bxp[0];
        dx = (bxp[3] - bxp[0]) * scale;
        y  =  bxp[4] + bxp[1];
        dy = (bxp[4] - bxp[1]) * scale;
        z  =  bxp[5] + bxp[2];
        dz = (bxp[5] - bxp[2]) * scale;

        bxp[0] = 0.5 * (x - dx);
        bxp[1] = 0.5 * (y - dy);
        bxp[2] = 0.5 * (z - dz);
        bxp[3] = 0.5 * (x + dx);
        bxp[4] = 0.5 * (y + dy);
        bxp[5] = 0.5 * (z + dz);
      }
    }
  }
  else
  {
    double  x, dx;

    for ( idx_t j = 0; j < n; j++ )
    {
      if ( boxes(0,j) > boxes(rank,j) )
      {
        continue;
      }

      for ( idx_t i = 0; i < rank; i++ )
      {
        x  =  boxes(i + rank,j) + boxes(i,j);
        dx = (boxes(i + rank,j) - boxes(i,j)) * scale;

        boxes(i,j)        = 0.5 * (x - dx);
        boxes(i + rank,j) = 0.5 * (x + dx);
      }
    }
  }
}


//-----------------------------------------------------------------------
//   scaleBoxes (varying scale)
//-----------------------------------------------------------------------


void                scaleBoxes

  ( const Matrix&     boxes,
    const Vector&     scales )

{
  JEM_PRECHECK2 ( boxes.size(0) % 2 == 0 &&
                  boxes.size(1)     == scales.size(),
                  "bounding box array has wrong shape" );

  const idx_t  rank = boxes.size(0) / 2;
  const idx_t  n    = boxes.size(1);


  if ( rank == 0 )
  {
    return;
  }

  if ( boxes.isContiguous() && rank < 4_idx )
  {
    double* JEM_RESTRICT  bxp = boxes.addr ();

    double  scale;
    double  x, dx;
    double  y, dy;
    double  z, dz;

    if      ( rank == 1_idx )
    {
      for ( idx_t j = 0; j < n; j++, bxp += 2 )
      {
        if ( bxp[0] > bxp[1] )
        {
          continue;
        }

        scale  = scales[j];

        x      =  bxp[1] + bxp[0];
        dx     = (bxp[1] - bxp[0]) * scales[j];

        bxp[0] = 0.5 * (x - dx);
        bxp[1] = 0.5 * (x + dx);
      }
    }
    else if ( rank == 2_idx )
    {
      for ( idx_t j = 0; j < n; j++, bxp += 4 )
      {
        if ( bxp[0] > bxp[2] )
        {
          continue;
        }

        scale  = scales[j];

        x      =  bxp[2] + bxp[0];
        dx     = (bxp[2] - bxp[0]) * scale;
        y      =  bxp[3] + bxp[1];
        dy     = (bxp[3] - bxp[1]) * scale;

        bxp[0] = 0.5 * (x - dx);
        bxp[1] = 0.5 * (y - dy);
        bxp[2] = 0.5 * (x + dx);
        bxp[3] = 0.5 * (y + dy);
      }
    }
    else if ( rank == 3_idx )
    {
      for ( idx_t j = 0; j < n; j++, bxp += 6 )
      {
        if ( bxp[0] > bxp[3] )
        {
          continue;
        }

        scale  = scales[j];

        x      =  bxp[3] + bxp[0];
        dx     = (bxp[3] - bxp[0]) * scale;
        y      =  bxp[4] + bxp[1];
        dy     = (bxp[4] - bxp[1]) * scale;
        z      =  bxp[5] + bxp[2];
        dz     = (bxp[5] - bxp[2]) * scale;

        bxp[0] = 0.5 * (x - dx);
        bxp[1] = 0.5 * (y - dy);
        bxp[2] = 0.5 * (z - dz);
        bxp[3] = 0.5 * (x + dx);
        bxp[4] = 0.5 * (y + dy);
        bxp[5] = 0.5 * (z + dz);
      }
    }
  }
  else
  {
    double  scale;
    double  x, dx;

    for ( idx_t j = 0; j < n; j++ )
    {
      if ( boxes(0,j) > boxes(rank,j) )
      {
        continue;
      }

      scale = scales[j];

      for ( idx_t i = 0; i < rank; i++ )
      {
        x  =  boxes(i + rank,j) + boxes(i,j);
        dx = (boxes(i + rank,j) - boxes(i,j)) * scale;

        boxes(i,j)        = 0.5 * (x - dx);
        boxes(i + rank,j) = 0.5 * (x + dx);
      }
    }
  }
}


//-----------------------------------------------------------------------
//   isInBox
//-----------------------------------------------------------------------


bool                isInBox

  ( const Vector&     point,
    const Vector&     box )

{
  JEM_ASSERT2 ( box.size() == 2 * point.size(),
                "Array size mismatch" );

  const idx_t  rank = point.size ();

  for ( idx_t i = 0; i < rank; i++ )
  {
    double  x = point[i];

    if ( (x - box[i]) * (x - box[i + rank]) > 0.0 )
    {
      return false;
    }
  }

  return true;
}


//-----------------------------------------------------------------------
//   findEnclosingBoxes
//-----------------------------------------------------------------------


idx_t               findEnclosingBoxes

  ( const IdxVector&  iboxes,
    const Vector&     point,
    const Matrix&     boxes )

{
  JEM_ASSERT2 ( boxes.size(0) == 2 * point.size(),
                "Array size mismatch" );

  const idx_t  rank  = point.size ();
  const idx_t  n     = boxes.size (1);

  IntVector    imask ( n );

  int*         JEM_RESTRICT  imp = imask.addr ();

  double       x, y, z;

  idx_t        k = 0;


  if ( rank == 0 )
  {
    return 0;
  }

  imask = 0;

  // Mark all boxes containing the specified point.

  if ( boxes.isContiguous() && rank < 4_idx )
  {
    const double* JEM_RESTRICT  bxp = boxes.addr ();

    if      ( rank == 1_idx )
    {
      x = point[0];

JEM_IVDEP

      for ( idx_t i = 0; i < n; i++, bxp += 2 )
      {
        imp[i] = (int) ((x - bxp[0]) * (x - bxp[1]) <= 0.0);
      }
    }
    else if ( rank == 2_idx )
    {
      x = point[0];
      y = point[1];

JEM_IVDEP

      for ( idx_t i = 0; i < n; i++, bxp += 4 )
      {
        imp[i] =
          ((int) ((x - bxp[0]) * (x - bxp[2]) <= 0.0) +
           (int) ((y - bxp[1]) * (y - bxp[3]) <= 0.0)) >> 1;
      }
    }
    else if ( rank == 3_idx )
    {
      x = point[0];
      y = point[1];
      z = point[2];

JEM_IVDEP

      for ( idx_t i = 0; i < n; i++, bxp += 6 )
      {
        imp[i] =
          ((int) ((x - bxp[0]) * (x - bxp[3]) <= 0.0) +
           (int) ((y - bxp[1]) * (y - bxp[4]) <= 0.0) +
           (int) ((z - bxp[2]) * (z - bxp[5]) <= 0.0) + 1) >> 2;
      }
    }

    if ( iboxes.isContiguous() && iboxes.size() >= n )
    {
      idx_t* JEM_RESTRICT  ibp = iboxes.addr ();

      // Collect al marked boxes.

      for ( idx_t i = 0; i < n; i++ )
      {
        ibp[k] = i;
        k     += (idx_t) imp[i];
      }

      return k;
    }
  }
  else
  {
    const double* JEM_RESTRICT  bx1 = boxes.addr ();
    const double* JEM_RESTRICT  bx2 = boxes.addr ();

    const idx_t   st0 = boxes.stride (0);
    const idx_t   st1 = boxes.stride (1);

    bx2 += (st0 * rank);

    for ( idx_t j = 0; j < rank; j++ )
    {
      x = point[j];

JEM_IVDEP

      for ( idx_t i = 0; i < n; i++ )
      {
        imp[i] += (int) ((x - bx1[i * st1]) *
                         (x - bx2[i * st1]) <= 0.0);
      }

      bx1 += st0;
      bx2 += st0;
    }
  }

  // Collect all marked boxes.

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( imp[i] )
    {
      if ( k < iboxes.size() )
      {
        iboxes[k++] = i;
      }
      else
      {
        break;
      }
    }
  }

  return k;
}


JIVE_END_PACKAGE( geom )
