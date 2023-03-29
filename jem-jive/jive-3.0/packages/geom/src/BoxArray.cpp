
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


#include <cstring>
#include <jem/pragmas.h>
#include <jem/base/assert.h>
#include <jem/base/Float.h>
#include <jem/base/RuntimeException.h>
#include <jem/base/ArithmeticException.h>
#include <jem/base/array/utilities.h>
#include <jem/io/DataInput.h>
#include <jem/io/DataOutput.h>
#include <jive/util/error.h>
#include <jive/util/Reordering.h>
#include <jive/geom/boxes.h>
#include <jive/geom/MaskMatrix.h>
#include <jive/geom/BoxArray.h>


JIVE_BEGIN_PACKAGE( geom )


//=======================================================================
//   class BoxArray
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


BoxArray::BoxArray ( idx_t rank ) :

  rank_  ( rank ),
  rank2_ ( rank * 2 )

{
  JEM_PRECHECK2 ( rank > 0, "invalid rank" );

  bounds_.resize ( rank2_ );

  validBounds_ = false;
}


//-----------------------------------------------------------------------
//   checkBoxes
//-----------------------------------------------------------------------


void BoxArray::checkBoxes ( const String& context ) const
{
  const double* JEM_RESTRICT  x = boxes_.addr ();

  const idx_t  n  = colors_.size ();
  const idx_t  r  = rank_;
  const idx_t  r2 = rank2_;

  double       t;
  idx_t        i, j, k;


  t = 0.0;

  for ( i = 0; i < n; i++, x += r2 )
  {
    for ( j = k = 0; j < r; j++ )
    {
      t += x[j] + x[j + r];

      if ( x[j] > x[j + r] )
      {
        k++;
      }
    }

    if ( k != 0 && k != r )
    {
      throw jem::RuntimeException (
        context,
        String::format ( "box %d has invalid corner points", i )
      );
    }
  }

  if ( jem::Float::isNaN( t ) )
  {
    throw jem::ArithmeticException (
      context,
      "invalid box corners: NaN"
    );
  }
}


//-----------------------------------------------------------------------
//   getEnclosingBox
//-----------------------------------------------------------------------


void BoxArray::getEnclosingBox ( const Vector& box ) const
{
  JEM_PRECHECK2 ( box.size() == rank2(),
                  "bounding box array has wrong size" );

  if ( ! validBounds_ )
  {
    const_cast<Self*>( this )->updateBounds_ ();
  }

  for ( idx_t i = 0; i < rank2_; i++ )
  {
    box[i] = bounds_[i];
  }
}


//-----------------------------------------------------------------------
//   addBox
//-----------------------------------------------------------------------


idx_t BoxArray::addBox

  ( const Vector&  box,
    idx_t          color )

{
  JEM_PRECHECK2 ( box.size() == rank2(),
                  "bounding box array has wrong size" );

  const idx_t  r2   = rank2_;
  const idx_t  ibox = colors_.size ();


  colors_ .pushBack ( color );

  for ( idx_t i = 0; i < r2; i += 2 )
  {
    boxes_.pushBack ( box[i + 0] );
    boxes_.pushBack ( box[i + 1] );
  }

  validBounds_ = false;

  return ibox;
}


//-----------------------------------------------------------------------
//   addBoxes
//-----------------------------------------------------------------------


idx_t BoxArray::addBoxes

  ( const Matrix&  boxes,
    idx_t          color )

{
  JEM_PRECHECK2 ( boxes.size(0) == rank2(),
                  "bounding box array has wrong shape" );

  const idx_t  n    = boxes  .size (1);
  const idx_t  ibox = colors_.size ();


  colors_.pushBack ( color, n );
  boxes_ .pushBack ( 0.0,   n * rank2_ );

  updateBoxes_     ( ibox, boxes );

  return ibox;
}


idx_t BoxArray::addBoxes

  ( const Matrix&     boxes,
    const IdxVector&  colors )

{
  JEM_PRECHECK2 ( boxes.size(1) == colors.size(),
                  "Array size mismatch" );

  const idx_t  n    = colors.size ();
  const idx_t  ibox = addBoxes    ( boxes, 0 );


  for ( idx_t i = 0; i < n; i++ )
  {
    colors_[ibox + i] = colors[i];
  }

  return ibox;
}


//-----------------------------------------------------------------------
//   reorderBoxes
//-----------------------------------------------------------------------


void BoxArray::reorderBoxes ( const Reordering& reord )
{
  using jem::util::Flex;

  JEM_PRECHECK2 ( reord.oldSize() == size(),
                  "invalid Reordering" );

  const idx_t*  JEM_RESTRICT  rmap   = reord.rmapPtr ();
  idx_t*        JEM_RESTRICT  colors = colors_.addr  ();
  double*       JEM_RESTRICT  boxes  = boxes_ .addr  ();

  const idx_t   r2      = rank2_;
  const idx_t   start   = reord.start   ();
  const idx_t   newSize = reord.newSize ();

  idx_t         ibox;
  idx_t         jbox;
  idx_t         i, j, k;


  if ( reord.isSimple() )
  {
    for ( ibox = start; ibox < newSize; ibox++ )
    {
      jbox = rmap[ibox];
      i    = r2 * ibox;
      j    = r2 * jbox;

      colors[ibox] = colors[jbox];

      for ( k = 0; k < r2; k += 2 )
      {
        boxes[i + k + 0] = boxes[j + k + 0];
        boxes[i + k + 1] = boxes[j + k + 1];
      }
    }

    boxes_ .resize ( newSize * r2 );
    colors_.resize ( newSize );
  }
  else
  {
    Flex<double>  tmpBoxes  ( newSize * r2 );
    Flex<idx_t>   tmpColors ( newSize );

    double*       JEM_RESTRICT  newBoxes  = tmpBoxes .addr ();
    idx_t*        JEM_RESTRICT  newColors = tmpColors.addr ();


    if ( start > 0 )
    {
      std::memcpy ( newBoxes, boxes, (size_t)
                    (start * r2) * sizeof(double) );

      newBoxes += (start * r2);
    }

    for ( ibox = start; ibox < newSize; ibox++, newBoxes += r2 )
    {
      jbox = rmap[ibox];
      j    = r2 * jbox;

      newColors[ibox] = colors[jbox];

      for ( k = 0; k < r2; k += 2 )
      {
        newBoxes[k + 0] = boxes[j + k + 0];
        newBoxes[k + 1] = boxes[j + k + 1];
      }
    }

    colors_.swap ( tmpColors );
    boxes_ .swap ( tmpBoxes  );
  }

  validBounds_ = false;
}


//-----------------------------------------------------------------------
//   getBox
//-----------------------------------------------------------------------


void BoxArray::getBox

  ( const Vector&  box,
    idx_t          ibox ) const

{
  JEM_PRECHECK2 ( box.size() == rank2(),
                  "bounding box array has wrong size" );

  checkIndex   ( ibox );

  const idx_t    r2 = rank2_;

  const double*  JEM_RESTRICT  x = boxes_.addr ( r2 * ibox );


  if ( box.isContiguous() )
  {
    std::memcpy ( box.addr(), x, (size_t) r2 * sizeof(double) );
  }
  else
  {
    for ( idx_t i = 0; i < r2; i += 2 )
    {
      box[i + 0] = x[i + 0];
      box[i + 1] = x[i + 1];
    }
  }
}


//-----------------------------------------------------------------------
//   getBoxes
//-----------------------------------------------------------------------


void BoxArray::getBoxes ( const Matrix& boxes ) const
{
  JEM_PRECHECK2 ( boxes.size(0) == rank2() &&
                  boxes.size(1) == size (),
                  "bounding box array has wrong shape" );

  if ( boxes.isContiguous() )
  {
    const idx_t  n = boxes_.size ();

    std::memcpy ( boxes.addr(), boxes_.addr(),
                  (size_t) n * sizeof(double) );
  }
  else
  {
    const double* JEM_RESTRICT  x = boxes_.addr ();

    const idx_t  n  = colors_.size ();
    const idx_t  r2 = rank2_;

    for ( idx_t i = 0; i < n; i++, x += r2 )
    {
      for ( idx_t k = 0; k < r2; k++ )
      {
        boxes(k,i) = x[k];
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getSomeBoxes
//-----------------------------------------------------------------------


void BoxArray::getSomeBoxes

  ( const Matrix&     boxes,
    const IdxVector&  iboxes ) const

{
  JEM_PRECHECK2 ( boxes.size(0) == rank2() &&
                  boxes.size(1) == iboxes.size(),
                  "Array shape mismatch" );

  const double* JEM_RESTRICT  bx = boxes_.addr ();
  const double* JEM_RESTRICT  x;

  const idx_t  n = iboxes.size ();

  idx_t        i;


  for ( i = 0; i < n; i++ )
  {
    checkIndex ( iboxes[i] );
  }

  if ( boxes.isContiguous() && rank_ < 4_idx )
  {
    double* JEM_RESTRICT  y = boxes.addr ();

    if      ( rank_ == 1_idx )
    {
      for ( i = 0; i < n; i++, y += 2 )
      {
        x = bx + (2 * iboxes[i]);

        y[0] = x[0];
        y[1] = x[1];
      }
    }
    else if ( rank_ == 2_idx )
    {
      for ( i = 0; i < n; i++, y += 4 )
      {
        x = bx + (4 * iboxes[i]);

        y[0] = x[0];
        y[1] = x[1];
        y[2] = x[2];
        y[3] = x[3];
      }
    }
    else if ( rank_ == 3_idx )
    {
      for ( i = 0; i < n; i++, y += 6 )
      {
        x = bx + (6 * iboxes[i]);

        y[0] = x[0];
        y[1] = x[1];
        y[2] = x[2];
        y[3] = x[3];
        y[4] = x[4];
        y[5] = x[5];
      }
    }
  }
  else
  {
    const idx_t  r2 = rank2_;

    for ( i = 0; i < n; i++ )
    {
      x = bx + (r2 * iboxes[i]);

      for ( idx_t k = 0; k < r2; k++ )
      {
        boxes(k,i) = x[k];
      }
    }
  }
}


//-----------------------------------------------------------------------
//   setColors
//-----------------------------------------------------------------------


void BoxArray::setColors ( const IdxVector& colors )
{
  JEM_PRECHECK2 ( colors.size() == size(),
                  "color array has wrong size" );

  idx_t* JEM_RESTRICT  dest = colors_.addr ();

  const idx_t  n = colors.size ();

  if ( colors.isContiguous() )
  {
    std::memcpy ( dest, colors.addr(),
                  (size_t) n * sizeof(idx_t) );
  }
  else
  {
JEM_IVDEP

    for ( idx_t i = 0; i < n; i++ )
    {
      dest[i] = colors[i];
    }
  }
}


//-----------------------------------------------------------------------
//   setSomeColors
//-----------------------------------------------------------------------


void BoxArray::setSomeColors

  ( const IdxVector&  iboxes,
    const IdxVector&  colors )

{
  JEM_PRECHECK2 ( iboxes.size() == colors.size(),
                  "Array size mismatch" );

  idx_t* JEM_RESTRICT  dest = colors_.addr ();

  const idx_t   n = colors.size ();

  idx_t         i;


#ifndef NDEBUG

  for ( i = 0; i < n; i++ )
  {
    checkIndex ( iboxes[i] );
  }

#endif

  if ( iboxes.stride() + colors.stride() == 2L )
  {
    const idx_t* JEM_RESTRICT  src = colors.addr ();
    const idx_t* JEM_RESTRICT  ib  = iboxes.addr ();

JEM_IVDEP

    for ( i = 0; i < n; i++ )
    {
      dest[ib[i]] = src[i];
    }
  }
  else
  {

JEM_IVDEP

    for ( i = 0; i < n; i++ )
    {
      dest[iboxes[i]] = colors[i];
    }
  }
}


//-----------------------------------------------------------------------
//   updateBoxes
//-----------------------------------------------------------------------


void BoxArray::updateBoxes ( const Matrix&  boxes )
{
  JEM_PRECHECK2 ( boxes.size(0) == rank2() &&
                  boxes.size(1) == size(),
                  "bounding box array has wrong shape" );

  updateBoxes_ ( 0, boxes );
}


//-----------------------------------------------------------------------
//   updateSomeBoxes
//-----------------------------------------------------------------------


void BoxArray::updateSomeBoxes

  ( const IdxVector&  iboxes,
    const Matrix&     boxes )

{
  JEM_PRECHECK2 ( boxes.size(0) == rank2() &&
                  boxes.size(1) == iboxes.size(),
                  "Array shape mismatch" );

  double* JEM_RESTRICT  bx = boxes_.addr ();
  double* JEM_RESTRICT  x;

  const idx_t  n = iboxes.size ();

  idx_t        i;


  for ( i = 0; i < n; i++ )
  {
    checkIndex ( iboxes[i] );
  }

  if ( boxes.isContiguous() && rank_ < 4_idx )
  {
    const double* JEM_RESTRICT  y = boxes.addr ();

    if      ( rank_ == 1_idx )
    {
JEM_IVDEP

      for ( i = 0; i < n; i++, y += 2 )
      {
        x    = bx + (2 * iboxes[i]);

        x[0] = y[0];
        x[1] = y[1];
      }
    }
    else if ( rank_ == 2_idx )
    {
JEM_IVDEP

      for ( i = 0; i < n; i++, y += 4 )
      {
        x    = bx + (4 * iboxes[i]);

        x[0] = y[0];
        x[1] = y[1];
        x[2] = y[2];
        x[3] = y[3];
      }
    }
    else if ( rank_ == 3_idx )
    {
JEM_IVDEP

      for ( i = 0; i < n; i++, y += 6 )
      {
        x    = bx + (6 * iboxes[i]);

        x[0] = y[0];
        x[1] = y[1];
        x[2] = y[2];
        x[3] = y[3];
        x[4] = y[4];
        x[5] = y[5];
      }
    }
  }
  else
  {
    const idx_t  r2 = rank2_;

    for ( i = 0; i < n; i++ )
    {
      x = bx + (r2 * iboxes[i]);

      for ( idx_t k = 0; k < r2; k++ )
      {
        x[k] = boxes(k,i);
      }
    }
  }

  validBounds_ = false;
}


//-----------------------------------------------------------------------
//   findChangedBoxes
//-----------------------------------------------------------------------


idx_t BoxArray::findChangedBoxes

  ( const IdxVector&  iboxes,
    const Matrix&     boxes ) const

{
  JEM_PRECHECK2 ( iboxes.size() >= size()  &&
                  boxes.size(0) == rank2() &&
                  boxes.size(1) == size(),
                  "Array shape mismatch" );

  const double* JEM_RESTRICT  x = boxes_.addr ();

  const idx_t  n = colors_.size ();

  idx_t        a, b;
  idx_t        i, j, k;


  // A box is marked as changed if one of the two following conditions
  // is true:
  //
  //  1) Both the current box and the new box are regular, and the new
  //     bounds exceed the current bounds.
  //
  //  2) The current box and the new box have different states
  //     (regular <-> empty).

  iboxes = 0;

  if ( boxes.isContiguous() && rank_ < 4_idx )
  {
    const double* JEM_RESTRICT  y = boxes.addr ();

    if      ( rank_ == 1_idx )
    {
JEM_IVDEP

      for ( i = 0; i < n; i++, x += 2, y += 2 )
      {
        a = (x[0] > x[1]) ? 0 : 1;
        b = (y[0] > y[1]) ? 0 : 1;

        // Check for two regular boxes first.

        if ( a + b == 2 )
        {
          if ( y[0] < x[0] ||
               y[1] > x[1] )
          {
            iboxes[i] = 1;
          }

          continue;
        }

        // Check for a "phase change" (regular <-> empty).

        if ( a + b == 1 )
        {
          iboxes[i] = 1;
        }
      }
    }
    else if ( rank_ == 2_idx )
    {
JEM_IVDEP

      for ( i = 0; i < n; i++, x += 4, y += 4 )
      {
        a = (x[0] > x[2]) ? 0 : 1;
        b = (y[0] > y[2]) ? 0 : 1;

        if ( a + b == 2 )
        {
          if ( y[0] < x[0] ||
               y[1] < x[1] ||
               y[2] > x[2] ||
               y[3] > x[3] )
          {
            iboxes[i] = 1;
          }

          continue;
        }

        if ( a + b == 1 )
        {
          iboxes[i] = 1;
        }
      }
    }
    else if ( rank_ == 3_idx )
    {
JEM_IVDEP

      for ( i = 0; i < n; i++, x += 6, y += 6 )
      {
        a = (x[0] > x[3]) ? 0 : 1;
        b = (y[0] > y[3]) ? 0 : 1;

        if ( a + b == 2 )
        {
          if ( y[0] < x[0] ||
               y[1] < x[1] ||
               y[2] < x[2] ||
               y[3] > x[3] ||
               y[4] > x[4] ||
               y[5] > x[5] )
          {
            iboxes[i] = 1;
          }

          continue;
        }

        if ( a + b == 1 )
        {
          iboxes[i] = 1;
        }
      }
    }
  }
  else
  {
    const idx_t  r  = rank_;
    const idx_t  r2 = rank2_;

    for ( i = 0; i < n; i++, x += r2 )
    {
      a = (x[0]       > x[r])       ? 0 : 1;
      b = (boxes(0,i) > boxes(r,i)) ? 0 : 1;

      if ( a + b == 2 )
      {
        for ( idx_t k = 0; k < r; k++ )
        {
          if ( boxes(k + 0,i) < x[k + 0] ||
               boxes(k + r,i) > x[k + r] )
          {
            iboxes[i] = 1;
            break;
          }
        }

        continue;
      }

      if ( a + b == 1 )
      {
        iboxes[i] = 1;
      }
    }
  }

  // Collect all marked boxes.

  for ( i = j = 0; i < n; i++ )
  {
    k         = iboxes[i];
    iboxes[j] = i;
    j        += k;
  }

  return j;
}


//-----------------------------------------------------------------------
//   findBoxNeighbors
//-----------------------------------------------------------------------


idx_t BoxArray::findBoxNeighbors

  ( idx_t*            iboxes,
    const Vector&     box,
    const idx_t*      ilist,
    idx_t             count ) const

{
  JEM_PRECHECK2 ( box.size() == rank2(),
                  "bounding box array has wrong size" );

  const double* JEM_RESTRICT  bx = boxes_.addr ();
  const double* JEM_RESTRICT  x;

  const idx_t  r  = rank_;
  const idx_t  r2 = rank2_;

  double       y0, y1, y2, y3, y4, y5;

  idx_t        ibox;
  idx_t        i, j, k;


#ifndef NDEBUG

  for ( i = 0; i < count; i++ )
  {
    checkIndex ( ilist[i] );
  }

#endif

  for ( i = 0; i < count; i++ )
  {
    iboxes[i] = 0;
  }

  if      ( r == 1_idx )
  {
    y0 = box[0];
    y1 = box[1];

    if ( y0 > y1 )
    {
      return 0;
    }

JEM_IVDEP

    for ( i = 0; i < count; i++ )
    {
      ibox = ilist[i];
      x    = bx + (2 * ibox);

      if ( x[0] > x[1] )
      {
        continue;
      }

      iboxes[i] = overlap ( y0, y1, x[0], x[1] );
    }
  }
  else if ( r == 2_idx )
  {
    y0 = box[0];
    y1 = box[1];
    y2 = box[2];
    y3 = box[3];

    if ( y0 > y2 )
    {
      return 0;
    }

JEM_IVDEP

    for ( i = 0; i < count; i++ )
    {
      ibox = ilist[i];
      x    = bx + (4 * ibox);

      if ( x[0] > x[2] )
      {
        continue;
      }

      iboxes[i] = (overlap( y0, y2, x[0], x[2] ) +
                   overlap( y1, y3, x[1], x[3] )) >> 1;
    }
  }
  else if ( r == 3_idx )
  {
    y0 = box[0];
    y1 = box[1];
    y2 = box[2];
    y3 = box[3];
    y4 = box[4];
    y5 = box[5];

    if ( y0 > y3 )
    {
      return 0;
    }

JEM_IVDEP

    for ( i = 0; i < count; i++ )
    {
      ibox = ilist[i];
      x    = bx + (6 * ibox);

      if ( x[0] > x[3] )
      {
        continue;
      }

      iboxes[i] = (overlap( y0, y3, x[0], x[3] ) +
                   overlap( y1, y4, x[1], x[4] ) +
                   overlap( y2, y5, x[2], x[5] ) + 1) >> 2;
    }
  }
  else if ( r >= 4_idx )
  {
    if ( box[0] > box[r] )
    {
      return 0;
    }

    for ( i = 0; i < count; i++ )
    {
      ibox = ilist[i];
      x    = bx + (r2 * ibox);

      if ( x[0] > x[r] )
      {
        continue;
      }

      for ( k = 0; k < r; k++ )
      {
        if ( ! overlap( box[k + 0], box[k + r],
                        x  [k + 0], x  [k + r] ) )
        {
          break;
        }
      }

      if ( k == r )
      {
        iboxes[i] = 1;
      }
    }
  }

  // Collect all marked boxes.

  for ( i = j = 0; i < count; i++ )
  {
    k         = iboxes[i];
    iboxes[j] = ilist [i];
    j        += k;
  }

  return j;
}


//-----------------------------------------------------------------------
//   findBoxNeighbors (given a MaskMatrix)
//-----------------------------------------------------------------------


idx_t BoxArray::findBoxNeighbors

  ( idx_t*             iboxes,
    const Vector&      box,
    idx_t              color,
    const MaskMatrix&  mask,
    const idx_t*       ilist,
    idx_t              count ) const

{
  const idx_t*  colors = colors_.addr ();

  idx_t         ibox;
  idx_t         i, j, n;


  n = findBoxNeighbors ( iboxes, box, ilist, count );

  for ( i = j = 0; i < n; i++ )
  {
    ibox = iboxes[i];

    if ( mask.getValue( color, colors[ibox] ) )
    {
      iboxes[j++] = ibox;
    }
  }

  return j;
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void BoxArray::readFrom ( DataInput& in )

{
  decode ( in, rank_, boxes_, colors_ );

  rank2_ = 2 * rank_;

  if ( rank2_ * colors_.size() != boxes_.size() )
  {
    jem::io::decodeError ( JEM_FUNC );
  }

  bounds_.resize ( rank2_ );

  validBounds_ = false;
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void BoxArray::writeTo ( DataOutput& out ) const
{
  encode ( out, rank_, boxes_, colors_ );
}


//-----------------------------------------------------------------------
//   updateBoxes_
//-----------------------------------------------------------------------


void BoxArray::updateBoxes_

  ( idx_t          from,
    const Matrix&  boxes )

{
  double* JEM_RESTRICT  x = boxes_.addr ( from * rank2_ );

  const idx_t  n  = boxes.size (1);


  if ( n == 0 )
  {
    return;
  }

  validBounds_ = false;

  if ( boxes.isContiguous() )
  {
    std::memcpy ( x, boxes.addr(), (size_t)
                  (n * rank2_) * sizeof(double) );
  }
  else
  {
    const idx_t  r2 = rank2_;

    for ( idx_t i = 0; i < n; i++, x += r2 )
    {
      for ( idx_t k = 0; k < r2; k += 2 )
      {
        x[k + 0] = boxes(k + 0,i);
        x[k + 1] = boxes(k + 1,i);
      }
    }
  }
}


//-----------------------------------------------------------------------
//   updateBounds_
//-----------------------------------------------------------------------


void BoxArray::updateBounds_ ()
{
  using jem::min;
  using jem::max;

  const double* JEM_RESTRICT  x = boxes_ .addr ();
  double*       JEM_RESTRICT  b = bounds_.addr ();

  const idx_t  n  = colors_.size ();
  const idx_t  r  = rank_;
  const idx_t  r2 = rank2_;

  idx_t        i, k;


  // Clear the current bounds.

  for ( k = 0; k < r; k++ )
  {
    b[k + 0] =  1.0;
    b[k + r] = -1.0;
  }

  // Find the first regular box and initialize the bounds.

  for ( i = 0; i < n; i++, x += r2 )
  {
    if ( x[0] > x[r] )
    {
      continue;
    }

    for ( k = 0; k < r; k += 2 )
    {
      b[k + 0] = x[k + 0];
      b[k + 1] = x[k + 1];
    }

    break;
  }

  // Loop over the remaining boxes and update the bounds.

  if      ( r == 1_idx )
  {
    for ( ; i < n; i++, x += 2 )
    {
      if ( x[0] > x[1] )
      {
        continue;
      }

      b[0] = min ( b[0], x[0] );
      b[1] = max ( b[1], x[1] );
    }
  }
  else if ( r == 2_idx )
  {
    for ( ; i < n; i++, x += 4 )
    {
      if ( x[0] > x[2] )
      {
        continue;
      }

      b[0] = min ( b[0], x[0] );
      b[1] = min ( b[1], x[1] );
      b[2] = max ( b[2], x[2] );
      b[3] = max ( b[3], x[3] );
    }
  }
  else if ( r == 3_idx )
  {
    for ( ; i < n; i++, x += 6 )
    {
      if ( x[0] > x[3] )
      {
        continue;
      }

      b[0] = min ( b[0], x[0] );
      b[1] = min ( b[1], x[1] );
      b[2] = min ( b[2], x[2] );
      b[3] = max ( b[3], x[3] );
      b[4] = max ( b[4], x[4] );
      b[5] = max ( b[5], x[5] );
    }
  }
  else if ( r >= 4_idx )
  {
    for ( ; i < n; i++, x += r2 )
    {
      if ( x[0] > x[r] )
      {
        continue;
      }

      for ( k = 0; k < r; k++ )
      {
        b[k + 0] = min ( b[k + 0], x[k + 0] );
        b[k + r] = max ( b[k + r], x[k + r] );
      }
    }
  }

  validBounds_ = true;
}


//-----------------------------------------------------------------------
//   indexError_
//-----------------------------------------------------------------------


void BoxArray::indexError_ ( idx_t ibox, idx_t ubound ) const
{
  util::indexError (
    "jive::geom::BoxArray",
    "bounding box",
    ibox,
    ubound
  );
}


JIVE_END_PACKAGE( geom )
