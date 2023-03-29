
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


#include <cmath>
#include <cstring>
#include <jem/base/System.h>
#include <jem/base/MemCache.h>
#include "MatrixStack.h"


JEM_BEGIN_PACKAGE( gl )


extern const GLdouble  RADS_PER_DEGREE;


//=======================================================================
//   class MatrixStack
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const idx_t  MatrixStack::MAX_DEPTH = 128;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


MatrixStack::MatrixStack ()
{
  stack_    = (GLdouble*) MemCache::alloc ( 16 * sizeof(*stack_) );
  depth_    = 0;
  capacity_ = 0;

  clear ();
}


MatrixStack::~MatrixStack ()
{
  size_t  n = (size_t) capacity_;

  MemCache::dealloc ( stack_, 16 * (n + 1) * sizeof(*stack_) );

  stack_ = nullptr;
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void MatrixStack::clear ()
{
  depth_ = 0;

  setIdentity ();
}


//-----------------------------------------------------------------------
//   rotate
//-----------------------------------------------------------------------


void MatrixStack::rotate

  ( GLdouble  angle,
    GLdouble  x,
    GLdouble  y,
    GLdouble  z )

{
  GLdouble  tr[16];
  GLdouble  a, b, c;


  a = (GLdouble) std::sqrt ( (double) (x * x + y * y + z * z) );

  if ( a <= 0.0 )
  {
    return;
  }

  a  = 1.0 / a;

  x *= a;
  y *= a;
  z *= a;

  a  = (GLdouble) std::cos ( (double) (angle * RADS_PER_DEGREE) );
  b  = (GLdouble) std::sin ( (double) (angle * RADS_PER_DEGREE) );
  c  = (GLdouble) (1.0 - a);

  tr[ 0] = x * x * c +     a;
  tr[ 1] = x * y * c + z * b;
  tr[ 2] = x * z * c - y * b;
  tr[ 3] = 0.0;

  tr[ 4] = x * y * c - z * b;
  tr[ 5] = y * y * c +     a;
  tr[ 6] = y * z * c + x * b;
  tr[ 7] = 0.0;

  tr[ 8] = x * z * c + y * b;
  tr[ 9] = y * z * c - x * b;
  tr[10] = z * z * c +     a;
  tr[11] = 0.0;

  tr[12] = 0.0;
  tr[13] = 0.0;
  tr[14] = 0.0;
  tr[15] = 1.0;

  transform ( tr );
}


//-----------------------------------------------------------------------
//   scale
//-----------------------------------------------------------------------


void MatrixStack::scale

  ( GLdouble  sx,
    GLdouble  sy,
    GLdouble  sz )

{
  GLdouble* JEM_RESTRICT  tr = stack_;

  tr[ 0] *= sx;
  tr[ 1] *= sx;
  tr[ 2] *= sx;
  tr[ 3] *= sx;

  tr[ 4] *= sy;
  tr[ 5] *= sy;
  tr[ 6] *= sy;
  tr[ 7] *= sy;

  tr[ 8] *= sx;
  tr[ 9] *= sy;
  tr[10] *= sz;
  tr[11] *= sz;
}


//-----------------------------------------------------------------------
//   transform
//-----------------------------------------------------------------------


void MatrixStack::transform ( const GLdouble* tr )
{
  GLdouble* JEM_RESTRICT  st = stack_;

  GLdouble                rt[16];


  rt[ 0] = (st[ 0] * tr[ 0] + st[ 4] * tr[ 1] +
            st[ 8] * tr[ 2] + st[12] * tr[ 3]);
  rt[ 1] = (st[ 1] * tr[ 0] + st[ 5] * tr[ 1] +
            st[ 9] * tr[ 2] + st[13] * tr[ 3]);
  rt[ 2] = (st[ 2] * tr[ 0] + st[ 6] * tr[ 1] +
            st[10] * tr[ 2] + st[14] * tr[ 3]);
  rt[ 3] = (st[ 3] * tr[ 0] + st[ 7] * tr[ 1] +
            st[11] * tr[ 2] + st[15] * tr[ 3]);

  rt[ 4] = (st[ 0] * tr[ 4] + st[ 4] * tr[ 5] +
            st[ 8] * tr[ 6] + st[12] * tr[ 7]);
  rt[ 5] = (st[ 1] * tr[ 4] + st[ 5] * tr[ 5] +
            st[ 9] * tr[ 6] + st[13] * tr[ 7]);
  rt[ 6] = (st[ 2] * tr[ 4] + st[ 6] * tr[ 5] +
            st[10] * tr[ 6] + st[14] * tr[ 7]);
  rt[ 7] = (st[ 3] * tr[ 4] + st[ 7] * tr[ 5] +
            st[11] * tr[ 6] + st[15] * tr[ 7]);

  rt[ 8] = (st[ 0] * tr[ 8] + st[ 4] * tr[ 9] +
            st[ 8] * tr[10] + st[12] * tr[11]);
  rt[ 9] = (st[ 1] * tr[ 8] + st[ 5] * tr[ 9] +
            st[ 9] * tr[10] + st[13] * tr[11]);
  rt[10] = (st[ 2] * tr[ 8] + st[ 6] * tr[ 9] +
            st[10] * tr[10] + st[14] * tr[11]);
  rt[11] = (st[ 3] * tr[ 8] + st[ 7] * tr[ 9] +
            st[11] * tr[10] + st[15] * tr[11]);

  rt[12] = (st[ 0] * tr[12] + st[ 4] * tr[13] +
            st[ 8] * tr[14] + st[12] * tr[15]);
  rt[13] = (st[ 1] * tr[12] + st[ 5] * tr[13] +
            st[ 9] * tr[14] + st[13] * tr[15]);
  rt[14] = (st[ 2] * tr[12] + st[ 6] * tr[13] +
            st[10] * tr[14] + st[14] * tr[15]);
  rt[15] = (st[ 3] * tr[12] + st[ 7] * tr[13] +
            st[11] * tr[14] + st[15] * tr[15]);

  std::memcpy ( st, rt, sizeof(rt) );
}


//-----------------------------------------------------------------------
//   setOrtho
//-----------------------------------------------------------------------


void MatrixStack::setOrtho

  ( GLdouble  left,
    GLdouble  right,
    GLdouble  bot,
    GLdouble  top,
    GLdouble  near,
    GLdouble  far )

{
  GLdouble* JEM_RESTRICT  tr = stack_;


  tr[ 0] =  2.0 / (right - left);
  tr[ 1] =  0.0;
  tr[ 2] =  0.0;
  tr[ 3] =  0.0;

  tr[ 4] =  0.0;
  tr[ 5] =  2.0 / (top - bot);
  tr[ 6] =  0.0;
  tr[ 7] =  0.0;

  tr[ 8] =  0.0;
  tr[ 9] =  0.0;
  tr[10] = -2.0 / (far - near);
  tr[11] =  0.0;

  tr[12] = -(right + left) / (right - left);
  tr[13] = -(top   + bot)  / (top   - bot);
  tr[14] = -(far   + near) / (far   - near);
  tr[15] =  1.0;
}


//-----------------------------------------------------------------------
//   popMatrix
//-----------------------------------------------------------------------


void MatrixStack::popMatrix ()
{
  if ( depth_ > 0 )
  {
    std::memcpy ( stack_, stack_ + (16 * depth_),
                  16 * sizeof(*stack_) );

    depth_--;
  }
}


//-----------------------------------------------------------------------
//   pushMatrix
//-----------------------------------------------------------------------


void MatrixStack::pushMatrix ()
{
  if ( depth_ >= MAX_DEPTH )
  {
    System::err() << JEM_FUNC << ": matrix stack overflow.\n";
  }
  else
  {
    if ( depth_ >= capacity_ )
    {
      idx_t  cap = 2 * depth_ + 1;

      if ( cap > MAX_DEPTH )
      {
        cap = MAX_DEPTH;
      }

      stack_    = (GLdouble*) MemCache::realloc

        ( stack_,
          16 * (size_t) (capacity_ + 1) * sizeof(*stack_),
          16 * (size_t) (cap       + 1) * sizeof(*stack_) );

      capacity_ = cap;
    }

    depth_++;

    std::memcpy ( stack_ + (16 * depth_),
                  stack_,   16 * sizeof(*stack_) );
  }
}


JEM_END_PACKAGE( gl )
