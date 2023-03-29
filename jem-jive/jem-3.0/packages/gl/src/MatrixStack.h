
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

#ifndef JEM_GL_MATRIXSTACK_H
#define JEM_GL_MATRIXSTACK_H

#include <jem/defines.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class MatrixStack
//-----------------------------------------------------------------------


class MatrixStack
{
 public:

  typedef MatrixStack     Self;

  static const idx_t      MAX_DEPTH;


                          MatrixStack   ();
                         ~MatrixStack   ();

  void                    clear         ();

  void                    rotate

    ( GLdouble              angle,
      GLdouble              x,
      GLdouble              y,
      GLdouble              z );

  void                    scale

    ( GLdouble              sx,
      GLdouble              sy,
      GLdouble              sz );

  inline void             translate

    ( GLdouble              dx,
      GLdouble              dy,
      GLdouble              dz = 0.0 );

  void                    transform

    ( const GLdouble*       tr );

  inline GLdouble*        getCurrent    ();

  inline void             setIdentity   ();

  void                    setOrtho

    ( GLdouble              left,
      GLdouble              right,
      GLdouble              bot,
      GLdouble              top,
      GLdouble              near,
      GLdouble              far );

  void                    popMatrix     ();
  void                    pushMatrix    ();


 private:

                          MatrixStack

    ( const Self&           rhs );

  MatrixStack&            operator =

    ( const Self&           rhs );


 private:

  GLdouble*               stack_;
  idx_t                   depth_;
  idx_t                   capacity_;

};



//#######################################################################
//  implementation
//#######################################################################

//-----------------------------------------------------------------------
//   translate
//-----------------------------------------------------------------------


inline void MatrixStack::translate

  ( GLdouble  dx,
    GLdouble  dy,
    GLdouble  dz )

{
  stack_[12] += stack_[0] * dx + stack_[4] * dy + stack_[ 8] * dz;
  stack_[13] += stack_[1] * dx + stack_[5] * dy + stack_[ 9] * dz;
  stack_[14] += stack_[2] * dx + stack_[6] * dy + stack_[10] * dz;
  stack_[15] += stack_[3] * dx + stack_[7] * dy + stack_[11] * dz;
}


//-----------------------------------------------------------------------
//   getCurrent
//-----------------------------------------------------------------------


inline GLdouble* MatrixStack::getCurrent ()
{
  return stack_;
}


//-----------------------------------------------------------------------
//   setIdentity
//-----------------------------------------------------------------------


inline void MatrixStack::setIdentity ()
{
  GLdouble* JEM_RESTRICT  tr = stack_;

  tr[ 0] = 1.0;
  tr[ 1] = 0.0;
  tr[ 2] = 0.0;
  tr[ 3] = 0.0;

  tr[ 4] = 0.0;
  tr[ 5] = 1.0;
  tr[ 6] = 0.0;
  tr[ 7] = 0.0;

  tr[ 8] = 0.0;
  tr[ 9] = 0.0;
  tr[10] = 1.0;
  tr[11] = 0.0;

  tr[12] = 0.0;
  tr[13] = 0.0;
  tr[14] = 0.0;
  tr[15] = 1.0;
}


JEM_END_PACKAGE( gl )

#endif
