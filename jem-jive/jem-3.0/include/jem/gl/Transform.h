
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

#ifndef JEM_GL_TRANSFORM_H
#define JEM_GL_TRANSFORM_H

#include <jem/base/assert.h>
#include <jem/gl/Vertex.h>


JEM_BEGIN_PACKAGE( gl )


class Rect;


//-----------------------------------------------------------------------
//   class Transform
//-----------------------------------------------------------------------


class Transform
{
 public:

                            Transform     ()       noexcept;

                            Transform

    ( const Transform&        rhs )                noexcept;

  Transform&                operator =

    ( const Transform&        rhs )                noexcept;

  void                      swap

    ( Transform&              rhs )                noexcept;

  inline GLdouble&          operator ()

    ( int                     i,
      int                     j );

  inline GLdouble           operator ()

    ( int                     i,
      int                     j )            const;

  void                      setIdentity   ()       noexcept;

  inline void               scale

    ( GLdouble                s );

  void                      scale

    ( GLdouble                sx,
      GLdouble                sy,
      GLdouble                sz = 1.0 );

  inline void               setScale

    ( GLdouble                s );

  void                      setScale

    ( GLdouble                sx,
      GLdouble                sy,
      GLdouble                sz = 1.0 );

  GLdouble                  getScale      () const noexcept;

  inline void               translate

    ( GLdouble                dx,
      GLdouble                dy,
      GLdouble                dz = 0.0 )           noexcept;

  inline void               setTranslation

    ( GLdouble                dx,
      GLdouble                dy,
      GLdouble                dz = 0.0 )           noexcept;

  void                      rotate

    ( GLdouble                a )                  noexcept;

  void                      rotate

    ( GLdouble                a,
      GLdouble                x,
      GLdouble                y );

  void                      rotate

    ( GLdouble                a,
      GLdouble                x,
      GLdouble                y,
      GLdouble                z );

  void                      setRotation

    ( GLdouble                a )                  noexcept;

  void                      setRotation

    ( GLdouble                a,
      GLdouble                x,
      GLdouble                y );

  void                      setRotation

    ( GLdouble                a,
      GLdouble                x,
      GLdouble                y,
      GLdouble                z );

  void                      transform

    ( const Transform&        tr )                 noexcept;

  Transform                 transpose     () const noexcept;

  static Transform          identity      ()       noexcept;

  static Transform          scaling

    ( GLdouble                s );

  static Transform          scaling

    ( GLdouble                sx,
      GLdouble                sy,
      GLdouble                sz = 1.0 );

  static Transform          translation

    ( GLdouble                dx,
      GLdouble                dy,
      GLdouble                dz = 0.0 )           noexcept;

  static Transform          rotation

    ( GLdouble                a )                  noexcept;

  static Transform          rotation

    ( GLdouble                a,
      GLdouble                x,
      GLdouble                y );

  static Transform          rotation

    ( GLdouble                a,
      GLdouble                x,
      GLdouble                y,
      GLdouble                z );

  void                      configure2D

    ( const Properties&       props );

  void                      getConfig2D

    ( const Properties&       props )        const;

  static void               listProps2D

    ( const Properties&       props );

  void                      configure3D

    ( const Properties&       props );

  void                      getConfig3D

    ( const Properties&       props )        const;

  static void               listProps3D

    ( const Properties&       props );

  inline       GLdouble*    addr          ()       noexcept;
  inline const GLdouble*    addr          () const noexcept;


 private:

  GLdouble                  data_[16];

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


inline void                 swap

  ( Transform&                lhs,
    Transform&                rhs )                noexcept;

void                        print

  ( TextOutput&               out,
    const Transform&          tr );

inline Vertex2d             matmul

  ( const Transform&          a,
    const Vertex2d&           b )                  noexcept;

inline Vertex3d             matmul

  ( const Transform&          a,
    const Vertex3d&           b )                  noexcept;

inline Vertex4d             matmul

  ( const Transform&          a,
    const Vertex4d&           b )                  noexcept;

inline Transform            matmul

  ( const Transform&          a,
    const Transform&          b )                  noexcept;

void                        matmul

  ( Vertex2d&                 r,
    const Transform&          a,
    const Vertex2d&           b )                  noexcept;

void                        matmul

  ( Vertex3d&                 r,
    const Transform&          a,
    const Vertex3d&           b )                  noexcept;

void                        matmul

  ( Vertex4d&                 r,
    const Transform&          a,
    const Vertex4d&           b )                  noexcept;

void                        matmul

  ( Transform&                r,
    const Transform&          a,
    const Transform&          b )                  noexcept;

Vertex3d                    project

  ( const Transform&          p,
    const Vertex3d&           v );

bool                        project

  ( Vertex3d&                 d,
    const Transform&          p,
    const Vertex3d&           v );

bool                        project

  ( Vertex2d&                 d,
    const Transform&          p,
    const Vertex3d&           v );

Vertex3d                    unproject

  ( const Transform&          p,
    const Vertex3d&           d );

bool                        unproject

  ( Vertex3d&                 v,
    const Transform&          p,
    const Vertex3d&           d );

void                        toWindow

  ( Vertex3d&                 v,
    const Rect&               vport );

void                        fromWindow

  ( Vertex2d&                 v,
    const Rect&               vport );

void                        fromWindow

  ( Vertex3d&                 v,
    const Rect&               vport );

Transform                   inverse

  ( const Transform&          tr );

bool                        invert

  ( Transform&                tr );

GLdouble                    reldiff

  ( const Transform&          a,
    const Transform&          b )                  noexcept;



//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Transform
//=======================================================================

//-----------------------------------------------------------------------
//   subscript operators
//-----------------------------------------------------------------------


inline GLdouble& Transform::operator () ( int i, int j )
{
  JEM_ASSERT ( i >= 0 && i < 4 && j >= 0 && j < 4 );

  return data_[i + 4 * j];
}


inline GLdouble Transform::operator () ( int i, int j ) const
{
  JEM_ASSERT ( i >= 0 && i < 4 && j >= 0 && j < 4 );

  return data_[i + 4 * j];
}


//-----------------------------------------------------------------------
//   scale
//-----------------------------------------------------------------------


inline void Transform::scale ( GLdouble s )
{
  scale ( s, s, s );
}


//-----------------------------------------------------------------------
//   setScale
//-----------------------------------------------------------------------


inline void Transform::setScale ( GLdouble s )
{
  setScale ( s, s, s );
}


//-----------------------------------------------------------------------
//   translate
//-----------------------------------------------------------------------


inline void Transform::translate

  ( GLdouble  dx,
    GLdouble  dy,
    GLdouble  dz ) noexcept

{
  data_[12] += dx;
  data_[13] += dy;
  data_[14] += dz;
}


//-----------------------------------------------------------------------
//   setTranslation
//-----------------------------------------------------------------------


void Transform::setTranslation

  ( GLdouble  dx,
    GLdouble  dy,
    GLdouble  dz ) noexcept

{
  data_[12] = dx;
  data_[13] = dy;
  data_[14] = dz;
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


inline GLdouble* Transform::addr () noexcept
{
  return data_;
}


inline const GLdouble* Transform::addr () const noexcept
{
  return data_;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void               swap

  ( Transform&              lhs,
    Transform&              rhs ) noexcept

{
  lhs.swap ( rhs );
}


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


inline Vertex2d           matmul

  ( const Transform&        a,
    const Vertex2d&         b ) noexcept

{
  Vertex2d  r;

  matmul ( r, a, b );

  return r;
}


inline Vertex3d           matmul

  ( const Transform&        a,
    const Vertex3d&         b ) noexcept

{
  Vertex3d  r;

  matmul ( r, a, b );

  return r;
}


inline Vertex4d           matmul

  ( const Transform&        a,
    const Vertex4d&         b ) noexcept

{
  Vertex4d  r;

  matmul ( r, a, b );

  return r;
}


inline Transform          matmul

  ( const Transform&        a,
    const Transform&        b ) noexcept

{
  Transform  r;

  matmul ( r, a, b );

  return r;
}


//-----------------------------------------------------------------------
//   unproject
//-----------------------------------------------------------------------


inline Vertex3d           unproject

  ( const Transform&        p,
    const Vertex3d&         d )

{
  return project ( inverse(p), d );
}


JEM_END_PACKAGE( gl )

#endif
