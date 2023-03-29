
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

#ifndef JEM_GL_VERTEX_H
#define JEM_GL_VERTEX_H

#include <cmath>
#include <jem/base/tuple/Tuple.h>
#include <jem/gl/gl.h>
#include <jem/gl/import.h>


namespace jem
{
  class String;
}


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class Vertex
//-----------------------------------------------------------------------


class Vertex
{
 public:

  template <class T>

    static inline bool      configure

    ( Tuple<T,2>&             vx,
      const String&           name,
      const Properties&       props );

  template <class T>

    static inline bool      configure

    ( Tuple<T,3>&             vx,
      const String&           name,
      const Properties&       props );

  template <class T>

    static inline void      getConfig

    ( const String&           name,
      const Properties&       props,
      const Tuple<T,2>&       vx );

  template <class T>

    static inline void      getConfig

    ( const String&           name,
      const Properties&       props,
      const Tuple<T,3>&       vx );


 private:

  static bool               configure_

    ( double&                 x,
      double&                 y,
      const String&           name,
      const Properties&       props );

  static bool               configure_

    ( double&                 x,
      double&                 y,
      double&                 z,
      const String&           name,
      const Properties&       props );

  static void               getConfig_

    ( const String&           name,
      const Properties&       props,
      double                  x,
      double                  y );

  static void               getConfig_

    ( const String&           name,
      const Properties&       props,
      double                  x,
      double                  y,
      double                  z );

};


//-----------------------------------------------------------------------
//   Vertex types
//-----------------------------------------------------------------------


typedef Tuple<GLint,2>      Vertex2i;
typedef Tuple<GLfloat,2>    Vertex2f;
typedef Tuple<GLdouble,2>   Vertex2d;
typedef Tuple<GLint,3>      Vertex3i;
typedef Tuple<GLfloat,3>    Vertex3f;
typedef Tuple<GLdouble,3>   Vertex3d;
typedef Tuple<GLint,4>      Vertex4i;
typedef Tuple<GLfloat,4>    Vertex4f;
typedef Tuple<GLdouble,4>   Vertex4d;



//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


GLdouble                    getAngle

  ( const Vertex2d&           e1,
    const Vertex2d&           e2 );

template <class T, int N>

  inline T                  normalize

  ( Tuple<T,N>&               vx );

template <class T>

  inline T                  getNormal

  ( Tuple<T,3>&               n,
    const Tuple<T,3>&         a,
    const Tuple<T,3>&         b );





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Vertex
//=======================================================================

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


template <class T>

  inline bool Vertex::configure

  ( Tuple<T,2>&        vx,
    const String&      name,
    const Properties&  props )

{
  double  x, y;

  if ( configure_( x, y, name, props ) )
  {
    vx[0] = (T) x;
    vx[1] = (T) y;

    return true;
  }
  else
  {
    return false;
  }
}


template <class T>

  inline bool Vertex::configure

  ( Tuple<T,3>&        vx,
    const String&      name,
    const Properties&  props )

{
  double  x, y, z;

  if ( configure_( x, y, z, name, props ) )
  {
    vx[0] = (T) x;
    vx[1] = (T) y;
    vx[2] = (T) z;

    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


template <class T>

  inline void Vertex::getConfig

  ( const String&      name,
    const Properties&  props,
    const Tuple<T,2>&  vx )

{
  getConfig_ ( name, props, vx[0], vx[1] );
}


template <class T>

  inline void Vertex::getConfig

  ( const String&      name,
    const Properties&  props,
    const Tuple<T,3>&  vx )

{
  getConfig_ ( name, props, vx[0], vx[1], vx[2] );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   normalize
//-----------------------------------------------------------------------


template <class T, int N>

  inline T normalize ( Tuple<T,N>& vx )

{
  double  a, b;

  a = (double) vx[0] * (double) vx[0];

  for ( int i = 1; i < N; i++ )
  {
    a += ((double) vx[i] * (double) vx[i]);
  }

  a = std::sqrt ( a );

  if ( a > 0.0 )
  {
    b = 1.0 / a;

    for ( int i = 0; i < N; i++ )
    {
      vx[i] = (T) (vx[i] * b);
    }
  }

  return (T) a;
}


//-----------------------------------------------------------------------
//   normalize ( Tuple<T,2> )
//-----------------------------------------------------------------------


template <class T>

  inline T normalize ( Tuple<T,2>& vx )

{
  double  a, b;

  a = ( (double) vx[0] * (double) vx[0] +
        (double) vx[1] * (double) vx[1] );

  a = std::sqrt ( a );

  if ( a > 0.0 )
  {
    b = 1.0 / a;

    vx[0] = (T) (vx[0] * b);
    vx[1] = (T) (vx[1] * b);
  }

  return (T) a;
}


//-----------------------------------------------------------------------
//   normalize ( Tuple<T,3> )
//-----------------------------------------------------------------------


template <class T>

  inline T normalize ( Tuple<T,3>& vx )

{
  double  a, b;

  a = ( (double) vx[0] * (double) vx[0] +
        (double) vx[1] * (double) vx[1] +
        (double) vx[2] * (double) vx[2] );

  a = std::sqrt ( a );

  if ( a > 0.0 )
  {
    b = 1.0 / a;

    vx[0] = (T) (vx[0] * b);
    vx[1] = (T) (vx[1] * b);
    vx[2] = (T) (vx[2] * b);
  }

  return (T) a;
}


//-----------------------------------------------------------------------
//   getNormal
//-----------------------------------------------------------------------


template <class T>

  inline T           getNormal

  ( Tuple<T,3>&         n,
    const Tuple<T,3>&   a,
    const Tuple<T,3>&   b )

{
  n[0] = a[1] * b[2] - a[2] * b[1];
  n[1] = a[2] * b[0] - a[0] * b[2];
  n[2] = a[0] * b[1] - a[1] * b[0];

  return normalize ( n );
}


JEM_END_PACKAGE( gl )

#endif
