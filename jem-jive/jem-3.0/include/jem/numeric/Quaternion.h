
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

#ifndef JEM_NUMERIC_QUATERNION_H
#define JEM_NUMERIC_QUATERNION_H

#include <jem/base/array/Array.h>


JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   class Quaternion
//-----------------------------------------------------------------------


class Quaternion
{
 public:

  typedef Tuple<double,3>     Vec3;
  typedef Tuple<double,4>     Vec4;
  typedef Tuple<double,3,3>   Mat3x3;
  typedef Array<double,1>     Vector;
  typedef Array<double,2>     Matrix;


  static inline void          qmult

    ( Vec4&                     r,
      const Vec4&               a,
      const Vec4&               b )  noexcept;

  static inline void          qmult

    ( const Vector&             r,
      const Vector&             a,
      const Vector&             b );

  static void                 qmult

    ( const Matrix&             r,
      const Vec4&               a,
      const Matrix&             b );

  static void                 qmult

    ( const Matrix&             r,
      const Vector&             a,
      const Matrix&             b );

  static void                 qmult

    ( const Matrix&             r,
      const Matrix&             a,
      const Matrix&             b );

  static void                 vmult

    ( Vec3&                     r,
      const Vec4&               a,
      const Vec3&               b )  noexcept;

  static void                 vmult

    ( const Vector&             r,
      const Vector&             a,
      const Vector&             b );

  static void                 vmult

    ( const Matrix&             r,
      const Vec4&               a,
      const Matrix&             b );

  static void                 vmult

    ( const Matrix&             r,
      const Vector&             a,
      const Matrix&             b );

  static void                 vmult

    ( const Matrix&             r,
      const Matrix&             a,
      const Matrix&             b );

  static inline void          invert

    ( Vec4&                     q )  noexcept;

  static inline void          invert

    ( const Vector&             q );

  static void                 invert

    ( const Matrix&             q );

  static void                 normalize

    ( Vec4&                     q );

  static void                 normalize

    ( const Vector&             q );

  static void                 normalize

    ( const Matrix&             q );

  static void                 toRotmat

    ( Mat3x3&                   rm,
      const Vec4&               q )  noexcept;

  static void                 toRotmat

    ( const Matrix&             rm,
      const Vector&             q );

  static void                 toRotvec

    ( Vec4&                     rv,
      const Vec4&               q )  noexcept;

  static void                 toRotvec

    ( const Vector&             rv,
      const Vector&             q );

  static void                 fromRotvec

    ( Vec4&                     q,
      const Vec4&               rv );

  static void                 fromRotvec

    ( const Vector&             q,
      const Vector&             rv );


 private:

  static void                 normalizeError_

    ( const char*               where );

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   JIVE_QUATERNION_QMULT
//-----------------------------------------------------------------------


#undef  JIVE_QUATERNION_QMULT
#define JIVE_QUATERNION_QMULT( r, a, b )                        \
                                                                \
  r[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]; \
  r[1] = a[0] * b[1] + b[0] * a[1] + a[2] * b[3] - a[3] * b[2]; \
  r[2] = a[0] * b[2] + b[0] * a[2] + a[3] * b[1] - a[1] * b[3]; \
  r[3] = a[0] * b[3] + b[0] * a[3] + a[1] * b[2] - a[2] * b[1]


//-----------------------------------------------------------------------
//   qmult
//-----------------------------------------------------------------------


inline void Quaternion::qmult

  ( Vec4&          r,
    const Vec4&    a,
    const Vec4&    b ) noexcept

{
  JIVE_QUATERNION_QMULT( r, a, b );
}


inline void Quaternion::qmult

  ( const Vector&  r,
    const Vector&  a,
    const Vector&  b )

{
  JIVE_QUATERNION_QMULT( r, a, b );
}


//-----------------------------------------------------------------------
//   invert
//-----------------------------------------------------------------------


inline void Quaternion::invert ( Vec4& q ) noexcept
{
  q[1] = -q[1];
  q[2] = -q[2];
  q[3] = -q[3];
}


inline void Quaternion::invert ( const Vector& q )
{
  q[1] = -q[1];
  q[2] = -q[2];
  q[3] = -q[3];
}


JEM_END_PACKAGE( numeric )

#endif
