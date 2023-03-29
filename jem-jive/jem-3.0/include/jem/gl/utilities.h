
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

#ifndef JEM_GL_UTILITIES_H
#define JEM_GL_UTILITIES_H

#include <jem/util/utilities.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


using jem::util::clamp;
using jem::util::clamped;


//-----------------------------------------------------------------------
//   global constants
//-----------------------------------------------------------------------


extern const GLdouble     RADS_PER_DEGREE;
extern const GLdouble     DEGREES_PER_RAD;


void                      round

  ( float&                  val,
    float                   step )         noexcept;

void                      round

  ( double&                 val,
    double                  step )         noexcept;

inline float              rounded

  ( float                   val,
    float                   step )         noexcept;

inline double             rounded

  ( double                  val,
    double                  step )         noexcept;

inline void               clamp

  ( float&                  t )            noexcept;

inline void               clamp

  ( double&                 t )            noexcept;

inline float              clamped

  ( float                   t )            noexcept;

inline double             clamped

  ( double                  t )            noexcept;

void                      adjustTickCount

  ( int&                    nticks )       noexcept;



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   rounded
//-----------------------------------------------------------------------


inline float rounded ( float val, float step ) noexcept
{
  round ( val, step );

  return val;
}


inline double rounded ( double val, double step ) noexcept
{
  round ( val, step );

  return val;
}


//-----------------------------------------------------------------------
//   clamp
//-----------------------------------------------------------------------


inline void clamp ( float& t ) noexcept
{
  clamp ( t, 0.0F, 1.0F );
}


inline void clamp ( double& t ) noexcept
{
  clamp ( t, 0.0, 1.0 );
}


//-----------------------------------------------------------------------
//   clamped
//-----------------------------------------------------------------------


inline float clamped ( float t ) noexcept
{
  return clamped ( t, 0.0F, 1.0F );
}


inline double clamped ( double t ) noexcept
{
  return clamped ( t, 0.0, 1.0 );
}


JEM_END_PACKAGE( gl )

#endif
