
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

#ifndef JEM_GL_BYTE_H
#define JEM_GL_BYTE_H

#include <jem/defines.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   conversion functions
//-----------------------------------------------------------------------


inline GLbyte   toByte    ( float     fval )              noexcept;
inline GLbyte   toByte    ( double    dval )              noexcept;
inline GLubyte  toUbyte   ( float     fval )              noexcept;
inline GLubyte  toUbyte   ( double    dval )              noexcept;
inline float    toFloat   ( GLbyte    bval )              noexcept;
inline float    toFloat   ( GLubyte   bval )              noexcept;
inline double   toDouble  ( GLbyte    bval )              noexcept;
inline double   toDouble  ( GLubyte   bval )              noexcept;

inline void     convert   ( GLbyte&   lhs, float    rhs ) noexcept;
inline void     convert   ( GLbyte&   lhs, double   rhs ) noexcept;
inline void     convert   ( GLubyte&  lhs, float    rhs ) noexcept;
inline void     convert   ( GLubyte&  lhs, double   rhs ) noexcept;
inline void     convert   ( float&    lhs, GLbyte   rhs ) noexcept;
inline void     convert   ( float&    lhs, GLubyte  rhs ) noexcept;
inline void     convert   ( double&   lhs, GLbyte   rhs ) noexcept;
inline void     convert   ( double&   lhs, GLubyte  rhs ) noexcept;




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   toByte
//-----------------------------------------------------------------------


inline GLbyte toByte ( float  fval ) noexcept
{
  int  i = (int) (255.0 * 0.5 * ((double) fval + 1.0)) - 128;

  if      ( i < -128 )
  {
    return (GLbyte) -128;
  }
  else if ( i >  127 )
  {
    return (GLbyte)  127;
  }
  else
  {
    return (GLbyte)    i;
  }
}


inline GLbyte toByte ( double dval ) noexcept
{
  int  i = (int) (255.0 * 0.5 * (dval + 1.0)) - 128;

  if      ( i < -128 )
  {
    return (GLbyte) -128;
  }
  else if ( i >  127 )
  {
    return (GLbyte)  127;
  }
  else
  {
    return (GLbyte)    i;
  }
}


//-----------------------------------------------------------------------
//   toUbyte
//-----------------------------------------------------------------------


inline GLubyte toUbyte ( float fval ) noexcept
{
  int  i = (int) (255.0 * (double) fval);

  if      ( i < 0 )
  {
    return (GLubyte)   0;
  }
  else if ( i > 255 )
  {
    return (GLubyte) 255;
  }
  else
  {
    return (GLubyte)   i;
  }
}


inline GLubyte toUbyte ( double dval ) noexcept
{
  int  i = (int) (255.0 * dval);

  if      ( i < 0 )
  {
    return (GLubyte)   0;
  }
  else if ( i > 255 )
  {
    return (GLubyte) 255;
  }
  else
  {
    return (GLubyte)   i;
  }
}


//-----------------------------------------------------------------------
//   toFloat
//-----------------------------------------------------------------------


inline float toFloat ( GLbyte bval ) noexcept
{
  return (float) (2.0 * (((int) bval + 128) / 255.0) - 1.0);
}


inline float toFloat ( GLubyte bval ) noexcept
{
  return (float) (bval / 255.0);
}


//-----------------------------------------------------------------------
//   toDouble
//-----------------------------------------------------------------------


inline double toDouble ( GLbyte bval ) noexcept
{
  return (double) (2.0 * (((int) bval + 128) / 255.0) - 1.0);
}


inline double toDouble ( GLubyte bval ) noexcept
{
  return (double) (bval / 255.0);
}


//-----------------------------------------------------------------------
//   convert
//-----------------------------------------------------------------------


inline void  convert ( GLbyte&   lhs, float    rhs ) noexcept
{
  lhs = toByte ( rhs );
}

inline void  convert ( GLbyte&   lhs, double   rhs ) noexcept
{
  lhs = toByte ( rhs );
}

inline void  convert ( GLubyte&  lhs, float    rhs ) noexcept
{
  lhs = toUbyte ( rhs );
}

inline void  convert ( GLubyte&  lhs, double   rhs ) noexcept
{
  lhs = toUbyte ( rhs );
}

inline void  convert ( float&    lhs, GLbyte   rhs ) noexcept
{
  lhs = toFloat ( rhs );
}

inline void  convert ( float&    lhs, GLubyte  rhs ) noexcept
{
  lhs = toFloat ( rhs );
}

inline void  convert ( double&   lhs, GLbyte   rhs ) noexcept
{
  lhs = toDouble ( rhs );
}

inline void  convert ( double&   lhs, GLubyte  rhs ) noexcept
{
  lhs = toDouble ( rhs );
}


JEM_END_PACKAGE( gl )

#endif
