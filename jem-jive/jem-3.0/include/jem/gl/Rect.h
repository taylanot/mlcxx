
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

#ifndef JEM_GL_RECT_H
#define JEM_GL_RECT_H

#include <jem/gl/import.h>
#include <jem/gl/Size.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class Rect
//-----------------------------------------------------------------------


class Rect
{
 public:

  inline                    Rect          ()       noexcept;

  explicit inline           Rect

    ( GLsizei                 size )               noexcept;

  explicit inline           Rect

    ( const Size2i&           size )               noexcept;

  inline                    Rect

    ( GLsizei                 width,
      GLsizei                 height )             noexcept;

  inline                    Rect

    ( GLint                   x,
      GLint                   y,
      const Size2i&           size )               noexcept;

  inline                    Rect

    ( GLint                   x,
      GLint                   y,
      GLsizei                 width,
      GLsizei                 height )             noexcept;

  inline                    Rect

    ( const Rect&             rhs )                noexcept;

  inline Rect&              operator =

    ( const Rect&             rhs )                noexcept;

  void                      swap

    ( Rect&                   rhs )                noexcept;

  inline Size2i             size          () const noexcept;

  inline void               resize

    ( const Size2i&           s )                  noexcept;

  inline bool               isValid       () const noexcept;
  inline bool               isEmpty       () const noexcept;
  inline void               makeEmpty     ()       noexcept;
  inline GLdouble           aspectRatio   () const noexcept;
  inline bool               contains

    ( GLint                   x,
      GLint                   y )            const noexcept;


 public:

  GLint                     x;
  GLint                     y;
  GLsizei                   width;
  GLsizei                   height;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


inline void                 swap

  ( Rect&                     lhs,
    Rect&                     rhs )                noexcept;

void                        print

  ( TextOutput&               out,
    const Rect&               rect );




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Rect
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline Rect::Rect () noexcept
{
  x = y  = 0;
  width  = 0;
  height = 0;
}


inline Rect::Rect ( GLsizei s ) noexcept
{
  x = y  = 0;
  width  = s;
  height = s;
}


inline Rect::Rect ( const Size2i& s ) noexcept
{
  x = y  = 0;
  width  = s.width;
  height = s.height;
}


inline Rect::Rect ( GLsizei w, GLsizei h ) noexcept
{
  x = y  = 0;
  width  = w;
  height = h;
}


inline Rect::Rect ( GLint x, GLint y, const Size2i& s ) noexcept
{
  this->x = x;
  this->y = y;
  width   = s.width;
  height  = s.height;
}


inline Rect::Rect ( GLint   x, GLint   y,
                    GLsizei w, GLsizei h ) noexcept
{
  this->x = x;
  this->y = y;
  width   = w;
  height  = h;
}


inline Rect::Rect ( const Rect& rhs ) noexcept
{
  x      = rhs.x;
  y      = rhs.y;
  width  = rhs.width;
  height = rhs.height;
}


//-----------------------------------------------------------------------
//   assignment operator
//-----------------------------------------------------------------------


inline Rect& Rect::operator = ( const Rect& rhs ) noexcept
{
  x      = rhs.x;
  y      = rhs.y;
  width  = rhs.width;
  height = rhs.height;

  return *this;
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline Size2i Rect::size () const noexcept
{
  return Size2i( width, height );
}


//-----------------------------------------------------------------------
//   resize
//-----------------------------------------------------------------------


inline void Rect::resize ( const Size2i& s ) noexcept
{
  width  = s.width;
  height = s.height;
}


//-----------------------------------------------------------------------
//   isValid
//-----------------------------------------------------------------------


inline bool Rect::isValid () const noexcept
{
  return (width >= 0 && height >= 0);
}


//-----------------------------------------------------------------------
//   isEmpty
//-----------------------------------------------------------------------


inline bool Rect::isEmpty () const noexcept
{
  return (width <= 0 || height <= 0);
}


//-----------------------------------------------------------------------
//   makeEmpty
//-----------------------------------------------------------------------


inline void Rect::makeEmpty () noexcept
{
  x = y  = 0;
  width  = 0;
  height = 0;
}


//-----------------------------------------------------------------------
//   aspectRatio
//-----------------------------------------------------------------------


inline GLdouble Rect::aspectRatio () const noexcept
{
  return (height + 0.001) / (width + 0.001);
}


//-----------------------------------------------------------------------
//   contains
//-----------------------------------------------------------------------


inline bool Rect::contains

  ( GLint  x,
    GLint  y ) const noexcept

{
  GLint  xMin = this->x;
  GLint  yMin = this->y;
  GLint  xMax = xMin + width;
  GLint  yMax = yMin + height;

  return (x >= xMin && x < xMax &&
          y >= yMin && y < yMax);
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void swap ( Rect& lhs, Rect& rhs ) noexcept
{
  lhs.swap ( rhs );
}


JEM_END_PACKAGE( gl )

#endif
