
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

#ifndef JEM_GL_TEXTPAINTER_H
#define JEM_GL_TEXTPAINTER_H

#include <jem/gl/Font.h>
#include <jem/gl/Size.h>
#include <jem/gl/Alignment.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class TextPainter
//-----------------------------------------------------------------------


class TextPainter
{
 public:

  explicit                  TextPainter

    ( const Font&             font   = Font (),
      Alignment               hAlign = LEFT_ALIGNED,
      Alignment               vAlign = BOT_ALIGNED );

  inline void               begin

    ( GLContext&              gtx );

  inline void               end

    ( GLContext&              gtx );

  GLsizei                   write

    ( GLContext&              gtx,
      char                    ch );

  inline GLsizei            writeLine

    ( GLContext&              gtx,
      const char*             str );

  inline GLsizei            writeLine

    ( GLContext&              gtx,
      const String&           str );

  GLsizei                   writeLine

    ( GLContext&              gtx,
      const char*             str,
      idx_t                   len );

  Size2i                    writeBlock

    ( GLContext&              gtx,
      const String&           text );

  void                      writeBlock

    ( GLContext&              gtx,
      const String&           text,
      const Size2i&           size );

  Size2i                    getSizeOf

    ( const String&           text )       const;

  inline void               setCursor

    ( GLint                   x,
      GLint                   y )                noexcept;

  inline void               setCursor

    ( GLint                   x,
      GLint                   y,
      GLfloat                 z )                noexcept;


 public:

  Font                      font;
  GLint                     xPos;
  GLint                     yPos;
  GLfloat                   zPos;
  GLfloat                   angle;
  Alignment                 hAlignment;
  Alignment                 vAlignment;


 private:

                            TextPainter

    ( const TextPainter&      rhs );

  TextPainter&              operator =

    ( const TextPainter&      rhs );

  inline void               rotate_

    ( GLContext&              gtx );


 private:

  static const GLfloat      HALIGN_[3];
  static const GLfloat      VALIGN_[3];
  static const GLfloat      LINE_SKIP_;

};


//-----------------------------------------------------------------------
//   class TextScope
//-----------------------------------------------------------------------


class TextScope : private FontScope
{
 public:

  explicit inline           TextScope

    ( GLContext&              gtx,
      TextPainter&            txp );

};




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class TextPainter
//=======================================================================

//-----------------------------------------------------------------------
//   begin
//-----------------------------------------------------------------------


inline void TextPainter::begin ( GLContext& gtx )
{
  font.begin ( gtx );
}


//-----------------------------------------------------------------------
//   end
//-----------------------------------------------------------------------


inline void TextPainter::end ( GLContext& gtx )
{
  font.end ( gtx );
}


//-----------------------------------------------------------------------
//   writeLine
//-----------------------------------------------------------------------


inline GLsizei TextPainter::writeLine

  ( GLContext&   gtx,
    const char*  str )

{
  return writeLine ( gtx, str, String::sizeOf( str ) );
}


inline GLsizei TextPainter::writeLine

  ( GLContext&     gtx,
    const String&  str )

{
  return writeLine ( gtx, str.addr(), str.size() );
}


//-----------------------------------------------------------------------
//   setCursor
//-----------------------------------------------------------------------


inline void TextPainter::setCursor

  ( GLint  x,
    GLint  y ) noexcept

{
  xPos = x;
  yPos = y;
  zPos = 0.0F;
}


inline void TextPainter::setCursor

  ( GLint    x,
    GLint    y,
    GLfloat  z ) noexcept

{
  xPos = x;
  yPos = y;
  zPos = z;
}


//=======================================================================
//   class TextScope
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline TextScope::TextScope

  ( GLContext&    gtx,
    TextPainter&  txp ) :

    FontScope ( gtx, txp.font )

{}


JEM_END_PACKAGE( gl )

#endif
