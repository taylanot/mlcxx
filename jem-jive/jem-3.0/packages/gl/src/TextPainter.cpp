
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


#include <jem/base/utilities.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/TextPainter.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class TextPainter
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const GLfloat  TextPainter::HALIGN_[3] = { 0.0F, -0.50F, -1.0F };
const GLfloat  TextPainter::VALIGN_[3] = { 0.0F, -0.50F, -1.0F };
const GLfloat  TextPainter::LINE_SKIP_ =   1.2F;


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


TextPainter::TextPainter

  ( const Font&  fnt,
    Alignment    hAlign,
    Alignment    vAlign ) :

    font ( fnt )

{
  xPos       = 0;
  yPos       = 0;
  zPos       = 0.0F;
  angle      = 0.0F;
  hAlignment = hAlign;
  vAlignment = vAlign;
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


GLsizei TextPainter::write

  ( GLContext&  gtx,
    char        ch )

{
  Size2i  size = font.getSizeOf ( ch );

  GLint   dx = (GLint) (HALIGN_[hAlignment] * (GLfloat) size.width);
  GLint   dy = (GLint) (VALIGN_[vAlignment] * (GLfloat) size.height);


  gtx .pushTransform ();
  gtx .translate     ( (GLfloat) xPos, (GLfloat) yPos,
                       (GLfloat) zPos );
  rotate_            ( gtx );
  gtx .translate     ( (GLfloat) dx, (GLfloat) dy, 0.0F );
  font.write         ( gtx, ch );
  gtx .popTransform  ();

  return size.width;
}


//-----------------------------------------------------------------------
//   writeLine
//-----------------------------------------------------------------------


GLsizei TextPainter::writeLine

  ( GLContext&   gtx,
    const char*  str,
    idx_t        len )

{
  Size2i  size = font.getSizeOf ( str, len );

  GLint   dx = (GLint) (HALIGN_[hAlignment] * (GLfloat) size.width);
  GLint   dy = (GLint) (VALIGN_[vAlignment] * (GLfloat) size.height);

  gtx .pushTransform ();
  gtx .translate     ( (GLfloat) xPos, (GLfloat) yPos,
                       (GLfloat) zPos );
  rotate_            ( gtx );
  gtx .translate     ( (GLfloat) dx, (GLfloat) dy, 0.0F );
  font.write         ( gtx, str, len );
  gtx .popTransform  ();

  return size.width;
}


//-----------------------------------------------------------------------
//   writeBlock
//-----------------------------------------------------------------------


Size2i TextPainter::writeBlock

  ( GLContext&     gtx,
    const String&  text )
{
  Size2i  size = getSizeOf ( text );

  writeBlock ( gtx, text, size );

  return size;
}


void TextPainter::writeBlock

  ( GLContext&     gtx,
    const String&  text,
    const Size2i&  size )

{
  const char*  str = text.addr ();
  const idx_t  len = text.size ();

  GLint    dx, dy;
  GLsizei  w,  h;
  idx_t    i,  j;


  h  = font.getHeight ();
  dx = (GLint) (HALIGN_[hAlignment] * (GLfloat) size.width);
  dy = 0;

  switch ( vAlignment )
  {
  case BOT_ALIGNED:

    dy = (GLint) (size.height - h);

    break;

  case CENTERED:

    dy = (GLint) (size.height - h) / 2 +
         (GLint) (VALIGN_[CENTERED] * (GLfloat) h);

    break;

  case TOP_ALIGNED:

    dy = - (GLint) h;

    break;
  }

  gtx.pushTransform ();
  gtx.translate     ( (GLfloat) xPos, (GLfloat) yPos,
                      (GLfloat) zPos );
  rotate_           ( gtx );
  gtx.translate     ( 0.0F, (GLfloat) dy, 0.0F );

  dy = (GLint) (LINE_SKIP_ * (GLfloat) h);

  for ( i = 0; i < len; i++ )
  {
    for ( j = i; i < len && str[i] != '\n'; i++ ) ;

    w  = font.getWidthOf ( str + j, i - j );
    dx = (GLint) (HALIGN_[hAlignment] * (GLfloat) w);

    gtx .translate ( (GLfloat) dx, 0.0F, 0.0F );
    font.write     ( gtx, str + j, i - j );
    gtx .translate ( (GLfloat) -dx, (GLfloat) -dy, 0.0F );
  }

  gtx.popTransform ();
}


//-----------------------------------------------------------------------
//   getSizeOf
//-----------------------------------------------------------------------


Size2i TextPainter::getSizeOf ( const String& text ) const
{
  const char*  str = text.addr ();
  const idx_t  len = text.size ();

  GLsizei      charWidth, fontHeight;
  GLsizei      lineWidth, lineSkip;
  GLsizei      textWidth, textHeight;
  int          lines;
  idx_t        i, j, k;


  if ( font.getType() == MONOSPACE_FONT )
  {
    charWidth = font.getWidthOf ( '0' );
  }
  else
  {
    charWidth = 0;
  }

  textWidth = 0;
  lines     = 1;
  j         = 0;

  for ( i = 0; i < len; i++ )
  {
    if ( str[i] == '\n' )
    {
      k = i - j;

      if ( charWidth )
      {
        lineWidth = (GLsizei) k * charWidth;
      }
      else
      {
        lineWidth  = font.getWidthOf ( str + j, k );
      }

      textWidth = max ( textWidth, lineWidth );
      j         = i + 1;

      lines++;
    }
  }

  k = i - j;

  if ( charWidth )
  {
    lineWidth = (GLsizei) k * charWidth;
  }
  else
  {
    lineWidth = font.getWidthOf ( str + j, k );
  }

  textWidth  = max ( textWidth, lineWidth );
  fontHeight = font.getHeight ();
  lineSkip   = (GLsizei) (LINE_SKIP_ * (GLfloat) fontHeight);
  textHeight = fontHeight + (lines - 1) * lineSkip;

  return Size2i ( textWidth, textHeight );
}


//-----------------------------------------------------------------------
//   rotate_
//-----------------------------------------------------------------------


inline void TextPainter::rotate_ ( GLContext& gtx )
{
  if ( (angle < -0.01F) || (angle > 0.01F) )
  {
    gtx.rotate ( angle, 0.0F, 0.0F, 1.0F );
  }
}


JEM_END_PACKAGE( gl )
