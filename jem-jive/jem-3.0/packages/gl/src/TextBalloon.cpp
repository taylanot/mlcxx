
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


#include <jem/base/Time.h>
#include <jem/base/ClassTemplate.h>
#include <jem/gl/render.h>
#include <jem/gl/Rect.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/ShaderDraw.h>
#include <jem/gl/TextPainter.h>
#include <jem/gl/TextBalloon.h>


JEM_DEFINE_CLASS( jem::gl::TextBalloon );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class TextBalloon
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


TextBalloon::TextBalloon () : Super ( "balloon" )
{
  xPos = yPos = -1;
}


TextBalloon::TextBalloon

  ( const String&  txt,
    int            x,
    int            y ) :

    Super ( "balloon" ),
    text  ( txt )

{
  xPos = x;
  yPos = y;
}


TextBalloon::~TextBalloon ()
{}


//-----------------------------------------------------------------------
//   show
//-----------------------------------------------------------------------


void TextBalloon::show

  ( RenderContext&       rtx,
    const Ref<GfxNode>&  node )

{
  double  dt = 4.0 + (double) text.size() / 10.0;

  Super::show   ( rtx, node );
  Super::hideAt ( rtx, Time::now() + Time( dt ) );
}


//-----------------------------------------------------------------------
//   redraw
//-----------------------------------------------------------------------


void TextBalloon::redraw ( RenderContext& rtx )
{
  const GLsizei  SPACE   = 4;
  GLContext&     gtx     = rtx.getGLContext ();
  GLFuncs1*      funcs   = gtx.findFunctions<GLFuncs1> ();

  CanvasScope    canvas  ( gtx, -1.0, 50.0 );
  TextPainter    txp     ( this->font );
  AttribScope    attribs ( gtx, GLContext::COLOR |
                                GLContext::LINE_WIDTH );

  GLfloat        vertPos[15];

  Color          col;
  Size2i         tsz;
  int            x0, x1;
  int            y0, y1;
  idx_t          i, k, n;


  if ( xPos < 0 || xPos > canvas.width ||
       yPos < 0 || yPos > canvas.height )
  {
    return;
  }

  n = text.size ();
  k = 1;

  for ( i = 0; i < n; i++ )
  {
    if ( text[i] == '\n' )
    {
      k++;
    }
  }

  txp.vAlignment = CENTERED;

  while ( txp.font.getSize() != SMALL_FONT )
  {
    GLsizei  h = txp.font.getHeight ();

    if ( k * h > canvas.height / 4 )
    {
      txp.font--;
    }
    else
    {
      break;
    }
  }

  tsz = txp.getSizeOf ( text );

  x0  = xPos + 2 * SPACE;
  x1  = xPos + 4 * SPACE + tsz.width;
  y0  = yPos + 2 * SPACE;
  y1  = yPos + 4 * SPACE + tsz.height;

  if ( x1 > canvas.width )
  {
    x0 = xPos - 4 * SPACE - tsz.width;
    x1 = xPos - 2 * SPACE;
  }

  if ( y1 > canvas.height )
  {
    y0 = yPos - 4 * SPACE - tsz.height;
    y1 = yPos - 2 * SPACE;
  }

  vertPos[ 0] = (GLfloat) x0; vertPos[ 1] = (GLfloat) y0;
  vertPos[ 2] = 0.0F;
  vertPos[ 3] = (GLfloat) x1; vertPos[ 4] = (GLfloat) y0;
  vertPos[ 5] = 0.0F;
  vertPos[ 6] = (GLfloat) x1; vertPos[ 7] = (GLfloat) y1;
  vertPos[ 8] = 0.0F;
  vertPos[ 9] = (GLfloat) x0; vertPos[10] = (GLfloat) y1;
  vertPos[11] = 0.0F;
  vertPos[12] = (GLfloat) x0; vertPos[13] = (GLfloat) y0;
  vertPos[14] = 0.0F;

  gtx.setLineWidth ( 1.0F );

  renderShadow ( gtx,
                 (GLfloat) x0, (GLfloat) y0,
                 (GLfloat) x1, (GLfloat) y1,
                 (GLfloat) SPACE );

  col = Color ( 0.98F, 0.94F, 0.80F, 0.90F );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    ShaderDraw::draw ( gtx, 12, vertPos, col,
                       ShaderDraw::TRIANGLE_FAN );
  }
  else
  {
    gtx.setColor  ( col );

    if ( funcs )
    {
      funcs->glRect ( x0, y0, x1, y1 );
    }
  }

  gtx.translate  ( 0.0F, 0.0F, (GLfloat)(-SPACE) );
  gtx.setColor   ( Color::black() );
  txp.begin      ( gtx );
  txp.setCursor  ( x0 + SPACE, y0 + SPACE + tsz.height / 2 );
  txp.writeBlock ( gtx, text, tsz );
  txp.end        ( gtx );

  col = Color ( 0.8F, 0.8F, 0.0F );

  if      ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    ShaderDraw::draw ( gtx, 15, vertPos, col, ShaderDraw::LINE_STRIP );
  }
  else if ( funcs )
  {
    gtx.setColor   ( col );
    funcs->glBegin ( GL_LINE_STRIP );

    for ( idx_t i = 0; i < 5; i++ )
    {
      funcs->glVertex3 ( &vertPos[3*i] );
    }

    funcs->glEnd ();
  }
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   showBalloon
//-----------------------------------------------------------------------


Ref<TextBalloon>        showBalloon

  ( RenderContext&        rtx,
    const Ref<GfxNode>&   node,
    const String&         text )

{
  Vertex2i  pos = rtx.getMousePos ();

  return showBalloon ( rtx, node, text, pos[0], pos[1] );
}


Ref<TextBalloon>         showBalloon

  ( RenderContext&        rtx,
    const Ref<GfxNode>&   node,
    const String&         text,
    int                   xPos,
    int                   yPos )

{
  Ref<TextBalloon>  balloon =

    newInstance<TextBalloon> ( text, xPos, yPos );

  balloon->show ( rtx, node );

  return balloon;
}


JEM_END_PACKAGE( gl )
