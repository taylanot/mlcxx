
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


#include <jem/base/ClassTemplate.h>
#include <jem/gl/render.h>
#include <jem/gl/Rect.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/ShaderDraw.h>
#include <jem/gl/TextPainter.h>
#include <jem/gl/MessageBox.h>


JEM_DEFINE_CLASS( jem::gl::MessageBox );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class MessageBox
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MessageBox::MessageBox () : Super ( "messageBox" )
{
  textAlignment = CENTERED;
  boxHAlignment = CENTERED;
  boxVAlignment = CENTERED;
}


MessageBox::MessageBox

  ( const String&  txt,
    Alignment      hAlign,
    Alignment      vAlign ) :

    Super ( "messageBox" ),
    text  ( txt )

{
  textAlignment = hAlign;
  boxHAlignment = hAlign;
  boxVAlignment = vAlign;
}


MessageBox::~MessageBox ()
{}


//-----------------------------------------------------------------------
//   redraw
//-----------------------------------------------------------------------


void MessageBox::redraw ( RenderContext& rtx )
{
  const GLsizei  SPACE       = 15;
  GLContext&     gtx         = rtx.getGLContext ();
  GLFunctions1*  funcs       = gtx.findFunctions<GLFunctions1> ();
  GLfloat        vertPos[15] = { 0.0F };

  CanvasScope    canvas      ( gtx, -1.0, 50.0 );
  TextPainter    txp         ( this->font );
  AttribScope    attribs     ( gtx, GLContext::COLOR |
                                    GLContext::LINE_WIDTH );

  Rect           mbox;
  Rect           tbox;

  GLfloat        x, y, w, h;
  idx_t          i, k, n;


  n = text.size ();
  k = 1;

  for ( i = 0; i < n; i++ )
  {
    if ( text[i] == '\n' )
    {
      k++;
    }
  }

  while ( txp.font.getSize() != SMALL_FONT )
  {
    GLsizei  h = txp.font.getHeight ();

    if ( k * h > canvas.height / 2 )
    {
      txp.font--;
    }
    else
    {
      break;
    }
  }

  txp.hAlignment = textAlignment;

  tbox.resize ( txp.getSizeOf( text ) );

  switch ( boxHAlignment )
  {
  case LEFT_ALIGNED:

    tbox.x = 2 * SPACE;
    break;

  case CENTERED:

    tbox.x = (canvas.width - tbox.width) / 2;
    break;

  case RIGHT_ALIGNED:

    tbox.x = canvas.width - tbox.width - 2 * SPACE;
    break;
  }

  switch ( boxVAlignment )
  {
  case BOT_ALIGNED:

    tbox.y = 2 * SPACE;
    break;

  case CENTERED:

    tbox.y = (canvas.height - tbox.height) / 2;
    break;

  case TOP_ALIGNED:

    tbox.y = canvas.height - tbox.height - 2 * SPACE;
    break;
  }

  if ( tbox.x < 2 * SPACE )
  {
    tbox.x = 2 * SPACE;
  }

  if ( tbox.y < 2 * SPACE )
  {
    tbox.y = 2 * SPACE;
  }

  mbox.x      = tbox.x      -     SPACE;
  mbox.y      = tbox.y      -     SPACE;
  mbox.width  = tbox.width  + 2 * SPACE;
  mbox.height = tbox.height + 2 * SPACE;

  if ( textAlignment != LEFT_ALIGNED )
  {
    if ( textAlignment == CENTERED )
    {
      tbox.x = tbox.x + tbox.width / 2;
    }
    else
    {
      tbox.x = tbox.x + tbox.width;
    }

    if ( tbox.x > canvas.width - 2 * SPACE )
    {
      tbox.x = canvas.width - 2 * SPACE;
    }
  }

  gtx.setLineWidth ( 2.0F );
  renderShadow     ( gtx,
                     (GLfloat)  mbox.x,
                     (GLfloat)  mbox.y,
                     (GLfloat) (mbox.x + mbox.width),
                     (GLfloat) (mbox.y + mbox.height), 8 );

  gtx.setColor    ( Color::white() );
  renderRect      ( gtx, mbox );
  gtx.setColor    ( Color::black() );
  txp.begin       ( gtx );
  txp.setCursor   ( tbox.x, tbox.y );
  txp.writeBlock  ( gtx, text,   tbox.size() );
  txp.end         ( gtx );

  x           = (GLfloat) mbox.x;
  y           = (GLfloat) mbox.y;
  w           = (GLfloat) mbox.width;
  h           = (GLfloat) mbox.height;

  vertPos[ 0] = x,     vertPos[ 1] = y;
  vertPos[ 3] = x + w; vertPos[ 4] = y;
  vertPos[ 6] = x + w; vertPos[ 7] = y + h;
  vertPos[ 9] = x,     vertPos[10] = y + h;
  vertPos[12] = x,     vertPos[13] = y;

  if      ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    ShaderDraw::draw ( gtx, 15, vertPos, Color::black(),
                       ShaderDraw::LINE_STRIP );
  }
  else if ( funcs )
  {
    // Unfortunately, GL_LINE_LOOP is broken with some Linux video
    // drivers. So use GL_LINE_STRIP instead.

    funcs->glBegin ( GL_LINE_STRIP );
    {
      funcs->glVertex2 ( &vertPos[ 0] );
      funcs->glVertex2 ( &vertPos[ 3] );
      funcs->glVertex2 ( &vertPos[ 6] );
      funcs->glVertex2 ( &vertPos[ 9] );
      funcs->glVertex2 ( &vertPos[12] );
    }
    funcs->glEnd ();
  }
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   showMessage
//-----------------------------------------------------------------------


Ref<MessageBox>         showMessage

  ( Viewer&               viewer,
    const Ref<GfxNode>&   node,
    const String&         text,
    Alignment             hAlign,
    Alignment             vAlign )

{
  Ref<MessageBox>  mbox =

    newInstance<MessageBox> ( text, hAlign, vAlign );

  mbox->show ( viewer, node );

  return mbox;
}


Ref<MessageBox>         showMessage

  ( RenderContext&        rtx,
    const Ref<GfxNode>&   node,
    const String&         text,
    Alignment             hAlign,
    Alignment             vAlign )

{
  Ref<MessageBox>  mbox =

    newInstance<MessageBox> ( text, hAlign, vAlign );

  mbox->show ( rtx, node );

  return mbox;
}


//-----------------------------------------------------------------------
//   showMessageUntil
//-----------------------------------------------------------------------


Ref<MessageBox>         showMessageUntil

  ( RenderContext&        rtx,
    const Ref<GfxNode>&   node,
    const String&         text,
    const Time&           deadline,
    Alignment             hAlign,
    Alignment             vAlign )

{
  Ref<MessageBox>  mbox =

    newInstance<MessageBox> ( text, hAlign, vAlign );

  mbox->show   ( rtx, node     );
  mbox->hideAt ( rtx, deadline );

  return mbox;
}


JEM_END_PACKAGE( gl )
