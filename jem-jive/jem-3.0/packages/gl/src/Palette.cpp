
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


#include <cmath>
#include <cstdio>
#include <jem/base/ClassTemplate.h>
#include <jem/gl/render.h>
#include <jem/gl/Rect.h>
#include <jem/gl/ColorMap.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/ShaderDraw.h>
#include <jem/gl/TextPainter.h>
#include <jem/gl/Palette.h>


JEM_DEFINE_CLASS( jem::gl::Palette );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class Palette
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Palette::~Palette ()
{}


//-----------------------------------------------------------------------
//   findValue
//-----------------------------------------------------------------------


double Palette::findValue

  ( double&       val,
    double        prec,
    const Color&  col ) const

{
  int     sampleCount = 2;

  double  eps;

  do
  {
    sampleCount *= 2;

    eps = findValue_ ( val, prec, col, sampleCount );
  }
  while ( eps > prec && sampleCount < 128 );

  return eps;
}


//-----------------------------------------------------------------------
//   makeColorMap
//-----------------------------------------------------------------------


void Palette::makeColorMap ( ColorMap& colorMap ) const
{
  makeLinearColorMap ( colorMap, 256, *this );
}


//-----------------------------------------------------------------------
//   configure & getConfig
//-----------------------------------------------------------------------


void Palette::configure ( const Properties& props )
{}


void Palette::getConfig ( const Properties& props ) const
{}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


void Palette::listProps ( const Properties& props ) const
{}


//-----------------------------------------------------------------------
//   render
//-----------------------------------------------------------------------


void Palette::render

  ( GLContext&     gtx,
    const Rect&    rect,
    const String&  title,
    Alignment      align,
    double         scale,
    double         bias ) const

{
  if ( rect.width < 40 || rect.height < 40 )
  {
    return;
  }

  CanvasScope  canvas  ( gtx );
  TextPainter  txp     ( Font(), align );
  AttribScope  attribs ( gtx, GLContext::COLOR |
                              GLContext::LINE_WIDTH );

  GLint        x, y;
  GLint        w, h;
  GLint        xtick;


  w = (GLint) rect.width / 4;
  h = (GLint) rect.height;
  y = rect.y;

  if ( align == LEFT_ALIGNED )
  {
    x     = rect.x;
    xtick = w + 4;
  }
  else
  {
    x     = rect.x + (GLint) rect.width - w;
    xtick = -4;
  }

  gtx.translate ( x, y, 0.0 );

  if ( title.size() && h > (6 * txp.font.getHeight()) )
  {
    TextScope  textScope  ( gtx, txp );
    GLsizei    charHeight = txp.font.getHeight ();

    h -= (GLint)     (2.5 * charHeight);
    y  = (GLint) (h + 1.5 * charHeight);

    if ( h < 40 )
    {
      return;
    }

    if ( align == LEFT_ALIGNED )
    {
      x = 0;
    }
    else
    {
      x = w;
    }

    gtx.setColor   ( Color::black() );
    txp.setCursor  ( x, y );
    txp.writeLine  ( gtx, title );
  }

  if ( h < 200 )
  {
    txp.font--;
  }

  renderShadow  ( gtx, 0.0F, 0.0F, (GLfloat) w, (GLfloat) h, 5 );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    drawPaletteV3_ ( gtx, txp, w, h, xtick, align, scale, bias );
  }
  else
  {
    drawPaletteV1_ ( gtx, txp, w, h, xtick, align, scale, bias );
  }
}


//-----------------------------------------------------------------------
//   fillRectangle
//-----------------------------------------------------------------------


void Palette::fillRectangle

  ( GLContext&  gtx,
    GLint       width,
    GLint       height ) const

{}


//-----------------------------------------------------------------------
//   drawTickMarks
//-----------------------------------------------------------------------


void Palette::drawTickMarks

  ( GLContext&    gtx,
    TextPainter&  txp,
    GLint         width,
    GLint         height,
    GLint         xtick,
    Alignment     hAlign,
    double        scale,
    double        bias ) const

{
  const double  minVal  = bias + scale * getMinValue ();
  const double  maxVal  = bias + scale * getMaxValue ();

  char          buf[32];
  double        dv, dy;
  GLint         y;
  int           i, k, n;


  n = (int) (height / (4.0 * txp.font.getHeight())) + 1;

  adjustTickCount ( n );

  if ( n < 2 )
  {
    return;
  }

  dy =  height           / (n - 1.0);
  dv = (maxVal - minVal) / (n - 1.0);

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    GLfloat        wf      = (GLfloat) width;
    GLfloatVector  vertPos ( 6 * (n - 2) );
    GLfloat        yf;

    vertPos = 0.0F;

    for ( i = 1; i < (n - 1); i++ )
    {
      GLfloatVector  part = vertPos[slice(6 * (i - 1),6 * i)];

      yf = (GLfloat) (i * dy);

                    part[1] = yf;
      part[3] = wf; part[4] = yf;
    }

    ShaderDraw::draw ( gtx, vertPos,
                       gtx.getColor(), ShaderDraw::LINES );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      funcs->glBegin ( GL_LINES );

      for ( i = 1; i < (n - 1); i++ )
      {
        y = (GLint) (i * dy);

        funcs->glVertex ( 0,     y );
        funcs->glVertex ( width, y );
      }

      funcs->glEnd ();
    }
  }

  {
    TextScope  textScope ( gtx, txp );

    k = std::sprintf ( buf, "%+.2e", minVal );

    txp.setCursor ( xtick, 0 );
    txp.writeLine ( gtx, buf,   k );

    for ( i = 1; i < (n - 1); i++ )
    {
      y = (GLint) (i * dy);
      k = std::sprintf ( buf, "%+.2e", (minVal + i * dv) );

      txp.setCursor ( xtick, y );
      txp.writeLine ( gtx, buf,   k );
    }

    k = std::sprintf ( buf, "%+.2e", maxVal );

    txp.setCursor ( xtick, height );
    txp.writeLine ( gtx, buf, k );
  }
}


//-----------------------------------------------------------------------
//   findValue_
//-----------------------------------------------------------------------


double Palette::findValue_

  ( double&       val,
    double        prec,
    const Color&  col,
    int           n ) const

{
  double   minVal = getMinValue ();
  double   maxVal = getMaxValue ();

  double   err, eps;
  double   delta;
  double   v, dv;

  Color    c;

  int      iter;
  int      i, j, k;


  // Make sure n is even

  n   += (n % 2);
  prec = prec * prec;

  getColor ( c, minVal );

  err = 0.0;

  for ( i = 0; i < 4; i++ )
  {
    delta = (double) c[i] - (double) col[i];
    err  += delta * delta;
  }

  val = minVal;
  eps = err;
  err = 0.0;
  k   = 0;

  getColor ( c, maxVal );

  for ( i = 0; i < 4; i++ )
  {
    delta = (double) c[i] - (double) col[i];
    err  += delta * delta;
  }

  if ( err < eps )
  {
    eps = err;
    val = maxVal;
    k   = n;
  }

  for ( iter = 0; iter < 10 && eps > prec; iter++ )
  {
    dv = (maxVal - minVal) / (double) n;

    for ( j = 1; j < n; j++ )
    {
      v = minVal + j * dv;

      getColor ( c, v );

      err = 0.0;

      for ( i = 0; i < 4; i++ )
      {
        delta = (double) c[i] - (double) col[i];
        err  += delta * delta;
      }

      if ( err < eps )
      {
        eps = err;
        val = v;
        k   = j;
      }
    }

    if      ( k == 0 )
    {
      maxVal = minVal + dv;
    }
    else if ( k == n )
    {
      minVal = maxVal - dv;
    }
    else
    {
      maxVal = minVal + (k + 1) * dv;
      minVal = minVal + (k - 1) * dv;
      k      = n / 2;
    }
  }

  return std::sqrt ( eps );
}


//-----------------------------------------------------------------------
//   drawPaletteV1_
//-----------------------------------------------------------------------


void Palette::drawPaletteV1_

  ( GLContext&    gtx,
    TextPainter&  txp,
    GLint         width,
    GLint         height,
    GLint         xtick,
    Alignment     aling,
    double        scale,
    double        bias ) const

{
  GLFuncs1*  funcs = gtx.findFunctions<GLFuncs1> ();


  // Draw a solid white rectangle as background so that transparent
  // palettes are rendered correctly.

  gtx.setColor ( Color::white() );

  if ( funcs )
  {
    funcs->glRect ( 0, 0, width, height );
  }

  fillRectangle    ( gtx, width, height );
  gtx.setColor     ( Color::black() );
  gtx.setLineWidth ( 2.0F );

  if ( funcs )
  {
    funcs->glBegin ( GL_LINE_STRIP );
    {
      funcs->glVertex  ( 0,     0 );
      funcs->glVertex  ( width, 0 );
      funcs->glVertex  ( width, height );
      funcs->glVertex  ( 0,     height );
      funcs->glVertex  ( 0,     0 );
    }
    funcs->glEnd ();
  }

  txp.vAlignment = CENTERED;

  drawTickMarks ( gtx, txp, width, height, xtick, aling, scale, bias );
}


//-----------------------------------------------------------------------
//   drawPaletteV3_
//-----------------------------------------------------------------------


void Palette::drawPaletteV3_

  ( GLContext&    gtx,
    TextPainter&  txp,
    GLint         width,
    GLint         height,
    GLint         xtick,
    Alignment     aling,
    double        scale,
    double        bias ) const

{
  const GLfloat  w           = (GLfloat) width;
  const GLfloat  h           = (GLfloat) height;
  GLfloat        vertPos[18] = { 0.0F };


  vertPos[ 0] = 0.0F; vertPos[ 1] = 0.0F; vertPos[ 2] = 0.0F;
  vertPos[ 3] = w;    vertPos[ 4] = 0.0F; vertPos[ 5] = 0.0F;
  vertPos[ 6] = w;    vertPos[ 7] = h;    vertPos[ 8] = 0.0F;

  vertPos[9]  = w;    vertPos[10] = h;    vertPos[11] = 0.0F;
  vertPos[12] = 0.0F; vertPos[13] = h;    vertPos[14] = 0.0F;
  vertPos[15] = 0.0F; vertPos[16] = 0.0F; vertPos[17] = 0.0F;

  ShaderDraw::draw ( gtx, 18, vertPos, Color::white(),
                     ShaderDraw::TRIANGLE_STRIP );

  fillRectangle    ( gtx, width, height );
  gtx.setLineWidth ( 2.0F );
  ShaderDraw::draw ( gtx, 18, vertPos, Color::black(),
                     ShaderDraw::LINE_STRIP );

  txp.vAlignment = CENTERED;

  gtx.setColor     ( Color::black() );
  drawTickMarks    ( gtx, txp, width, height, xtick,
                     aling, scale, bias );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   mapColorToValue
//-----------------------------------------------------------------------


double            mapColorToValue

  ( double&         val,
    double          prec,
    const Color&    color,
    const Color&    minColor,
    const Color&    maxColor )

{
  double  t[4];
  double  u[4];
  double  d, x;
  int     i;


  d = 0.0;
  x = 0.0;

  for ( i = 0; i < 4; i++ )
  {
    t[i] = (double) maxColor[i] - (double) minColor[i];
    u[i] = (double)    color[i] - (double) minColor[i];

    d   += t[i] * t[i];
    x   += t[i] * u[i];
  }

  if ( d <= (0.1 * prec * prec) )
  {
    x = 0.0;
  }
  else
  {
    x = x / d;

    if      ( x < 0.0 )
    {
      x = 0.0;
    }
    else if ( x > 1.0 )
    {
      x = 1.0;
    }
  }

  d = 0.0;

  for ( i = 0; i < 4; i++ )
  {
    u[i] -= x    * t[i];
    d    += u[i] * u[i];
  }

  val = x;

  return std::sqrt ( d );
}


//-----------------------------------------------------------------------
//   makeLinearColorMap
//-----------------------------------------------------------------------


void                makeLinearColorMap

  ( ColorMap&         colorMap,
    idx_t             colorCount,
    const Palette&    palette )

{
  const double    minVal = palette.getMinValue ();
  const double    maxVal = palette.getMaxValue ();

  GLfloatMatrix   colors;
  Color           c;
  double          dv;
  idx_t           i, j, m, n;


  if ( isOpaque( palette.getMinAlpha() ) )
  {
    m = 3;
  }
  else
  {
    m = 4;
  }

  if ( maxVal <= minVal )
  {
    n = 2;

    colors.resize ( m, n );

    palette.getColor ( c, minVal );

    for ( j = 0; j < n; j++ )
    {
      for ( i = 0; i < m; i++ )
      {
        colors(i,j) = c[i];
      }
    }
  }
  else
  {
    for ( n = 2; n < colorCount && n < 1024; n = n << 2 ) ;

    dv = (maxVal - minVal) / ((double) n - 1.0);

    colors.resize ( m, n );

    for ( j = 0; j < n; j++ )
    {
      palette.getColor ( c, (minVal + (double) j * dv) );

      for ( i = 0; i < m; i++ )
      {
        colors(i,j) = c[i];
      }
    }
  }

  colorMap.setColors ( colors );
}


JEM_END_PACKAGE( gl )
