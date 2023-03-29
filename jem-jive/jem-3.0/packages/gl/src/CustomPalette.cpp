
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
#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/util/Properties.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/PropertyNames.h>
#include <jem/gl/ShaderDraw.h>
#include <jem/gl/TextPainter.h>
#include <jem/gl/CustomPalette.h>


JEM_DEFINE_CLASS( jem::gl::CustomPalette );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class CustomPalette
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


CustomPalette::CustomPalette () :

  colors_ ( 3, 1 ),
  ranges_ (    2 )

{
  colors_    = 0.5F;
  ranges_[0] = 0.0;
  ranges_[1] = 1.0;
}


CustomPalette::CustomPalette

  ( const GLfloatMatrix&  colors,
    const Array<double>&  ranges )

{
  setColorRanges ( colors, ranges );
}


CustomPalette::~CustomPalette ()
{}


//-----------------------------------------------------------------------
//   getColor
//-----------------------------------------------------------------------


void CustomPalette::getColor

  ( Color&  col,
    double  val ) const

{
  const bool   flat = (ranges_.size() != colors_.size(1));

  const idx_t  m    = colors_.size(0);

  idx_t        i, j, k;


  i = 0;
  j = ranges_.size () - 2;

  while ( i < j )
  {
    k = i + (j - i + 1) / 2;

    if ( val >= ranges_[k] )
    {
      i = k;
    }
    else
    {
      j = k - 1;
    }
  }

  col[3] = 1.0;

  if ( flat )
  {
    col[0] = colors_(0,j);
    col[1] = colors_(1,j);
    col[2] = colors_(2,j);

    if ( m == 4 )
    {
      col[3] = colors_(3,j);
    }
  }
  else
  {
    double  a, b;

    k = j + 1;

    if      ( val <= ranges_[j] )
    {
      a = 1.0;
    }
    else if ( val >= ranges_[k] )
    {
      a = 0.0;
    }
    else
    {
      a = (ranges_[k] - val) / (ranges_[k] - ranges_[j]);
    }

    b = 1.0 - a;

    for ( i = 0; i < m; i++ )
    {
      col[i] = (GLfloat) (a * colors_(i,j) + b * colors_(i,k));
    }
  }
}


//-----------------------------------------------------------------------
//   findValue
//-----------------------------------------------------------------------


double CustomPalette::findValue

  ( double&       val,
    double        prec,
    const Color&  col ) const

{
  const idx_t  m = colors_.size (0);
  const idx_t  n = colors_.size (1);

  double       eps, eps0;
  double       val0;
  idx_t        i, j;


  eps0 = 10.0;
  val0 = ranges_.front ();

  if ( n == ranges_.size() )
  {
    Color  minColor;
    Color  maxColor;

    for ( i = 0; i < m; i++ )
    {
      maxColor[i] = colors_(i,0);
    }

    for ( j = 1; j < n; j++ )
    {
      minColor = maxColor;

      for ( i = 0; i < m; i++ )
      {
        maxColor[i] = colors_(i,j);
      }

      eps = mapColorToValue ( val, prec, col, minColor, maxColor );

      if ( eps < eps0 )
      {
        eps0 = eps;
        val0 = ranges_[j - 1] + val * (ranges_[j] - ranges_[j - 1]);
      }
    }
  }
  else
  {
    double  err0;
    double  delta;

    err0 = 0.0;
    eps0 = eps0 * eps0;

    for ( i = m; i < 4; i++ )
    {
      delta = 1.0 - (double) col[i];
      err0 += delta * delta;
    }

    for ( j = 0; j < n; j++ )
    {
      eps = err0;

      for ( i = 0; i < m; i++ )
      {
        delta = (double) colors_(i,j) - (double) col[i];
        eps  += delta * delta;
      }

      if ( eps < eps0 )
      {
        eps0 = eps;
        val0 = ranges_[j];
      }
    }

    eps0 = std::sqrt ( eps0 );
  }

  val = val0;

  return eps0;
}


//-----------------------------------------------------------------------
//   getMinValue & getMaxValue
//-----------------------------------------------------------------------


double CustomPalette::getMinValue () const
{
  return ranges_.front ();
}


double CustomPalette::getMaxValue () const
{
  return ranges_.back ();
}


//-----------------------------------------------------------------------
//   getMinAlpha
//-----------------------------------------------------------------------


GLfloat CustomPalette::getMinAlpha () const
{
  GLfloat  minAlpha = 1.0F;

  if ( colors_.size(0) == 4 )
  {
    const idx_t  n  = colors_.size (1);

    for ( idx_t j = 0; j < n; j++ )
    {
      if ( colors_(3,j) < minAlpha )
      {
        minAlpha = colors_(3,j);
      }
    }
  }

  return minAlpha;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void CustomPalette::configure ( const Properties& props )
{
  using util::findBool;

  Array<String>  colors;
  Array<double>  ranges;

  GLfloatMatrix  cmat;

  bool           newRanges;
  bool           newColors;

  idx_t          m, n;


  newRanges = props.find ( ranges, PropNames::RANGES );
  newColors = props.find ( colors, PropNames::COLORS );

  m = newRanges ? ranges.size () : ranges_.size ();
  n = newColors ? colors.size () : colors_.size (1);

  if ( newRanges )
  {
    if ( m <= 1 )
    {
      props.propertyError (
        PropNames::RANGES,
        "range array must contain at least two elements"
      );
    }

    if ( m < n || m > n + 1 )
    {
      props.propertyError (
        PropNames::RANGES,
        String::format (
          "range array has size %d but should have size %d or %d",
          m, n, (n + 1)
        )
      );
    }

    if ( ! checkRanges_( ranges ) )
    {
      props.propertyError (
        PropNames::RANGES,
        "range array must contain strictly ascending elements"
      );
    }
  }

  if ( newColors )
  {
    Color  c;

    if ( n < m - 1 || n > m )
    {
      props.propertyError (
        PropNames::COLORS,
        String::format (
          "color array has size %d but should have size %d or %d",
          n, (m - 1), m
        )
      );
    }

    cmat.resize ( 4, n );

    try
    {
      for ( idx_t j = 0; j < n; j++ )
      {
        c = Color::parse ( colors[j] );

        for ( idx_t i = 0; i < 4; i++ )
        {
          cmat(i,j) = c[i];
        }
      }
    }
    catch ( const IllegalInputException& ex )
    {
      props.propertyError ( PropNames::COLORS, ex.what() );
    }
  }

  // No errors, so commit the changes

  if ( newRanges )
  {
    ranges_.ref ( ranges );
  }

  if ( newColors )
  {
    colors_.ref ( cmat );
  }

  findBool ( options_,  HIDE_MIN_MAX, props,
             PropNames::HIDE_MIN_MAX );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void CustomPalette::getConfig ( const Properties& props ) const
{
  using util::setBool;

  const idx_t    m = colors_.size (0);
  const idx_t    n = colors_.size (1);

  Array<String>  colors ( n );

  Color          c;

  idx_t          i, j;


  c[3] = 1.0;

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      c[i] = colors_(i,j);
    }

    colors[j] = c.toString ();
  }

  props.set ( PropNames::COLORS, colors  );
  props.set ( PropNames::RANGES, ranges_ );

  setBool   ( props,
              PropNames::HIDE_MIN_MAX,
              options_,  HIDE_MIN_MAX );
}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


void CustomPalette::listProps ( const Properties& props ) const
{
  props.set ( PropNames::COLORS,
              "An array with colors" );
  props.set ( PropNames::RANGES,
              "An array specifying the range for each color" );
  props.set ( PropNames::HIDE_MIN_MAX,
              "Whether the minimum and maximum values "
              "should be hidden" );
}


//-----------------------------------------------------------------------
//   setColorRanges
//-----------------------------------------------------------------------


void CustomPalette::setColorRanges

  ( const GLfloatMatrix&  colors,
    const Array<double>&  ranges )

{
  JEM_PRECHECK ( colors.size(0) == 3 || colors.size(0) == 4 );

  const idx_t  m = ranges.size ();
  const idx_t  n = colors.size (1);

  if ( m <= 1 )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      "range array must contain at least two elements"
    );
  }

  if ( ! checkRanges_( ranges ) )
  {
    throw IllegalInputException (
      JEM_FUNC,
      "range array must contain strictly ascending elements"
    );
  }

  if ( n < m - 1 || n > m )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      String::format (
        "color array has size %d but should have size %d or %d",
        n, (m - 1), m
      )
    );
  }

  colors_.ref ( colors );
  ranges_.ref ( ranges );
}


//-----------------------------------------------------------------------
//   setOption
//-----------------------------------------------------------------------


void CustomPalette::setOption

  ( Option  option,
    bool    yesno )

{
  options_.set ( option, yesno );
}


//-----------------------------------------------------------------------
//   setOptions
//-----------------------------------------------------------------------


void CustomPalette::setOptions ( Options options )
{
  options_ = options;
}


//-----------------------------------------------------------------------
//   getOptions
//-----------------------------------------------------------------------


CustomPalette::Options CustomPalette::getOptions () const noexcept
{
  return options_;
}


//-----------------------------------------------------------------------
//   fillRectangle
//-----------------------------------------------------------------------


void CustomPalette::fillRectangle

  ( GLContext&  gtx,
    GLint       w,
    GLint       h ) const

{
  const bool     flat      = (ranges_.size() != colors_.size(1));

  const idx_t    m         = colors_.size(0);
  const idx_t    n         = ranges_.size() - 1;
  const idx_t    bufSize   = (18 + 18 + 24) * n;

  const double   minVal    =      ranges_[0];
  const double   scale     = h / (ranges_[n] - minVal);

  GLfloatVector  positions;
  GLfloatVector  normals;
  GLfloatVector  colors;

  GLfloat        r, g, b, a;
  GLfloat        y0, y1;


  if ( scratch_.buf.size() < bufSize )
  {
    const_cast<Self*>( this )->scratch_.buf.resize ( bufSize );
  }

  positions.ref ( scratch_.buf[slice(BEGIN, 18 * n)] );
  normals  .ref ( scratch_.buf[slice(18 * n,36 * n)] );
  colors   .ref ( scratch_.buf[slice(36 * n,END)]    );

  // Fill the normal array.

  normals = 0.0F;

  for ( idx_t j = 2; j < 18 * n; j += 3 )
  {
    normals[j] = 1.0F;
  }

  // Fill the vertex array.

  y0        = 0.0F;
  positions = 0.0F;

  for ( idx_t j = 0; j < n; j++ )
  {
    GLfloatVector  partPos = positions[slice(18 * j,18 * (j + 1))];

    if ( j + 1 == n )
    {
      y1 = (GLfloat) h;
    }
    else
    {
      y1 = (GLfloat) (scale * (ranges_[j + 1] - minVal));
    }

                               partPos[ 1] = y0;
    partPos[ 3] = (GLfloat) w; partPos[ 4] = y0;
    partPos[ 6] = (GLfloat) w; partPos[ 7] = y1;

    partPos[ 9] = (GLfloat) w; partPos[10] = y1;
                               partPos[13] = y1;
                               partPos[16] = y0;

    y0 = y1;
  }

  // Fill the color array.

  if ( flat )
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      GLfloatVector  part = colors[slice(24 * j,24 * (j + 1))];

      r = colors_(0,j);
      g = colors_(1,j);
      b = colors_(2,j);

      if ( m == 4 )
      {
        a = colors_(3,j);
      }
      else
      {
        a = 1.0F;
      }

      for ( idx_t k = 0; k < 24; k += 4 )
      {
        part[k]     = r;
        part[k + 1] = g;
        part[k + 2] = b;
        part[k + 3] = a;
      }
    }
  }
  else
  {
    GLfloat  r0, g0, b0, a0;

    r0 = colors_(0,0);
    g0 = colors_(1,0);
    b0 = colors_(2,0);

    if ( m == 4 )
    {
      a0 = colors_(3,0);
    }
    else
    {
      a0 = 1.0F;
    }

    for ( idx_t j = 0; j < n; j++ )
    {
      GLfloatVector  part = colors[slice(24 * j,24 * (j + 1))];

      r  = colors_(0,j + 1);
      g  = colors_(1,j + 1);
      b  = colors_(2,j + 1);

      if ( m == 4 )
      {
        a = colors_(3,j + 1);
      }
      else
      {
        a = 1.0F;
      }

      part[ 0] = r0; part[ 1] = g0; part[ 2] = b0; part[ 3] = a0;
      part[ 4] = r0; part[ 5] = g0; part[ 6] = b0; part[ 7] = a0;
      part[ 8] = r;  part[ 9] = g;  part[10] = b;  part[11] = a;
      part[12] = r;  part[13] = g;  part[14] = b;  part[15] = a;
      part[16] = r;  part[17] = g;  part[18] = b;  part[19] = a;
      part[20] = r0; part[21] = g0; part[22] = b0; part[23] = a0;

      r0 = r;
      g0 = g;
      b0 = b;
      a0 = a;
    }
  }

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    ShaderDraw::draw ( gtx, positions, normals, colors,
                       ShaderDraw::TRIANGLES );
  }
  else
  {
    GLFuncs1*  funcs = gtx.findFunctions<GLFuncs1> ();

    if ( funcs )
    {
      funcs->glBegin  ( GL_TRIANGLES );

      funcs->glNormal ( normals[0], normals[1], normals[2] );

      for ( idx_t j = 0; j < n; j++ )
      {
        GLfloatVector  partPos =

          positions[slice(18 * j,18 * (j + 1))];

        GLfloatVector  partCol =

          colors   [slice(24 * j,24 * (j + 1))];

        for ( idx_t k = 0; k < 6; k++ )
        {
          funcs->glColor4  ( partCol.addr( 4 * k ) );
          funcs->glVertex3 ( partPos.addr( 3 * k ) );
        }
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   drawTickMarks
//-----------------------------------------------------------------------


void CustomPalette::drawTickMarks

  ( GLContext&    gtx,
    TextPainter&  txp,
    GLint         w,
    GLint         h,
    GLint         xtick,
    Alignment     hAlign,
    double        scale,
    double        bias ) const

{
  const idx_t    n      = ranges_.size() - 1;

  const double   minVal =      ranges_[0];
  const double   hScale = h / (ranges_[n] - minVal);

  char           buf[32];
  GLint          minSpace;
  GLint          y, y0;


  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    const idx_t    bufSize = 6 * (n - 1);

    GLfloatVector  vertPos;

    if ( scratch_.buf.size() < bufSize )
    {
      const_cast<Self*>( this )->scratch_.buf.resize ( bufSize );
    }

    vertPos.ref ( scratch_.buf[slice(BEGIN,bufSize)] );

    vertPos = 0.0F;

    for ( idx_t j = 1; j < n; j++ )
    {
      GLfloatVector  part = vertPos[slice(6 * (j - 1),6 * j)];

      y = (GLint) (hScale * (ranges_[j] - minVal));

                             part[1] = (GLfloat) y;
      part[3] = (GLfloat) w; part[4] = (GLfloat) y;
    }

    ShaderDraw::draw ( gtx, vertPos,
                       gtx.getColor(), ShaderDraw::LINES );
  }
  else
  {
    GLFuncs1*  funcs = gtx.findFunctions<GLFuncs1> ();

    if ( funcs )
    {
      funcs->glBegin ( GL_LINES );

      for ( idx_t j = 1; j < n; j++ )
      {
        y = (GLint) (hScale * (ranges_[j] - minVal));

        funcs->glVertex ( 0, y );
        funcs->glVertex ( w, y );
      }

      funcs->glEnd ();
    }
  }

  minSpace = 2 * txp.font.getHeight ();

  {
    TextScope  textScope ( gtx, txp );

    int        k;

    if ( ! (options_ & HIDE_MIN_MAX) )
    {
      k = std::sprintf ( buf, "%+.2e", (bias + scale * minVal) );

      txp.setCursor ( xtick, 0 );
      txp.writeLine ( gtx, buf, k );
    }

    y0 = 0;

    for ( idx_t j = 1; j < n; j++ )
    {
      y = (GLint) (hScale * (ranges_[j] - minVal));

      if ( y - y0 < minSpace || h - y < minSpace )
      {
        continue;
      }

      k = std::sprintf ( buf, "%+.2e", (bias + scale * ranges_[j]) );

      txp.setCursor ( xtick, y );
      txp.writeLine ( gtx, buf, k );

      y0 = y;
    }

    if ( ! (options_ & HIDE_MIN_MAX) )
    {
      k = std::sprintf ( buf, "%+.2e", (bias + scale * ranges_[n]) );

      txp.setCursor ( xtick, h );
      txp.writeLine ( gtx, buf, k );
    }
  }
}


//-----------------------------------------------------------------------
//   checkRanges_
//-----------------------------------------------------------------------


bool CustomPalette::checkRanges_ ( const Array<double>& r )
{
  const idx_t  n = r.size() - 1;

  for ( idx_t j = 0; j < n; j++ )
  {
    if ( r[j] >= r[j + 1] )
    {
      return false;
    }
  }

  return true;
}


JEM_END_PACKAGE( gl )
