
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
#include <jem/base/rtti.h>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/Float.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/utilities.h>
#include <jem/util/None.h>
#include <jem/util/Properties.h>
#include <jem/gl/render.h>
#include <jem/gl/Box.h>
#include <jem/gl/Symbols.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/PropertyNames.h>
#include <jem/gl/ShaderDraw.h>
#include <jem/gl/NumberMarker.h>
#include <jem/gl/SymbolMarker.h>
#include <jem/gl/Transform.h>
#include "XYPlotSet.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class XYPlot::DataSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


XYPlot::DataSet::DataSet ()
{
  lnWidth_    = 0.0;
  maxPoints_  = maxOf ( maxPoints_ );
  baseNumber_ = 0_idx;
}


XYPlot::DataSet::~DataSet ()
{}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void XYPlot::DataSet::clear ()
{
  Array<double>  empty;

  setPoints ( empty, empty );
}


//-----------------------------------------------------------------------
//   setStyle
//-----------------------------------------------------------------------


void XYPlot::DataSet::setStyle

  ( GLfloat                lwidth,
    const Color&           lcolor,
    const Ref<PntStyle>&   pstyle )

{
  lnWidth_ = clamped ( lwidth, (GLfloat) 64.0 );
  lnColor_ = clamped ( lcolor );
  ptStyle_ = pstyle;
}


//-----------------------------------------------------------------------
//   setLineWidth
//-----------------------------------------------------------------------


void XYPlot::DataSet::setLineWidth ( GLfloat width )
{
  lnWidth_ = clamped ( width, (GLfloat) 64.0 );
}


//-----------------------------------------------------------------------
//   setLineColor
//-----------------------------------------------------------------------


void XYPlot::DataSet::setLineColor ( const Color& col )
{
  lnColor_ = clamped ( col );

  ColoredMarker*  cm = dynamicCast<ColoredMarker*> ( ptStyle_ );

  if ( cm )
  {
    cm->setColor ( lnColor_ );
  }
}


//-----------------------------------------------------------------------
//   setPointStyle
//-----------------------------------------------------------------------


void XYPlot::DataSet::setPointStyle ( const Ref<PntStyle>& style )
{
  ptStyle_ = style;
}


//-----------------------------------------------------------------------
//   setBaseNumber
//-----------------------------------------------------------------------


void XYPlot::DataSet::setBaseNumber ( idx_t nr0 )
{
  baseNumber_ = nr0;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void XYPlot::DataSet::configure ( const Properties& props )
{
  String   ps;
  Color    lc;
  double   lw;
  idx_t    mp;
  idx_t    nr0;


  props.find ( key, PropNames::KEY );

  if ( props.find( mp, PropNames::MAX_POINTS ) )
  {
    setMaxPoints ( mp );
  }

  if ( props.find( ps, PropNames::POINT_STYLE ) )
  {
    Ref<PntStyle>  style;

    if ( ps.size() )
    {
      if ( ps.equalsIgnoreCase( "number" ) )
      {
        Ref<NumberMarker>  nm =

          newInstance<NumberMarker> ( lnColor_ );

        nm->writer.hAlignment = CENTERED;
        nm->writer.vAlignment = CENTERED;

        style = nm;
      }
      else
      {
        char  symbol = Symbols::byName ( ps.toLower() );

        if ( ! symbol )
        {
          String  choice = Symbols::getChoice ();

          props.propertyError (
            PropNames::POINT_STYLE,
            String::format (
              "invalid point style: %s; should be %s",
              ps,
              choice
            )
          );
        }

        style = newInstance<SymbolMarker> ( symbol, lnColor_ );
      }
    }

    setPointStyle ( style );
  }

  if ( props.find( lw, PropNames::LINE_WIDTH, 0.0, 64.0 ) )
  {
    setLineWidth ( (GLfloat) lw );
  }

  if ( lc.configure( PropNames::LINE_COLOR, props ) )
  {
    setLineColor ( lc );
  }

  if ( dynamicCast<NumberMarker*>( ptStyle_ ) )
  {
    if ( props.find( nr0, PropNames::BASE_NUMBER ) )
    {
      setBaseNumber ( nr0 );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void XYPlot::DataSet::getConfig ( const Properties& conf )  const
{
  using jem::util::None;

  String  ps;
  idx_t   mp;


  mp = maxPoints_;

  if ( mp == maxOf( mp ) )
  {
    mp = -1;
  }

  conf.set ( PropNames::KEY,        key );
  conf.set ( PropNames::MAX_POINTS, mp  );

  if ( ! ptStyle_ )
  {
    conf.set ( PropNames::POINT_STYLE, None::getInstance() );
  }
  else
  {
    conf.set ( PropNames::POINT_STYLE, ptStyle_->toString() );
  }

  conf.set ( PropNames::LINE_WIDTH, (double) lnWidth_ );

  lnColor_.getConfig ( PropNames::LINE_COLOR, conf );

  if ( dynamicCast<NumberMarker*>( ptStyle_ ) )
  {
    conf.set ( PropNames::BASE_NUMBER, baseNumber_ );
  }
}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


void XYPlot::DataSet::listProps ( const Properties& conf )  const
{
  conf.set ( PropNames::KEY,
             "The key (title)" );
  conf.set ( PropNames::MAX_POINTS,
             "The maximum number of points" );
  conf.set ( PropNames::POINT_STYLE,
             "The point style" );
  conf.set ( PropNames::LINE_WIDTH,
             "The line width (in pixels)" );
  conf.set ( PropNames::LINE_COLOR,
             "The line color" );

  if ( dynamicCast<NumberMarker*>( ptStyle_ ) )
  {
    conf.set ( PropNames::BASE_NUMBER,
               "The starting number" );
  }
}


//=======================================================================
//   class XYPlot::DataSet_
//=======================================================================

//-----------------------------------------------------------------------
//   setPoints
//-----------------------------------------------------------------------


void XYPlot::DataSet_::setPoints

  ( const Array<double>&  x,
    const Array<double>&  y )

{
  JEM_PRECHECK ( x.size() == y.size() );

  if ( Float::isNaN( sum( x ) + sum( y ) ) )
  {
    throw IllegalInputException (
      plot_->getContext (),
      "invalid input data: NaN"
    );
  }

  xData_.ref ( makeContiguous( x ) );
  yData_.ref ( makeContiguous( y ) );

  size_ = x.size ();

  if ( size_ > maxPoints_ )
  {
    maxPoints_ = size_;
  }

  update_ ();
}


//-----------------------------------------------------------------------
//   addPoint
//-----------------------------------------------------------------------


void XYPlot::DataSet_::addPoint

  ( double  x,
    double  y )

{
  if ( Float::isNaN( x + y ) )
  {
    throw IllegalInputException (
      plot_->getContext (),
      "invalid input data: NaN"
    );
  }

  if ( size_ == xData_.size() )
  {
    idx_t  n = size_ + size_ / 2 + 8;

    if ( n > maxPoints_ )
    {
      n = maxPoints_;
    }

    xData_.reshape ( n );
    yData_.reshape ( n );
  }

  if ( size_ == maxPoints_ )
  {
    shift ( xData_, -1 );
    shift ( yData_, -1 );

    size_--;
    baseNumber_++;

    update_ ();
  }

  if ( size_ == 0 )
  {
    xRange_ = x;
    yRange_ = y;

    plot_->updated_ = false;
  }
  else
  {
    if      ( x < xRange_[0] )
    {
      xRange_[0] = x;
    }
    else if ( x > xRange_[1] )
    {
      xRange_[1] = x;
    }

    if      ( y < yRange_[0] )
    {
      yRange_[0] = y;
    }
    else if ( y > yRange_[1] )
    {
      yRange_[1] = y;
    }

    if ( plot_->updated_        &&
         x >= plot_->xRange_[0] &&
         x <= plot_->xRange_[1] &&
         y >= plot_->yRange_[0] &&
         y <= plot_->yRange_[1] )
    {
      pointList_.pushBack ( size_ );
    }
    else
    {
      plot_->updated_ = false;
    }
  }

  xData_[size_] = x;
  yData_[size_] = y;

  size_++;

  last_ = size_;
}


//-----------------------------------------------------------------------
//   addPoints
//-----------------------------------------------------------------------


void XYPlot::DataSet_::addPoints

  ( const Array<double>&  x,
    const Array<double>&  y )

{
  JEM_PRECHECK ( x.size() == y.size() );

  if ( Float::isNaN( sum( x ) + sum( y ) ) )
  {
    throw IllegalInputException (
      plot_->getContext (),
      "invalid input data: NaN"
    );
  }

  const idx_t  n = x.size ();

  double       u, v;
  idx_t        k;


  if ( n >= maxPoints_ )
  {
    k = n - maxPoints_;

    setPoints ( x[slice(k,END)], y[slice(k,END)] );

    return;
  }

  if ( size_ + n > xData_.size() )
  {
    k = size_ + n + xData_.size() / 2 + 8;

    if ( k > maxPoints_ )
    {
      k = maxPoints_;
    }

    xData_.reshape ( k );
    yData_.reshape ( k );
  }

  if ( size_ + n > maxPoints_ )
  {
    k = size_ + n - maxPoints_;

    shift ( xData_, -k );
    shift ( yData_, -k );

    size_       -= k;
    baseNumber_ += k;

    update_ ();
  }

  if ( size_ == 0 && n > 0 )
  {
    xRange_ = x[0];
    yRange_ = y[0];

    plot_->updated_ = false;
  }

  for ( idx_t i = 0; i < n; i++ )
  {
    u = x[i];
    v = y[i];

    if      ( u < xRange_[0] )
    {
      xRange_[0] = u;
    }
    else if ( u > xRange_[1] )
    {
      xRange_[1] = u;
    }

    if      ( v < yRange_[0] )
    {
      yRange_[0] = v;
    }
    else if ( v > yRange_[1] )
    {
      yRange_[1] = v;
    }

    if ( plot_->updated_        &&
         u >= plot_->xRange_[0] &&
         u <= plot_->xRange_[1] &&
         v >= plot_->yRange_[0] &&
         v <= plot_->yRange_[1] )
    {
      pointList_.pushBack ( size_ );
    }
    else
    {
      plot_->updated_ = false;
    }

    xData_[size_] = u;
    yData_[size_] = v;

    size_++;
  }

  last_ = size_;
}


//-----------------------------------------------------------------------
//   pointCount
//-----------------------------------------------------------------------


idx_t XYPlot::DataSet_::pointCount () const
{
  return size_;
}


//-----------------------------------------------------------------------
//   setMaxPoints
//-----------------------------------------------------------------------


void XYPlot::DataSet_::setMaxPoints ( idx_t count )
{
  if      ( count <= 0 )
  {
    count = maxOf ( count );
  }
  else if ( count < size_ )
  {
    size_ = count;

    update_ ();
  }

  maxPoints_ = count;
}


//-----------------------------------------------------------------------
//   getXRange
//-----------------------------------------------------------------------


XYPlot::Range XYPlot::DataSet_::getXRange

  ( const Range&  yRange ) const

{
  if ( this->yRange_[0] >= yRange[0] &&
       this->yRange_[1] <= yRange[1] )
  {
    return this->xRange_;
  }
  else
  {
    return getRange_ ( yRange, xData_, yData_ );
  }
}


//-----------------------------------------------------------------------
//   getYRange
//-----------------------------------------------------------------------


XYPlot::Range XYPlot::DataSet_::getYRange

  ( const Range&  xRange ) const

{
  if ( this->xRange_[0] >= xRange[0] &&
       this->xRange_[1] <= xRange[1] )
  {
    return this->yRange_;
  }
  else
  {
    return getRange_ ( xRange, yData_, xData_ );
  }
}


//-----------------------------------------------------------------------
//   sieve
//-----------------------------------------------------------------------


void XYPlot::DataSet_::sieve

  ( const Range&  xr,
    const Range&  yr )

{
  using jem::min;
  using jem::max;

  const double*  xp = xData_.addr ();
  const double*  yp = yData_.addr ();

  const idx_t    np = size_;

  double         t;
  idx_t          i, j;


  pointList_.resize ( np );

  // Check for special case: all points are within the specified
  // ranges.

  if ( xr[0] <= xRange_[0] &&
       xr[1] >= xRange_[1] &&
       yr[0] <= yRange_[0] &&
       yr[1] >= yRange_[1] )
  {
    for ( i = 0; i < np; i++ )
    {
      pointList_[i] = i;
    }

    first_ = 0;
    last_  = size_;

    return;
  }

  // Find all points within the specified ranges.

  for ( i = j = 0; i < np; i++ )
  {
    t = xp[i];

    if ( t < xr[0] || t > xr[1] )
    {
      continue;
    }

    t = yp[i];

    if ( t < yr[0] || t > yr[1] )
    {
      continue;
    }

    pointList_[j++] = i;
  }

  pointList_.resize ( j );

  // Find the first line segment within the x-range.

  for ( i = 0; i < np; i++ )
  {
    if ( xp[i] >= xr[0] )
    {
      break;
    }
  }

  first_ = max ( 0_idx, i - 1 );

  // Find the last line segment within the x-range.

  for ( i = np - 1; i >= 0; i-- )
  {
    if ( xp[i] <= xr[1] )
    {
      break;
    }
  }

  last_ = min ( np, i + 2 );
}


//-----------------------------------------------------------------------
//   drawFast
//-----------------------------------------------------------------------


void XYPlot::DataSet_::drawFast

  ( GLContext&    gtx,
    const double  scale[2],
    const double  bias [2],
    double        lod ) const

{
  const idx_t*  plist = pointList_.addr ();
  const idx_t   np    = pointList_.size ();

  const double  st    = 1.0 / lod;
  GLfloat*      vbuf  = plot_->scratch_.alloc ( 3 * np );

  idx_t         i, j, n;


  gtx.setColor ( lnColor_ );

  for ( i = j = n = 0; j < np;
        i++, j = (idx_t) ((double) i * st), n += 3 )
  {
    idx_t  k = plist[j];

    vbuf[n + 0] = (GLfloat) (bias[0] + scale[0] * xData_[k]);
    vbuf[n + 1] = (GLfloat) (bias[1] + scale[1] * yData_[k]);
    vbuf[n + 2] = 0.0F;
  }

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    ShaderDraw::draw ( gtx, n, vbuf, lnColor_, ShaderDraw::POINTS );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      funcs->glBegin ( GL_POINTS );

      for ( i = 0; i < n; i += 3 )
      {
        funcs->glVertex2 ( vbuf + i );
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   drawLines
//-----------------------------------------------------------------------


void XYPlot::DataSet_::drawLines

  ( GLContext&    gtx,
    const double  scale[2],
    const double  bias [2] ) const

{
  GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();
  const idx_t    count = last_ - first_;

  if ( (count > 1) && (lnWidth_ > 0.1) )
  {
    DisableScope   blendScope ( gtx, GL_BLEND, lnColor_.isOpaque() );

    const double*  xp   = xData_.addr ();
    const double*  yp   = yData_.addr ();
    GLfloat*       vbuf = plot_->scratch_.alloc ( 3 * count );

    idx_t          i, n;

    gtx.setColor ( lnColor_ );

    n = 0;

    for ( i = first_; i < last_; i++, n += 3 )
    {
      vbuf[n + 0] = (GLfloat) (bias[0] + scale[0] * xp[i]);
      vbuf[n + 1] = (GLfloat) (bias[1] + scale[1] * yp[i]);
      vbuf[n + 2] = 0.0F;
    }

    gtx.setLineWidth ( lnWidth_ );

    if      ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
    {
      ShaderDraw::draw ( gtx, n, vbuf, lnColor_,
                         ShaderDraw::LINE_STRIP );
    }
    else if ( funcs )
    {
      funcs->glBegin ( GL_LINE_STRIP );

      for ( idx_t i = 0; i < n; i += 3 )
      {
        funcs->glVertex2 ( vbuf + i );
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   drawMarkers
//-----------------------------------------------------------------------


void XYPlot::DataSet_::drawMarkers

  ( GLContext&    gtx,
    const double  scale[2],
    const double  bias [2],
    const Box&    bbox )     const

{
  if ( ptStyle_ )
  {
    const double*  xp    = xData_    .addr ();
    const double*  yp    = yData_    .addr ();
    const idx_t*   plist = pointList_.addr ();
    const idx_t    np    = pointList_.size ();
    const idx_t    nr0   = baseNumber_;

    GLfloat        pos[3];


    pos[2] = 0.0;

    ptStyle_->beginDraw ( gtx, bbox, np );

    for ( idx_t i = 0; i < np; i++ )
    {
      idx_t j =  plist[i];
      pos[0] = (GLfloat) (bias[0] + scale[0] * xp[j]);
      pos[1] = (GLfloat) (bias[1] + scale[1] * yp[j]);

      ptStyle_->drawMarker ( gtx, nr0 + j, pos );
    }

    ptStyle_->endDraw ( gtx );
  }
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void XYPlot::DataSet_::update_ ()
{
  const double*  xp = xData_.addr ();
  const double*  yp = yData_.addr ();

  const idx_t    np = size_;


  if ( np == 0 )
  {
    xRange_ = yRange_ = Range ( 0.0, 1.0 );
  }
  else
  {
    double  xmin = xp[0];
    double  xmax = xp[0];
    double  ymin = yp[0];
    double  ymax = yp[0];

    double  t;


    for ( idx_t i = 1; i < np; i++ )
    {
      t = xp[i];

      if      ( t < xmin )
      {
        xmin = t;
      }
      else if ( t > xmax )
      {
        xmax = t;
      }

      t = yp[i];

      if      ( t < ymin )
      {
        ymin = t;
      }
      else if ( t > ymax )
      {
        ymax = t;
      }
    }

    xRange_[0] = xmin;
    xRange_[1] = xmax;
    yRange_[0] = ymin;
    yRange_[1] = ymax;
  }

  pointList_.resize ( 0 );

  first_ = 0;
  last_  = np;

  plot_->updated_ = false;
}


//-----------------------------------------------------------------------
//   getRange_
//-----------------------------------------------------------------------


XYPlot::Range XYPlot::DataSet_::getRange_

  ( const Range&          vRange,
    const Array<double>&  uData,
    const Array<double>&  vData ) const

{
  const double*  up    = uData.addr ();
  const double*  vp    = vData.addr ();

  const idx_t    np    = size_;

  const double   vmin  = vRange[0];
  const double   vmax  = vRange[1];

  double         umin  = 0.0;
  double         umax  = 1.0;
  bool           empty = true;
  double         u, v;


  for ( idx_t i = 0; i < np; i++ )
  {
    v = vp[i];

    if ( v < vmin || v > vmax )
    {
      continue;
    }

    u = up[i];

    if      ( empty )
    {
      umin  = umax = u;
      empty = false;
    }
    else if ( u < umin )
    {
      umin  = u;
    }
    else if ( u > umax )
    {
      umax  = u;
    }
  }

  return Range ( umin, umax );
}


JEM_END_PACKAGE( gl )
