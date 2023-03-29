
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
#include <jem/base/rtti.h>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/utilities.h>
#include <jem/util/Properties.h>
#include <jem/gl/render.h>
#include <jem/gl/Box.h>
#include <jem/gl/Menu.h>
#include <jem/gl/Symbols.h>
#include <jem/gl/Commands.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/PropertyNames.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/ShaderDraw.h>
#include <jem/gl/NumberMarker.h>
#include <jem/gl/SymbolMarker.h>
#include <jem/gl/TextProjector.h>
#include <jem/gl/Transform.h>
#include <jem/gl/XYPlot.h>
#include "XYPlotSet.h"


JEM_DEFINE_CLASS( jem::gl::XYPlot );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class XYPlot
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  XYPlot::TICK_FORMAT_ = "%.3g";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


XYPlot::XYPlot ( const String& name ) :

  Super ( name )

{
  init_ ( Size2f( 1.0F ) );
}


XYPlot::XYPlot

  ( const String&  name,
    const Size2f&  sz ) :

    Super ( name )

{
  init_ ( sz );
}


XYPlot::~XYPlot ()
{}


//-----------------------------------------------------------------------
//   getBBox
//-----------------------------------------------------------------------


void XYPlot::getBBox ( Box& box ) const
{
  GLfloat  a[3];
  GLfloat  b[3];

  a[0] =  bbox_[0];
  a[1] =  bbox_[1];
  a[2] = -depth_;

  b[0] =  bbox_[2];
  b[1] =  bbox_[3];
  b[2] =  depth_;

  box.setCorners3D ( a, b );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void XYPlot::configure ( const Properties& props )
{
  if ( ! props.contains( myName_ ) )
  {
    return;
  }

  Properties  myProps = props.findProps ( myName_ );

  bool        newWidth;
  bool        newHeight;
  Vertex2f    keyPos;
  double      w, h;
  bool        choice;


  Super::configure ( props );

  w = (double) width_;
  h = (double) height_;

  newWidth  = myProps.find ( w, PropNames::WIDTH,  1.0e-10, 1.0e10 );
  newHeight = myProps.find ( h, PropNames::HEIGHT, 1.0e-10, 1.0e10 );

  if ( newWidth || newHeight )
  {
    resize ( Size2f( (GLfloat) w, (GLfloat) h ) );
  }

  keyPos = 0.0F;

  if ( Vertex::configure( keyPos, PropNames::KEY_POS, myProps ) )
  {
    setKeyPos ( keyPos[0], keyPos[1] );
  }

  if ( myProps.find( choice, PropNames::SHOW_KEY ) )
  {
    setOption ( SHOW_KEY, choice );
  }

  myProps.find ( title,      PropNames::TITLE  );
  myProps.find ( xLabel,     PropNames::XLABEL );
  myProps.find ( yLabel,     PropNames::YLABEL );
  myProps.find ( scaleStep_, PropNames::SCALE_STEP, 0.0, 100.0 );

  setRanges_   ( myProps );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void XYPlot::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  Vertex2f    keyPos  ( keyPos_[0], keyPos_[1] );

  bool        showKey = (bool) (options_ & SHOW_KEY);

  double      width   = width_;
  double      height  = height_;


  Super::getConfig  ( props );

  myProps.set ( PropNames::WIDTH,      width   );
  myProps.set ( PropNames::HEIGHT,     height  );
  myProps.set ( PropNames::SHOW_KEY,   showKey );

  Vertex::getConfig ( PropNames::KEY_POS, myProps, keyPos );

  myProps.set ( PropNames::TITLE,      title      );
  myProps.set ( PropNames::XLABEL,     xLabel     );
  myProps.set ( PropNames::YLABEL,     yLabel     );
  myProps.set ( PropNames::SCALE_STEP, scaleStep_ );

  if ( xAutoScale_ )
  {
    myProps.set ( PropNames::XRANGE, "auto" );
  }
  else
  {
    Array<double>  r ( 2 );

    r[0] = xRange_[0];
    r[1] = xRange_[1];

    myProps.set ( PropNames::XRANGE, r );
  }

  if ( yAutoScale_ )
  {
    myProps.set ( PropNames::YRANGE, "auto" );
  }
  else
  {
    Array<double>  r ( 2 );

    r[0] = yRange_[0];
    r[1] = yRange_[1];

    myProps.set ( PropNames::YRANGE, r );
  }
}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


void XYPlot::listProps ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  Super::listProps ( props );

  myProps.set ( PropNames::WIDTH,
                "The width of the plot" );
  myProps.set ( PropNames::HEIGHT,
                "The height of the plot" );
  myProps.set ( PropNames::SHOW_KEY,
                "Whether to show the plot key(s)" );
  myProps.set ( PropNames::KEY_POS,
                "The relative position of the plot key(s)" );
  myProps.set ( PropNames::TITLE,
                "The title of the plot" );
  myProps.set ( PropNames::XLABEL,
                "The label below the x-axis" );
  myProps.set ( PropNames::YLABEL,
                "The label left of the y-axis" );
  myProps.set ( PropNames::SCALE_STEP,
                "The relative rounding (on a log scale) of the "
                "range of the axes" );
  myProps.set ( PropNames::XRANGE,
                "The range of the x-axis" );
  myProps.set ( PropNames::YRANGE,
                "The range of the y-axis" );
}


//-----------------------------------------------------------------------
//   redraw
//-----------------------------------------------------------------------


void XYPlot::redraw ( RenderContext& rtx )
{
  GLContext&     gtx     = rtx.getGLContext            ();
  GLFuncs1*      funcs   = gtx.findFunctions<GLFuncs1> ();

  AttribScope    attribs ( gtx, GLContext::COLOR    |
                                GLContext::LIGHTING |
                                GLContext::LINE_WIDTH );
  Box            bbox;
  GLfloat        psize[2];


  gtx.setLighting ( false );

  if ( ! updated_ )
  {
    update_ ();
  }

  getBBox      ( bbox  );
  getPlotSize_ ( gtx, psize );

  if ( psize[0] < 100.0 || psize[1] < 100.0 )
  {
    setPlotRect_ ();
    drawFrame_   ( gtx );
    drawData_    ( gtx, bbox );
  }
  else
  {
    TextProjector  txp;

    GLfloat  space[3];
    int      ticks[2];

    GLint    stpRepeat;
    GLint    stpPattern;

    if ( ! gtx.checkVersion( GLVersion::VERSION_3_3 ) && funcs )
    {
      funcs->glGetParameter ( GL_LINE_STIPPLE_REPEAT,  &stpRepeat  );
      funcs->glGetParameter ( GL_LINE_STIPPLE_PATTERN, &stpPattern );
      funcs->glLineStipple  ( 1, 255 );
    }

    psize[0] = width_  / psize[0];
    psize[1] = height_ / psize[1];

    setFontSize_  ( txp.font, psize );

    space[0] = 0.5F * psize[1] * (GLfloat) txp.font.getHeight ();
    space[1] =        space[0];
    space[2] = depth_ / 3.0F;

    setPlotRect_  ( ticks, psize, space, txp.font );
    drawFrame_    ( gtx );
    drawData_     ( gtx, bbox );

    if ( ticks[0] > 0 )
    {
      drawXTicks_ ( gtx, txp, ticks[0], psize, space );
    }

    if ( ticks[1] > 0 )
    {
      drawYTicks_ ( gtx, txp, ticks[1], psize, space );
    }

    if ( title.size() + xLabel.size() + yLabel.size() > 0 )
    {
      drawLabels_ ( gtx, txp, psize, space );
    }

    if ( options_ & SHOW_KEY )
    {
      drawKey_    ( gtx, txp, bbox, psize, space );
    }

    bbox_[0] = 0.0F;
    bbox_[1] = 0.0F;
    bbox_[2] = width_;
    bbox_[3] = height_;

    if ( ! gtx.checkVersion( GLVersion::VERSION_3_3 ) && funcs )
    {
      funcs->glLineStipple ( stpRepeat, (GLushort) stpPattern );
    }
  }
}


//-----------------------------------------------------------------------
//   redrawFast
//-----------------------------------------------------------------------


void XYPlot::redrawFast

  ( RenderContext&  rtx,
    double          lod )

{
  if ( lod <= 0.0 )
  {
    return;
  }

  if ( lod >= 1.0 )
  {
    Self::redraw ( rtx ); return;
  }

  GLContext&     gtx        = rtx.getGLContext ();
  const idx_t    n          = dataSets_.size   ();

  DisableScope   blendScope ( gtx, GL_BLEND );
  AttribScope    attribs    ( gtx, GLContext::COLOR      |
                                   GLContext::LIGHTING   |
                                   GLContext::LINE_WIDTH |
                                   GLContext::POINT_SIZE );

  double         scale[2];
  double         bias [2];


  gtx.setLighting ( false );

  if ( ! updated_ )
  {
    update_ ();
  }

  scale[0] = (prect_[2] - prect_[0]) / (xRange_[1] - xRange_[0]);
  scale[1] = (prect_[3] - prect_[1]) / (yRange_[1] - yRange_[0]);

  bias [0] = prect_[0] - scale[0] * xRange_[0];
  bias [1] = prect_[1] - scale[1] * yRange_[0];

  for ( idx_t i = 0; i < n; i++ )
  {
    dataSets_.getAs<DataSet_>(i)->drawFast ( gtx, scale, bias, lod );
  }

  drawFrame_ ( gtx );
}


//-----------------------------------------------------------------------
//   exeCommand
//-----------------------------------------------------------------------


bool XYPlot::exeCommand

  ( RenderContext&     rtx,
    const String&      cmd,
    const Properties&  params )

{
  if ( cmd == Commands::AUTO_SCALE )
  {
    setAutoScale ();
    return true;
  }

  if ( cmd == Commands::HIDE_KEY )
  {
    options_ &= ~SHOW_KEY;
    return true;
  }

  if ( cmd == Commands::SHOW_KEY )
  {
    options_ |= SHOW_KEY;
    return true;
  }

  if ( cmd == Commands::HIDE_XTICKS )
  {
    options_ &= ~SHOW_XTICKS;
    return true;
  }

  if ( cmd == Commands::SHOW_XTICKS )
  {
    options_ |= SHOW_XTICKS;
    return true;
  }

  if ( cmd == Commands::HIDE_YTICKS )
  {
    options_ &= ~SHOW_YTICKS;
    return true;
  }

  if ( cmd == Commands::SHOW_YTICKS )
  {
    options_ |= SHOW_YTICKS;
    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   listOptions
//-----------------------------------------------------------------------


void XYPlot::listOptions ( Menu& menu ) const
{
  bool  showXTicks = ((options_ & SHOW_XTICKS) != 0);
  bool  showYTicks = ((options_ & SHOW_YTICKS) != 0);

  if ( ! (xAutoScale_ && yAutoScale_) )
  {
    menu.addEntry ( Commands::AUTO_SCALE );
  }

  if ( dataSets_.size() > 0 )
  {
    bool  showKey = (bool) (options_ & SHOW_KEY);

    menu.addOption ( Commands::SHOW_KEY,
                     Commands::HIDE_KEY, showKey );
  }

  menu.addOption ( Commands::SHOW_XTICKS,
                   Commands::HIDE_XTICKS, showXTicks );
  menu.addOption ( Commands::SHOW_YTICKS,
                   Commands::HIDE_YTICKS, showYTicks );
}


//-----------------------------------------------------------------------
//   setFont
//-----------------------------------------------------------------------


bool XYPlot::setFont ()
{
  const idx_t  n = dataSets_.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    DataSet_*     dset  = dataSets_.getAs<DataSet_> ( i );
    MarkerStyle*  style = dset    ->getPointStyle   ();

    if ( style )
    {
      style->setFont ();
    }
  }

  return true;
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void XYPlot::clear ()
{
  dataSets_.clear ();

  if ( xAutoScale_ )
  {
    xRange_ = Range ( 0.0, 1.0 );
  }

  if ( yAutoScale_ )
  {
    yRange_ = Range ( 0.0, 1.0 );
  }

  updated_   = true;
  nextStyle_ = 0;
}


//-----------------------------------------------------------------------
//   newDataSet
//-----------------------------------------------------------------------


XYPlot::DataSet* XYPlot::newDataSet

  ( const String&  key,
    Style          style )

{
  Ref<DataSet_>  dset  = newInstance<DataSet_> ( this );
  Color          color = Color::black          ();


  switch ( nextStyle_ % 7 )
  {
  case 1: color = Color::red    (); break;
  case 2: color = Color::green  (); break;
  case 3: color = Color::blue   (); break;
  case 4: color = Color::yellow (); break;
  case 5: color = Color::orange (); break;
  case 6: color = Color::purple (); break;
  }

  dset->setLineColor ( color );

  if ( style & LINES )
  {
    dset->setLineWidth ( 1.2F );
  }

  if ( style & POINTS )
  {
    char  symbol = Symbols::PLUS;

    switch ( nextStyle_ % 7 )
    {
    case 2: symbol = Symbols::DIAMOND;  break;
    case 3: symbol = Symbols::TRIANGLE; break;
    case 4: symbol = Symbols::CIRCLE;   break;
    case 1: symbol = Symbols::SQUARE;   break;
    case 5: symbol = Symbols::STAR;     break;
    case 6: symbol = Symbols::DISC;     break;
    }

    dset->setPointStyle (
      newInstance<SymbolMarker> ( symbol, color )
    );
  }

  nextStyle_++;

  if ( nextStyle_ > 100 )
  {
    nextStyle_ = 0;
  }

  dset->key = key;
  updated_  = false;

  dataSets_.pushBack ( dset.get() );

  return dset.get ();
}


XYPlot::DataSet* XYPlot::newDataSet

  ( const String&         key,
    const Array<double>&  xData,
    const Array<double>&  yData,
    Style                 style )

{
  JEM_PRECHECK ( xData.size() == yData.size() );

  DataSet*  dset = newDataSet ( key, style );

  dset->setPoints ( xData, yData );

  return dset;
}


//-----------------------------------------------------------------------
//   findDataSet
//-----------------------------------------------------------------------


XYPlot::DataSet*
  XYPlot::findDataSet ( const String& key ) const noexcept
{
  const idx_t  n = dataSets_.size ();

  DataSet_*    dset;


  for ( idx_t i = 0; i < n; i++ )
  {
    dset = dataSets_.getAs<DataSet_> ( i );

    if ( dset->key == key )
    {
      return dset;
    }
  }

  return nullptr;
}


//-----------------------------------------------------------------------
//   eraseDataSet
//-----------------------------------------------------------------------


bool XYPlot::eraseDataSet ( DataSet* dset )
{
  const idx_t  n = dataSets_.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( dataSets_.getAs<DataSet_>(i) == dset )
    {
      dataSets_.erase ( i );

      updated_ = false;

      if ( dataSets_.size() == 0 )
      {
        nextStyle_ = 0;
      }

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   getDataSets
//-----------------------------------------------------------------------


Array<XYPlot::DataSet*> XYPlot::getDataSets () const
{
  const idx_t      n = dataSets_.size ();

  Array<DataSet*>  dsets ( n );


  for ( idx_t i = 0; i < n; i++ )
  {
    dsets[i] = dataSets_.getAs<DataSet_> ( i );
  }

  return dsets;
}


//-----------------------------------------------------------------------
//   resize
//-----------------------------------------------------------------------


void XYPlot::resize ( const Size2f& sz )
{
  JEM_PRECHECK ( sz.width > 0.0 && sz.height > 0.0 );

  width_  = sz.width;
  height_ = sz.height;

  setPlotRect_ ();
}


//-----------------------------------------------------------------------
//   setOption
//-----------------------------------------------------------------------


void XYPlot::setOption

  ( Option  option,
    bool    choice )

{
  options_.set ( option, choice );
}


//-----------------------------------------------------------------------
//   setOptions
//-----------------------------------------------------------------------


void XYPlot::setOptions ( Options options )
{
  options_ = options;
}


//-----------------------------------------------------------------------
//   toggleOption
//-----------------------------------------------------------------------


void XYPlot::toggleOption ( Option option )
{
  options_.toggle ( option );
}


//-----------------------------------------------------------------------
//   setKeyPos
//-----------------------------------------------------------------------


void XYPlot::setKeyPos ( GLfloat x, GLfloat y )
{
  keyPos_[0] = clamped ( x, 1.0F );
  keyPos_[1] = clamped ( y, 1.0F );
}


//-----------------------------------------------------------------------
//   set[X|Y]Range
//-----------------------------------------------------------------------


void XYPlot::setXRange ( const Range& range )
{
  JEM_PRECHECK ( (range[1] - range[0]) >= 0.0 );

  xRange_     = range;
  xAutoScale_ = false;
  updated_    = false;

  adjustRange_ ( xRange_ );
}


void XYPlot::setYRange ( const Range& range )
{
  JEM_PRECHECK ( (range[1] - range[0]) >= 0.0 );

  yRange_     = range;
  yAutoScale_ = false;
  updated_    = false;

  adjustRange_ ( yRange_ );
}


//-----------------------------------------------------------------------
//   get[X|Y]Range
//-----------------------------------------------------------------------


XYPlot::Range XYPlot::getXRange () const
{
  if ( xAutoScale_ && ! updated_ )
  {
    const_cast<Self*>(this)->update_ ();
  }

  return xRange_;
}


XYPlot::Range XYPlot::getYRange () const
{
  if ( yAutoScale_ && ! updated_ )
  {
    const_cast<Self*>(this)->update_ ();
  }

  return yRange_;
}


//-----------------------------------------------------------------------
//   set[X|Y]AutoScale
//-----------------------------------------------------------------------


void XYPlot::setAutoScale ()
{
  setXAutoScale ();
  setYAutoScale ();
}


void XYPlot::setXAutoScale ()
{
  if ( ! xAutoScale_ )
  {
    xAutoScale_ = true;
    updated_    = false;
  }
}


void XYPlot::setYAutoScale ()
{
  if ( ! yAutoScale_ )
  {
    yAutoScale_ = true;
    updated_    = false;
  }
}


//-----------------------------------------------------------------------
//   setScaleStep
//-----------------------------------------------------------------------


void XYPlot::setScaleStep ( double step )
{
  JEM_PRECHECK ( step >= 0.0 );

  scaleStep_ = step;
}


//-----------------------------------------------------------------------
//   mapPoint
//-----------------------------------------------------------------------


Vertex2d XYPlot::mapPoint ( const Vertex2d& p ) const
{
  if ( ! updated_ )
  {
    const_cast<Self*>(this)->update_ ();
  }

  Vertex2d  v;
  double    scale[2];
  double    bias [2];

  scale[0] = (prect_[2] - prect_[0]) / (xRange_[1] - xRange_[0]);
  scale[1] = (prect_[3] - prect_[1]) / (yRange_[1] - yRange_[0]);

  bias [0] = prect_[0] - scale[0] * xRange_[0];
  bias [1] = prect_[1] - scale[1] * yRange_[0];

  v[0] = (p[0] - bias[0]) / scale[0];
  v[1] = (p[1] - bias[1]) / scale[1];

  return v;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void XYPlot::init_ ( const Size2f& sz )
{
  hidden      = false;
  options_    = DEFAULT_OPTIONS;
  xRange_     = Range ( 0.0, 1.0 );
  yRange_     = xRange_;
  xAutoScale_ = true;
  yAutoScale_ = true;
  scaleStep_  = 0.1;
  updated_    = true;
  nextStyle_  = 0;
  keyPos_[0]  = 0.90F;
  keyPos_[1]  = 0.90F;

  resize ( sz );
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void XYPlot::update_ ()
{
  const idx_t  n = dataSets_.size ();


  if ( xAutoScale_ )
  {
    Range  r ( 0.0, 1.0 );

    xRange_ = r;

    for ( idx_t i = 0; i < n; i++ )
    {
      DataSet_*  dset = dataSets_.getAs<DataSet_> ( i );

      if ( yAutoScale_ )
      {
        r = dset->getXRange ();
      }
      else
      {
        r = dset->getXRange ( yRange_ );
      }

      if ( i == 0 )
      {
        xRange_ = r;
      }
      else
      {
        if ( r[0] < xRange_[0] )
        {
          xRange_[0] = r[0];
        }

        if ( r[1] > xRange_[1] )
        {
          xRange_[1] = r[1];
        }
      }
    }

    adjustRange_ ( xRange_, scaleStep_ );
  }

  if ( yAutoScale_ )
  {
    Range  r ( 0.0, 1.0 );

    yRange_ = r;

    for ( idx_t i = 0; i < n; i++ )
    {
      DataSet_*  dset = dataSets_.getAs<DataSet_> ( i );

      if ( xAutoScale_ )
      {
        r = dset->getYRange ();
      }
      else
      {
        r = dset->getYRange ( xRange_ );
      }

      if ( i == 0 )
      {
        yRange_ = r;
      }
      else
      {
        if ( r[0] < yRange_[0] )
        {
          yRange_[0] = r[0];
        }

        if ( r[1] > yRange_[1] )
        {
          yRange_[1] = r[1];
        }
      }
    }

    adjustRange_ ( yRange_, scaleStep_ );
  }

  for ( idx_t i = 0; i < n; i++ )
  {
    dataSets_.getAs<DataSet_>(i)->sieve ( xRange_, yRange_ );
  }

  updated_ = true;
}


//-----------------------------------------------------------------------
//   drawData_
//-----------------------------------------------------------------------


void XYPlot::drawData_

  ( GLContext&  gtx,
    const Box&  bbox ) const

{
  GLFuncs1Core*  funcs = gtx.findFunctions<GLFuncs1Core> ();
  const idx_t    n     = dataSets_.size ();

  double         scale[2];
  double         bias [2];


  if ( funcs )
  {
    Rect     vport = gtx.getViewport ();

    GLfloat  mvp[16];

    GLfloat  x0, x1, y0, y1;
    GLsizei  width, height;
    GLint    x, y;

    // Calculate the plot rectangle corner points in screen
    // coordinates.

    gtx.getModelViewProj ( mvp );

    x0 = mvp[0] * prect_[0] + mvp[4] * prect_[1] + mvp[12];
    y0 = mvp[1] * prect_[0] + mvp[5] * prect_[1] + mvp[13];

    x1 = mvp[0] * prect_[2] + mvp[4] * prect_[3] + mvp[12];
    y1 = mvp[1] * prect_[2] + mvp[5] * prect_[3] + mvp[13];

    x0 = 0.5F * x0 + 0.5F;
    x1 = 0.5F * x1 + 0.5F;
    y0 = 0.5F * y0 + 0.5F;
    y1 = 0.5F * y1 + 0.5F;

    x      = (GLint) ((GLfloat) vport.width  * x0) + vport.x;
    y      = (GLint) ((GLfloat) vport.height * y0) + vport.y;
    width  = (GLint) ((GLfloat) vport.width  * (x1 - x0));
    height = (GLint) ((GLfloat) vport.height * (y1 - y0));

    funcs->glScissor ( x, y, width, height );
    funcs->glEnable  ( GL_SCISSOR_TEST );
  }

  // Draw the data sets.

  scale[0] = (prect_[2] - prect_[0]) / (xRange_[1] - xRange_[0]);
  scale[1] = (prect_[3] - prect_[1]) / (yRange_[1] - yRange_[0]);

  bias [0] = prect_[0] - scale[0] * xRange_[0];
  bias [1] = prect_[1] - scale[1] * yRange_[0];

  // First draw the lines.

  for ( idx_t i = 0; i < n; i++ )
  {
    dataSets_.getAs<DataSet_>(i)->drawLines ( gtx, scale, bias );
  }

  // Disable the scissor rectangle.

  if ( funcs )
  {
    funcs->glDisable ( GL_SCISSOR_TEST );
  }

  // Then draw the markers. This must be done with the clipping planes
  // disabled because a marker may set up its own projection or
  // model-view matrix.

  for ( idx_t i = 0; i < n; i++ )
  {
    dataSets_.getAs<DataSet_>(i) ->

      drawMarkers ( gtx, scale, bias, bbox );
  }
}


//-----------------------------------------------------------------------
//   drawFrame_
//-----------------------------------------------------------------------


void XYPlot::drawFrame_ ( GLContext& gtx ) const
{
  gtx.setColor     ( Color::black() );
  gtx.setLineWidth ( 2.0F );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    GLfloat  pos[15];

    pos[ 0] = prect_[0]; pos[ 1] = prect_[1]; pos[ 2] = 0.0F;
    pos[ 3] = prect_[2]; pos[ 4] = prect_[1]; pos[ 5] = 0.0F;
    pos[ 6] = prect_[2]; pos[ 7] = prect_[3]; pos[ 8] = 0.0F;
    pos[ 9] = prect_[0]; pos[10] = prect_[3]; pos[11] = 0.0F;
    pos[12] = prect_[0]; pos[13] = prect_[1]; pos[14] = 0.0F;

    ShaderDraw::draw ( gtx, 15, pos, Color::black(),
                       ShaderDraw::LINE_STRIP );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      funcs->glBegin ( GL_LINE_STRIP );
      {
        funcs->glVertex ( prect_[0], prect_[1] );
        funcs->glVertex ( prect_[2], prect_[1] );
        funcs->glVertex ( prect_[2], prect_[3] );
        funcs->glVertex ( prect_[0], prect_[3] );
        funcs->glVertex ( prect_[0], prect_[1] );
      }
      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   drawXTicks_
//-----------------------------------------------------------------------


void XYPlot::drawXTicks_

  ( GLContext&      gtx,
    TextProjector&  txp,
    int             count,
    const GLfloat   psize[2],
    const GLfloat   space[3] )


{
  GLFunctions1*  funcs    = gtx.findFunctions<GLFunctions1> ();
  const GLfloat  fnHeight = psize[1] * (GLfloat) txp.font.getHeight ();
  GLfloat*       vertPos  = nullptr;

  char           buf[64];
  Color          col;
  double         v, dv;
  GLfloat        x, dx, y, dy;
  GLfloat        w;
  int            i, k;


  txp.hAlignment = CENTERED;
  txp.vAlignment = TOP_ALIGNED;

  // Draw the values along the x-axis.

  {
    OrthoTextScope  textScope ( gtx, txp );

    gtx.setColor ( Color::black() );

    w  =  prect_[2] - prect_[0];
    y  =  bbox_ [1];
    dx =  w                        / ((GLfloat) count - 1.0F);
    dv = (xRange_[1] - xRange_[0]) / ((GLfloat) count - 1.0F);
    k  =  std::sprintf ( buf, TICK_FORMAT_, xRange_[0] );

    txp.setCursor ( prect_[0], y );
    txp.writeLine ( gtx,  buf, k );

    for ( i = 1; i < count - 1; i++ )
    {
      x = prect_ [0] + (GLfloat) i * dx;
      v = xRange_[0] + (GLfloat) i * dv;
      k = std::sprintf ( buf, TICK_FORMAT_, v );

      txp.setCursor ( x, y );
      txp.writeLine ( gtx, buf, k );
    }

    k = std::sprintf ( buf, TICK_FORMAT_, xRange_[1] );

    txp.setCursor ( prect_[2], y );
    txp.writeLine ( gtx,  buf, k );
  }

  bbox_[1] -= space[1] + fnHeight;

  // Draw the tick marks.

  col = Color ( 0.25F, 0.25F, 0.25F, 0.5F );

  gtx.setColor     ( col );
  gtx.setLineWidth ( 1.F );

  vertPos = scratch_.alloc ( 6 * (count - 2) );

  for ( i = 0; i < count - 2; i++ )
  {
    x = prect_[0] + (GLfloat)(i+1) * dx;

    vertPos[6 * i + 0] = x;
    vertPos[6 * i + 1] = prect_[1];
    vertPos[6 * i + 2] = -space[2];
    vertPos[6 * i + 3] = x;
    vertPos[6 * i + 4] = prect_[3];
    vertPos[6 * i + 5] = -space[2];
  }

  if      ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    ShaderDraw::drawStippleLines ( gtx, 6 * (count - 2), vertPos,
                                   col, 1, 255 );
  }
  else if ( funcs )
  {
    funcs->glEnable ( GL_LINE_STIPPLE );
    funcs->glBegin  ( GL_LINES );

    for ( i = 0; i < (count - 2); i++ )
    {
      funcs->glVertex3 ( vertPos + (6 * i) );
      funcs->glVertex3 ( vertPos + (6 * i + 3) );
    }

    funcs->glEnd     ();
    funcs->glDisable ( GL_LINE_STIPPLE );
  }

  // Draw the y-axis if visible.

  if ( xRange_[0] < 0.0 && xRange_[1] > 0.0 )
  {
    vertPos = scratch_.alloc ( 6 );

    x = prect_[0] - (GLfloat)

      ( (xRange_[0] * w) / (xRange_[1] - xRange_[0]) );

    vertPos[0] = x; vertPos[1] = prect_[1]; vertPos[2] = -space[2];
    vertPos[3] = x; vertPos[4] = prect_[3]; vertPos[5] = -space[2];

    if      ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
    {
      ShaderDraw::draw ( gtx, 6, vertPos, col, ShaderDraw::LINES );
    }
    else if ( funcs )
    {
      funcs->glBegin ( GL_LINES );
      {
        funcs->glVertex3 ( vertPos + 0 );
        funcs->glVertex3 ( vertPos + 3 );
      }
      funcs->glEnd ();
    }
  }

  col = Color::black ();

  gtx.setColor     ( col );
  gtx.setLineWidth ( 2.F );

  count   = 2 * (count - 1);
  dx      = w / (GLfloat) count;
  dy      = 0.5F * space[1];
  vertPos = scratch_.alloc ( 6 * (count - 1) );

  for ( i = 0; i < (count - 1); i++ )
  {
    x = prect_[0] + (GLfloat) (i + 1) * dx;

    vertPos[6 * i + 0] = x;
    vertPos[6 * i + 1] = prect_[1];
    vertPos[6 * i + 2] = 0.0F;
    vertPos[6 * i + 3] = x;
    vertPos[6 * i + 4] = prect_[1] + dy;
    vertPos[6 * i + 5] = 0.0F;
  }

  if      ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    ShaderDraw::draw ( gtx, 6 * (count - 1), vertPos, col,
                       ShaderDraw::LINES );
  }
  else if ( funcs )
  {
    funcs->glBegin ( GL_LINES );

    for ( i = 0; i < (count - 1); i++ )
    {
      funcs->glVertex2 ( vertPos + (6 * i) );
      funcs->glVertex2 ( vertPos + (6 * i + 3) );
    }

    glEnd ();
  }
}


//-----------------------------------------------------------------------
//   drawYTicks_
//-----------------------------------------------------------------------


void XYPlot::drawYTicks_

  ( GLContext&      gtx,
    TextProjector&  txp,
    int             count,
    const GLfloat   psize[2],
    const GLfloat   space[3] )

{
  GLFunctions1*  funcs   = gtx.findFunctions<GLFunctions1> ();
  GLfloat*       vertPos = nullptr;

  char           buf[64];
  Color          col;
  double         v,  dv;
  GLfloat        x,  dx, y, dy, h;
  GLsizei        w,  wmax;
  int            i,  k;


  txp.hAlignment = RIGHT_ALIGNED;
  txp.vAlignment = CENTERED;

  // Draw the y-values.

  {
    OrthoTextScope  textScope ( gtx, txp );

    gtx.setColor ( Color::black() );

    h  =  prect_[3]  - prect_[1];
    x  =  bbox_ [0];
    dy =  h                        / ((GLfloat) count - 1.0F);
    dv = (yRange_[1] - yRange_[0]) / ((GLfloat) count - 1.0F);
    k  =  std::sprintf ( buf, TICK_FORMAT_, yRange_[0] );

    txp.setCursor ( x, prect_[1] );

    wmax = txp.writeLine ( gtx, buf, k );

    for ( i = 1; i < count - 1; i++ )
    {
      y = prect_ [1] + (GLfloat) i * dy;
      v = yRange_[0] + (GLfloat) i * dv;
      k = std::sprintf ( buf, TICK_FORMAT_, v );

      txp.setCursor ( x, y );

      w = txp.writeLine ( gtx, buf, k );

      if ( w > wmax )
      {
        wmax = w;
      }
    }

    k = std::sprintf ( buf, TICK_FORMAT_, yRange_[1] );

    txp.setCursor ( x,  prect_[3] );

    w = txp.writeLine ( gtx, buf, k );

    if ( w > wmax )
    {
      wmax = w;
    }
  }

  bbox_[0] -= space[0] + (GLfloat) wmax * psize[0];

  // Draw the tick marks.

  vertPos = scratch_.alloc ( 6 * (count - 2) );
  col     = Color ( 0.25F, 0.25F, 0.25F, 0.5F );

  gtx.setColor     ( col );
  gtx.setLineWidth ( 1.F );

  for ( i = 0; i < (count - 2); i++ )
  {
    y = prect_[1] + (GLfloat) (i + 1) * dy;

    vertPos[6 * i + 0] =  prect_[0];
    vertPos[6 * i + 1] =  y;
    vertPos[6 * i + 2] = -space[2];
    vertPos[6 * i + 3] =  prect_[2];
    vertPos[6 * i + 4] =  y;
    vertPos[6 * i + 5] = -space[2];
  }

  if      ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    ShaderDraw::drawStippleLines ( gtx, 6 * (count - 2), vertPos, col,
                                   1, 255 );
  }
  else if ( funcs )
  {
    funcs->glEnable ( GL_LINE_STIPPLE );
    funcs->glBegin  ( GL_LINES );

    for ( i = 0; i < (count - 2); i++ )
    {
      funcs->glVertex3 ( vertPos + (6 * i) );
      funcs->glVertex3 ( vertPos + (6 * i + 3) );
    }

    funcs->glEnd     ();
    funcs->glDisable ( GL_LINE_STIPPLE );
  }

  // Draw the x-axis if visible.

  if ( yRange_[0] < 0.0 && yRange_[1] > 0.0 )
  {
    vertPos = scratch_.alloc ( 6 );

    y = prect_[1] - (GLfloat)

      ( (yRange_[0] * h) / (yRange_[1] - yRange_[0]) );

    vertPos[0] = prect_[0]; vertPos[1] = y; vertPos[2] = -space[2];
    vertPos[3] = prect_[2]; vertPos[4] = y; vertPos[5] = -space[2];

    if      ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
    {
      ShaderDraw::draw ( gtx, 6, vertPos, col, ShaderDraw::LINES );
    }
    else if ( funcs )
    {
      funcs->glBegin ( GL_LINES );
      {
        funcs->glVertex3 ( vertPos + 0 );
        funcs->glVertex3 ( vertPos + 3 );
      }
      funcs->glEnd ();
    }
  }

  col = Color::black ();

  gtx.setColor     ( col );
  gtx.setLineWidth ( 2.F );

  count   = 2    * (count - 1);
  dx      = 0.5F * space[0];
  dy      = h    / (GLfloat) count;
  vertPos = scratch_.alloc ( 6 * (count - 1) );

  for ( i = 0; i < (count - 1); i++ )
  {
    y = prect_[1] + (GLfloat) (i + 1) * dy;

    vertPos[6 * i + 0] = prect_[0];
    vertPos[6 * i + 1] = y;
    vertPos[6 * i + 2] = 0.0F;
    vertPos[6 * i + 3] = prect_[0] + dx;
    vertPos[6 * i + 4] = y;
    vertPos[6 * i + 5] = 0.0F;
  }

  if      ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    ShaderDraw::draw ( gtx, 6 * (count - 1), vertPos, col,
                       ShaderDraw::LINES );
  }
  else if ( funcs )
  {
    funcs->glBegin ( GL_LINES );

    for ( i = 0; i < count-1; i++ )
    {
      funcs->glVertex2 ( vertPos + (6 * i) );
      funcs->glVertex2 ( vertPos + (6 * i + 3) );
    }

    glEnd ();
  }
}


//-----------------------------------------------------------------------
//   drawLabels_
//-----------------------------------------------------------------------


void XYPlot::drawLabels_

  ( GLContext&      gtx,
    TextProjector&  txp,
    const GLfloat   psize[2],
    const GLfloat   space[3] )

{
  OrthoTextScope  textScope ( gtx, txp );

  const GLfloat   fnHeight = psize[1] * (GLfloat) txp.font.getHeight ();

  GLfloat         x, y, dy;
  idx_t           i, n;


  gtx.setColor ( Color::black() );

  txp.hAlignment = CENTERED;

  if ( xLabel.size() )
  {
    x = 0.5F * (prect_[0] + prect_[2]);
    y = bbox_[1];

    txp.vAlignment = TOP_ALIGNED;

    txp.setCursor ( x, y );
    txp.writeLine ( gtx, xLabel );

    bbox_[1] -= space[1] + fnHeight;
  }

  n = yLabel.size ();

  if ( n )
  {
    x  = bbox_[0] - 0.5F * fnHeight;
    dy = 0.95F * fnHeight;
    y  = 0.50F * (prect_[1] + prect_[3] + (GLfloat) n * dy);

    txp.vAlignment = CENTERED;

    for ( i = 0; i < n; i++ )
    {
      txp.setCursor ( x, y );
      txp.write     ( gtx, yLabel[i] );

      y -= dy;
    }

    bbox_[0] -= space[0] + fnHeight;
  }

  if ( title.size() )
  {
    x = 0.5F * width_;
    y = bbox_[3];

    txp.vAlignment = BOT_ALIGNED;

    txp.setCursor ( x, y );
    txp.writeLine ( gtx, title );

    bbox_[3] += space[1] + fnHeight;
  }
}


//-----------------------------------------------------------------------
//   drawKey_
//-----------------------------------------------------------------------


void XYPlot::drawKey_

  ( GLContext&      gtx,
    TextProjector&  txp,
    const Box&      bbox,
    const GLfloat   psize[2],
    const GLfloat   space[3] )

{
  GLFunctions1*  funcs     = gtx.findFunctions<GLFunctions1> ();
  const idx_t    dsetCount = dataSets_.size ();
  const GLfloat  fnHeight  = psize[1] * (GLfloat) txp.font.getHeight ();

  MarkerStyle*   ps;
  DataSet_*      dset;
  Color          col;
  GLfloat        x1, x2, x3, x4;
  GLfloat        y,  dy;
  GLfloat        z;
  GLfloat        w,  wmax;
  GLfloat        lw;
  GLfloat        xKey, yKey;
  GLfloat        pos[3];
  idx_t          keyCount;


  txp.hAlignment = RIGHT_ALIGNED;
  txp.vAlignment = CENTERED;

  xKey     = prect_[0] + keyPos_[0] * (prect_[2] - prect_[0]);
  yKey     = prect_[1] + keyPos_[1] * (prect_[3] - prect_[1]);

  x1       = xKey   -        space[0];
  x2       = x1     - 2.0F * space[0];
  x3       = x2     - 2.0F * space[0];
  x4       = x3     -        space[0];

  dy       = fnHeight + 0.5F * space[1];
  y        = yKey - space[1] - 0.5F * dy;
  z        = 2.0F * space[2];

  pos[0]   = x2;
  pos[2]   = z;

  keyCount = 0;
  wmax     = 0.0F;

  for ( idx_t i = 0; i < dsetCount; i++ )
  {
    if ( y < space[1] )
    {
      break;
    }

    dset = dataSets_.getAs<DataSet_> ( i );

    if ( dset->key.size() == 0 )
    {
      continue;
    }

    lw = dset->getLineWidth ();

    if ( lw > 0.1F )
    {
      col = dset->getLineColor ();

      gtx.setColor     ( col );
      gtx.setLineWidth ( lw );

      if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
      {
        GLfloat  vertPos[6] = { x1, y, z, x3, y, z };

        ShaderDraw::draw ( gtx, 6, vertPos, col, ShaderDraw::LINES );
      }
      else if ( funcs )
      {
        funcs->glBegin ( GL_LINES );
        {
          funcs->glVertex ( x1, y, z );
          funcs->glVertex ( x3, y, z );
        }
        funcs->glEnd ();
      }
    }

    ps = dset->getPointStyle ();

    if ( ps )
    {
      pos[1] = y;

      ps->beginDraw  ( gtx, bbox, 1 );
      ps->drawMarker ( gtx, i, pos );
      ps->endDraw    ( gtx );
    }

    gtx.setColor   ( Color::black() );
    txp.begin      ( gtx );
    txp.setCursor  ( x4, y, z );

    w = (GLfloat) txp.writeLine ( gtx, dset->key );

    txp.end        ( gtx );

    if ( w > wmax )
    {
      wmax = w;
    }

    keyCount++;

    y = y - dy;
  }

  if ( keyCount == 0 )
  {
    return;
  }

  x1  = x4 - wmax * psize[0] - space[0];
  y   = y  + 0.5F *       dy - space[1];
  z   = space[2];
  col = Color ( 1.0F, 1.0F, 1.0F, 0.85F );

  gtx.setColor ( col );

  if      ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    GLfloat  vertPos[12] = { x1,   y,    z,
                             xKey, y,    z,
                             xKey, yKey, z,
                             x1,   yKey, z };

    ShaderDraw::draw ( gtx, 12, vertPos, col,
                       ShaderDraw::TRIANGLE_FAN );
  }
  else if ( funcs )
  {
    funcs->glBegin ( GL_QUADS );
    {
      funcs->glVertex (   x1,    y, z );
      funcs->glVertex ( xKey,    y, z );
      funcs->glVertex ( xKey, yKey, z );
      funcs->glVertex (   x1, yKey, z );
    }
    funcs->glEnd ();
  }

  gtx.translate ( 0.0F, 0.0F,  z );
  renderShadow  ( gtx, x1, y, xKey, yKey, space[0] );
  gtx.translate ( 0.0F, 0.0F, -z );

  // A rendering problem has been experienced with some video drivers
  // (Intel) on Windows. A work-around is to enable/disable lighting.

  col = Color::black ();

  gtx.setColor     ( col );
  gtx.setLineWidth ( 2.0F );

  z += space[2];

  if      ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    GLfloat  vertPos[15] = { x1,   y,    z,
                             xKey, y,    z,
                             xKey, yKey, z,
                             x1,   yKey, z,
                             x1,   y,    z };

    ShaderDraw::draw ( gtx, 15, vertPos, col, ShaderDraw::LINE_STRIP );
  }
  else if ( funcs )
  {
    funcs->glBegin     ( GL_LINE_STRIP );
    {
      funcs->glVertex  (   x1,    y, z );
      funcs->glVertex  ( xKey,    y, z );
      funcs->glVertex  ( xKey, yKey, z );
      funcs->glVertex  (   x1, yKey, z );
      funcs->glVertex  (   x1,    y, z );
    }
    funcs->glEnd ();
  }
}


//-----------------------------------------------------------------------
//   getPlotSize_
//-----------------------------------------------------------------------


void XYPlot::getPlotSize_

  ( GLContext&  gtx,
    GLfloat     psize[2] ) const

{
  Rect       vport = gtx.getViewport  ();

  Transform  p     = gtx.getTransform ( GLContext::PROJECTION );
  Transform  t     = gtx.getTransform ();

  Vertex3d   a,  b, c;
  double     dx, dy;


  psize[0] = 100.0;
  psize[1] = 100.0;

  a        = 0.0;

  matmul ( b, t, a );

  if ( ! project( a, p, b ) )
  {
    return;
  }

  toWindow ( a, vport );

  b[0] = width_;
  b[1] = 0.0;
  b[2] = 0.0;

  matmul ( c, t, b );

  if ( ! project( b, p, c ) )
  {
    return;
  }

  toWindow ( b, vport );

  dx = b[0] - a[0];
  dy = b[1] - a[1];

  psize[0] = (GLfloat) std::sqrt ( dx * dx + dy * dy );

  b[0] = 0.0;
  b[1] = height_;
  b[2] = 0.0;

  matmul ( c, t, b );

  if ( ! project( b, p, c ) )
  {
    return;
  }

  toWindow ( b, vport );

  dx = b[0] - a[0];
  dy = b[1] - a[1];

  psize[1] = (GLfloat) std::sqrt ( dx * dx + dy * dy );
}


//-----------------------------------------------------------------------
//   setFontSize_
//-----------------------------------------------------------------------


void XYPlot::setFontSize_

  ( Font&          font,
    const GLfloat  psize[2] ) const

{
  GLfloat  fnWidth  = psize[0] * (GLfloat) font.getWidthOf ( '0' );
  GLfloat  fnHeight = psize[1] * (GLfloat) font.getHeight  ();

  if ( 30.0F * fnHeight > height_ ||
       50.0F * fnWidth  > width_ )
  {
    font--;
  }
}


//-----------------------------------------------------------------------
//   setPlotRect_
//-----------------------------------------------------------------------


void XYPlot::setPlotRect_ ()
{
  bbox_ [0] = 0.0F;
  bbox_ [1] = 0.0F;
  bbox_ [2] = width_;
  bbox_ [3] = height_;
  depth_    = 0.05F * jem::max ( width_, height_ );

  prect_[0] = 0.05F * width_;
  prect_[1] = 0.05F * height_;
  prect_[2] = 0.95F * width_;
  prect_[3] = 0.95F * height_;
}


void XYPlot::setPlotRect_

  ( int            ticks[2],
    const GLfloat  psize[2],
    const GLfloat  space[3],
    const Font&    font )

{
  const GLfloat  fnWidth  = psize[0] * (GLfloat) font.getWidthOf ( '0' );
  const GLfloat  fnHeight = psize[1] * (GLfloat) font.getHeight  ();

  char           buf[64];
  GLfloat        h,  w;
  GLfloat        dx, dy;
  double         v,  dv;
  int            i,  k, n;


  // Initial plot rectangle.

  prect_[0] = 0.0F;
  prect_[1] = 0.0F;
  prect_[2] = width_;
  prect_[3] = height_;

  // Space for the labels and title.

  if ( xLabel.size() )
  {
    prect_[1] += space[1] + fnHeight;
  }

  if ( yLabel.size() )
  {
    prect_[0] += space[0] + fnHeight;
  }

  if ( title.size() )
  {
    prect_[3] -= space[1] + fnHeight;
  }

  // Determine the number of tick marks.

  if ( options_ & SHOW_XTICKS )
  {
    dx = fnWidth;
    w  = prect_[2] - prect_[0] - 2.0F * dx;

    if ( w <= 0.0 )
    {
      ticks[0] = 0;
    }
    else
    {
      v = 0.5 * (xRange_[0] + xRange_[1]);
      n = 3;
      k = std::sprintf ( buf, TICK_FORMAT_, xRange_[0] );
      n = max          ( n, k );
      k = std::sprintf ( buf, TICK_FORMAT_, v );
      n = max          ( n, k );
      k = std::sprintf ( buf, TICK_FORMAT_, xRange_[1] );
      n = max          ( n, k );
      n = n + 2;

      ticks[0] = (int) (0.5F + w / ((GLfloat) n * dx));

      adjustTickCount ( ticks[0] );
    }
  }
  else
  {
    ticks[0] = 0;
  }

  if ( options_ & SHOW_YTICKS )
  {
    dy = fnHeight;
    h  = prect_[3] - prect_[1] - 2.0F * dy;

    if ( h <= 0.0 )
    {
      ticks[1] = 0;
    }
    else
    {
      ticks[1] = (int) (0.5 + h / (4.0 * dy));

      adjustTickCount ( ticks[1] );
    }
  }
  else
  {
    ticks[1] = 0;
  }

  // Space for the tick marks (phase 1).

  if ( ticks[0] > 0 )
  {
    prect_[1] += space[1] + fnHeight;
  }

  if ( ticks[1] > 0 )
  {
    n  = ticks[1];
    dv = (yRange_[1] - yRange_[0]) / (n - 1.0);

    k  = std::sprintf ( buf, TICK_FORMAT_, yRange_[0] );
    w  = (GLfloat) k * fnWidth;

    for ( i = 1; i < n - 1; i++ )
    {
      v = yRange_[0] + i * dv;
      k = std::sprintf ( buf, TICK_FORMAT_, v );
      w = jem::max     ( w, (GLfloat) k * fnWidth );
    }

    k = std::sprintf ( buf, TICK_FORMAT_, yRange_[1] );
    w = jem::max     ( w, (GLfloat) k * fnWidth );

    // Note that the average character is less wide than a '-'; hence
    // the multiplication factor.

    prect_[0] += space[0] + w * 0.9F;
  }

  prect_[0] += space[0];
  prect_[1] += space[1];
  prect_[2] -= space[0];
  prect_[3] -= space[1];

  // Space for the tick marks (phase 2).

  if ( ticks[0] > 0 )
  {
    k = std::sprintf ( buf, TICK_FORMAT_, xRange_[0] );
    w = 0.5F * (GLfloat) k * fnWidth + space[0];

    if ( w > prect_[0] )
    {
      prect_[0] = w;
    }

    k = std::sprintf ( buf, TICK_FORMAT_, xRange_[1] );
    w = 0.5F * (GLfloat) k * fnWidth + space[0];

    if ( w > width_ - prect_[2] )
    {
      prect_[2] = width_ - w;
    }
  }

  if ( ticks[1] > 0 )
  {
    h = space[1] + 0.5F * fnHeight;

    if ( h > prect_[1] )
    {
      prect_[1] = h;
    }

    if ( h > height_ - prect_[3] )
    {
      prect_[3] = height_ - h;
    }
  }

  // Make sure the plot rectangle is valid.

  if ( prect_[0] >= prect_[2] )
  {
    prect_[0] = prect_[2] - 0.01F * width_;
  }

  if ( prect_[1] >= prect_[3] )
  {
    prect_[1] = prect_[3] - 0.01F * height_;
  }

  // Initialize the bounding box.

  bbox_[0] = prect_[0] - space[0];
  bbox_[1] = prect_[1] - space[1];
  bbox_[2] = prect_[2] + space[0];
  bbox_[3] = prect_[3] + space[1];
}


//-----------------------------------------------------------------------
//   setRanges_
//-----------------------------------------------------------------------


void XYPlot::setRanges_ ( const Properties& props )
{
  const char*  RANGE_PROPS[2] = { PropNames::XRANGE,
                                  PropNames::YRANGE };

  for ( int i = 0; i < 2; i++ )
  {
    Ref<Object>  obj;

    if ( ! props.find( obj, RANGE_PROPS[i] ) )
    {
      continue;
    }

    Range  newRange;
    bool   autoScale;

    autoScale = false;
    newRange  = 0.0;

    if ( isInstance<String>( obj ) )
    {
      String  str = toValue<String> ( obj );

      if ( str.equalsIgnoreCase( "auto" ) )
      {
        autoScale = true;
      }
      else
      {
        props.propertyError (
          RANGE_PROPS[i],
          String::format (
            "invalid range: %s; should be `auto\' or "
            "an array of length 2",
            str
          )
        );
      }
    }
    else
    {
      Array<double>  r;

      props.get ( r, RANGE_PROPS[i] );

      if ( r.size() == 0 )
      {
        autoScale = true;
      }
      else
      {
        if ( r.size() != 2 )
        {
          props.propertyError (
            RANGE_PROPS[i],
            String::format (
              "invalid range: array has length %d but "
              "should have length 2 or zero (for auto scale)",
              r.size ()
            )
          );
        }

        if ( (r[1] - r[0]) < 0.0 )
        {
          props.propertyError (
            RANGE_PROPS[i],
            String::format ( "invalid range: [%g,%g]", r[0], r[1] )
          );
        }

        newRange[0] = r[0];
        newRange[1] = r[1];
      }
    }

    if ( autoScale )
    {
      if ( i == 0 )
      {
        setXAutoScale ();
      }
      else
      {
        setYAutoScale ();
      }
    }
    else
    {
      if ( i == 0 )
      {
        setXRange ( newRange );
      }
      else
      {
        setYRange ( newRange );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   adjustRange_
//-----------------------------------------------------------------------


void XYPlot::adjustRange_

  ( Range&  r,
    double  step )

{
  const double  SMALL_VAL = 1.0e-30;
  const double  LARGE_VAL = 1.0e+30;

  double        x;


  if ( (r[1] - r[0]) < SMALL_VAL )
  {
    if ( std::fabs( r[0] ) < SMALL_VAL )
    {
      r[0] = -SMALL_VAL;
      r[1] =  SMALL_VAL;
    }
    else
    {
      r[1] = r[0] + 0.1 * std::fabs( r[0] );
    }
  }

  if ( step < 1.0e-5 )
  {
    return;
  }

  // Adjust the range so that the lower and upper points are visible.

  x     = 0.25 * step * (r[1] - r[0]);
  r[0] -= x;
  r[1] += x;
  x     = r[1] - r[0];

  if ( x >= SMALL_VAL && x <= LARGE_VAL )
  {
    x = ::log10( x );
    x = ::pow( 10.0, ::floor( x + 0.5 ) );
    x = x * step;

    if ( r[0] >= 0.0 )
    {
      r[0] = r[0] / x;
      r[0] = x * ::floor( r[0] );
    }
    else
    {
      r[0] = -r[0] / x;
      r[0] = -x * ::ceil( r[0] );
    }

    if ( r[1] >= 0.0 )
    {
      r[1] = r[1] / x;
      r[1] = x * ::ceil( r[1] );
    }
    else
    {
      r[1] = -r[1] / x;
      r[1] = -x * ::floor( r[1] );
    }
  }
}


JEM_END_PACKAGE( gl )
