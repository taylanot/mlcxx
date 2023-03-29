
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
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/io/list.h>
#include <jem/io/utilities.h>
#include <jem/io/FileOutputStream.h>
#include <jem/io/BufferedOutputStream.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jem/gl/input.h>
#include <jem/gl/Actions.h>
#include <jem/gl/Camera2D.h>
#include <jem/gl/Commands.h>
#include <jem/gl/Display.h>
#include <jem/gl/EventNode.h>
#include <jem/gl/GLSetup2D.h>
#include <jem/gl/Menu.h>
#include <jem/gl/MenuNode.h>
#include <jem/gl/MessageBox.h>
#include <jem/gl/MultiNode.h>
#include <jem/gl/PNGGrabber.h>
#include <jem/gl/PPMGrabber.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/RubberBand.h>
#include <jem/gl/TextBalloon.h>
#include <jem/gl/Viewer.h>
#include <jem/gl/Window.h>
#include <jem/gl/XYPlot.h>
#include <jem/gl/GraphViewer.h>


JEM_DEFINE_CLASS( jem::gl::GraphViewer );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GraphViewer::ReadLock_
//=======================================================================


class GraphViewer::ReadLock_
{
 public:

  inline                  ReadLock_

    ( const GraphViewer*    gv );

  inline                 ~ReadLock_ ();


 private:

  GraphViewer*            gv_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline GraphViewer::ReadLock_::ReadLock_ ( const GraphViewer* gv )
{
  gv_ = const_cast<GraphViewer*> ( gv );

  gv_->freeze ();
}


inline GraphViewer::ReadLock_::~ReadLock_ ()
{
  gv_->resume ();
}


//=======================================================================
//   class GraphViewer::WriteLock_
//=======================================================================


class GraphViewer::WriteLock_
{
 public:

  inline                  WriteLock_

    ( GraphViewer*          gv );

  inline                 ~WriteLock_ ();


 private:

  GraphViewer*            gv_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline GraphViewer::WriteLock_::WriteLock_ ( GraphViewer* gv )
{
  gv_ = gv;

  gv_->freeze ();
}


inline GraphViewer::WriteLock_::~WriteLock_ ()
{
  gv_->changed_ = true;

  gv_->resume ();
}


//=======================================================================
//   class GraphViewer::DataSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GraphViewer::DataSet::DataSet ()
{}


GraphViewer::DataSet::~DataSet ()
{}


//=======================================================================
//   class GraphViewer::DataSet_
//=======================================================================


class GraphViewer::DataSet_ : public Object, public DataSet
{
 public:

                            DataSet_

    ( GraphViewer*            gv,
      XYPlot::DataSet*        dset );

  void                      destroy        ();

  virtual void              setKey

    ( const String&           key )                  override;

  virtual String            getKey          () const override;

  inline bool               hasKey

    ( const String&           key )            const;

  virtual void              setPoints

    ( const Array<double>&    x,
      const Array<double>&    y )                    override;

  virtual void              addPoint

    ( double                  x,
      double                  y )                    override;

  virtual void              addPoints

    ( const Array<double>&    x,
      const Array<double>&    y )                    override;

  virtual void              setMaxPoints

    ( idx_t                   n )                    override;

  virtual idx_t             getMaxPoints    () const override;

  virtual void              setStyle

    ( GLfloat                 lwidth,
      const Color&            lcolor,
      const Ref<PntStyle>&    pstyle )               override;

  virtual void              setLineWidth

    ( GLfloat                 width )                override;

  virtual GLfloat           getLineWidth    () const override;

  virtual void              setLineColor

    ( const Color&            color )                override;

  virtual Color             getLineColor    () const override;

  virtual void              setPointStyle

    ( const Ref<PntStyle>&    style )                override;

  virtual PntStyle*         getPointStyle   () const override;


 protected:

  virtual                  ~DataSet_        ();


 private:

  GraphViewer*              gv_;
  XYPlot::DataSet*          dset_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GraphViewer::DataSet_::DataSet_

  ( GraphViewer*      gv,
    XYPlot::DataSet*  dset )

{
  gv_   = gv;
  dset_ = dset;
}


GraphViewer::DataSet_::~DataSet_ ()
{
  dset_ = 0;
}


//-----------------------------------------------------------------------
//   destroy
//-----------------------------------------------------------------------


void GraphViewer::DataSet_::destroy ()
{
  if ( dset_ )
  {
    WriteLock_  lock ( gv_ );

    gv_->plot_->eraseDataSet ( dset_ );

    dset_ = 0;
  }
}


//-----------------------------------------------------------------------
//   setKey
//-----------------------------------------------------------------------


void GraphViewer::DataSet_::setKey ( const String& key )
{
  WriteLock_  lock ( gv_ );

  dset_->key = key;
}


//-----------------------------------------------------------------------
//   getKey
//-----------------------------------------------------------------------


String GraphViewer::DataSet_::getKey () const
{
  return dset_->key;
}


//-----------------------------------------------------------------------
//   hasKey
//-----------------------------------------------------------------------


inline bool GraphViewer::DataSet_::hasKey ( const String& key ) const
{
  return (key == dset_->key);
}


//-----------------------------------------------------------------------
//   setPoints
//-----------------------------------------------------------------------


void GraphViewer::DataSet_::setPoints

  ( const Array<double>&  x,
    const Array<double>&  y )

{
  WriteLock_  lock ( gv_ );

  dset_->setPoints ( x.clone(), y.clone() );
}


//-----------------------------------------------------------------------
//   addPoint
//-----------------------------------------------------------------------


void GraphViewer::DataSet_::addPoint

  ( double  x,
    double  y )

{
  WriteLock_  lock ( gv_ );

  dset_->addPoint ( x, y );
}


//-----------------------------------------------------------------------
//   addPoints
//-----------------------------------------------------------------------


void GraphViewer::DataSet_::addPoints

  ( const Array<double>&  x,
    const Array<double>&  y )

{
  WriteLock_  lock ( gv_ );

  dset_->addPoints ( x, y );
}


//-----------------------------------------------------------------------
//   setMaxPoints
//-----------------------------------------------------------------------


void GraphViewer::DataSet_::setMaxPoints ( idx_t n )
{
  WriteLock_  lock ( gv_ );

  dset_->setMaxPoints ( n );
}


//-----------------------------------------------------------------------
//   getMaxPoints
//-----------------------------------------------------------------------


idx_t GraphViewer::DataSet_::getMaxPoints () const
{
  ReadLock_  lock ( gv_ );

  return dset_->getMaxPoints ();
}


//-----------------------------------------------------------------------
//   setStyle
//-----------------------------------------------------------------------


void GraphViewer::DataSet_::setStyle

  ( GLfloat               lwidth,
    const Color&          lcolor,
    const Ref<PntStyle>&  pstyle )

{
  WriteLock_  lock ( gv_ );

  dset_->setStyle ( lwidth, lcolor, pstyle );
}


//-----------------------------------------------------------------------
//   setLineWidth
//-----------------------------------------------------------------------


void GraphViewer::DataSet_::setLineWidth ( GLfloat lwidth )
{
  WriteLock_  lock ( gv_ );

  dset_->setLineWidth ( lwidth );
}


//-----------------------------------------------------------------------
//   getLineWidth
//-----------------------------------------------------------------------


GLfloat GraphViewer::DataSet_::getLineWidth () const
{
  ReadLock_  lock ( gv_ );

  return dset_->getLineWidth ();
}


//-----------------------------------------------------------------------
//   setLineColor
//-----------------------------------------------------------------------


void GraphViewer::DataSet_::setLineColor ( const Color& color )
{
  WriteLock_  lock ( gv_ );

  dset_->setLineColor ( color );
}


//-----------------------------------------------------------------------
//   getLineColor
//-----------------------------------------------------------------------


Color GraphViewer::DataSet_::getLineColor () const
{
  ReadLock_  lock ( gv_ );

  return dset_->getLineColor ();
}


//-----------------------------------------------------------------------
//   setPointStype
//-----------------------------------------------------------------------


void GraphViewer::DataSet_::setPointStyle

  ( const Ref<PntStyle>&  style )

{
  WriteLock_  lock ( gv_ );

  dset_->setPointStyle ( style );
}


//-----------------------------------------------------------------------
//   getPointStyle
//-----------------------------------------------------------------------


MarkerStyle* GraphViewer::DataSet_::getPointStyle () const
{
  ReadLock_  lock ( gv_ );

  return dset_->getPointStyle ();
}


//=======================================================================
//   class GraphViewer
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


GraphViewer::GraphViewer

  ( const Ref<Viewer>&  viewer ) :

    viewer_ ( viewer )

{
  using jem::util::connect;

  JEM_PRECHECK ( viewer );

  Ref<GLSetup>    setup = newInstance<GLSetup2D> ();
  Ref<EventNode>  evNode;
  Ref<MultiNode>  world;
  Ref<Menu>       menu;

  // The XYPlot node is child of a MultiNode so that it is possible to
  // add and remove nodes (like a TextBalloon) at runtime.

  world    = newInstance<MultiNode>  ( "world" );
  plot_    = newInstance<XYPlot>     ( "graph" );
  evNode   = newInstance<EventNode>  ( "event",  world );
  camera_  = newInstance<Camera2D>   ( "camera", evNode );
  window_  = newInstance<Window>     ( "window", camera_, setup );
  rubber_  = newInstance<RubberBand> ( "rubber" );

  frozen_  = 0;
  changed_ = false;

  connect ( evNode->keyEvent,
            this, & Self::keyPressed_ );

  connect ( evNode->mouseEvent,
            this, & Self::mouseClicked_ );

  connect ( evNode->dragEvent,
            this, & Self::mouseDragged_ );

  connect ( evNode->resizeEvent,
            this, & Self::resizeHandler_ );

  connect ( rubber_->selectEvent,
            this, & Self::rectSelected_ );

  world  ->addChild  ( plot_   );
  world  ->addChild  ( rubber_ );
  evNode ->takeFocus ();
  viewer_->setScene  ( window_ );
}


GraphViewer::~GraphViewer ()
{}


//-----------------------------------------------------------------------
//   open
//-----------------------------------------------------------------------


Ref<GraphViewer> GraphViewer::open ( const String& name )
{
  Display::open ( Display::NEW_THREAD );

  Ref<Viewer>  viewer = Display::newWindow ( name );

  Display::mainLoop ();
  viewer ->resume   ();

  return newInstance<Self> ( viewer );
}


Ref<GraphViewer> GraphViewer::open

  ( const String&  name,
    int&           argc,
    char**         argv )

{
  Display::open ( argc, argv, Display::NEW_THREAD );

  Ref<Viewer>  viewer = Display::newWindow ( name );

  Display::mainLoop ();
  viewer ->resume   ();

  return newInstance<Self> ( viewer );
}


//-----------------------------------------------------------------------
//   freeze
//-----------------------------------------------------------------------


void GraphViewer::freeze ()
{
  if ( ! frozen_ )
  {
    viewer_->freeze ();
  }

  frozen_++;
}


//-----------------------------------------------------------------------
//   resume
//-----------------------------------------------------------------------


void GraphViewer::resume ()
{
  frozen_--;

  if ( frozen_ <= 0 )
  {
    if ( changed_ )
    {
      viewer_->postRedraw ();

      changed_ = false;
    }

    viewer_->resume ();

    frozen_ = 0;
  }
}


//-----------------------------------------------------------------------
//   replot
//-----------------------------------------------------------------------


void GraphViewer::replot ()
{
  viewer_->postRedraw  ();
  viewer_->flushEvents ();
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void GraphViewer::clear ()
{
  WriteLock_  lock ( this );

  dataSets_.clear ();
  plot_   ->clear ();
}


//-----------------------------------------------------------------------
//   shouldClose
//-----------------------------------------------------------------------


bool GraphViewer::shouldClose ()
{
  return viewer_->shouldClose ();
}


//-----------------------------------------------------------------------
//   newDataSet
//-----------------------------------------------------------------------


GraphViewer::DataSet* GraphViewer::newDataSet

  ( const String&  key,
    Style          style )

{
  WriteLock_     lock ( this );

  Ref<DataSet_>  dset =

    newInstance<DataSet_> ( this, plot_->newDataSet( key, style ) );

  dataSets_.pushBack ( dset.get() );

  return dset.get ();
}


GraphViewer::DataSet* GraphViewer::newDataSet

  ( const String&         key,
    const Array<double>&  xData,
    const Array<double>&  yData,
    Style                 style )

{
  WriteLock_     lock ( this );

  Ref<DataSet_>  dset =

    newInstance<DataSet_> (
      this,
      plot_->newDataSet ( key, xData, yData, style )
    );

  dataSets_.pushBack ( dset.get() );

  return dset.get ();
}


//-----------------------------------------------------------------------
//   eraseDataSet
//-----------------------------------------------------------------------


bool GraphViewer::eraseDataSet ( DataSet* target )
{
  const idx_t  n = dataSets_.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    DataSet_*  dset = dataSets_.getAs<DataSet_> ( i );

    if ( dset == target )
    {
      dset->destroy ();

      dataSets_.erase ( i );

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   findDataSet
//-----------------------------------------------------------------------


GraphViewer::DataSet* GraphViewer::findDataSet

  ( const String& key ) const noexcept

{
  const idx_t  n = dataSets_.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    DataSet_*  dset = dataSets_.getAs<DataSet_> ( i );

    if ( dset->hasKey( key ) )
    {
      return dset;
    }
  }

  return 0;
}


//-----------------------------------------------------------------------
//   getDataSets
//-----------------------------------------------------------------------


Array<GraphViewer::DataSet*> GraphViewer::getDataSets () const
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
//   setQuality
//-----------------------------------------------------------------------


void GraphViewer::setQuality ( int level )
{
  WriteLock_  lock ( this );

  camera_->setQuality ( level );
}


//-----------------------------------------------------------------------
//   getQuality
//-----------------------------------------------------------------------


int GraphViewer::getQuality () const
{
  ReadLock_  lock ( this );

  return camera_->getQuality ();
}


//-----------------------------------------------------------------------
//   setOption
//-----------------------------------------------------------------------


void GraphViewer::setOption

  ( Option  option,
    bool    choice )

{
  WriteLock_  lock ( this );

  plot_->setOption ( option, choice );
}


//-----------------------------------------------------------------------
//   setOptions
//-----------------------------------------------------------------------


void GraphViewer::setOptions ( Options options )
{
  WriteLock_  lock ( this );

  plot_->setOptions ( options );
}


//-----------------------------------------------------------------------
//   getOptions
//-----------------------------------------------------------------------


GraphViewer::Options GraphViewer::getOptions () const
{
  ReadLock_  lock ( this );

  return plot_->getOptions ();
}


//-----------------------------------------------------------------------
//   setTitle
//-----------------------------------------------------------------------


void GraphViewer::setTitle ( const String& title )
{
  WriteLock_  lock ( this );

  plot_->title = title;
}


//-----------------------------------------------------------------------
//   getTitle
//-----------------------------------------------------------------------


String GraphViewer::getTitle () const
{
  ReadLock_  lock ( this );

  return plot_->title;
}


//-----------------------------------------------------------------------
//   setLabels
//-----------------------------------------------------------------------


void GraphViewer::setLabels

  ( const String&  xLabel,
    const String&  yLabel )

{
  WriteLock_  lock ( this );

  plot_->xLabel = xLabel;
  plot_->yLabel = yLabel;
}


//-----------------------------------------------------------------------
//   getXLabel
//-----------------------------------------------------------------------


String GraphViewer::getXLabel () const
{
  ReadLock_  lock ( this );

  return plot_->xLabel;
}


//-----------------------------------------------------------------------
//   getYLabel
//-----------------------------------------------------------------------


String GraphViewer::getYLabel () const
{
  ReadLock_  lock ( this );

  return plot_->yLabel;
}


//-----------------------------------------------------------------------
//   setKeyPos
//-----------------------------------------------------------------------


void GraphViewer::setKeyPos

  ( GLfloat  xpos,
    GLfloat  ypos )

{
  WriteLock_  lock ( this );

  plot_->setKeyPos ( xpos, ypos );
}


//-----------------------------------------------------------------------
//   getKey[X|Y]Pos
//-----------------------------------------------------------------------


GLfloat GraphViewer::getKeyXPos () const
{
  ReadLock_  lock ( this );

  return plot_->getKeyXPos ();
}


GLfloat GraphViewer::getKeyYPos () const
{
  ReadLock_  lock ( this );

  return plot_->getKeyYPos ();
}


//-----------------------------------------------------------------------
//   setBgColor
//-----------------------------------------------------------------------


void GraphViewer::setBgColor ( const Color& col )
{
  WriteLock_  lock ( this );

  window_->setBgColor ( col );
}


//-----------------------------------------------------------------------
//   getBgColor
//-----------------------------------------------------------------------


Color GraphViewer::getBgColor () const
{
  ReadLock_  lock ( this );

  return window_->getBgColor ();
}


//-----------------------------------------------------------------------
//   set[X|Y]Range
//-----------------------------------------------------------------------


void GraphViewer::setXRange ( const Range& range )
{
  WriteLock_  lock ( this );

  plot_->setXRange ( range );
}


void GraphViewer::setXRange ( double lo, double hi )
{
  WriteLock_  lock ( this );

  plot_->setXRange ( Range( lo, hi ) );
}


void GraphViewer::setYRange ( const Range& range )
{
  WriteLock_  lock ( this );

  plot_->setYRange ( range );
}


void GraphViewer::setYRange ( double lo, double hi )
{
  WriteLock_  lock ( this );

  plot_->setYRange ( Range( lo, hi ) );
}


//-----------------------------------------------------------------------
//   get[X|Y]Range
//-----------------------------------------------------------------------


GraphViewer::Range GraphViewer::getXRange () const
{
  ReadLock_  lock ( this );

  return plot_->getXRange ();
}


GraphViewer::Range GraphViewer::getYRange () const
{
  ReadLock_  lock ( this );

  return plot_->getXRange ();
}


//-----------------------------------------------------------------------
//   setAutoScale
//-----------------------------------------------------------------------


void GraphViewer::setAutoScale ()
{
  WriteLock_  lock ( this );

  plot_->setAutoScale ();
}


//-----------------------------------------------------------------------
//   set[X|Y]AutoScale
//-----------------------------------------------------------------------


void GraphViewer::setXAutoScale ()
{
  WriteLock_  lock ( this );

  plot_->setXAutoScale ();
}


void GraphViewer::setYAutoScale ()
{
  WriteLock_  lock ( this );

  plot_->setYAutoScale ();
}


//-----------------------------------------------------------------------
//   get[X|Y]AutoScale
//-----------------------------------------------------------------------


bool GraphViewer::getXAutoScale () const
{
  ReadLock_  lock ( this );

  return plot_->getXAutoScale ();
}


bool GraphViewer::getYAutoScale () const
{
  ReadLock_  lock ( this );

  return plot_->getYAutoScale ();
}


//-----------------------------------------------------------------------
//   setScaleStep
//-----------------------------------------------------------------------


void GraphViewer::setScaleStep ( double step )
{
  WriteLock_  lock ( this );

  plot_->setScaleStep ( step );
}


//-----------------------------------------------------------------------
//   getScaleStep
//-----------------------------------------------------------------------


double GraphViewer::getScaleStep () const
{
  ReadLock_  lock ( this );

  return plot_->getScaleStep ();
}


//-----------------------------------------------------------------------
//   saveSnapshot
//-----------------------------------------------------------------------


String GraphViewer::saveSnapshot ( const String& fname ) const
{
  using jem::io::canZip;
  using jem::io::FileOutputStream;
  using jem::io::BufferedOutputStream;

  String  fileName = fname;

  Ref<OutputStream>  output;
  Ref<FrameGrabber>  grabber;


  try
  {
    grabber = FrameGrabber::byFile ( fname );
  }
  catch ( const IllegalInputException& )
  {
    if ( canZip() )
    {
      grabber  = newInstance<PNGGrabber> ();
      fileName = fname + ".png";
    }
    else
    {
      grabber  = newInstance<PPMGrabber> ();
      fileName = fname + ".ppm";
    }
  }

  output = newInstance<BufferedOutputStream> (
    newInstance<FileOutputStream> ( fileName )
  );

  saveSnapshot ( *output, *grabber );

  return fileName;
}


void GraphViewer::saveSnapshot

  ( OutputStream&  output,
    FrameGrabber&  grabber ) const

{
  viewer_->grabFrame ( output, grabber );
}


//-----------------------------------------------------------------------
//   keyPressed_
//-----------------------------------------------------------------------


void GraphViewer::keyPressed_

  ( RenderContext&   rtx,
    const KeyPress&  key )

{
  bool  handled = false;
  int   keyMods = key.mods & ~Keys::SHIFT;


  switch ( key.code )
  {
  case '?':

    if ( keyMods == 0 )
    {
      showHelp_ ();

      return;
    }

    break;

  case 'c':

    if ( keyMods == Keys::ALT )
    {
      plot_->setAutoScale ();

      handled = true;
    }

    break;

  case 'k':

    if ( keyMods == Keys::ALT )
    {
      plot_->toggleOption ( XYPlot::SHOW_KEY );

      handled = true;
    }

    break;

  case 'x':

    if ( keyMods == Keys::ALT )
    {
      plot_->toggleOption ( XYPlot::SHOW_XTICKS );

      handled = true;
    }

    break;

  case 'y':

    if ( keyMods == Keys::ALT )
    {
      plot_->toggleOption ( XYPlot::SHOW_YTICKS );

      handled = true;
    }

    break;

  case Keys::ESCAPE:

    handled = true;

    break;

  default:

    handled = false;
  }

  if ( handled )
  {
    rtx.postRedraw ( window_ );
  }
  else
  {
    showMessageUntil (
      rtx, plot_,
      String::format ( "Unbound key: %c;\n"
                       "press `?\' for help.", key.code ),
      Time::now() + 4.0_sec
    );
  }
}


//-----------------------------------------------------------------------
//   mouseClicked_
//-----------------------------------------------------------------------


void GraphViewer::mouseClicked_

  ( RenderContext&     rtx,
    const MouseClick&  mouse )

{
  if ( mouse.button == Buttons::LEFT )
  {
    if ( mouse.mods == 0 && mouse.what == MouseClick::SINGLE )
    {
      showMousePos_ ( rtx, mouse.x, mouse.y );
    }
  }

  if ( mouse.button == Buttons::RIGHT )
  {
    if ( mouse.mods == 0 && mouse.what == MouseClick::DOWN )
    {
      showMenu_ ( rtx );
    }
  }
}


//-----------------------------------------------------------------------
//   mouseDragged_
//-----------------------------------------------------------------------


void GraphViewer::mouseDragged_

  ( RenderContext&    rtx,
    const MouseDrag&  mouse )

{
  if ( mouse.button == Buttons::LEFT )
  {
    if      ( mouse.mods == 0 && mouse.seqnr == 0 )
    {
      rubber_->start ( rtx, mouse );
    }
    else if ( mouse.mods == Keys::ALT )
    {
      showMousePos_  ( rtx, mouse.x, mouse.y );
    }
  }
}


//-----------------------------------------------------------------------
//   exeCommand_
//-----------------------------------------------------------------------


void GraphViewer::exeCommand_

  ( RenderContext&  rtx,
    const String&   cmd )

{
  bool  postRedraw = false;

  if ( cmd == Commands::HELP )
  {
    showHelp_ ();
  }
  else
  {
    Properties  params;

    params.set ( GLActionParams::COMMAND, cmd );

    postRedraw =

      window_->takeGLAction ( rtx, (int) GLActions::EXE_COMMAND, params );
  }

  if ( postRedraw )
  {
    rtx.postFastRedraw ( window_ );
  }
}


//-----------------------------------------------------------------------
//   rectSelected_
//-----------------------------------------------------------------------


void GraphViewer::rectSelected_

  ( RenderContext&  rtx,
    const Rect&     rect )

{
  GLContext&  gtx = rtx.getGLContext ();

  rtx.postRedraw ( window_ );

  if ( rect.width == 0 || rect.height == 0 )
  {
    return;
  }

  Vertex3d  u ( rect.x,              rect.y,               0.5 );
  Vertex3d  v ( rect.x + rect.width, rect.y + rect.height, 0.5 );

  if ( fromWindowToNode( gtx, u, *plot_ ) &&
       fromWindowToNode( gtx, v, *plot_ ) )
  {
    Vertex2d  a = plot_->mapPoint ( Vertex2d( u[0], u[1] ) );
    Vertex2d  b = plot_->mapPoint ( Vertex2d( v[0], v[1] ) );

    Range     rx ( (double) a[0], (double) b[0] );
    Range     ry ( (double) a[1], (double) b[1] );

    plot_->adjustRange ( rx );
    plot_->setXRange   ( rx );
    plot_->adjustRange ( ry );
    plot_->setYRange   ( ry );
  }
}


//-----------------------------------------------------------------------
//   showHelp_
//-----------------------------------------------------------------------


void GraphViewer::showHelp_ ()
{
  using jem::io::endl;
  using jem::io::indent;
  using jem::io::beginList;
  using jem::io::endList;
  using jem::io::beginItem;
  using jem::io::endItem;

  Ref<PrintWriter>  pr = newInstance<PrintWriter> ( & System::out() );


  pr->setPageWidth ( 72 );

  print ( *pr, endl );
  print ( *pr, "Key and mouse bindings:" );
  print ( *pr, indent );
  print ( *pr, beginList );

  print ( *pr, beginItem( "Right mouse click" ) );
  print ( *pr,   "Show the options menu." );
  print ( *pr, endItem );

  print ( *pr, beginItem( "Left mouse drag" ) );
  print ( *pr,   "Zoom into the selected area." );
  print ( *pr, endItem );

  print ( *pr, beginItem( "Left mouse click\nAlt+Left mouse drag" ) );
  print ( *pr,   "Show the current mouse position." );
  print ( *pr, endItem );

  print ( *pr, beginItem( "Esc" ) );
  print ( *pr,   "Cancel the current zoom operation." );
  print ( *pr, endItem );

  print ( *pr, beginItem( "?" ) );
  print ( *pr,   "Show this help." );
  print ( *pr, endItem );

  print ( *pr, beginItem( "Alt+c" ) );
  print ( *pr,   "Select auto scale (view full graph)." );
  print ( *pr, endItem );

  print ( *pr, beginItem( "Alt+k" ) );
  print ( *pr,   "Toggle the data key." );
  print ( *pr, endItem );

  print ( *pr, beginItem( "Alt+x" ) );
  print ( *pr,   "Toggle the tick marks along the x-axis." );
  print ( *pr, endItem );

  print ( *pr, beginItem( "Alt+y" ) );
  print ( *pr,   "Toggle the tick marks along the y-axis." );
  print ( *pr, endItem );

  print ( *pr, endList );

  pr->flush ();
}


//-----------------------------------------------------------------------
//   showMenu_
//-----------------------------------------------------------------------


void GraphViewer::showMenu_ ( RenderContext& rtx )
{
  using jem::util::connect;

  Ref<Menu>   menu = newInstance<Menu> ();
  Properties  params;


  params.set ( ActionParams::OPTIONS, menu );

  window_->takeAction ( Actions::LIST_OPTIONS, params );

  menu->addLine  ();
  menu->addEntry ( Commands::HELP );

  Ref<MenuNode>  node = newInstance<MenuNode> ( "menu", *menu );

  connect    ( node->selectEvent, this, & Self::exeCommand_ );
  node->show ( rtx, window_ );
}


//-----------------------------------------------------------------------
//   showMousePos_
//-----------------------------------------------------------------------


void GraphViewer::showMousePos_

  ( RenderContext&  rtx,
    int             xPos,
    int             yPos )

{
  Vertex3d  v ( xPos, yPos, 0.5 );

  if ( fromWindowToNode( rtx.getGLContext(), v, *plot_ ) )
  {
    Vertex2d  a = plot_->mapPoint ( Vertex2d( v[0], v[1] ) );

    if ( ! balloon_ )
    {
      balloon_ = newInstance<TextBalloon> ();
    }

    balloon_->text = String::format  ( "(%+.2e,%+.2e)", a[0], a[1] );
    balloon_->xPos = xPos;
    balloon_->yPos = yPos;

    balloon_->show ( rtx, plot_ );
  }
}


//-----------------------------------------------------------------------
//   resizeHandler_
//-----------------------------------------------------------------------


void GraphViewer::resizeHandler_

  ( RenderContext&  rtx,
    const Rect&     rect )

{
  GLfloat  width;
  GLfloat  height;


  if ( rect.width * rect.height == 0 )
  {
    return;
  }

  if ( rect.width > rect.height )
  {
    width  = 1.0F;
    height = (GLfloat) rect.height / (GLfloat) rect.width;
  }
  else
  {
    width  = (GLfloat) rect.width  / (GLfloat) rect.height;
    height = 1.0F;
  }

  plot_  ->resize    ( Size2f( width, height ) );
  camera_->lookAtObj ( *plot_ );
}


JEM_END_PACKAGE( gl )
