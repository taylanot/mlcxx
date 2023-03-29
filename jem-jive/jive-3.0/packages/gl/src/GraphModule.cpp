
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */


#include <jem/base/Time.h>
#include <jem/base/System.h>
#include <jem/base/Exception.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/list.h>
#include <jem/gl/input.h>
#include <jem/gl/utilities.h>
#include <jem/gl/Menu.h>
#include <jem/gl/MenuNode.h>
#include <jem/gl/Actions.h>
#include <jem/gl/Commands.h>
#include <jem/gl/Display.h>
#include <jem/gl/Viewer.h>
#include <jem/gl/Window.h>
#include <jem/gl/XYPlot.h>
#include <jem/gl/MultiNode.h>
#include <jem/gl/Camera2D.h>
#include <jem/gl/GLSetup2D.h>
#include <jem/gl/EventNode.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/RubberBand.h>
#include <jem/gl/MessageBox.h>
#include <jem/gl/TextBalloon.h>
#include <jem/util/None.h>
#include <jem/util/Event.h>
#include <jem/util/Timer.h>
#include <jem/util/ObjFlex.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/Runtime.h>
#include <jive/util/FuncUtils.h>
#include <jive/mp/Globdat.h>
#include <jive/app/Actions.h>
#include <jive/app/ModuleFactory.h>
#include <jive/gl/Names.h>
#include <jive/gl/GraphModule.h>


JEM_DEFINE_CLASS( jive::gl::GraphModule );


JIVE_BEGIN_PACKAGE( gl )


using jem::newInstance;
using jem::Collectable;
using jem::gl::Rect;
using jem::gl::Menu;
using jem::gl::Viewer;
using jem::gl::XYPlot;
using jem::gl::Window;
using jem::gl::StdCamera;
using jem::gl::KeyPress;
using jem::gl::MouseClick;
using jem::gl::MouseDrag;
using jem::gl::RubberBand;
using jem::gl::TextBalloon;
using jem::util::Timer;
using jem::util::ObjFlex;
using jive::util::joinNames;
using jive::util::FuncUtils;


//=======================================================================
//   class GraphModule::DataSet_
//=======================================================================


class GraphModule::DataSet_ : public NamedObject
{
 public:

  typedef DataSet_          Self;
  typedef NamedObject       Super;

  static const int          MAX_SAMPLES = 256;


  explicit                  DataSet_

    ( const String&           name );

  void                      open

    ( const Ref<XYPlot>&      plot );

  void                      close       ();

  idx_t                     sample

    ( const Properties&       globdat );

  inline void               flush       ();
  inline void               reset       ();

  void                      configure

    ( const Properties&       props,
      const Properties&       globdat );

  void                      getConfig

    ( const Properties&       conf )       const;

  virtual String            getContext  () const override;


 protected:

  virtual                  ~DataSet_    ();


 private:

  Ref<XYPlot>               plot_;
  XYPlot::DataSet*          pdata_;

  Ref<Function>             xFunc_;
  Ref<Function>             yFunc_;

  Vector                    xBuf_;
  Vector                    yBuf_;
  idx_t                     last_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GraphModule::DataSet_::DataSet_ ( const String& name ) :

  Super ( name )

{
  pdata_ = 0;
  last_  = 0;

  xBuf_.resize ( 8 );
  yBuf_.resize ( 8 );
}


GraphModule::DataSet_::~DataSet_ ()
{
  pdata_ = 0;
}


//-----------------------------------------------------------------------
//   open
//-----------------------------------------------------------------------


void GraphModule::DataSet_::open ( const Ref<XYPlot>&  plot )
{
  if ( plot_ )
  {
    close ();
  }

  pdata_ = plot->newDataSet ( myName_ );
  plot_  = plot;
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void GraphModule::DataSet_::close ()
{
  if ( plot_ )
  {
    plot_->eraseDataSet ( pdata_ );

    plot_  = nullptr;
    pdata_ = 0;
    xFunc_ = nullptr;
    yFunc_ = nullptr;
    last_  = 0;
  }
}


//-----------------------------------------------------------------------
//   sample
//-----------------------------------------------------------------------


idx_t GraphModule::DataSet_::sample ( const Properties& globdat )
{
  using jem::System;
  using jem::Exception;

  if ( pdata_ == 0 || xFunc_ == nullptr || yFunc_ == nullptr )
  {
    return last_;
  }

  double  x, y;
  idx_t   n;


  try
  {
    x = FuncUtils::evalFunc ( *xFunc_, globdat );
    y = FuncUtils::evalFunc ( *yFunc_, globdat );
  }
  catch ( const Exception& e )
  {
    print ( System::warn(), getContext(), " : ", e.what(), '\n' );

    return last_;
  }

  n = xBuf_.size ();

  if ( last_ >= n )
  {
    if ( n >= MAX_SAMPLES )
    {
      last_ = 0;
    }
    else
    {
      n = last_ + n / 2 + 8;

      if ( n > MAX_SAMPLES )
      {
        n = MAX_SAMPLES;
      }

      xBuf_.reshape ( n );
      yBuf_.reshape ( n );
    }
  }

  xBuf_[last_] = x;
  yBuf_[last_] = y;

  last_++;

  return last_;
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


inline void GraphModule::DataSet_::flush ()
{
  const idx_t  n = last_;

  last_ = 0;

  if      ( n == 1 )
  {
    pdata_->addPoint  ( xBuf_[0], yBuf_[0] );
  }
  else if ( n > 1 )
  {
    pdata_->addPoints ( xBuf_[slice(BEGIN,n)],
                        yBuf_[slice(BEGIN,n)] );
  }
}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


inline void GraphModule::DataSet_::reset ()
{
  last_ = 0;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void GraphModule::DataSet_::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  if ( pdata_ && props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );
    bool        clear   = false;


    pdata_->configure ( myProps );

    if ( FuncUtils::configFunc( xFunc_, PropNames::XDATA,
                                myProps, globdat ) )
    {
      clear = true;
    }

    if ( FuncUtils::configFunc( yFunc_, PropNames::YDATA,
                                myProps, globdat ) )
    {
      clear = true;
    }

    if ( clear )
    {
      last_ = 0;

      pdata_->clear ();
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void GraphModule::DataSet_::getConfig

  ( const Properties&  conf ) const

{
  if ( pdata_ )
  {
    Properties  myConf = conf.makeProps ( myName_ );

    pdata_   ->getConfig ( myConf );
    FuncUtils::getConfig ( myConf, xFunc_, PropNames::XDATA );
    FuncUtils::getConfig ( myConf, yFunc_, PropNames::YDATA );
  }
}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String GraphModule::DataSet_::getContext () const
{
  return NamedObject::makeContext ( "data set", myName_ );
}


//=======================================================================
//   class GraphModule::RunData_
//=======================================================================


class GraphModule::RunData_ : public Collectable
{
 public:

  typedef RunData_          Self;

  static const char*        CLEAR_CMD;


  explicit                  RunData_

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  inline void               freeze        ();
  inline void               resume        ();
  inline void               replot        ();

  inline void               sample

    ( const Properties&       globdat );

  inline void               flush

    ( double                  runtime );

  void                      shutdown      ();

  void                      configure

    ( const Properties&       props,
      const Properties&       globdat );

  void                      getConfig

    ( const Properties&       props )        const;

  inline bool               mustFlush

    ( double                  runtime )      const;


 private:

  void                      openViewer_

    ( const Properties&       conf,
      const Properties&       props );

  void                      keyPressed_

    ( RenderContext&          rtx,
      const KeyPress&         key );

  void                      mouseClicked_

    ( RenderContext&          rtx,
      const MouseClick&       mouse );

  void                      mouseDragged_

    ( RenderContext&          rtx,
      const MouseDrag&        mouse );

  void                      exeCommand_

    ( RenderContext&          rtx,
      const String&           cmd );

  void                      rectSelected_

    ( RenderContext&          rtx,
      const Rect&             rect );

  void                      clearGraph_   ();
  void                      showHelp_     ();

  void                      showMenu_

    ( RenderContext&          rtx );

  void                      showMousePos_

    ( RenderContext&          rtx,
      int                     xPos,
      int                     yPos );

  void                      resizeHandler_

    ( RenderContext&          rtx,
      const Rect&             rect );


 private:

  const String              myName_;

  double                    tflush_;
  idx_t                     samples_;
  Timer                     gfxTimer_;

  volatile bool             cleared_;

  Ref<Viewer>               viewer_;
  Ref<Window>               window_;
  Ref<StdCamera>            camera_;
  Ref<XYPlot>               plot_;
  Ref<TextBalloon>          balloon_;
  Ref<RubberBand>           rubber_;

  ObjFlex                   dataSets_;

};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  GraphModule::RunData_::CLEAR_CMD = "Clear Graph";


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


GraphModule::RunData_::RunData_

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat ) :

    myName_ ( name )

{
  using jem::gl::GLSetup;
  using jem::gl::GLSetup2D;
  using jem::gl::Camera2D;
  using jem::gl::EventNode;
  using jem::gl::MultiNode;

  Properties      myConf  = conf  .makeProps ( myName_ );
  Properties      myProps = props .findProps ( myName_ );

  StringVector    names;

  Ref<GLSetup>    setup   = newInstance<GLSetup2D> ();
  Ref<EventNode>  evNode;
  Ref<MultiNode>  world;


  tflush_  = 0.0;
  samples_ = 0;
  cleared_ = false;

  myProps.find ( names, PropNames::DATA_SETS );
  myConf .set  ( PropNames::DATA_SETS, names );

  // The XYPlot node is child of a MultiNode so that it is possible to
  // add and remove nodes (like a TextBalloon) at runtime.

  world    = newInstance<MultiNode>  ( "world" );
  plot_    = newInstance<XYPlot>     ( PropNames::GRAPH );
  evNode   = newInstance<EventNode>  ( "event",  world );
  camera_  = newInstance<Camera2D>   ( "camera", evNode );
  window_  = newInstance<Window>     ( PropNames::WINDOW,
                                       camera_, setup );
  rubber_  = newInstance<RubberBand> ( "rubber" );

  world  ->addChild  ( plot_   );
  world  ->addChild  ( rubber_ );
  window_->configure ( myProps );
  window_->getConfig ( myConf  );

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

  evNode->takeFocus ();
  dataSets_.reserve ( names.size() );

  for ( idx_t i = 0; i < names.size(); i++ )
  {
    Ref<DataSet_>  dset =

      newInstance<DataSet_> ( joinNames( myName_, names[i] ) );

    dset->open         ( plot_ );
    dset->configure    ( props, globdat );
    dset->getConfig    ( conf );
    dataSets_.pushBack ( dset );
  }

  openViewer_ ( myConf. makeProps ( PropNames::WINDOW ),
                myProps.findProps ( PropNames::WINDOW ) );

  viewer_->setScene ( window_ );
  viewer_->resume   ();
}


//-----------------------------------------------------------------------
//   freeze
//-----------------------------------------------------------------------


inline void GraphModule::RunData_::freeze ()
{
  gfxTimer_.start ();
  viewer_->freeze ();
}


//-----------------------------------------------------------------------
//   resume
//-----------------------------------------------------------------------


inline void GraphModule::RunData_::resume ()
{
  viewer_->resume ();
  gfxTimer_.stop  ();
}


//-----------------------------------------------------------------------
//   replot
//-----------------------------------------------------------------------


inline void GraphModule::RunData_::replot ()
{
  viewer_->postRedraw ();
}


//-----------------------------------------------------------------------
//   sample
//-----------------------------------------------------------------------


inline void GraphModule::RunData_::sample

  ( const Properties&  globdat )

{
  // Here is a possible race condition as two threads might try to
  // read/write the data member "cleared_" concurrently. However, this
  // is not really a problem.

  bool  cleared = cleared_;

  cleared_ = false;

  if ( cleared )
  {
    samples_ = 0;
  }

  samples_++;

  for ( idx_t i = 0; i < dataSets_.size(); i++ )
  {
    Ref<DataSet_>  dset = dataSets_.getAs<DataSet_> ( i );

    if ( cleared )
    {
      dset->reset ();
    }

    dset->sample ( globdat );
  }
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


inline void GraphModule::RunData_::flush ( double runtime )
{
  tflush_  = runtime;
  samples_ = 0;

  for ( idx_t i = 0; i < dataSets_.size(); i++ )
  {
    Ref<DataSet_>  dset = dataSets_.getAs<DataSet_> ( i );

    dset->flush ();
  }
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void GraphModule::RunData_::shutdown ()
{
  for ( idx_t i = 0; i < dataSets_.size(); i++ )
  {
    Ref<DataSet_>  dset = dataSets_.getAs<DataSet_> ( i );

    dset->close ();
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void GraphModule::RunData_::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  using jem::util::isNone;

  Properties    myProps = props.findProps ( myName_ );

  StringVector  names;
  StringVector  names2;

  idx_t         i, n;


  window_->configure ( myProps );

  // Erase all or selected data sets.

  if ( myProps.find( names, PropNames::DATA_SETS ) )
  {
    for ( i = 0; i < dataSets_.size(); i++ )
    {
      Ref<DataSet_>  dset = dataSets_.getAs<DataSet_> ( i );

      dset->close ();
    }

    dataSets_.clear ();
  }
  else
  {
    Ref<Object>  obj;

    for ( i = 0; i < dataSets_.size(); i++ )
    {
      Ref<DataSet_>  dset = dataSets_.getAs<DataSet_> ( i );

      if ( ! props.find( obj, dset->getName() ) )
      {
        continue;
      }

      if ( isNone( obj ) )
      {
        dset    ->close ();
        dataSets_.erase ( i-- );
      }
    }
  }

  // Configure the current data sets.

  for ( i = 0; i < dataSets_.size(); i++ )
  {
    Ref<DataSet_>  dset = dataSets_.getAs<DataSet_> ( i );

    dset->configure ( props, globdat );
  }

  if ( myProps.find( names2, PropNames::NEW_DATA_SETS ) )
  {
    n = names.size ();

    if ( n )
    {
      names.reshape ( n + names2.size() );
      names[slice(n,END)] = names2;
    }
    else
    {
      names.ref ( names2 );
    }
  }

  // Add new data sets.

  n = names.size ();

  if ( n )
  {
    dataSets_.reserve ( dataSets_.size() + n );

    for ( i = 0; i < n; i++ )
    {
      Ref<DataSet_>  dset =

        newInstance<DataSet_> ( joinNames( myName_, names[i] ) );

      dset->open      ( plot_ );
      dset->configure ( props, globdat );

      dataSets_.pushBack ( dset );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void GraphModule::RunData_::getConfig

  ( const Properties&  conf ) const

{
  Properties  myConf = conf.makeProps ( myName_ );

  window_->getConfig ( myConf );

  for ( idx_t i = 0; i < dataSets_.size(); i++ )
  {
    Ref<DataSet_>  dset = dataSets_.getAs<DataSet_> ( i );

    dset->getConfig ( conf );
  }
}


//-----------------------------------------------------------------------
//   mustFlush
//-----------------------------------------------------------------------


inline bool GraphModule::RunData_::mustFlush

  ( double runtime ) const

{
  if ( samples_ >= DataSet_::MAX_SAMPLES )
  {
    return true;
  }

  if ( samples_ == 0 )
  {
    return false;
  }

  double  dt = runtime - tflush_;

  if ( dt < 0.05 )
  {
    return false;
  }

  double  load = gfxTimer_.toDouble() / (runtime + 0.001);

  return ( load < 0.05 || dt > 5.0 );
}


//-----------------------------------------------------------------------
//   openViewer_
//-----------------------------------------------------------------------


void GraphModule::RunData_::openViewer_

  ( const Properties&  conf,
    const Properties&  props )

{
  using jem::gl::Display;

  String  title  = myName_;
  int     width  = 400;
  int     height = 200;

  props.find ( title,  PropNames::TITLE );
  props.find ( width,  PropNames::WIDTH,  10, 10000 );
  props.find ( height, PropNames::HEIGHT, 10, 10000 );

  conf .set  ( PropNames::TITLE,  title  );
  conf .set  ( PropNames::WIDTH,  width  );
  conf .set  ( PropNames::HEIGHT, height );

  Display::open ( Display::NEW_THREAD );

  viewer_ = Display::newWindow ( title, width, height );

  Display::mainLoop ();
}


//-----------------------------------------------------------------------
//   keyPressed_
//-----------------------------------------------------------------------


void GraphModule::RunData_::keyPressed_

  ( RenderContext&   rtx,
    const KeyPress&  key )

{
  using jem::Time;
  using jem::gl::Keys;
  using jem::gl::showMessageUntil;

  bool    handled = false;
  idx_t   keyMods = key.mods & ~Keys::SHIFT;


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
      Time::now() + 4_sec
    );
  }
}


//-----------------------------------------------------------------------
//   mouseClicked_
//-----------------------------------------------------------------------


void GraphModule::RunData_::mouseClicked_

  ( RenderContext&     rtx,
    const MouseClick&  mouse )

{
  using jem::gl::Buttons;

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


void GraphModule::RunData_::mouseDragged_

  ( RenderContext&    rtx,
    const MouseDrag&  mouse )

{
  using jem::gl::Keys;
  using jem::gl::Buttons;

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


void GraphModule::RunData_::exeCommand_

  ( RenderContext&  rtx,
    const String&   cmd )

{
  using jem::gl::Commands;
  using jem::gl::GLActions;
  using jem::gl::GLActionParams;

  bool  postRedraw = false;

  if      ( cmd == CLEAR_CMD )
  {
    clearGraph_ ();

    postRedraw = true;
  }
  else if ( cmd == Commands::HELP )
  {
    showHelp_   ();
  }
  else
  {
    Properties  params;

    params.set ( GLActionParams::COMMAND, cmd );

    postRedraw =

      window_->takeGLAction ( rtx, (int) GLActions::EXE_COMMAND,
                              params );
  }

  if ( postRedraw )
  {
    rtx.postFastRedraw ( window_ );
  }
}


//-----------------------------------------------------------------------
//   rectSelected_
//-----------------------------------------------------------------------


void GraphModule::RunData_::rectSelected_

  ( RenderContext&  rtx,
    const Rect&     rect )

{
  typedef XYPlot::Range  Range;

  using jem::gl::fromWindowToNode;
  using jem::gl::Vertex2d;
  using jem::gl::Vertex3d;

  rtx.postRedraw ( plot_ );

  if ( rect.width == 0 || rect.height == 0 )
  {
    return;
  }

  Vertex3d  u ( rect.x,              rect.y,               0.5 );
  Vertex3d  v ( rect.x + rect.width, rect.y + rect.height, 0.5 );

  if ( fromWindowToNode( rtx.getGLContext(), u, *plot_ ) &&
       fromWindowToNode( rtx.getGLContext(), v, *plot_ ) )
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
//   clearGraph_
//-----------------------------------------------------------------------


void GraphModule::RunData_::clearGraph_ ()
{
  jem::Array<XYPlot::DataSet*>  dsets = plot_->getDataSets ();

  for ( idx_t i = 0; i < dsets.size(); i++ )
  {
    // Save and set the correct base number for number-style point
    // markers.

    idx_t  nr0 = dsets[i]->getBaseNumber() + dsets[i]->pointCount ();

    dsets[i]->clear         ();
    dsets[i]->setBaseNumber ( nr0 );
  }

  cleared_ = true;
}


//-----------------------------------------------------------------------
//   showHelp_
//-----------------------------------------------------------------------


void GraphModule::RunData_::showHelp_ ()
{
  using jem::System;
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

  print ( *pr, beginItem( "Left mouse drag" ) );
  print ( *pr,   "Zoom into the selected area." );
  print ( *pr, endItem );

  print ( *pr, beginItem( "Left mouse click\n"
			  "Ctrl+Left mouse drag" ) );
  print ( *pr,   "Show the current mouse position." );
  print ( *pr, endItem );

  print ( *pr, beginItem( "Esc" ) );
  print ( *pr,   "Cancel the current zoom operation." );
  print ( *pr, endItem );

  print ( *pr, beginItem( "?" ) );
  print ( *pr,   "Show this help." );
  print ( *pr, endItem );

  print ( *pr, beginItem( "Alt+c" ) );
  print ( *pr,   "Select autoscale (view full graph)." );
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


void GraphModule::RunData_::showMenu_ ( RenderContext& rtx )
{
  using jem::util::connect;
  using jem::gl::Commands;
  using jem::gl::Actions;
  using jem::gl::ActionParams;
  using jem::gl::MenuNode;

  Ref<Menu>   menu = newInstance<Menu> ();
  Properties  params;


  params.set ( ActionParams::OPTIONS, menu );

  window_->takeAction ( Actions::LIST_OPTIONS, params );

  menu->addLine  ();
  menu->addEntry ( CLEAR_CMD );
  menu->addLine  ();
  menu->addEntry ( Commands::HELP );

  Ref<MenuNode>  node = newInstance<MenuNode> ( "menu", *menu );

  connect    ( node->selectEvent, this, & Self::exeCommand_ );
  node->show ( rtx, window_ );
}


//-----------------------------------------------------------------------
//   showMousePos_
//-----------------------------------------------------------------------


void GraphModule::RunData_::showMousePos_

  ( RenderContext&  rtx,
    int             xPos,
    int             yPos )

{
  using jem::gl::fromWindowToNode;
  using jem::gl::Vertex2d;
  using jem::gl::Vertex3d;

  Vertex3d  v ( (double) xPos, (double) yPos, 0.5 );

  if ( fromWindowToNode( rtx.getGLContext(), v, *plot_ ) )
  {
    Vertex2d  a, b;

    a[0] = v[0];
    a[1] = v[1];
    b    = plot_->mapPoint ( a );

    if ( ! balloon_ )
    {
      balloon_ = newInstance<TextBalloon> ();
    }

    balloon_->text = String::format  ( "(%+.2e,%+.2e)", b[0], b[1] );
    balloon_->xPos = xPos;
    balloon_->yPos = yPos;

    balloon_->show ( rtx, plot_ );
  }
}


//-----------------------------------------------------------------------
//   resizeHandler_
//-----------------------------------------------------------------------


void GraphModule::RunData_::resizeHandler_

  ( RenderContext&  rtx,
    const Rect&     rect )

{
  using jem::gl::Size2f;

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


//=======================================================================
//   class GraphModule::ReadLock_
//=======================================================================


class GraphModule::ReadLock_
{
 public:

  inline                    ReadLock_

    ( const GraphModule*      gm );

  inline                   ~ReadLock_   ();


 public:

  RunData_&                 rundat;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline GraphModule::ReadLock_::ReadLock_

  ( const GraphModule* gm ) : rundat ( *(gm->rundat_) )

{
  rundat.freeze ();
}


inline GraphModule::ReadLock_::~ReadLock_ ()
{
  rundat.resume ();
}


//=======================================================================
//   class GraphModule::WriteLock_
//=======================================================================


class GraphModule::WriteLock_
{
 public:

  inline                    WriteLock_

    ( const GraphModule*      gm );

  inline                   ~WriteLock_   ();


 public:

  RunData_&                 rundat;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline GraphModule::WriteLock_::WriteLock_

  ( const GraphModule* gm ) : rundat ( *(gm->rundat_) )

{
  rundat.freeze ();
}


inline GraphModule::WriteLock_::~WriteLock_ ()
{
  rundat.replot ();
  rundat.resume ();
}


//=======================================================================
//   class GraphModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  GraphModule::TYPE_NAME = "Graph";

const int    GraphModule::FG_MODE_  = 1 << 0;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GraphModule::GraphModule ( const String& name ) :

  Super ( name )

{
  mode_       = 0;
  sampleCond_ = FuncUtils::newCond ();
}


GraphModule::~GraphModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status GraphModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  rundat_ = nullptr;

  if ( mp::Globdat::myRank( globdat ) == 0 )
  {
    rundat_ = newInstance<RunData_> ( myName_, conf,
                                      props,   globdat );
  }

  return OK;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status GraphModule::run ( const Properties& globdat )
{
  using jem::Exception;
  using jive::util::Runtime;

  if ( ! rundat_ )
  {
    return OK;
  }

  RunData_&  rundat   = * rundat_;
  bool       doSample = false;

  try
  {
    doSample = FuncUtils::evalCond ( *sampleCond_, globdat );
  }
  catch ( Exception& ex )
  {
    ex.setContext ( getContext() );
    throw;
  }

  double  rt = Runtime::getValue ( globdat );

  if ( doSample )
  {
    rundat.sample ( globdat );
  }

  if ( (mode_ & FG_MODE_) || rundat.mustFlush( rt ) )
  {
    WriteLock_  lock ( this );

    rundat.flush ( rt );
  }

  return OK;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void GraphModule::shutdown ( const Properties& globdat )
{
  if ( rundat_ )
  {
    ReadLock_  lock ( this );

    lock.rundat.shutdown ();
  }

   rundat_ = nullptr;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void GraphModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    FuncUtils::configCond ( sampleCond_, PropNames::SAMPLE_COND,
                            myProps,     globdat );

    if ( rundat_ )
    {
      WriteLock_  lock ( this );

      lock.rundat.configure ( props, globdat );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void GraphModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );

  FuncUtils::getConfig ( myConf, sampleCond_,
                         PropNames::SAMPLE_COND );

  if ( rundat_ )
  {
    ReadLock_  lock ( this );

    lock.rundat.getConfig ( conf );
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool GraphModule::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::app::Actions;

  bool  result = true;

  if      ( action == Actions::FG )
  {
    mode_ |=  FG_MODE_;
  }
  else if ( action == Actions::BG )
  {
    mode_ &= ~FG_MODE_;
  }
  else
  {
    result = false;
  }

  return result;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> GraphModule::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void GraphModule::declare ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( gl )
