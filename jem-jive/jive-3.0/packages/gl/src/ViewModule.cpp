
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


#include <cstdlib>
#include <jem/base/assert.h>
#include <jem/base/Time.h>
#include <jem/base/Float.h>
#include <jem/base/System.h>
#include <jem/base/Thread.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/io/list.h>
#include <jem/io/utilities.h>
#include <jem/io/FileName.h>
#include <jem/io/FileWriter.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/IOException.h>
#include <jem/gl/Viewer.h>
#include <jem/gl/Display.h>
#include <jem/gl/Window.h>
#include <jem/gl/Camera2D.h>
#include <jem/gl/Camera3D.h>
#include <jem/gl/OrthoCamera3D.h>
#include <jem/gl/GLSetup2D.h>
#include <jem/gl/GLSetup3D.h>
#include <jem/gl/EventNode.h>
#include <jem/gl/GroupNode.h>
#include <jem/gl/LightsNode.h>
#include <jem/gl/Controller.h>
#include <jem/gl/PPMGrabber.h>
#include <jem/gl/PNGGrabber.h>
#include <jem/gl/PrintErrHandler.h>
#include <jem/util/None.h>
#include <jem/util/Flex.h>
#include <jem/util/Event.h>
#include <jem/util/Timer.h>
#include <jem/util/Properties.h>
#include <jem/util/DictionaryEnumerator.h>
#include <jem/numeric/func/Function.h>
#include <jive/util/utilities.h>
#include <jive/util/Globdat.h>
#include <jive/util/FuncUtils.h>
#include <jive/app/Actions.h>
#include <jive/app/ProgramArgs.h>
#include <jive/app/ModuleFactory.h>
#include <jive/gl/Names.h>
#include <jive/gl/Viewdat.h>
#include <jive/gl/DataSet.h>
#include <jive/gl/AppContext.h>
#include <jive/gl/ChainPlugin.h>
#include <jive/gl/UserconfPlugin.h>
#include <jive/gl/DataSetFactory.h>
#include <jive/gl/ViewModule.h>


JEM_DEFINE_CLASS( jive::gl::ViewModule );


JIVE_BEGIN_PACKAGE( gl )


using jem::newInstance;
using jem::Array;
using jem::System;
using jem::Exception;
using jem::gl::Viewer;
using jem::gl::Window;
using jem::gl::Camera;
using jem::gl::GroupNode;
using jem::gl::Controller;
using jem::util::Timer;
using jive::util::FuncUtils;
using jive::app::ProgramArgs;


//=======================================================================
//   class ViewModule::RunData_
//=======================================================================


class ViewModule::RunData_ : public jem::Collectable
{
 public:

  typedef RunData_        Self;


  inline                  RunData_

    ( const String&         nam,
      const String&         ctx,
      const Ref<Plugin>&    plugins );

  void                    init

    ( const Properties&     conf,
      const Properties&     props,
      const Properties&     globdat );

  void                    update

    ( const Properties&     globdat );

  void                    shutdown

    ( const Properties&     globdat );

  void                    configure

    ( const Properties&     props,
      const Properties&     globdat );

  void                    getConfig

    ( const Properties&     props );

  void                    setSnapFile

    ( const String&         fname );

  void                    runForever

    ( const Properties&     globdat );

  void                    saveConfig0   ();
  void                    showHelp      ();

  bool                    exeCommand

    ( const String&         cmd,
      const String&         arg,
      const Properties&     globdat );

  void                    setRunvars

    ( const Properties&     globdat );


 public:

  const String            myName;
  const String            myContext;

  Properties              viewdat;

  Ref<Viewer>             viewer;
  Ref<Controller>         controller;
  Timer                   runTimer;
  Timer                   workTimer;

  int                     mode;
  String                  confFile;
  bool                    mustExit;


 private:

  void                    openDisplay_

    ( ProgramArgs&          args,
      bool&                 offScreen );

  void                    initViewer_

    ( const Properties&     myConf,
      const Properties&     myProps,
      const ProgramArgs&    args );

  void                    initScene_

    ( const String&         myName,
      const Properties&     myConf,
      const Properties&     myProps );

  void                    initCamera3D_

    ( const String&         camName,
      const Properties&     camConf,
      const Properties&     camProps );

  void                    exit_       ();

  void                    addDataSets_

    ( const String&         myName,
      const StringVector&   names,
      const Properties&     conf,
      const Properties&     props,
      const Properties&     globdat );


 private:

  Ref<Window>             window_;
  Ref<Camera>             camera_;
  Ref<GroupNode>          world_;

  Properties              dataSets_;
  Ref<Plugin>             plugins_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline ViewModule::RunData_::RunData_

  ( const String&       nam,
    const String&       ctx,
    const Ref<Plugin>&  plugins ) :

    myName    (        nam ),
    myContext (        ctx ),
    viewdat   ( "viewData" ),
    dataSets_ ( "dataSets" ),
    plugins_  (    plugins )

{
  mode     = 0;
  mustExit = false;

  viewdat.set ( Viewdat::DATA_SETS, dataSets_ );
}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void ViewModule::RunData_::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jem::gl::Display;
  using jem::gl::EventNode;
  using jem::gl::newController2D;
  using jem::gl::newController3D;
  using jem::util::connect;

  Properties        myConf  = conf   .makeProps ( myName );
  Properties        myProps = props  .findProps ( myName );

  Properties        config0 = viewdat.makeProps ( Viewdat::CONFIG0 );

  Ref<ProgramArgs>  args    = ProgramArgs::get  ( globdat );
  Ref<EventNode>    evNode;

  StringVector      names;
  AppContext        apx;
  bool              offScreen;


  if ( mode & OFF_SCREEN_MODE )
  {
    offScreen = true;
  }
  else
  {
    offScreen = false;

    myProps.find ( offScreen, PropNames::OFF_SCREEN );
  }

  if ( ! Display::isOpen() )
  {
    openDisplay_ ( *args, offScreen );
  }
  else
  {
    offScreen = ((Display::getMode() & Display::OFF_SCREEN) != 0);
  }

  myConf .set ( PropNames::OFF_SCREEN, offScreen );
  viewdat.set ( Viewdat::GLOBDAT,      globdat   );

  initScene_  ( myName, myConf, myProps );

  if ( myProps.find( names, PropNames::DATA_SETS ) )
  {
    addDataSets_ ( myName, names, conf, props, globdat );
  }

  myConf.set ( PropNames::DATA_SETS, names );

  window_->configure ( props   );
  window_->getConfig ( conf    );
  window_->getConfig ( config0 );

  initViewer_        ( myConf, myProps, *args );
  viewer->setScene   ( window_ );

  if ( mode & MODE_2D )
  {
    controller = newController2D ( camera_, world_ );
  }
  else
  {
    controller = newController3D ( camera_, world_ );
  }

  evNode = controller->getEventNode ();

  controller->setConfig0 ( config0 );

  viewdat.set ( Viewdat::EVENT_NODE, evNode );

  connect ( controller->exitEvent,    this, & Self::exit_       );
  connect ( controller->helpEvent,    this, & Self::showHelp    );
  connect ( controller->config0Event, this, & Self::saveConfig0 );

  plugins_->init ( conf, props, viewdat, apx );

  camera_->updateZRange ();
  evNode->takeFocus     ();
  viewer->resume        ();
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void ViewModule::RunData_::update ( const Properties& globdat )
{
  using jem::checkedCast;
  using jem::util::DictEnum;
  using jive::util::Globdat;

  Ref<DictEnum>  e;
  Ref<DataSet>   ds;
  AppContext     apx;


  viewdat.set ( Viewdat::GLOBDAT, globdat );

  for ( e = dataSets_.enumerate(); ! e->atEnd(); e->toNext() )
  {
    ds = checkedCast<DataSet> ( e->getValue() );

    try
    {
      ds->update ( globdat, apx );
    }
    catch ( const jem::IllegalInputException& ex )
    {
      print ( System::warn(), ex, '\n' );
    }
  }

  plugins_->update ( viewdat, apx );
  camera_ ->updateZRange ();
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void ViewModule::RunData_::shutdown ( const Properties& globdat )
{
  viewdat.set ( Viewdat::GLOBDAT, globdat );

  plugins_->shutdown ( viewdat, AppContext() );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void ViewModule::RunData_::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  using jem::checkedCast;
  using jem::util::isNone;
  using jem::util::Flex;
  using jem::util::DictEnum;

  Properties    myProps = props.findProps ( myName );

  StringVector  names;
  AppContext    apx;


  viewdat.set ( Viewdat::GLOBDAT, globdat );

  window_->configure ( props );

  if ( myProps.find( names, PropNames::DATA_SETS ) )
  {
    Properties  conf;

    dataSets_.clear ();

    addDataSets_ ( myName, names, conf, props, globdat );
  }
  else
  {
    Ref<DictEnum>  e;
    Ref<DataSet>   ds;
    Ref<Object>    obj;
    Flex<String>   hitList;
    String         name;

    for ( e = dataSets_.enumerate(); ! e->atEnd(); e->toNext() )
    {
      ds   = checkedCast<DataSet> ( e->getValue() );
      name = ds->getName ();

      if ( ! props.find( obj, name ) )
      {
        continue;
      }

      if ( isNone( obj ) )
      {
        hitList.pushBack ( std::move( name ) );
      }
      else
      {
        ds->configure ( props, globdat, apx );
      }
    }

    for ( idx_t i = 0; i < hitList.size(); i++ )
    {
      dataSets_.erase ( hitList[i] );
    }
  }

  if ( myProps.find( names, PropNames::NEW_DATA_SETS ) )
  {
    Properties  conf;

    addDataSets_ ( myName, names, conf, props, globdat );
  }

  plugins_->configure ( props, viewdat, apx );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void ViewModule::RunData_::getConfig ( const Properties& props )
{
  using jem::checkedCast;
  using jem::util::DictEnum;

  Ref<DictEnum>  e;
  Ref<DataSet>   ds;


  for ( e = dataSets_.enumerate(); ! e->atEnd(); e->toNext() )
  {
    ds = checkedCast<DataSet> ( e->getValue() );
    ds->getConfig ( props );
  }

  window_ ->getConfig ( props );
  plugins_->getConfig ( props );
}


//-----------------------------------------------------------------------
//   setSnapFile
//-----------------------------------------------------------------------


void ViewModule::RunData_::setSnapFile ( const String& fname )
{
  using jem::gl::FrameGrabber;
  using jem::gl::PPMGrabber;
  using jem::gl::PNGGrabber;

  try
  {
    controller->setSnapFile ( fname );
  }
  catch ( const Exception& ex )
  {
    Ref<FrameGrabber>  grabber;
    String             type;


    print ( System::warn(), myContext, " : ", ex.what(), '\n' );

    if ( jem::io::canZip() )
    {
      type    = "PNG";
      grabber = newInstance<PNGGrabber> ();
    }
    else
    {
      type    = "PPM";
      grabber = newInstance<PPMGrabber> ();
    }

    print ( System::warn(), myContext,
            " : snapshot files will be saved in ",
            type, " format\n\n" );

    controller->setSnapFile ( fname, grabber );
  }
}


//-----------------------------------------------------------------------
//   runForever
//-----------------------------------------------------------------------


void ViewModule::RunData_::runForever ( const Properties& globdat )
{
  jem::Time  timeout ( 10.0 );

  mode |= FG_MODE;

  viewer->resume ();

  while ( true )
  {
    jem::Thread::sleep ( timeout );
  }
}


//-----------------------------------------------------------------------
//   saveConfig0
//-----------------------------------------------------------------------


void ViewModule::RunData_::saveConfig0 ()
{
  using jem::io::FileWriter;
  using jem::gl::Size2i;

  if ( ! confFile.size() )
  {
    return;
  }

  Properties  config0 = controller->getConfig0 ();
  Properties  winConf = config0.makeProps ( window_->getName() );

  Size2i      winSize = window_->size ();


  winConf.set ( PropNames::WIDTH,  (idx_t) winSize.width  );
  winConf.set ( PropNames::HEIGHT, (idx_t) winSize.height );

  try
  {
    Ref<PrintWriter>  prn =

      newInstance<PrintWriter> (
        newInstance<FileWriter> ( confFile )
      );

    config0.printTo ( *prn );
  }
  catch ( const Exception& ex )
  {
    print ( System::warn(), myContext, " : ", ex.what(), '\n' );

    return;
  }

  print ( System::info( myName ), myName,
          " : saved configuration data in `", confFile, "\'\n" );
}


//-----------------------------------------------------------------------
//   showHelp
//-----------------------------------------------------------------------


void ViewModule::RunData_::showHelp ()
{
  using jem::io::endl;
  using jem::io::indent;
  using jem::io::beginList;
  using jem::io::endList;

  Ref<PrintWriter>  pr = newInstance<PrintWriter> ( & System::out() );

  pr->setPageWidth ( 72 );

  print ( *pr, endl );
  print ( *pr, "Key and mouse bindings:" );
  print ( *pr, indent );
  print ( *pr, beginList );

  controller->printHelp ( *pr );
  plugins_  ->printHelp ( *pr );

  print ( *pr, endList );

  pr->flush ();
}


//-----------------------------------------------------------------------
//   exeCommand
//-----------------------------------------------------------------------


bool ViewModule::RunData_::exeCommand

  ( const String&      cmd,
    const String&      arg,
    const Properties&  globdat )

{
  if ( cmd == "explode" )
  {
    double  scale = 0.1;

    if ( arg.size() )
    {
      scale = jem::Float::parse ( arg );
    }

    jem::gl::explode   ( *world_, scale );
    viewer->postRedraw ();

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   setRunvars
//-----------------------------------------------------------------------


void ViewModule::RunData_::setRunvars ( const Properties& globdat )
{
  using jive::util::Globdat;

  if ( ! Globdat::hasVariable( myName, globdat ) )
  {
    return;
  }

  Properties  myVars = Globdat::getVariables ( myName, globdat );

  if ( myVars.contains( VarNames::UPDATE_LOAD ) )
  {
    double  rt = runTimer .toDouble ();
    double  wt = workTimer.toDouble ();
    double  wl = 0.0;

    if ( rt > 0.0 )
    {
      wl = wt / rt;
    }

    myVars.set ( VarNames::UPDATE_LOAD, wl );
  }

  if ( myVars.contains( VarNames::RENDER_LOAD ) )
  {
    myVars.set ( VarNames::RENDER_LOAD, viewer->getWorkLoad() );
  }
}


//-----------------------------------------------------------------------
//   openDisplay_
//-----------------------------------------------------------------------


void ViewModule::RunData_::openDisplay_

  ( ProgramArgs&  args,
    bool&         offScreen )

{
  using jem::gl::Display;

  int  mode = Display::NEW_THREAD;


  if ( offScreen )
  {
    mode |= Display::OFF_SCREEN;
  }

  mode = Display::open ( args.argc(), args.argv(), mode );

  args.commit ();

  offScreen = ((mode & Display::OFF_SCREEN) != 0);
}


//-----------------------------------------------------------------------
//   initViewer_
//-----------------------------------------------------------------------


void ViewModule::RunData_::initViewer_

  ( const Properties&   myConf,
    const Properties&   myProps,
    const ProgramArgs&  args )

{
  using jem::io::FileName;
  using jem::gl::Display;
  using jem::gl::PrintErrHandler;
  using jive::util::joinNames;

  Properties  winConf  = myConf .makeProps ( PropNames::WINDOW );
  Properties  winProps = myProps.findProps ( PropNames::WINDOW );

  String      title;
  int         width    = -1;
  int         height   = -1;


  if ( ! winProps.find( title, PropNames::TITLE ) )
  {
    title = FileName::getFileName ( args.getProgramName() );
    title = joinNames ( title, myName );
  }

  winProps.find ( width,  PropNames::WIDTH,  -1, 6000 );
  winProps.find ( height, PropNames::HEIGHT, -1, 6000 );

  winConf .set  ( PropNames::TITLE,  title  );
  winConf .set  ( PropNames::WIDTH,  width  );
  winConf .set  ( PropNames::HEIGHT, height );

  viewer = Display::newWindow (
    title,
    width, height,
    newInstance<PrintErrHandler> ( myContext )
  );

  viewdat.set ( Viewdat::VIEWER, viewer );

  Display::mainLoop ();
}


//-----------------------------------------------------------------------
//   initScene_
//-----------------------------------------------------------------------


void ViewModule::RunData_::initScene_

  ( const String&      myName,
    const Properties&  myConf,
    const Properties&  myProps )

{
  using jem::gl::Camera2D;
  using jem::gl::GLSetup;
  using jem::gl::GLSetup2D;
  using jem::gl::GLSetup3D;
  using jem::gl::LightsNode;
  using jive::util::joinNames;

  Ref<GLSetup>  setup;
  String        camName;


  world_ = newInstance<GroupNode>

    ( joinNames( myName, PropNames::WORLD ) );

  world_->addChild (
    newInstance<LightsNode> ( joinNames( myName, PropNames::LIGHTS ) )
  );

  camName = joinNames ( myName, PropNames::CAMERA );

  if ( mode & MODE_2D )
  {
    setup   = newInstance<GLSetup2D> ();
    camera_ = newInstance<Camera2D>  ( camName, world_ );
  }
  else
  {
    setup   = newInstance<GLSetup3D> ();

    initCamera3D_ ( camName,
                    myConf .makeProps ( PropNames::CAMERA ),
                    myProps.findProps ( PropNames::CAMERA ) );
  }

  window_ = newInstance<Window> (
    joinNames ( myName, PropNames::WINDOW ),
    camera_,
    setup
  );

  viewdat.set ( Viewdat::WORLD_NODE, world_ );
}


//-----------------------------------------------------------------------
//   initCamera3D_
//-----------------------------------------------------------------------


void ViewModule::RunData_::initCamera3D_

  ( const String&      camName,
    const Properties&  camConf,
    const Properties&  camProps )

{
  using jem::gl::Camera3D;
  using jem::gl::OrthoCamera3D;

  String  type = "Perspective";


  camProps.find ( type, "type" );

  if      ( type == "Ortho" )
  {
    camera_ = newInstance<OrthoCamera3D> ( camName, world_ );
  }
  else if ( type == "Perspective" )
  {
    camera_ = newInstance<Camera3D>      ( camName, world_ );
  }
  else
  {
    camProps.propertyError (
      "type",
      String::format (
        "invalid camera type: %s; valid types are "
        "`Ortho\' and `Perspective\'",
        type
      )
    );
  }

  camConf.set ( "type", type );
}


//-----------------------------------------------------------------------
//   exit_
//-----------------------------------------------------------------------


void ViewModule::RunData_::exit_ ()
{
  if ( mode & FG_MODE)
  {
    std::exit ( 0 );
  }
  else
  {
    mustExit = true;
  }
}


//-----------------------------------------------------------------------
//   addDataSets_
//-----------------------------------------------------------------------


void ViewModule::RunData_::addDataSets_

  ( const String&        myName,
    const StringVector&  names,
    const Properties&    conf,
    const Properties&    props,
    const Properties&    globdat )

{
  using jive::util::joinNames;

  const idx_t   n = names.size ();

  AppContext    apx;
  Ref<DataSet>  ds;
  String        name;


  for ( idx_t i = 0; i < n; i++ )
  {
    name = joinNames ( myName, names[i] );
    ds   =

      DataSetFactory::newInstance ( name, conf, props, globdat, apx );

    ds->configure ( props, globdat, apx );
    ds->getConfig ( conf );

    dataSets_.set ( names[i], ds );
  }
}


//=======================================================================
//   class ViewModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int    ViewModule::MODE_2D         = 1 << 0;
const int    ViewModule::FG_MODE         = 1 << 1;
const int    ViewModule::OFF_SCREEN_MODE = 1 << 2;

const char*  ViewModule::TYPE_NAME       = "View";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ViewModule::ViewModule

  ( const String&  name,
    int            mode ) :

    Super ( name ),
    mode_ ( mode )

{
  String  prefix;
  String  cond;

  if ( myName_.size() )
  {
    prefix = myName_ + ".";
  }

  cond = String::format (
    "(%supdateLoad < 0.1) && (%srenderLoad < 0.5)",
    prefix,
    prefix
  );

  if ( jem::io::canZip() )
  {
    snapFile_ = "snapshot%2i.png";
  }
  else
  {
    snapFile_ = "snapshot%2i.ppm";
  }

  if ( myName_.size() )
  {
    confFile_ = myName_ + ".conf";
  }

  autoSnap_ = false;
  updCond_  = FuncUtils::newCond ( cond  );
  snapCond_ = FuncUtils::newCond ( false );
  plugins_  = newInstance<ChainPlugin> ( myName_ );
}


ViewModule::~ViewModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status ViewModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  rundat_ = nullptr;

  if ( confFile_.size() )
  {
    loadConfig0_ ( props );
  }

  Ref<RunData_>  newdat =

    newInstance<RunData_> ( myName_, getContext(), plugins_ );

  newdat->mode = mode_;

  newdat->init        ( conf, props, globdat );
  newdat->setSnapFile ( snapFile_ );

  newdat->confFile = confFile_;

  // Init succeeded, switch to running mode

  rundat_.swap ( newdat );

  // This is to make sure that all runtime variables used in the
  // default update expression are properly declared.

  FuncUtils::declareVars ( *updCond_, globdat );

  return OK;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status ViewModule::run ( const Properties& globdat )
{
  if ( ! rundat_ )
  {
    return DONE;
  }

  RunData_&  d = * rundat_;

  bool       doSnap, doUpdate;


  if ( d.mustExit )
  {
    return EXIT;
  }

  // The runtime variables must be set before evaluating the update
  // and snap conditions.

  d.setRunvars ( globdat );

  doSnap   = autoSnap_;
  doUpdate = false;

  try
  {
    if ( ! doSnap )
    {
      doSnap = FuncUtils::evalCond ( *snapCond_, globdat );
    }

    doUpdate = doSnap || ((mode_ & FG_MODE) != 0);

    if ( ! doUpdate )
    {
      doUpdate = FuncUtils::evalCond ( *updCond_, globdat );
    }
  }
  catch ( Exception& ex )
  {
    ex.setContext ( d.myContext );
    throw;
  }

  if ( doUpdate )
  {
    d.runTimer .reset    ();
    d.workTimer.reset    ();
    d.workTimer.start    ();

    d.viewer->freeze     ();
    d.viewer->postRedraw ();

    try
    {
      d.update ( globdat );

      if ( doSnap )
      {
        d.controller->makeSnapshot ( *d.viewer );
      }
    }
    catch ( const Exception& )
    {
      d.viewer->resume ();
      d.workTimer.stop ();
      throw;
    }

    d.viewer->resume ();
    d.workTimer.stop ();
  }

  return OK;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void ViewModule::shutdown ( const Properties& globdat )
{
  if ( rundat_ )
  {
    rundat_->viewer->freeze ();
    rundat_->shutdown ( globdat );

    rundat_ = nullptr;
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void ViewModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  Properties   myProps = props.findProps ( myName_ );

  bool         newSnapFile;


  myProps.find ( confFile_, PropNames::CONF_FILE );

  FuncUtils::configCond ( updCond_,  PropNames::UPDATE_COND,
                          myProps,   globdat );
  FuncUtils::configCond ( snapCond_, PropNames::SNAP_COND,
                          myProps,   globdat );

  newSnapFile = myProps.find ( snapFile_, PropNames::SNAP_FILE );

  if ( ! rundat_ )
  {
    Properties  viewdat;
    AppContext  apx;

    viewdat.set ( Viewdat::GLOBDAT, globdat );

    plugins_->configure ( props, viewdat, apx );
  }
  else
  {
    RunData_&  d = * rundat_;

    d.workTimer.start    ();
    d.viewer->freeze     ();
    d.viewer->postRedraw ();

    try
    {
      d.confFile = confFile_;

      if ( newSnapFile )
      {
        d.setSnapFile ( snapFile_ );
      }

      d.configure ( props, globdat );
      d.update    ( globdat );
    }
    catch ( const Exception& )
    {
      d.viewer->resume ();
      d.workTimer.stop ();
      throw;
    }

    d.viewer->resume ();
    d.workTimer.stop ();
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void ViewModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );

  myConf.set ( PropNames::SNAP_FILE, snapFile_ );
  myConf.set ( PropNames::CONF_FILE, confFile_ );

  FuncUtils::getConfig ( myConf, updCond_,  PropNames::UPDATE_COND );
  FuncUtils::getConfig ( myConf, snapCond_, PropNames::SNAP_COND   );

  if ( ! rundat_ )
  {
    plugins_->getConfig ( conf );
  }
  else
  {
    RunData_&  d = * rundat_;

    d.viewer->freeze ();
    d.getConfig      ( conf );
    d.viewer->resume ();
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool ViewModule::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::app::Actions;
  using jive::app::ActionParams;

  bool  result = false;


  if      ( action == Actions::FG )
  {
    setMode ( mode_ | FG_MODE );

    result = true;
  }
  else if ( action == Actions::BG )
  {
    setMode ( mode_ & ~FG_MODE );

    result = true;
  }
  else if ( action == Actions::EXE_COMMAND )
  {
    result = exeCommand_ ( params, globdat );
  }
  else if ( action == Actions::PRINT_HELP )
  {
    Ref<PrintWriter>  pr;

    params.get ( pr, ActionParams::PRINTER );
    printHelp  ( *pr );

    result = true;
  }
  else if ( action == Actions::GET_COMMANDS )
  {
    params.set ( "explode",  "0.1" );
    params.set ( "autosnap", "on off" );

    result = true;
  }

  return result;
}


//-----------------------------------------------------------------------
//   runForever
//-----------------------------------------------------------------------


void ViewModule::runForever ( const Properties& globdat )
{
  if ( rundat_ )
  {
    rundat_->viewer->freeze ();

    rundat_->runForever ( globdat );
  }
}


//-----------------------------------------------------------------------
//   setMode
//-----------------------------------------------------------------------


void ViewModule::setMode ( int mode )
{
  mode_ = mode;

  if ( rundat_ )
  {
    rundat_->mode = mode;
  }
}


//-----------------------------------------------------------------------
//   getMode
//-----------------------------------------------------------------------


int ViewModule::getMode () const
{
  return mode_;
}


//-----------------------------------------------------------------------
//   addPlugin
//-----------------------------------------------------------------------


void ViewModule::addPlugin ( const Ref<Plugin>&  plugin )
{
  JEM_PRECHECK ( plugin );

  plugins_->pushBack ( plugin );
}


void ViewModule::addPlugin

  ( const Ref<Plugin>&  plugin,
    const Properties&   conf,
    const Properties&   props )

{
  JEM_PRECHECK ( plugin );

  if ( rundat_ )
  {
    RunData_&   d = * rundat_;

    AppContext  apx;

    d.viewer->freeze     ();
    d.viewer->postRedraw ();

    try
    {
      plugin->init   ( conf, props, d.viewdat, apx );
      plugin->update ( d.viewdat, apx );
    }
    catch ( const Exception& )
    {
      d.viewer->resume ();
      throw;
    }

    d.viewer->resume ();
  }

  plugins_->pushBack ( plugin );
}


//-----------------------------------------------------------------------
//   addUserPlugin
//-----------------------------------------------------------------------


Ref<Plugin> ViewModule::addUserPlugin ()
{
  Ref<Plugin>  p = newInstance<UserconfPlugin> ( myName_ );

  plugins_->pushBack ( p );

  return p;
}


//-----------------------------------------------------------------------
//   erasePlugin
//-----------------------------------------------------------------------


void ViewModule::erasePlugin ( const Ref<Plugin>&  plugin )
{
  plugins_->erase ( plugin );

  if ( rundat_ && plugin )
  {
    RunData_&  d = * rundat_;

    d.viewer->freeze     ();
    d.viewer->postRedraw ();

    try
    {
      plugin->shutdown ( d.viewdat, AppContext() );
    }
    catch ( const Exception& )
    {
      d.viewer->resume ();
      throw;
    }

    d.viewer->resume ();
  }
}


//-----------------------------------------------------------------------
//   printHelp
//-----------------------------------------------------------------------


void ViewModule::printHelp ( PrintWriter& out ) const
{
  using jem::io::beginItem;
  using jem::io::endItem;

  print ( out, beginItem( "autosnap [state]" ) );
  print ( out,   "Turn on/off autosnap mode. If present, the " );
  print ( out,   "argument <state> should be either `on\' or " );
  print ( out,   "`off\'. If <state> is omitted, then the " );
  print ( out,   "current state is printed. If autosnap is " );
  print ( out,   "turned on, a snapshot will be saved " );
  print ( out,   "every new time step/iteration number." );
  print ( out, endItem );

  print ( out, beginItem ( "explode [frac]" ) );
  print ( out,   "Move all objects away from each other. " );
  print ( out,   "The optional argument <frac> specifies " );
  print ( out,   "the distance relative to the size of the " );
  print ( out,   "world. If omitted, <frac> is set to 0.1." );
  print ( out, endItem );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void ViewModule::declare ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> ViewModule::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  Ref<Self>  self = newInstance<Self> ( name );

  self->addUserPlugin ();

  return self;
}


//-----------------------------------------------------------------------
//   loadConfig0_
//-----------------------------------------------------------------------


void ViewModule::loadConfig0_ ( const Properties& props )
{
  Properties  conf0;

  try
  {
    conf0.parseFile ( confFile_, Properties::PARSE_INCLUDE );
  }
  catch ( const jem::io::IOException& )
  {
    return;
  }
  catch ( const Exception& ex )
  {
    print ( System::warn(), getContext(), " : ", ex, '\n' );

    return;
  }

  print ( System::info( myName_ ), myName_,
          " : loaded configuration data from `",
          confFile_, "\'\n" );

  jem::util::mergeNoReplace ( props, conf0 );
}


//-----------------------------------------------------------------------
//   exeCommand_
//-----------------------------------------------------------------------


bool ViewModule::exeCommand_

  ( const Properties&  params,
    const Properties&  globdat )

{
  using jive::app::ActionParams;

  String  cmd;
  String  arg;


  params.get ( cmd, ActionParams::COMMAND     );
  params.get ( arg, ActionParams::COMMAND_ARG );


  if ( cmd == "autosnap" )
  {
    if      ( arg.size() == 0 )
    {
      print ( System::out(), "autosnap = ", autoSnap_, '\n' );
    }
    else if ( arg == "on" )
    {
      autoSnap_ = true;
    }
    else if ( arg == "off" )
    {
      autoSnap_ = false;
    }
    else
    {
      throw Exception (
        getContext(),
        String::format (
          "invalid autosnap state: %s; should be "
          "`on\' or `off\'",
          arg
        )
      );
    }

    return true;
  }

  if ( rundat_ )
  {
    RunData_&  d      = * rundat_;
    bool       result = false;

    d.viewer->freeze ();

    try
    {
      result = d.exeCommand ( cmd, arg, globdat );
    }
    catch ( const Exception& )
    {
      d.viewer->resume ();
      throw;
    }

    d.viewer->resume ();

    return result;
  }

  return false;
}


JIVE_END_PACKAGE( gl )
