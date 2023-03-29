
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


#include <cctype>
#include <cstdlib>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/Time.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/io/list.h>
#include <jem/io/IOException.h>
#include <jem/io/FileOutputStream.h>
#include <jem/io/BufferedOutputStream.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jem/util/StringUtils.h>
#include <jem/gl/input.h>
#include <jem/gl/render.h>
#include <jem/gl/utilities.h>
#include <jem/gl/Box.h>
#include <jem/gl/Menu.h>
#include <jem/gl/MenuNode.h>
#include <jem/gl/Color.h>
#include <jem/gl/Actions.h>
#include <jem/gl/Commands.h>
#include <jem/gl/Viewer.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/EventNode.h>
#include <jem/gl/FrameGrabber.h>
#include <jem/gl/Pivotable.h>
#include <jem/gl/RedrawRoot.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/Transformable.h>
#include <jem/gl/MessageBox.h>
#include <jem/gl/MultiNode.h>
#include <jem/gl/TextPainter.h>
#include <jem/gl/EventHandler.h>
#include <jem/gl/Controller.h>


JEM_DEFINE_CLASS( jem::gl::Controller );


JEM_BEGIN_PACKAGE( gl )


using jem::util::StringUtils;


//=======================================================================
//   class Controller
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Controller::Controller

  ( Ref<MultiNode>     world,
    Ref<EventHandler>  handler ) :

    world_     ( world ),
    evHandler_ ( handler ),
    config0_   ( "config" )

{
  using jem::util::connect;

  JEM_PRECHECK ( handler && world );

  GfxNode*     node;
  Pivotable*   pivot;


  snapCount_ = 0;
  showNames_ = false;
  focus_     = world_;

  // Find an existing event node, or insert a new one in the scene
  // graph.

  node  = world->getParent ();
  pivot = 0;

  while ( node )
  {
    evNode_ = dynamic_cast<EventNode*> ( node );

    if ( evNode_ )
    {
      break;
    }

    Pivotable*  tmpPivot = dynamic_cast<Pivotable*> ( node );

    if ( tmpPivot )
    {
      pivot = tmpPivot;
    }

    if ( dynamic_cast<RedrawRoot*>( node ) )
    {
      break;
    }

    node = node->getParent ();
  }

  if ( ! evNode_ )
  {
    if ( ! pivot )
    {
      throw IllegalInputException (
        JEM_FUNC,
        "unable to insert an EventNode in the scene graph"
      );
    }

    Ref<GfxNode>  child = pivot->getChild ();

    pivot->setChild ( nullptr );

    evNode_ = newInstance<EventNode> (
      child->getName() + ".eventNode",
      child
    );

    pivot->setChild ( evNode_ );
  }

  connect ( evNode_->resizeEvent,
            evHandler_, & EventHandler::newView );

  connect ( evNode_->dragEvent,
            evHandler_, & EventHandler::mouseDragged );

  connect ( evNode_->mouseEvent,
            this, & Self::mouseClicked_ );

  connect ( evNode_->redrawEvent,
            this, & Self::redraw_ );

  connect ( evNode_->focusEvent,
            this, & Self::newFocus_ );

  connect ( evNode_->keyEvent,
            this, & Self::keyPressed_ );

  connect ( evNode_->commandEvent,
            this, & Self::exeCommand_ );

  connect ( world_->childErasedEvent,
            this, & Self::nodeErased_ );
}


Controller::~Controller ()
{}


//-----------------------------------------------------------------------
//   setSnapFile
//-----------------------------------------------------------------------


void Controller::setSnapFile ( const String& name )
{
  if ( ! name.size() )
  {
    grabber_ = nullptr;
  }
  else
  {
    grabber_ = FrameGrabber::byFile ( name );
  }

  if ( name != snapFile_ )
  {
    snapCount_ = 0;
  }

  snapFile_ = name;
}


void Controller::setSnapFile

  ( const String&      name,
    Ref<FrameGrabber>  grabber )

{
  JEM_PRECHECK ( name.size() > 0 && grabber );

  if ( name != snapFile_ )
  {
    snapCount_ = 0;
  }

  snapFile_ = name;
  grabber_  = grabber;
}


//-----------------------------------------------------------------------
//   makeSnapshot
//-----------------------------------------------------------------------


String Controller::makeSnapshot ( Viewer& viewer )
{
  if ( ! grabber_ )
  {
    return String ();
  }
  else
  {
    Ref<OutputStream>  output;
    String             fname;

    initSnapshot_    ( fname, output );
    viewer.grabFrame ( world_, *output, *grabber_ );

    if ( snapCount_ == maxOf( snapCount_ ) )
    {
      snapCount_ = 1;
    }
    else
    {
      snapCount_++;
    }

    return fname;
  }
}


//-----------------------------------------------------------------------
//   printHelp
//-----------------------------------------------------------------------


void Controller::printHelp ( PrintWriter& pr ) const
{
  using jem::io::beginItem;
  using jem::io::endItem;

  evHandler_->printHelp ( pr );

  print ( pr, beginItem( "Right mouse click" ) );
  print ( pr,   "Show the main menu." );
  print ( pr, endItem );

  print ( pr, beginItem( "Ctrl+l" ) );
  print ( pr,   "Redraw." );
  print ( pr, endItem );

  if ( exitEvent.isConnected() )
  {
    print ( pr, beginItem( "Ctrl+q" ) );
    print ( pr,   "Quit." );
    print ( pr, endItem );
  }

  print ( pr, beginItem( "Ctrl+r" ) );
  print ( pr,   "Reset to the last saved configuration." );
  print ( pr, endItem );

  print ( pr, beginItem( "Tab" ) );
  print ( pr,   "Select the next object." );
  print ( pr, endItem );

  print ( pr, beginItem( "Alt+Tab" ) );
  print ( pr,   "Select the previous object." );
  print ( pr, endItem );

  if ( grabber_ )
  {
    print ( pr, beginItem( "Alt+s" ) );
    print ( pr,   "Make a screenshot." );
    print ( pr, endItem );
  }

  print ( pr, beginItem( "Ctrl+s" ) );
  print ( pr,   "Save the current configuration." );
  print ( pr, endItem );

  print ( pr, beginItem( "?" ) );
  print ( pr,   "Print help." );
  print ( pr, endItem );
}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void Controller::reset ()
{
  GfxNode*  root = findRedrawRoot ( world_ );

  evHandler_->reset ();

  if ( ! root )
  {
    root = evNode_.get ();
  }

  if ( config0_.size() )
  {
    root->configure ( config0_ );
  }
}


//-----------------------------------------------------------------------
//   setConfig0
//-----------------------------------------------------------------------


void Controller::setConfig0 ( const Properties& props )
{
  config0_ = props;
}


//-----------------------------------------------------------------------
//   getConfig0
//-----------------------------------------------------------------------


Properties Controller::getConfig0 () const
{
  return config0_;
}


//-----------------------------------------------------------------------
//   getEventNode
//-----------------------------------------------------------------------


EventNode* Controller::getEventNode () const
{
  return evNode_.get ();
}


//-----------------------------------------------------------------------
//   redraw_
//-----------------------------------------------------------------------


void Controller::redraw_

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params )

{
  if ( showNames_ )
  {
    double  lod = 1.0;

    if ( action == (int)GLActions::REDRAW_FAST )
    {
      params.find ( lod, GLActionParams::DETAIL_LEVEL );
    }

    if ( lod > 0.25 )
    {
      drawNames_ ( rtx );
    }
  }

  evHandler_->redraw ( rtx, action, params );
}


//-----------------------------------------------------------------------
//   newFocus_
//-----------------------------------------------------------------------


void Controller::newFocus_ ( bool state )
{
  if ( state )
  {
    evHandler_->setFocus ( focus_ );
  }
  else
  {
    evHandler_->setFocus ( nullptr );
  }
}


//-----------------------------------------------------------------------
//   drawNames_
//-----------------------------------------------------------------------


void Controller::drawNames_ ( RenderContext& rtx )
{
  const idx_t     worldSize  = world_->size     ();
  GLContext&      gtx        = rtx.getGLContext ();
  Rect            vport      = gtx.getViewport  ();
  Transform       p          =

    gtx.getTransform ( GLContext::PROJECTION );

  AttribScope     attribs    ( gtx, GLContext::COLOR |
                                    GLContext::LIGHTING );
  TextPainter     txp;

  Transformable*  tf;
  Transform       t;
  Ref<GfxNode>    node;
  Alignment       hAlign;
  Alignment       vAlign;
  String          name;
  Vertex3d        a, b, c, d;
  Vertex3d        v;
  Box             box;


  gtx.setColor      ( Color::black() );
  gtx.setLighting   ( false );
  gtx.pushTransform ( GLContext::PROJECTION );
  gtx.setOrtho      ( vport, 0.0, -1.0,
                      GLContext::PROJECTION );
  gtx.pushTransform ();
  gtx.setIdentity   ();

  for ( idx_t i = 0; i < worldSize; i++ )
  {
    node = world_->getChild ( i );
    tf   = toTransformable  ( node );

    if ( ! tf )
    {
      continue;
    }

    tf->getBareBBox ( box );

    if ( box.isEmpty() )
    {
      continue;
    }

    t = getTransformOf  ( node );
    c = box.center      ();

    box.transform       ( t );
    box.project         ( p );
    box.getCorners      ( a, b );

    // If the object is not visible, then continue

    if ( a[0] > 1.0 || b[0] < -1.0 ||
         a[1] > 1.0 || b[1] < -1.0 ||
         a[2] > 1.0 || b[2] < -1.0 )
    {
      continue;
    }

    if ( project( d, p, c ) )
    {
      v[0] = d[0];
    }
    else
    {
      v[0] = 0.5 * (a[0] + b[0]);
    }

    hAlign = CENTERED;

    if      ( v[0] < -0.99 )
    {
      v[0]   = jem::min ( (GLdouble) -0.99, b[0] );
      hAlign = LEFT_ALIGNED;
    }
    else if ( v[0] > 0.99 )
    {
      v[0]   = jem::max (  (GLdouble) 0.99, a[0] );
      hAlign = RIGHT_ALIGNED;
    }

    if      ( a[1] < -0.99 && b[1] > 0.99 )
    {
      v[1]   = -0.99;
      vAlign = BOT_ALIGNED;
    }
    else if ( a[1] < -0.99 )
    {
      v[1]   = b[1] + 0.01;
      vAlign = BOT_ALIGNED;
    }
    else
    {
      v[1]   = a[1] - 0.01;
      vAlign = TOP_ALIGNED;
    }

    if      ( a[2] >= -1.0 )
    {
      v[2] = a[2];
    }
    else if ( b[2] <= 1.0 )
    {
      v[2] = b[2];
    }
    else
    {
      v[2] = 0.5 * (a[2] + b[2]);
    }

    toWindow ( v, vport );

    name = StringUtils::shorten ( node->getName() );

    txp.hAlignment = hAlign;
    txp.vAlignment = vAlign;

    txp.begin     ( gtx );
    txp.setCursor ( (GLint) v[0], (GLint) v[1], (GLfloat) v[2] );
    txp.writeLine ( gtx, name );
    txp.end       ( gtx );
  }

  gtx.popTransform ( GLContext::PROJECTION );
  gtx.popTransform ();
}


//-----------------------------------------------------------------------
//   keyPressed_
//-----------------------------------------------------------------------


void Controller::keyPressed_

  ( RenderContext&   rtx,
    const KeyPress&  key )

{
  bool  postRedraw = false;
  bool  handled    = false;
  int   keyMods    = key.mods & ~Keys::SHIFT;


  switch ( key.code )
  {
  case 'l':

    if ( keyMods == Keys::CTRL )
    {
      handled = postRedraw = true;
    }

    break;

  case 'q':

    if ( keyMods == Keys::CTRL )
    {
      if ( exitEvent.isConnected() )
      {
        exitEvent.emit ( *this, rtx );

        handled = true;
      }
      else
      {
        ::exit ( 0 );
      }
    }

    break;

  case 'r':

    if ( keyMods == Keys::CTRL )
    {
      reset           ();
      resetEvent.emit ( *this, rtx );

      handled = postRedraw = true;
    }

    break;

  case '\t':

    if      ( keyMods == Keys::ALT )
    {
      selectPrev_ ();

      handled = true;
    }
    else if ( keyMods == 0 )
    {
      selectNext_ ();

      handled = true;
    }

    postRedraw = handled;

    break;

  case 's':

    if      ( keyMods == Keys::CTRL )
    {
      saveConfig0_      ( rtx );
      config0Event.emit ( *this, rtx );

      handled = true;
    }
    else if ( keyMods == Keys::ALT )
    {
      makeSnapshot_ ( rtx );

      handled = true;
    }

    break;

  case '?':

    if ( keyMods == 0 )
    {
      handled = true;

      if ( helpEvent.isConnected() )
      {
        helpEvent.emit ( *this, rtx );
      }
      else
      {
        showHelp_ ();
      }
    }

    break;
  }

  if ( ! handled )
  {
    evHandler_->keyPressed ( rtx, key );
  }

  if ( postRedraw )
  {
    rtx.postFastRedraw ( world_ );
  }
}


//-----------------------------------------------------------------------
//   mouseClicked_
//-----------------------------------------------------------------------


void Controller::mouseClicked_

  ( RenderContext&     rtx,
    const MouseClick&  mouse )

{
  if ( mouse.mods   == 0              &&
       mouse.button == Buttons::RIGHT &&
       mouse.what   == MouseClick::DOWN )
  {
    showMenu_ ( rtx );
  }
  else
  {
    evHandler_->mouseClicked ( rtx, mouse );
  }
}


//-----------------------------------------------------------------------
//   exeCommand_
//-----------------------------------------------------------------------


void Controller::exeCommand_

  ( RenderContext&  rtx,
    const String&   cmd )

{
  bool  postRedraw = false;

  if      ( cmd == Commands::SAVE_STATE )
  {
    saveConfig0_      ( rtx );
    config0Event.emit ( *this, rtx );
  }
  else if ( cmd == Commands::RESTORE_STATE )
  {
    reset             ();
    resetEvent.emit   ( *this, rtx );
  }
  else if ( cmd == Commands::SAVE_SNAPSHOT )
  {
    makeSnapshot_     ( rtx );
  }
  else if ( cmd == Commands::HELP )
  {
    if ( helpEvent.isConnected() )
    {
      helpEvent.emit  ( *this, rtx );
    }
    else
    {
      showHelp_ ();
    }
  }
  else if ( cmd == Commands::QUIT )
  {
    if ( exitEvent.isConnected() )
    {
      exitEvent.emit  ( *this, rtx );
    }
    else
    {
      ::exit ( 0 );
    }
  }
  else if ( cmd == Commands::SHOW_OPTIONS )
  {
    showOptions_ ( rtx );
  }
  else if ( cmd == Commands::SELECT )
  {
    showNodes_ ( rtx );
  }
  else if ( cmd == Commands::SHOW_NAMES )
  {
    postRedraw = ! showNames_;
    showNames_ = true;
  }
  else if ( cmd == Commands::HIDE_NAMES )
  {
    postRedraw = showNames_;
    showNames_ = false;
  }
  else
  {
    postRedraw = evHandler_->exeCommand ( rtx, cmd );
  }

  if ( postRedraw )
  {
    rtx.postFastRedraw ( world_ );
  }
}


//-----------------------------------------------------------------------
//   showHelp_
//-----------------------------------------------------------------------


void Controller::showHelp_ ()
{
  using jem::io::endl;
  using jem::io::flush;
  using jem::io::indent;
  using jem::io::beginList;
  using jem::io::endList;

  Ref<PrintWriter>  pr = newInstance<PrintWriter> ( & System::out() );

  pr->setPageWidth ( 72 );

  print     ( *pr, endl );
  print     ( *pr, "Key and mouse bindings:" );
  print     ( *pr, indent, beginList );
  printHelp ( *pr );
  print     ( *pr, endList, flush );
}


//-----------------------------------------------------------------------
//   showMenu_
//-----------------------------------------------------------------------


void Controller::showMenu_ ( RenderContext& rtx )
{
  using jem::util::connect;

  Ref<Menu>  menu = newInstance<Menu> ();

  evHandler_->initMenu ( *menu );

  menu->addLine  ();
  menu->addEntry ( Commands::SAVE_STATE );
  menu->addEntry ( Commands::RESTORE_STATE );

  if ( grabber_ )
  {
    menu->addEntry ( Commands::SAVE_SNAPSHOT );
  }

  menu->addLine  ();
  menu->addEntry ( Commands::SHOW_OPTIONS );
  menu->addEntry ( Commands::SELECT );
  menu->addLine  ();
  menu->addEntry ( Commands::HELP );
  menu->addEntry ( Commands::QUIT );

  Ref<MenuNode>  node = newInstance<MenuNode> ( "menu", *menu );

  connect ( node->selectEvent, this, & Self::exeCommand_ );

  node->show ( rtx, world_ );
}


//-----------------------------------------------------------------------
//   showOptions_
//-----------------------------------------------------------------------


void Controller::showOptions_ ( RenderContext& rtx )
{
  using jem::util::connect;

  Ref<Menu>  menu = newInstance<Menu> ();

  evHandler_->listOptions ( *menu );

  menu->addOption ( Commands::SHOW_NAMES,
                    Commands::HIDE_NAMES, showNames_ );

  Ref<MenuNode>  node = newInstance<MenuNode> ( "options", *menu );

  connect    ( node->selectEvent, this, & Self::exeCommand_ );
  node->show ( rtx, world_ );
}


//-----------------------------------------------------------------------
//   showNodes_
//-----------------------------------------------------------------------


void Controller::showNodes_ ( RenderContext& rtx )
{
  using jem::util::connect;

  const idx_t  worldSize = world_->size      ();
  Ref<Menu>    menu      = newInstance<Menu> ();

  menu->addEntry ( "World" );

  if ( worldSize > 0 )
  {
    menu->addLine  ();

    for ( idx_t i = 0; i < worldSize; i++ )
    {
      Ref<GfxNode>  node = world_->getChild ( i );

      if ( isTransformable( node ) )
      {
        menu->addEntry ( node->getName() );
      }
    }
  }

  Ref<MenuNode>  node = newInstance<MenuNode> ( "menu", *menu );

  connect    ( node->selectEvent, this, & Self::selectNode_ );
  node->show ( rtx, world_ );
}


//-----------------------------------------------------------------------
//   nodeErased_
//-----------------------------------------------------------------------


void Controller::nodeErased_ ( const Ref<GfxNode>& node )
{
  if ( node == focus_ )
  {
    focus_ = world_;

    evHandler_->setFocus ( focus_ );
  }
}


//-----------------------------------------------------------------------
//   selectNode_
//-----------------------------------------------------------------------


void Controller::selectNode_

  ( RenderContext&  rtx,
    const String&   name )

{
  Ref<GfxNode>  oldFocus = focus_;

  if ( name == "World" )
  {
    focus_ = world_;
  }
  else
  {
    const idx_t  worldSize = world_->size ();

    for ( idx_t i = 0; i < worldSize; i++ )
    {
      Ref<GfxNode>  node = world_->getChild ( i );

      if ( isTransformable( node ) &&
           node->getName() == name )
      {
        focus_ = node;

        break;
      }
    }
  }

  if ( focus_ != oldFocus )
  {
    evHandler_->setFocus ( focus_ );
    rtx.postFastRedraw   ( world_ );
  }
}


//-----------------------------------------------------------------------
//   selectNext_
//-----------------------------------------------------------------------


void Controller::selectNext_ ()
{
  const idx_t   worldSize = world_->size ();

  Ref<GfxNode>  oldFocus  = focus_;

  idx_t  i = world_->findChild ( focus_ );

  if ( i >= 0 )
  {
    i++;
  }
  else
  {
    i = 0;
  }

  for ( ; i < worldSize; i++ )
  {
    Ref<GfxNode>  node = world_->getChild ( i );

    if ( isTransformable( node ) )
    {
      focus_ = node;

      break;
    }
  }

  if ( i >= worldSize )
  {
    focus_ = world_;
  }

  if ( focus_ != oldFocus )
  {
    evHandler_->setFocus ( focus_ );
  }
}


//-----------------------------------------------------------------------
//   selectPrev_
//-----------------------------------------------------------------------


void Controller::selectPrev_ ()
{
  const idx_t   worldSize = world_->size ();

  Ref<GfxNode>  oldFocus  = focus_;

  idx_t  i = world_->findChild ( focus_ );

  if ( i >= 0 )
  {
    i--;
  }
  else
  {
    i = worldSize - 1;
  }

  for ( ; i >= 0; i-- )
  {
    Ref<GfxNode>  node = world_->getChild ( i );

    if ( isTransformable( node ) )
    {
      focus_ = node;

      break;
    }
  }

  if ( i < 0 )
  {
    focus_ = world_;
  }

  if ( focus_ != oldFocus )
  {
    evHandler_->setFocus ( focus_ );
  }
}


//-----------------------------------------------------------------------
//   makeSnapshot_
//-----------------------------------------------------------------------


void Controller::makeSnapshot_ ( RenderContext& rtx )
{
  if ( ! grabber_ )
  {
    return;
  }

  GfxNode*  root = findRedrawRoot ( world_ );

  if ( ! root )
  {
    return;
  }

  Ref<OutputStream>  output;
  Properties         params;
  String             fname;
  String             text;

  try
  {
    initSnapshot_ ( fname, output );

    params.set ( GLActionParams::FRAME_GRABBER, grabber_ );
    params.set ( GLActionParams::FRAME_OUTPUT,  output   );

    root->takeGLAction ( rtx, (int)GLActions::GRAB_FRAME, params );

    text = String::format ( "Saved snapshot to\n`%s\'", fname );

    if ( snapCount_ == maxOf( snapCount_ ) )
    {
      snapCount_ = 1;
    }
    else
    {
      snapCount_++;
    }
  }
  catch ( const io::IOException& ex )
  {
    text = "Snapshot failed:\n" + ex.what ();
  }

  showMessageUntil ( rtx, world_, text,
                     Time::now() + 4_sec );
}


//-----------------------------------------------------------------------
//   initSnapshot_
//-----------------------------------------------------------------------


void Controller::initSnapshot_

  ( String&             fname,
    Ref<OutputStream>&  output ) const

{
  using jem::io::FileOutputStream;
  using jem::io::BufferedOutputStream;

  fname  = StringUtils::expand ( snapFile_, snapCount_ );

  output = newInstance<BufferedOutputStream> (
    newInstance<FileOutputStream> ( fname )
  );
}


//-----------------------------------------------------------------------
//   saveConfig0_
//-----------------------------------------------------------------------


void Controller::saveConfig0_ ( RenderContext& rtx )
{
  GfxNode*  root = findRedrawRoot ( world_ );

  String    text = "Saved current configuration;\n"
                   "hit Ctrl+r to restore";

  if ( ! root )
  {
    root = evNode_.get ();
  }

  root->getConfig  ( config0_ );
  showMessageUntil ( rtx, world_, text,
                     Time::now() + 4_sec );
}


JEM_END_PACKAGE( gl )
