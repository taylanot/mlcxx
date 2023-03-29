
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


#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/ArithmeticException.h>
#include <jem/util/Properties.h>
#include <jem/io/list.h>
#include <jem/gl/input.h>
#include <jem/gl/Box.h>
#include <jem/gl/Viewer.h>
#include <jem/gl/Camera.h>
#include <jem/gl/Axes3D.h>
#include <jem/gl/Actions.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/Transformer.h>
#include <jem/gl/Transformable.h>
#include <jem/gl/EventHandler3D.h>


JEM_DEFINE_CLASS( jem::gl::EventHandler3D );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class EventHandler3D
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


EventHandler3D::EventHandler3D

  ( const Ref<Camera>&   camera,
    const Ref<GfxNode>&  world ) :

    Super ( camera, world )

{
  axes_ = newInstance<Axes3D> ();

  axes_->hAlignment = RIGHT_ALIGNED;
  axes_->vAlignment = TOP_ALIGNED;
}


EventHandler3D::~EventHandler3D ()
{}


//-----------------------------------------------------------------------
//   redraw
//-----------------------------------------------------------------------


void EventHandler3D::redraw

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params )

{
  Super::redraw ( rtx, action, params );

  if ( focus_ == noFocus_ )
  {
    return;
  }

  if ( ! axes_->hidden )
  {
    axes_->redraw ( rtx, action, params );
  }

  if ( transformer_ != nullptr && transformer_->isActive() )
  {
    transformer_->redraw ( *camera_, rtx );
  }
}


//-----------------------------------------------------------------------
//   keyPressed
//-----------------------------------------------------------------------


void EventHandler3D::keyPressed

  ( RenderContext&   rtx,
    const KeyPress&  key )

{
  try
  {
    bool  handled = false;

    switch ( key.code )
    {
    case Keys::RIGHT:
    case Keys::LEFT:
    case Keys::UP:
    case Keys::DOWN:
    case Keys::PAGE_UP:
    case Keys::PAGE_DOWN:

      if ( (key.mods & (Keys::SHIFT | Keys::ALT)) == key.mods )
      {
        if      ( mode_ == ROTATION_MODE )
        {
          rotate_ ( key );

          handled = true;
        }
        else if ( mode_ == TRANSLATION_MODE )
        {
          translate_ ( key );

          handled = true;
        }
      }

      break;

    default:

      handled = false;
    }

    if ( handled )
    {
      rtx.postFastRedraw ( world_ );
    }
    else
    {
      Super::keyPressed  ( rtx, key );
    }
  }
  catch ( const ArithmeticException& )
  {}
}


//-----------------------------------------------------------------------
//   exeCommand
//-----------------------------------------------------------------------


bool EventHandler3D::exeCommand

  ( RenderContext&  rtx,
    const String&   cmd )

{
  bool  result = axes_->exeCommand ( rtx, cmd );

  if ( ! result )
  {
    result = Super::exeCommand ( rtx, cmd );
  }

  return result;
}


//-----------------------------------------------------------------------
//   mouseClicked
//-----------------------------------------------------------------------


void EventHandler3D::mouseClicked

  ( RenderContext&     rtx,
    const MouseClick&  mouse )

{
  if ( transformer_ != nullptr && transformer_->isActive() )
  {
    transformer_->reset ();
    rtx.postFastRedraw  ( world_ );
  }

  Super::mouseClicked ( rtx, mouse );
}


//-----------------------------------------------------------------------
//   mouseDragged
//-----------------------------------------------------------------------


void EventHandler3D::mouseDragged

  ( RenderContext&    rtx,
    const MouseDrag&  mouse )

{
  if ( (mode_        == ROTATION_MODE ||
        mode_        == TRANSLATION_MODE)  &&
       (mouse.button == Buttons::LEFT ||
        mouse.button == Buttons::MIDDLE)   &&
       (mouse.mods   == (mouse.mods & (Keys::ALT | Keys::SHIFT))) )
  {
    if ( ! transformer_ )
    {
      transformer_ = newInstance<Transformer> ();
    }

    if ( ! transformer_->isActive() )
    {
      int  trMode;

      if ( mode_ == ROTATION_MODE )
      {
        if ( (mouse.button == Buttons::MIDDLE) ||
             (mouse.mods & Keys::ALT) )
        {
          trMode = Transformer::ROT_Z;
        }
        else
        {
          trMode = Transformer::ROT_XY;
        }
      }
      else
      {
        if ( (mouse.button == Buttons::MIDDLE) ||
             (mouse.mods & Keys::ALT) )
        {
          trMode = Transformer::TRANS_Z;
        }
        else
        {
          trMode = Transformer::TRANS_XY;
        }
      }

      if ( mouse.mods & Keys::SHIFT )
      {
        trMode |= Transformer::SNAPPY;
      }

      try
      {
        transformer_->init ( trMode, *focus_, *camera_, mouse );
      }
      catch ( const ArithmeticException& )
      {
        transformer_->reset ();
        return;
      }
    }

    try
    {
      transformer_->transform ( *focus_, *camera_, mouse );
    }
    catch ( const ArithmeticException& )
    {}

    rtx.postFastRedraw ( world_ );
  }
  else
  {
    Super::mouseDragged ( rtx, mouse );
  }
}


//-----------------------------------------------------------------------
//   printHelp
//-----------------------------------------------------------------------


void EventHandler3D::printHelp ( PrintWriter& pr )
{
  using jem::io::endItem;
  using jem::io::beginItem;

  if ( mode_ == ROTATION_MODE )
  {
    print ( pr, beginItem( "Left mouse drag" ) );
    print ( pr,   "Rotate the object around the x- and y-axis. " );
    print ( pr,   "Hold the Alt key to rotate the object around " );
    print ( pr,   "the z-axis. Hold the Shift key to limit the " );
    print ( pr,   "rotation to specific angles." );
    print ( pr, endItem );

    print ( pr, beginItem( "Middle mouse drag" ) );
    print ( pr,   "Rotate the object around the z-axis. " );
    print ( pr,   "Hold the Shift key to limit the rotation to " );
    print ( pr,   "specific angles." );
    print ( pr, endItem );

    print ( pr, beginItem( "Arrow keys" ) );
    print ( pr,   "Rotate the object around the x- and y-axis. " );
    print ( pr,   "Hold the Alt key to rotate the object within " );
    print ( pr,   "its own coordinate system. Hold the Shift key " );
    print ( pr,   "for fine-grained control." );
    print ( pr, endItem );

    print ( pr, beginItem( "Page Up/Page Down" ) );
    print ( pr,   "Rotate the object around the z-axis. " );
    print ( pr,   "Hold the Alt key to rotate the object in its " );
    print ( pr,   "own coordinate system. Hold the Shift key for " );
    print ( pr,   "fine-grained control." );
    print ( pr, endItem );
  }

  if ( mode_ == TRANSLATION_MODE )
  {
    print ( pr, beginItem( "Left mouse drag" ) );
    print ( pr,   "Translate the object in the x-y plane. " );
    print ( pr,   "Hold the Alt key to translate the object " );
    print ( pr,   "along the z-axis. Hold the Shift key to limit " );
    print ( pr,   "the translation to specific directions." );
    print ( pr, endItem );

    print ( pr, beginItem( "Middle mouse drag" ) );
    print ( pr,   "Translate the object along the z-axis. " );
    print ( pr,   "Hold the Shift key to limit the translation " );
    print ( pr,   "to specific directions." );
    print ( pr, endItem );

    print ( pr, beginItem( "Arrow keys" ) );
    print ( pr,   "Translate the object in the x-y plane. " );
    print ( pr,   "Hold the Alt key to translate the object " );
    print ( pr,   "within its own coordinate system. Hold the " );
    print ( pr,   "Shift key for fine-grained control." );
    print ( pr, endItem );

    print ( pr, beginItem( "Page Up/Page Down" ) );
    print ( pr,   "Translate the object along the z-axis. " );
    print ( pr,   "Hold the Alt key to translate the object in " );
    print ( pr,   "its own coordinate system. Hold the Shift key " );
    print ( pr,   "for fine-grained control." );
    print ( pr, endItem );
  }

  Super::printHelp ( pr );
}


//-----------------------------------------------------------------------
//   setFocus
//-----------------------------------------------------------------------


void EventHandler3D::setFocus ( const Ref<GfxNode>& node )
{
  transformer_ = nullptr;

  Super::setFocus ( node );
  axes_->setBuddy ( focus_ );
}


//-----------------------------------------------------------------------
//   listOptions
//-----------------------------------------------------------------------


void EventHandler3D::listOptions ( Menu& menu ) const
{
  axes_->listOptions ( menu );
  Super::listOptions ( menu );
}


//-----------------------------------------------------------------------
//   rotate_
//-----------------------------------------------------------------------


void EventHandler3D::rotate_ ( const KeyPress& key )
{
  Transformable*  tf = toTransformable ( focus_ );

  if ( ! tf )
  {
    return;
  }

  Transform  t, u;
  Box        box;
  Vertex3d   c;

  double     x = 0.0;
  double     y = 0.0;
  double     z = 0.0;
  double     a = 5.0;


  switch ( key.code )
  {
  case Keys::RIGHT:      y =  1.0; break;
  case Keys::LEFT:       y = -1.0; break;
  case Keys::UP:         x = -1.0; break;
  case Keys::DOWN:       x =  1.0; break;
  case Keys::PAGE_UP:    z = -1.0; break;
  case Keys::PAGE_DOWN:  z =  1.0; break;

  default: return;
  }

  if ( key.mods & Keys::SHIFT )
  {
    a /= 5.0;
  }

  tf->getBareBBox ( box );

  if ( box.isEmpty() )
  {
    return;
  }

  t = tf->getTransform ();

  if ( key.mods & Keys::ALT )
  {
    c = box.center ();

    u.setIdentity ();
    u.translate   ( -c[0], -c[1], -c[2] );
    u.rotate      (  a, x, y, z );
    u.translate   (  c[0],  c[1],  c[2] );

    t = matmul ( t, u );
  }
  else
  {
    u = getTransformOf ( focus_->getParent() );

    t.transform ( u );

    c = matmul  ( t, box.center() );

    t.translate ( -c[0], -c[1], -c[2] );
    t.rotate    (  a, x, y, z );
    t.translate (  c[0],  c[1],  c[2] );

    if ( ! invert( u ) )
    {
      return;
    }

    t.transform ( u );
  }

  tf->setTransform   ( t );
  camera_->updateZRange ();
}


//-----------------------------------------------------------------------
//   translate_
//-----------------------------------------------------------------------


void EventHandler3D::translate_ ( const KeyPress& key )
{
  GLdouble  dx = 0.0;
  GLdouble  dy = 0.0;
  GLdouble  dz = 0.0;
  GLdouble  s  = 0.1;


  switch ( key.code )
  {
  case Keys::RIGHT:      dx =  1.0; break;
  case Keys::LEFT:       dx = -1.0; break;
  case Keys::UP:         dy =  1.0; break;
  case Keys::DOWN:       dy = -1.0; break;
  case Keys::PAGE_UP:    dz =  1.0; break;
  case Keys::PAGE_DOWN:  dz = -1.0; break;

  default: return;
  }

  if ( key.mods & Keys::SHIFT )
  {
    s /= 10.0;
  }

  dx *= s;
  dy *= s;
  dz *= s;

  if ( key.mods & Keys::ALT )
  {
    Transformable*  tf = toTransformable ( focus_ );

    if ( ! tf )
    {
      return;
    }

    Transform  t, u;
    Box        box;
    GLdouble   sz;

    tf->getBareBBox ( box );

    if ( box.isEmpty() )
    {
      return;
    }

    sz  = box.size ();

    dx *= sz;
    dy *= sz;
    dz *= sz;

    t   = tf->getTransform ();

    u.setIdentity ();
    u.translate   ( dx, dy, dz );

    tf->setTransform ( matmul( t, u ) );

    camera_->updateZRange ();
  }
  else
  {
    camera_->translateObj ( *focus_, dx, dy, dz );

    if ( ! isTiny( dz ) )
    {
      camera_->updateZRange ();
    }
  }
}


JEM_END_PACKAGE( gl )
