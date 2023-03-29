
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
#include <jem/base/ClassTemplate.h>
#include <jem/base/ArithmeticException.h>
#include <jem/io/list.h>
#include <jem/gl/input.h>
#include <jem/gl/utilities.h>
#include <jem/gl/Box.h>
#include <jem/gl/Rect.h>
#include <jem/gl/Viewer.h>
#include <jem/gl/Camera.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/Transformer.h>
#include <jem/gl/Transformable.h>
#include <jem/gl/EventHandler2D.h>


JEM_DEFINE_CLASS( jem::gl::EventHandler2D );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class EventHandler2D
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


EventHandler2D::EventHandler2D

  ( const Ref<Camera>&   camera,
    const Ref<GfxNode>&  world ) :

    Super ( camera, world )

{}


EventHandler2D::~EventHandler2D ()
{}


//-----------------------------------------------------------------------
//   redraw
//-----------------------------------------------------------------------


void EventHandler2D::redraw

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params )

{
  Super::redraw ( rtx, action, params );

  if ( focus_ == noFocus_ )
  {
    return;
  }

  if ( transformer_ != nullptr && transformer_->isActive() )
  {
    transformer_->redraw ( *camera_, rtx );
  }
}


//-----------------------------------------------------------------------
//   keyPressed
//-----------------------------------------------------------------------


void EventHandler2D::keyPressed

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

      if ( key.mods == (key.mods & Keys::SHIFT) )
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
//   mouseClicked
//-----------------------------------------------------------------------


void EventHandler2D::mouseClicked

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


void EventHandler2D::mouseDragged

  ( RenderContext&    rtx,
    const MouseDrag&  mouse )

{
  if ( (mode_        == ROTATION_MODE ||
        mode_        == TRANSLATION_MODE)  &&
       (mouse.button == Buttons::LEFT)     &&
       (mouse.mods   == (mouse.mods & Keys::SHIFT)) )
  {
    if ( ! transformer_ )
    {
      transformer_ = newInstance<Transformer> ();
    }

    if ( ! transformer_->isActive() )
    {
      int  trMode = (mode_ == ROTATION_MODE) ?

        Transformer::ROT_Z : Transformer::TRANS_XY;

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


void EventHandler2D::printHelp ( PrintWriter& pr )
{
  using jem::io::endItem;
  using jem::io::beginItem;

  if ( mode_ == ROTATION_MODE )
  {
    print ( pr, beginItem( "Left mouse drag" ) );
    print ( pr,   "Rotate the object around the z-axis. " );
    print ( pr,   "Hold the Shift key to limit the rotation to " );
    print ( pr,   "specific angles." );
    print ( pr, endItem );

    print ( pr, beginItem( "Arrow keys" ) );
    print ( pr,   "Rotate the object around the z-axis. " );
    print ( pr,   "Hold the Shift key for fine-grained control." );
    print ( pr, endItem );
  }

  if ( mode_ == TRANSLATION_MODE )
  {
    print ( pr, beginItem( "Left mouse drag" ) );
    print ( pr,   "Translate the object in the x-y plane. " );
    print ( pr,   "Hold the Shift key to limit the translation " );
    print ( pr,   "to specific directions." );
    print ( pr, endItem );

    print ( pr, beginItem( "Arrow keys" ) );
    print ( pr,   "Translate the object in the x-y plane. " );
    print ( pr,   "Hold the Shift key for fine-grained control." );
    print ( pr, endItem );
  }

  Super::printHelp ( pr );
}


//-----------------------------------------------------------------------
//   setFocus
//-----------------------------------------------------------------------


void EventHandler2D::setFocus ( const Ref<GfxNode>& node )
{
  transformer_ = nullptr;

  Super::setFocus ( node );
}


//-----------------------------------------------------------------------
//   rotate_
//-----------------------------------------------------------------------


void EventHandler2D::rotate_ ( const KeyPress& key )
{
  Transformable*  tf = toTransformable ( focus_ );

  if ( ! tf )
  {
    return;
  }

  Transform   t, u;
  Box         box;
  Vertex3d    c;
  GLdouble    a;


  tf->getBareBBox ( box );

  if ( box.isEmpty() )
  {
    return;
  }

  t = tf->getTransform ();

  if ( key.mods & Keys::SHIFT )
  {
    a = 1.0;
  }
  else
  {
    a = 5.0;
  }

  if ( key.code == Keys::LEFT || key.code == Keys::DOWN )
  {
    a = -a;
  }

  u = getTransformOf ( focus_->getParent() );

  t.transform ( u );

  c = matmul  ( t, box.center() );

  t.translate ( -c[0], -c[1] );
  t.rotate    (  a );
  t.translate (  c[0],  c[1] );

  if ( ! invert( u ) )
  {
    return;
  }

  t.transform      ( u );
  tf->setTransform ( t );
}


//-----------------------------------------------------------------------
//   translate_
//-----------------------------------------------------------------------


void EventHandler2D::translate_ ( const KeyPress& key )
{
  GLdouble  dx = 0.0;
  GLdouble  dy = 0.0;
  GLdouble  s  = 0.1;


  switch ( key.code )
  {
  case Keys::RIGHT: dx =  0.1; break;
  case Keys::LEFT:  dx = -0.1; break;
  case Keys::UP:    dy =  0.1; break;
  case Keys::DOWN:  dy = -0.1; break;

  default: return;
  }

  if ( key.mods & Keys::SHIFT )
  {
    s *= 10.0;
  }

  dx *= s;
  dy *= s;

  camera_->translateObj ( *focus_, dx, dy, 0.0 );
}


JEM_END_PACKAGE( gl )
