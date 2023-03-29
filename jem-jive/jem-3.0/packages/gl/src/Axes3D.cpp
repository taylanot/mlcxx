
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


#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jem/gl/render.h>
#include <jem/gl/Actions.h>
#include <jem/gl/Alignment.h>
#include <jem/gl/Box.h>
#include <jem/gl/Color.h>
#include <jem/gl/Commands.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/Menu.h>
#include <jem/gl/PropertyNames.h>
#include <jem/gl/Rect.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/TextPainter.h>
#include <jem/gl/Transform.h>
#include <jem/gl/Transformable.h>
#include <jem/gl/Axes3D.h>


JEM_DEFINE_CLASS( jem::gl::Axes3D );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class Axes3D
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Axes3D::Axes3D ( const String& name ) : Super ( name )
{
  hidden     = false;
  hAlignment = RIGHT_ALIGNED;
  vAlignment = BOT_ALIGNED;
}


Axes3D::~Axes3D ()
{}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool Axes3D::takeAction

  ( const String&      action,
    const Properties&  params )

{
  if ( ! buddy_ )
  {
    return false;
  }

  if ( action == Actions::HIDE )
  {
    hidden = true;
    return   true;
  }

  if ( action == Actions::SHOW )
  {
    hidden = false;
    return   true;
  }

  if ( action == Actions::LIST_OPTIONS )
  {
    Ref<Menu>  menu;

    params.get ( menu, ActionParams::OPTIONS );

    listOptions ( *menu );

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   takeGLAction
//-----------------------------------------------------------------------


bool Axes3D::takeGLAction

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params )

{
  if ( ! buddy_ )
  {
    return false;
  }

  if ( ! hidden )
  {
    if ( action == (int)GLActions::REDRAW ||
         action == (int)GLActions::REDRAW_FAST )
    {
      redraw ( rtx, action, params );

      return true;
    }
  }

  if ( action == (int)GLActions::EXE_COMMAND )
  {
    String  cmd;

    params.get ( cmd, GLActionParams::COMMAND );

    return exeCommand ( rtx, cmd );
  }

  return false;
}


//-----------------------------------------------------------------------
//   getBBox
//-----------------------------------------------------------------------


void Axes3D::getBBox ( Box& box ) const
{
  box.makeEmpty ();
}


//-----------------------------------------------------------------------
//   getVisibility
//-----------------------------------------------------------------------


void Axes3D::getVisibility

  ( bool&             viz,
    Box&              vbox,
    const Transform&  proj ) const

{
  viz = false;

  vbox.makeEmpty ();
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void Axes3D::configure ( const Properties& props )
{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    myProps.find ( hidden, PropNames::HIDDEN );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void Axes3D::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set ( PropNames::HIDDEN, hidden );
}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


void Axes3D::listProps ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set ( PropNames::HIDDEN,
                "The non-visibility state" );
}


//-----------------------------------------------------------------------
//   redraw
//-----------------------------------------------------------------------


void Axes3D::redraw

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params ) const

{
  if ( buddy_ )
  {
    GLContext&  gtx   = rtx.getGLContext ();
    Rect        vport = gtx.getViewport  ();

    double      lod   = 1.0;
    double      dx    = 0.0;
    double      dy    = 0.0;

    if ( action == (int) GLActions::REDRAW )
    {
      params.find ( dx, GLActionParams::XJITTER );
      params.find ( dy, GLActionParams::YJITTER );
    }
    else
    {
      params.find ( lod, GLActionParams::DETAIL_LEVEL );
    }

    if ( lod > 0.02 && vport.width > 99 && vport.height > 99 )
    {
      redraw_ ( gtx, dx, dy, vport );
    }
  }
}


//-----------------------------------------------------------------------
//   exeCommand
//-----------------------------------------------------------------------


bool Axes3D::exeCommand

  ( RenderContext&  rtx,
    const String&   cmd )

{

  if ( cmd == Commands::SHOW_AXES )
  {
    hidden = false;
    return   true;
  }

  if ( cmd == Commands::HIDE_AXES )
  {
    hidden = true;
    return   true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   listOptions
//-----------------------------------------------------------------------


void Axes3D::listOptions ( Menu& menu ) const
{
  menu.addOption ( Commands::SHOW_AXES,
                   Commands::HIDE_AXES, ! hidden );
}


//-----------------------------------------------------------------------
//   setBuddy
//-----------------------------------------------------------------------


void Axes3D::setBuddy ( const Ref<GfxNode>& buddy )
{
  buddy_ = buddy;
}


//-----------------------------------------------------------------------
//   redraw_
//-----------------------------------------------------------------------


void Axes3D::redraw_

  ( GLContext&   gtx,
    double       dx,
    double       dy,
    const Rect&  vport ) const

{
  const char   LABELS[3] = { 'x', 'y', 'z' };

  TextPainter  txp     ( Font(), CENTERED, CENTERED );
  AttribScope  attribs ( gtx, GLContext::COLOR    |
                              GLContext::LIGHTING |
                              GLContext::LINE_WIDTH );

  Transform    tr      = getTransformOf ( buddy_ );

  GLdouble     scale;
  GLsizei      vsize;
  GLint        x, y;


  // Set the viewport.

  vsize = jem::min ( vport.width, vport.height );
  vsize = (GLsizei) (0.2 * vsize);

  if ( vsize > 128 )
  {
    vsize = 128;
  }

  if ( vsize < 64 )
  {
    txp.font--;
  }

  x  = vport.x + 2;
  x += (GLint) (((vport.width  - vsize - 4) * hAlignment) / 2);
  y  = vport.y + 2;
  y += (GLint) (((vport.height - vsize - 4) * vAlignment) / 2);

  gtx.setViewport ( Rect( x, y, vsize, vsize ) );

  // Modify the transformation matrix. First, adjust the scaling.

  scale = tr.getScale ();

  if ( ! isTiny( scale ) )
  {
    scale = 1.0 / scale;
  }

  tr.scale ( 0.5 * vsize / scale );

  // Set the translation.

  tr(0,3) = 0.5 * vsize + dx;
  tr(1,3) = 0.5 * vsize + dy;
  tr(2,3) = 0.5 * vsize;

  // Set the projection and model view matrix.

  gtx.pushTransform ( GLContext::PROJECTION );
  gtx.setIdentity   ( GLContext::PROJECTION );
  gtx.setOrtho      ( 0.0, vsize, 0.0, vsize, -vsize, 20 * vsize,
                      GLContext::PROJECTION );
  gtx.pushTransform ();
  gtx.setTransform  ( tr.addr() );

  // Render the axes.

  gtx.setLineWidth  ( 1.0F );
  gtx.scale         (   0.7, 0.7, 0.7 );
  gtx.rotate        (  90.0, 0.0, 1.0, 0.0 );
  renderArrow3D     ( gtx, Color::red() );

  gtx.rotate        ( -90.0, 1.0, 0.0, 0.0 );
  renderArrow3D     ( gtx, Color::green() );

  gtx.rotate        ( -90.0, 0.0, 1.0, 0.0 );
  renderArrow3D     ( gtx, Color::blue() );

  // Remove the jitter offsets and render the labels.

  tr(0,3) = 0.5 * vsize;
  tr(1,3) = 0.5 * vsize;

  // Draw in orthogonal mode to make sure that the text is
  // pixel-aligned.

  gtx.setIdentity   ();
  gtx.setColor      ( Color::black() );

  {
    TextScope  textScope  ( gtx, txp );

    Vertex3d   v, w;

    gtx.setLighting ( false );

    v = 0.0;

    for ( int i = 0; i < 3; i++ )
    {
      v[i] = 0.85;

      matmul        ( w, tr, v );
      txp.setCursor ( (GLint) w[0], (GLint) w[1], (GLfloat) w[2] );
      txp.write     ( gtx, LABELS[i] );

      v[i] = 0.0;
    }
  }

  // Restore the viewport and all other previous settings.

  gtx.setViewport   ( vport );
  gtx.popTransform  ( GLContext::PROJECTION );
  gtx.popTransform  ();

  checkGLError      ( gtx, JEM_FUNC );
}


JEM_END_PACKAGE( gl )
