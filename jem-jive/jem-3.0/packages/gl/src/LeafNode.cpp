
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


#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jem/gl/render.h>
#include <jem/gl/Box.h>
#include <jem/gl/Menu.h>
#include <jem/gl/Actions.h>
#include <jem/gl/Commands.h>
#include <jem/gl/PropertyNames.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/Transform.h>
#include <jem/gl/LeafNode.h>


JEM_DEFINE_CLASS( jem::gl::LeafNode );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class LeafNode
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


LeafNode::LeafNode ( const String& name ) : Super ( name )
{
  hidden = false;
}


LeafNode::~LeafNode ()
{}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool LeafNode::takeAction

  ( const String&      action,
    const Properties&  params )

{
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

    params.get  (  menu, ActionParams::OPTIONS );
    listOptions ( *menu );

    return true;
  }

  if ( action == Actions::SET_FONT )
  {
    return setFont ();
  }

  return false;
}


//-----------------------------------------------------------------------
//   takeGLAction
//-----------------------------------------------------------------------


bool LeafNode::takeGLAction

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params )

{
  if ( ! hidden )
  {
    if ( action == (int) GLActions::REDRAW_FAST )
    {
      double  lod = 1.0;

      params.find ( lod, GLActionParams::DETAIL_LEVEL );

      if      ( lod >= 1.0 )
      {
        redraw     ( rtx );
      }
      else if ( lod > 0.0 )
      {
        redrawFast ( rtx, lod );
      }

      return true;
    }

    if ( action == (int) GLActions::REDRAW )
    {
      redraw ( rtx );

      return true;
    }

    if ( action == (int) GLActions::DRAW_CDATA )
    {
      GLContext&   gtx   = rtx.getGLContext ();
      AttribScope  scope ( gtx, GLContext::LIGHTING );

      gtx.setLighting ( false );

      redraw ( rtx );

      return true;
    }
  }

  if ( action == (int) GLActions::EXE_COMMAND )
  {
    String  cmd;

    params.get ( cmd, GLActionParams::COMMAND );

    return exeCommand ( rtx, cmd, params );
  }

  if ( action == (int) GLActions::INIT_GL )
  {
    return initGL ( rtx );
  }

  return false;
}


//-----------------------------------------------------------------------
//   getVisibility
//-----------------------------------------------------------------------


void LeafNode::getVisibility

  ( bool&             viz,
    Box&              vbox,
    const Transform&  proj ) const

{
  if ( hidden )
  {
    viz = false;
    vbox.makeEmpty ();
    return;
  }

  getBBox ( vbox );

  if ( vbox.isEmpty() )
  {
    viz = true;
    return;
  }

  if ( isVisible( vbox, proj ) )
  {
    viz = true;
  }
  else
  {
    viz = false;
    vbox.makeEmpty ();
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void LeafNode::configure ( const Properties& props )
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


void LeafNode::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set ( PropNames::HIDDEN, hidden );
}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


void LeafNode::listProps ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set ( PropNames::HIDDEN,
                "The non-visibility state" );
}


//-----------------------------------------------------------------------
//   initGL
//-----------------------------------------------------------------------


bool LeafNode::initGL ( RenderContext& rtx )
{
  return false;
}


//-----------------------------------------------------------------------
//   exeCommand
//-----------------------------------------------------------------------


bool LeafNode::exeCommand

  ( RenderContext&     rtx,
    const String&      cmd,
    const Properties&  params )

{
  return false;
}


//-----------------------------------------------------------------------
//   listOptions
//-----------------------------------------------------------------------


void LeafNode::listOptions ( Menu& menu ) const
{}


//-----------------------------------------------------------------------
//   setFont
//-----------------------------------------------------------------------


bool LeafNode::setFont ()
{
  return false;
}


JEM_END_PACKAGE( gl )
