
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
#include <jem/base/RuntimeException.h>
#include <jem/gl/Box.h>
#include <jem/gl/Transform.h>
#include <jem/gl/GfxNode.h>


JEM_DEFINE_CLASS( jem::gl::GfxNode );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GfxNode
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GfxNode::GfxNode ( const String& name ) :

  Super ( name )

{
  parent_ = nullptr;
}


GfxNode::~GfxNode ()
{
  parent_ = nullptr;
}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String GfxNode::getContext () const
{
  return makeContext ( "scene-graph node", myName_ );
}


//-----------------------------------------------------------------------
//   findNode
//-----------------------------------------------------------------------


GfxNode* GfxNode::findNode ( const String& name ) const
{
  if ( name == myName_ )
  {
    return const_cast<Self*> ( this );
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool GfxNode::takeAction

  ( const String&      action,
    const Properties&  params )

{
  return false;
}


//-----------------------------------------------------------------------
//   getBBox
//-----------------------------------------------------------------------


void GfxNode::getBBox ( Box& box ) const
{
  box.makeEmpty ();
}


//-----------------------------------------------------------------------
//   getVisibility
//-----------------------------------------------------------------------


void GfxNode::getVisibility

  ( bool&             viz,
    Box&              vbox,
    const Transform&  proj ) const

{
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


void GfxNode::configure ( const Properties& props )
{}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void GfxNode::getConfig ( const Properties& props ) const
{}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


void GfxNode::listProps ( const Properties& props ) const
{}


//-----------------------------------------------------------------------
//   setParent
//-----------------------------------------------------------------------


void GfxNode::setParent ( GfxNode* p )
{
  if ( parent_ && p )
  {
    throw RuntimeException (
      getContext (),
      String::format (
        "attempt to change the parent of node `%s\' from "
        "`%s\' to `%s\'",
        myName_,
        parent_->myName_,
        p->myName_
      )
    );
  }

  parent_ = p;
}


JEM_END_PACKAGE( gl )
