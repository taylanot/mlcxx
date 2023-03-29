
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
#include <jem/gl/Box.h>
#include <jem/gl/WrapperNode.h>


JEM_DEFINE_CLASS( jem::gl::WrapperNode );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class WrapperNode
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


WrapperNode::WrapperNode

  ( const String&        name,
    const Ref<GfxNode>&  child ) :

    Super  ( name  ),
    child_ ( child )

{
  if ( child_ )
  {
    child_->setParent ( this );
  }
}


WrapperNode::~WrapperNode ()
{
  if ( child_ )
  {
    child_->clearParent ();
  }
}


//-----------------------------------------------------------------------
//   findNode
//-----------------------------------------------------------------------


GfxNode* WrapperNode::findNode ( const String& name ) const
{
  if      ( name == myName_ )
  {
    return const_cast<Self*> ( this );
  }
  else if ( child_ )
  {
    return child_->findNode ( name );
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool WrapperNode::takeAction

  ( const String&      action,
    const Properties&  params )

{
  return passAction ( child_, action, params );
}


//-----------------------------------------------------------------------
//   takeGLAction
//-----------------------------------------------------------------------


bool WrapperNode::takeGLAction

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params )

{
  return passGLAction ( child_, rtx, action, params );
}


//-----------------------------------------------------------------------
//   getBBox
//-----------------------------------------------------------------------


void WrapperNode::getBBox ( Box& box ) const
{
  if ( ! child_ )
  {
    box.makeEmpty ();
  }
  else
  {
    child_->getBBox ( box );
  }
}


//-----------------------------------------------------------------------
//   getVisibility
//-----------------------------------------------------------------------


void WrapperNode::getVisibility

  ( bool&             viz,
    Box&              vbox,
    const Transform&  proj ) const

{
  if ( ! child_ )
  {
    viz = false;
    vbox.makeEmpty ();
  }
  else
  {
    child_->getVisibility ( viz, vbox, proj );
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void WrapperNode::configure ( const Properties& props )
{
  configChild ( child_, props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void WrapperNode::getConfig ( const Properties& props ) const
{
  if ( child_ )
  {
    child_->getConfig ( props );
  }
}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


void WrapperNode::listProps ( const Properties& props ) const
{
  if ( child_ )
  {
    child_->listProps ( props );
  }
}


//-----------------------------------------------------------------------
//   setChild_
//-----------------------------------------------------------------------


void WrapperNode::setChild_ ( const Ref<GfxNode>& node )
{
  if ( node != child_ )
  {
    if ( child_ )
    {
      child_->clearParent ();
    }

    child_ = node;

    if ( child_ )
    {
      child_->setParent ( this );
    }
  }
}


JEM_END_PACKAGE( gl )
