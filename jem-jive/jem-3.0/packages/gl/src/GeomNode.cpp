
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
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jem/gl/Box.h>
#include <jem/gl/Actions.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/GeomNode.h>


JEM_DEFINE_CLASS( jem::gl::GeomNode );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GeomNode
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GeomNode::GeomNode

  ( const String&        name,
    const Ref<GfxNode>&  child,
    Rank                 rank ) :

    Super    ( name ),
    rank_    ( rank ),
    child_   ( child )

{
  using jem::util::connect;

  JEM_PRECHECK ( child );

  child_ ->setParent   ( this );
  transfm_.setIdentity ();

  hidden = false;
}


GeomNode::~GeomNode ()
{
  child_->clearParent ();
}


//-----------------------------------------------------------------------
//   findNode
//-----------------------------------------------------------------------


GfxNode* GeomNode::findNode ( const String& name ) const
{
  if ( name == myName_ )
  {
    return const_cast<Self*> ( this );
  }
  else
  {
    return child_->findNode ( name );
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool GeomNode::takeAction

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

  return passAction ( child_, action, params );
}


//-----------------------------------------------------------------------
//   takeGLAction
//-----------------------------------------------------------------------


bool GeomNode::takeGLAction

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params )

{
  if ( hidden && (action == (int) GLActions::REDRAW ||
                  action == (int) GLActions::REDRAW_FAST) )
  {
    return false;
  }

  GLContext&  gtx = rtx.getGLContext ();

  bool        result;

  gtx.pushTransform ();
  gtx.transform     ( transfm_ );

  result = passGLAction ( child_, rtx, action, params );

  gtx.popTransform  ();

  return result;
}


//-----------------------------------------------------------------------
//   getBBox
//-----------------------------------------------------------------------


void GeomNode::getBBox ( Box& box ) const
{
  child_->getBBox ( box );

  if ( rank_ == RANK2D )
  {
    box.transform2D ( transfm_ );
  }
  else
  {
    box.transform   ( transfm_ );
  }
}


//-----------------------------------------------------------------------
//   getBareBBox
//-----------------------------------------------------------------------


void GeomNode::getBareBBox ( Box& box ) const
{
  child_->getBBox ( box );
}


//-----------------------------------------------------------------------
//   getVisibility
//-----------------------------------------------------------------------


void GeomNode::getVisibility

  ( bool&             viz,
    Box&              vbox,
    const Transform&  proj ) const

{
  if ( hidden )
  {
    viz = false;
    vbox.makeEmpty ();
  }
  else
  {
    Transform  t;

    matmul ( t, proj, transfm_ );

    child_->getVisibility ( viz, vbox, t );

    if ( rank_ == RANK2D )
    {
      vbox.transform2D ( transfm_ );
    }
    else
    {
      vbox.transform   ( transfm_ );
    }
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void GeomNode::configure ( const Properties& props )
{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    if ( rank_ == RANK2D )
    {
      transfm_.configure2D ( myProps );
    }
    else
    {
      transfm_.configure3D ( myProps );
    }
  }

  configChild ( child_, props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void GeomNode::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  if ( rank_ == RANK2D )
  {
    transfm_.getConfig2D ( myProps );
  }
  else
  {
    transfm_.getConfig3D ( myProps );
  }

  child_->getConfig ( props );
}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


void GeomNode::listProps ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  if ( rank_ == RANK2D )
  {
    Transform::listProps2D ( myProps );
  }
  else
  {
    Transform::listProps3D ( myProps );
  }

  child_->listProps ( props );
}


//-----------------------------------------------------------------------
//   getTransform
//-----------------------------------------------------------------------


Transform GeomNode::getTransform () const
{
  return transfm_;
}


//-----------------------------------------------------------------------
//   setTransform
//-----------------------------------------------------------------------


void GeomNode::setTransform ( const Transform& tr )
{
  transfm_ = tr;
}


JEM_END_PACKAGE( gl )
