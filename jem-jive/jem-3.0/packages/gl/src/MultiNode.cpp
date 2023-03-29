
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
#include <jem/gl/error.h>
#include <jem/gl/Box.h>
#include <jem/gl/Actions.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/MultiNode.h>


JEM_DEFINE_CLASS( jem::gl::MultiNode );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class MultiNode::PosSaver_
//=======================================================================


class MultiNode::PosSaver_
{
 public:

  explicit inline         PosSaver_

    ( MultiNode*            mnode );

  inline                 ~PosSaver_ ();


 private:

  MultiNode*              mnode_;
  idx_t                    pos_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline MultiNode::PosSaver_::PosSaver_ ( MultiNode* mnode )
{
  mnode_ = mnode;
  pos_   = mnode->pos_;
}


inline MultiNode::PosSaver_::~PosSaver_ ()
{
  mnode_->pos_ = pos_;
}


//=======================================================================
//   class MultiNode
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


MultiNode::MultiNode ( const String& name ) :

  Super ( name )

{
  pos_ = -1;
}


MultiNode::~MultiNode ()
{
  for ( idx_t i = 0; i < children_.size(); i++ )
  {
    children_.getAs<GfxNode> ( i ) -> clearParent ();
  }
}


//-----------------------------------------------------------------------
//   findNode
//-----------------------------------------------------------------------


GfxNode* MultiNode::findNode ( const String& name ) const
{
  GfxNode*  node = 0;

  if ( myName_ == name )
  {
    node = const_cast<Self*> ( this );
  }
  else
  {
    for ( idx_t i = 0; i < children_.size() && ! node; i++ )
    {
      node = children_.getAs<GfxNode>(i)->findNode ( name );
    }
  }

  return node;
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool MultiNode::takeAction

  ( const String&      action,
    const Properties&  params )

{
  PosSaver_  psaver ( this );

  bool       r = false;


  // This loop is a bit complicated because children may
  // be added and deleted while executing the loop.

  pos_ = 0;

  while ( pos_ < children_.size() )
  {
    // Hold a reference to the child so that it is not deleted
    // prematurely when it removes itself from this node.

    Ref<GfxNode>  child = children_.getAs<GfxNode> ( pos_++ );

    if ( child->takeAction( action, params ) )
    {
      r = true;
    }
  }

  return r;
}


//-----------------------------------------------------------------------
//   takeGLAction
//-----------------------------------------------------------------------


bool MultiNode::takeGLAction

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params )

{
  PosSaver_  psaver ( this );

  bool       r = false;


  // This loop is a bit complicated because children may
  // be added and deleted while executing the loop.

  pos_ = 0;

  while ( pos_ < children_.size() )
  {
    // Hold a reference to the child so that it is not deleted
    // prematurely when it removes itself from this node.

    Ref<GfxNode>  child = children_.getAs<GfxNode> ( pos_++ );

    if ( child->takeGLAction( rtx, action, params ) )
    {
      r = true;
    }
  }

  return r;
}


//-----------------------------------------------------------------------
//   getBBox
//-----------------------------------------------------------------------


void MultiNode::getBBox ( Box& box ) const
{
  Box  childBox;


  box.makeEmpty ();

  for ( idx_t i = 0; i < children_.size(); i++ )
  {
    Ref<GfxNode>  child = children_.getAs<GfxNode> ( i );

    childBox.makeEmpty ();

    child->getBBox  ( childBox );
    box.enlargeWith ( childBox );
  }
}


//-----------------------------------------------------------------------
//   getVisibility
//-----------------------------------------------------------------------


void MultiNode::getVisibility

  ( bool&             viz,
    Box&              vbox,
    const Transform&  proj ) const

{
  Box   childBox;
  bool  childViz;


  viz = false;
  vbox.makeEmpty ();

  for ( idx_t i = 0; i < children_.size(); i++ )
  {
    Ref<GfxNode>  child = children_.getAs<GfxNode> ( i );

    childBox.makeEmpty ();

    child->getVisibility ( childViz, childBox, proj );

    viz = viz || childViz;

    vbox.enlargeWith ( childBox );
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void MultiNode::configure ( const Properties& props )
{
  PosSaver_  psaver ( this );

  pos_ = 0;

  while ( pos_ < children_.size() )
  {
    Ref<GfxNode>  child = children_.getAs<GfxNode> ( pos_++ );

    child->configure ( props );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void MultiNode::getConfig ( const Properties& props ) const
{
  for ( idx_t i = 0; i < children_.size(); i++ )
  {
    Ref<GfxNode>  child = children_.getAs<GfxNode> ( i );

    child->getConfig ( props );
  }
}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


void MultiNode::listProps ( const Properties& props ) const
{
  for ( idx_t i = 0; i < children_.size(); i++ )
  {
    Ref<GfxNode>  child = children_.getAs<GfxNode> ( i );

    child->listProps ( props );
  }
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void MultiNode::clear ()
{
  while ( children_.size() > 0 )
  {
    Ref<GfxNode>  child = static_cast<GfxNode*> ( children_.back() );

    child   ->clearParent ();
    children_.popBack     ();
    childErasedEvent.emit ( child, *this );
  }
}


//-----------------------------------------------------------------------
//   childCount
//-----------------------------------------------------------------------


idx_t MultiNode::childCount () const
{
  return children_.size ();
}


//-----------------------------------------------------------------------
//   addChild
//-----------------------------------------------------------------------


void MultiNode::addChild ( const Ref<GfxNode>& child )
{
  JEM_PRECHECK ( child );

  children_.pushBack  ( child.get() );
  child   ->setParent ( this  );
  newChildEvent.emit  ( child, *this );
}


//-----------------------------------------------------------------------
//   eraseChild
//-----------------------------------------------------------------------


bool MultiNode::eraseChild ( const Ref<GfxNode>& child )
{
  // Typically child nodes are removed from the end of the list

  for ( idx_t i = children_.size() - 1; i >= 0; i-- )
  {
    if ( children_.getAs<GfxNode>(i) == child.get() )
    {
      children_.erase    ( i );
      child->clearParent ();

      if ( pos_ > i )
      {
        pos_--;
      }

      childErasedEvent.emit ( child, *this );

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   eraseChild (by index)
//-----------------------------------------------------------------------


void MultiNode::eraseChild ( idx_t idx )
{
  if ( idx < 0 || idx >= children_.size() )
  {
    indexError ( JEM_FUNC, "child node", idx, children_.size() );
  }

  Ref<GfxNode>  child = children_.getAs<GfxNode> ( idx );

  child->clearParent ();
  children_.erase    ( idx );

  if ( pos_ > idx )
  {
    pos_--;
  }

  childErasedEvent.emit ( child, *this );
}


//-----------------------------------------------------------------------
//   getChild
//-----------------------------------------------------------------------


GfxNode* MultiNode::getChild ( idx_t idx ) const
{
  if ( idx < 0 || idx >= children_.size() )
  {
    indexError ( JEM_FUNC, "child node", idx, children_.size() );
  }

  return children_.getAs<GfxNode> ( idx );
}


//-----------------------------------------------------------------------
//   findChild
//-----------------------------------------------------------------------


idx_t MultiNode::findChild ( const Ref<GfxNode>& child ) const noexcept
{
  const idx_t  n = children_.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( children_[i] == child )
    {
      return i;
    }
  }

  return -1;
}


JEM_END_PACKAGE( gl )
