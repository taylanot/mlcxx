
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
#include <jem/xml/DOMElement.h>
#include <jem/xml/DOMIterator.h>


JEM_BEGIN_PACKAGE( xml )


//=======================================================================
//   class DOMIterator
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


DOMIterator::DOMIterator () noexcept :

  maxDepth_ ( maxOf<int>() ),
  parent_   ( 0 ),
  elem_     ( 0 ),
  depth_    ( 0 ),
  dir_      ( 0 )

{}


DOMIterator::DOMIterator ( Ref<DOMElement> root ) :

  maxDepth_ ( maxOf<int>() )

{
  start ( root );
}


//-----------------------------------------------------------------------
//   operator ==
//-----------------------------------------------------------------------


bool DOMIterator::operator == ( const Self& rhs ) const noexcept
{
  return ( root_ == rhs.root_ &&
           elem_ == rhs.elem_ &&
           dir_  == rhs.dir_ );
}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void DOMIterator::start ( Ref<DOMElement> root )
{
  idx_t  i, n;


  stack_.clear ();

  root_   = root;
  elem_   = root.get ();
  parent_ = 0;

  if ( elem_ && maxDepth_ > 0 )
  {
    depth_ =  1;
    dir_   = -1;
    n      = elem_->childCount ();

    if ( n > 0 && maxDepth_ > 1 )
    {
      stack_.reserve  ( 16 );
      stack_.pushBack ( elem_ );

      for ( i = n - 1; i >= 0; i-- )
      {
        stack_.pushBack ( elem_->getChild(i) );
      }
    }
  }
  else
  {
    depth_ = dir_ = 0;
  }
}


//-----------------------------------------------------------------------
//   next
//-----------------------------------------------------------------------


DOMElement* DOMIterator::next ()
{
  idx_t  i, n;

  if ( stack_.size () == 0 )
  {
    elem_  = parent_ = 0;
    depth_ = dir_    = 0;
  }
  else
  {
    elem_   = stack_.back ();

    stack_.popBack ();

    dir_    = (elem_ == parent_) ? 1 : -1;
    parent_ = elem_->getParent ();
    depth_ -= dir_;

    if ( dir_ < 0 )
    {
      n = elem_->childCount ();

      if ( n > 0 && depth_ < maxDepth_ )
      {
        stack_.pushBack ( elem_ );

        for ( i = n - 1; i >= 0; i-- )
        {
          stack_.pushBack ( elem_->getChild(i) );
        }
      }
    }
  }

  return elem_;
}


//-----------------------------------------------------------------------
//   goUp
//-----------------------------------------------------------------------


DOMElement* DOMIterator::goUp ()
{
  while ( stack_.size() > 0 )
  {
    elem_ = stack_.back ();

    stack_.popBack ();

    if ( elem_ == parent_ )
    {
      parent_ = elem_->getParent ();
      dir_    = 1;

      depth_--;

      return elem_;
    }
  }

  elem_  = parent_ = 0;
  depth_ = dir_    = 0;

  return elem_;
}


//-----------------------------------------------------------------------
//   stop
//-----------------------------------------------------------------------


void DOMIterator::stop ()
{
  stack_.clear ();

  elem_  = parent_ = 0;
  depth_ = dir_    = 0;
}


//-----------------------------------------------------------------------
//   setMaxDepth
//-----------------------------------------------------------------------


void DOMIterator::setMaxDepth ( int maxDepth )
{
  JEM_PRECHECK ( maxDepth >= 0 );

  maxDepth_ = maxDepth;

  while ( depth_ > maxDepth_ )
  {
    goUp ();
  }
}


JEM_END_PACKAGE( xml )
