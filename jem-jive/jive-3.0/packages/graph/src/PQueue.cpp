
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */


#include <cstring>
#include <jem/base/assert.h>
#include <jem/base/memory.h>
#include <jem/base/MemCache.h>
#include <jem/numeric/utilities.h>
#include <jive/graph/PQueue.h>


JIVE_BEGIN_PACKAGE( graph )


using jem::MemCache;
using jem::numeric::abs;


//=======================================================================
//   class PQueue::Node_
//=======================================================================


class PQueue::Node_
{
 public:

  inline          Node_   ();

  inline void     pushBack

    ( Node_*        n );

  inline void     unlink  ();


 public:

  Node_*          next;
  Node_*          prev;

  idx_t           index;
  idx_t           gain;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline PQueue::Node_::Node_ ()
{
  next  =  this;
  prev  =  this;
  index = -1;
  gain  =  0;
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


inline void PQueue::Node_::pushBack ( Node_* n )
{
  n->next    = next;
  n->prev    = this;
  next->prev = n;
  next       = n;
}


//-----------------------------------------------------------------------
//   unlink
//-----------------------------------------------------------------------


inline void PQueue::Node_::unlink ()
{
  next->prev = prev;
  prev->next = next;
  next       = this;
  prev       = this;
}


//=======================================================================
//   class PQueue
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  PQueue::MAX_CHAINS_ = 32;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


PQueue::PQueue ()
{
  init_ ( 0, 0 );
}


PQueue::PQueue

  ( idx_t  nodeCount,
    idx_t  maxGain )

{
  JEM_PRECHECK ( nodeCount >= 0 && maxGain >= 0 );

  init_ ( nodeCount, maxGain );
}


PQueue::~PQueue ()
{
  free_ ();
}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void PQueue::init

  ( idx_t  nodeCount,
    idx_t  maxGain )

{
  JEM_PRECHECK ( nodeCount >= 0 && maxGain >= 0 );

  PQueue  tmp ( nodeCount, maxGain );

  swap ( tmp );
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


void PQueue::swap ( PQueue& rhs )
{
  void*      self  = static_cast<void*> ( this );
  void*      other = static_cast<void*> ( &rhs );

  jem::byte  buf[sizeof(Self)];

  std::memcpy ( buf,   self,  sizeof(Self) );
  std::memcpy ( self,  other, sizeof(Self) );
  std::memcpy ( other, buf,   sizeof(Self) );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void PQueue::clear ()
{
  Node_*  node;

  while ( head_ >= chains_ )
  {
    while ( head_->next != head_ )
    {
      node        = head_->next;
      node->index = -1;
      node->unlink ();
    }

    head_--;
  }

  size_ = 0;
}


//-----------------------------------------------------------------------
//   popFront
//-----------------------------------------------------------------------


idx_t PQueue::popFront ()
{
  JEM_ASSERT ( size() > 0 );

  Node_*  node  = head_->next;
  idx_t   inode = node->index;

  node->index = -1;
  node->unlink ();

  size_--;

  if ( head_->next == head_ )
  {
    while ( head_ >= chains_ && head_->next == head_ )
    {
      head_--;
    }
  }

  return inode;
}


//-----------------------------------------------------------------------
//   remove
//-----------------------------------------------------------------------


void PQueue::remove ( idx_t inode )
{
  Node_*  node = nodes_ + inode;

  if ( node->index >= 0 )
  {
    Node_*  chain = getChain_ ( node->gain );

    node->index = -1;
    node->unlink ();

    size_--;

    if ( chain == head_ )
    {
      while ( head_ >= chains_ && head_->next == head_ )
      {
        head_--;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


void PQueue::insert ( idx_t inode, idx_t gain )
{
  JEM_ASSERT ( abs( gain ) <= maxGain() );

  Node_*  node = nodes_ + inode;

  if ( node->index < 0 )
  {
    Node_*  chain = getChain_ ( gain );

    node->index = inode;
    node->gain  = gain;

    chain->pushBack ( node );
    size_++;

    if ( chain > head_ )
    {
      head_ = chain;
    }
  }
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void PQueue::update ( idx_t inode, idx_t gain )
{
  JEM_ASSERT ( abs( gain ) <= maxGain() );

  Node_*  node = nodes_ + inode;

  if ( node->index >= 0 )
  {
    Node_*  oldChain = getChain_ ( node->gain );
    Node_*  newChain = getChain_ ( gain );

    node->gain = gain;

    if ( oldChain != newChain )
    {
      node->unlink ();
      newChain->pushBack ( node );

      if      ( newChain > head_ )
      {
        head_ = newChain;
      }
      else if ( oldChain == head_ )
      {
        while ( head_->next == head_ )
        {
          head_--;
        }
      }
    }
  }
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void PQueue::init_

  ( idx_t  nodeCount,
    idx_t  maxGain )

{
  Node_*  nodes;
  idx_t   chainCount;
  idx_t   i, n;


  chainCount = 2 * maxGain + 1;

  if ( chainCount > MAX_CHAINS_ )
  {
    chainCount = MAX_CHAINS_;
  }

  n     = nodeCount + chainCount + 1;

  nodes = (Node_*) MemCache::alloc ( (size_t) n * sizeof(*nodes) );

  try
  {
    jem::construct ( nodes, n );
  }
  catch ( ... )
  {
    MemCache::dealloc ( nodes, (size_t) n * sizeof(*nodes) );
    throw;
  }

  nodes_      = nodes;
  head_       = nodes + nodeCount;
  chains_     = head_ + 1;
  nodeCount_  = nodeCount;
  chainCount_ = chainCount;
  scale_      = (double) chainCount / (double) (2 * maxGain + 1);
  maxGain_    = maxGain;
  size_       = 0;

  for ( i = 0; i < chainCount; i++ )
  {
    chains_[i].gain  = (idx_t) ((double) (i - maxGain_) / scale_);
    chains_[i].index = i;
  }
}


//-----------------------------------------------------------------------
//   free_
//-----------------------------------------------------------------------


void PQueue::free_ ()
{
  const idx_t  n = nodeCount_ + chainCount_ + 1;

  MemCache::dealloc ( nodes_, (size_t) n * sizeof(*nodes_) );

  nodes_  = 0;
  chains_ = 0;
  head_   = 0;
}


//-----------------------------------------------------------------------
//   getChain_
//-----------------------------------------------------------------------


inline PQueue::Node_* PQueue::getChain_ ( idx_t gain ) const
{
  return (chains_ + (idx_t) ((double) (gain + maxGain_) * scale_));
}


JIVE_END_PACKAGE( graph )
