
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
#include <jem/base/Thread.h>
#include <jem/base/Error.h>
#include <jem/base/CancelledException.h>
#include <jem/util/Flex.h>
#include <jem/util/List.h>
#include <jem/util/FastAllocator.h>
#include <jem/numeric/sparse/Reorder.h>
#include <jem/numeric/sparse/SparseStructure.h>


using jem::util::Flex;
using jem::util::List;
using jem::util::FastAllocator;


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   class NDOrderer
//=======================================================================

// The NDOrderer class implements a simple nested dissection algorithm.

class NDOrderer
{
 public:

  typedef NDOrderer         Self;
  typedef SparseStructure   SparseStruct;


  inline                    NDOrderer       ();

  inline                   ~NDOrderer       ();

  void                      reorder

    ( const Array<idx_t>&     iperm,
      const SparseStruct&     mstruc );


 private:

  class                     Node_
  {
   public:

    inline                    Node_         ();

    inline void               unlink        ();

    inline void               insert

      ( Node_*                   node );


   public:

    Node_*                    next;
    Node_*                    prev;
    idx_t                     inode;

  };


 private:

  void                      init_

    ( const SparseStruct&     mstruct );

  void                      clear_          ();

  void                      reorder_

    ( const Array<idx_t>&     iperm,
      List<idx_t>&            parts );

  idx_t                     traverse_

    ( idx_t                   inode,
      idx_t&                  count );

  inline void               freeNode_

    ( Node_**                 node );

  inline void               testCancelled_  ();


 private:

  FastAllocator             allocator_;

  Array<idx_t>              xadj_;
  Array<idx_t>              adjncy_;

  Flex<Node_*>              nodes_;
  Array<idx_t>              inodes_;
  Array<idx_t>              jnodes_;
  Array<idx_t>              colors_;
  Array<idx_t>              mask_;

  Node_                     freeList_;

  idx_t                     nodeCount_;
  idx_t                     nextColor_;
  idx_t                     nextMask_;
  lint                      opCount_;
  idx_t                     ifirst_;
  idx_t                     ilast_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline NDOrderer::NDOrderer () :

  allocator_ ( sizeof(Node_) )

{
  nodeCount_ = 0;
  nextColor_ = 0;
  nextMask_  = 0;
  opCount_   = 0;
  ifirst_    = 0;
  ilast_     = 0;
}


inline NDOrderer::~NDOrderer ()
{
  clear_ ();
}


//-----------------------------------------------------------------------
//   reorder
//-----------------------------------------------------------------------


void NDOrderer::reorder

  ( const Array<idx_t>&  iperm,
    const SparseStruct&  mstruc )

{
  List<idx_t>  parts;

  init_ ( mstruc );

  while ( ifirst_ < ilast_ )
  {
    idx_t  inode = freeList_.next->inode;

    if ( inode < 0 )
    {
      throw Error ( JEM_FUNC, "no more free nodes left" );
    }

    parts.pushBack ( inode );

    while ( parts.size() > 0 )
    {
      reorder_ ( iperm, parts );
    }
  }

  if ( ifirst_ > ilast_ )
  {
    throw Error ( JEM_FUNC, "too many nodes reordered" );
  }
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void NDOrderer::init_ ( const SparseStruct&  mstruct )
{
  const idx_t  nodeCount = mstruct.size(0);
  Node_*       node      = nullptr;


  if ( nodeCount_ > 0 )
  {
    clear_ ();
  }

  xadj_   .ref    ( mstruct.getRowOffsets()    );
  adjncy_ .ref    ( mstruct.getColumnIndices() );

  nodes_  .resize ( nodeCount );
  inodes_ .resize ( nodeCount );
  jnodes_ .resize ( nodeCount );
  colors_ .resize ( nodeCount );
  mask_   .resize ( nodeCount );

  nodeCount_ = nodeCount;
  nextColor_ = 0_idx;
  nextMask_  = 0_idx;
  opCount_   = 0;
  ifirst_    = 0;
  ilast_     = nodeCount_;

  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    node           = (Node_*) allocator_.alloc ();
    node->inode    = inode;
    nodes_ [inode] = node;
    colors_[inode] = -1_idx;
    mask_  [inode] = -1_idx;

    freeList_.insert ( node );
  }
}


//-----------------------------------------------------------------------
//   clear_
//-----------------------------------------------------------------------


void NDOrderer::clear_ ()
{
  Node_*  list = & freeList_;
  Node_*  node = list->next;

  while ( node != list )
  {
    node     ->unlink  ();
    allocator_.dealloc ( node );

    node = list->next;
  }
}


//-----------------------------------------------------------------------
//   reorder_
//-----------------------------------------------------------------------


void NDOrderer::reorder_

  ( const Array<idx_t>&  iperm,
    List<idx_t>&         parts )

{
  JEM_PRECHECK ( parts.size() > 0 );

  Array<idx_t>  xadj   ( xadj_   );
  Array<idx_t>  adjncy ( adjncy_ );
  Array<idx_t>  inodes ( inodes_ );
  Array<idx_t>  jnodes ( jnodes_ );
  Array<idx_t>  colors ( colors_ );
  Array<idx_t>  mask   ( mask_   );

  const idx_t   mask0  = nextMask_++;

  idx_t         color0;
  idx_t         color1;
  idx_t         color2;
  idx_t         color3;
  idx_t         icount;
  idx_t         jcount;
  idx_t         count;
  idx_t         inode;
  idx_t         jnode;
  idx_t         iseed;
  idx_t         k, n;


  iseed = inode = parts.front ();

  parts.popFront ();

  if ( inode < 0 )
  {
    inode = -inode - 1;
  }

  color0      = colors[inode];
  inode       = traverse_ ( inode, count );
  icount      = 1;
  jcount      = 0;
  inodes[0]   = inode;
  mask[inode] = mask0;

  if ( (count <= 8) || (iseed < 0) )
  {
    // This part is small enough; order the nodes it contains.

    while ( icount )
    {
      jcount = 0;

      for ( idx_t i = 0; i < icount; i++ )
      {
        inode            = inodes[i];
        n                = xadj[inode + 1];
        iperm[ifirst_++] = inode;

        freeNode_ ( &nodes_[inode] );

        for ( idx_t j = xadj[inode]; j < n; j++ )
        {
          jnode = adjncy[j];

          if ( (colors[jnode] == color0) && (mask[jnode] != mask0) )
          {
            jnodes[jcount++] = jnode;
            mask  [jnode]    = mask0;
          }

          opCount_++;
        }
      }

      testCancelled_ ();

      jem::swap ( icount, jcount );
      jem::swap ( inodes, jnodes );
    }

    return;
  }

  color1 = nextColor_++;
  iseed  = inode;
  k      = 0;

  // Collect the nodes in the first part.

  while ( icount )
  {
    jcount = 0;

    for ( idx_t i = 0; i < icount; i++ )
    {
      inode         = inodes[i];
      n             = xadj[inode + 1];
      colors[inode] = color1;

      k++;

      for ( idx_t j = xadj[inode]; j < n; j++ )
      {
        jnode = adjncy[j];

        if ( (colors[jnode] == color0) && (mask[jnode] != mask0) )
        {
          jnodes[jcount++] = jnode;
          mask  [jnode]    = mask0;
        }

        opCount_++;
      }
    }

    testCancelled_ ();

    jem::swap ( icount, jcount );
    jem::swap ( inodes, jnodes );

    if ( k >= ((count - icount) / 2) )
    {
      break;
    }
  }

  jcount = 0;

  if ( (k + icount) >= count )
  {
    // There are not enough nodes to form a second part. Add the
    // separator nodes to the first part and schedule that part to
    // be ordered as a whole.

    for ( idx_t i = 0; i < icount; i++ )
    {
      colors[inodes[i]] = color1;
    }

    parts.pushFront ( -iseed - 1 );

    return;
  }

  parts.pushBack ( iseed );

  color2 = nextColor_++;
  color3 = nextColor_++;

  // Order the nodes in the separator and collect the first nodes
  // in the second part.

  for ( idx_t i = 0; i < icount; i++ )
  {
    inode           = inodes[i];
    n               = xadj[inode + 1];
    colors[inode]   = color2;
    iperm[--ilast_] = inode;

    freeNode_ ( &nodes_[inode] );

    for ( idx_t j = xadj[inode]; j < n; j++ )
    {
      jnode = adjncy[j];

      if ( (colors[jnode] == color0) && (mask[jnode] != mask0) )
      {
        jnodes[jcount++] = jnode;
        mask  [jnode]    = mask0;
      }

      opCount_++;
    }
  }

  jem::swap ( icount, jcount );
  jem::swap ( inodes, jnodes );

  // Collect the nodes in the second part.

  if ( ! icount )
  {
    throw Error ( JEM_FUNC, "oops, lost a node" );
  }

  parts.pushBack ( inodes[0] );

  while ( icount )
  {
    jcount = 0;

    for ( idx_t i = 0; i < icount; i++ )
    {
      inode         = inodes[i];
      n             = xadj[inode + 1];
      colors[inode] = color3;

      for ( idx_t j = xadj[inode]; j < n; j++ )
      {
        jnode = adjncy[j];

        if ( (colors[jnode] == color0) && (mask[jnode] != mask0) )
        {
          jnodes[jcount++] = jnode;
          mask  [jnode]    = mask0;
        }

        opCount_++;
      }
    }

    testCancelled_ ();

    jem::swap ( icount, jcount );
    jem::swap ( inodes, jnodes );
  }
}


//-----------------------------------------------------------------------
//   traverse_
//-----------------------------------------------------------------------


idx_t NDOrderer::traverse_

  ( idx_t   inode,
    idx_t&  count )

{
  JEM_PRECHECK ( inode >= 0 && inode < nodeCount_ );

  Array<idx_t>  xadj   ( xadj_   );
  Array<idx_t>  adjncy ( adjncy_ );
  Array<idx_t>  inodes ( inodes_ );
  Array<idx_t>  jnodes ( jnodes_ );
  Array<idx_t>  colors ( colors_ );
  Array<idx_t>  mask   ( mask_   );

  const idx_t   color0 = colors[inode];
  const idx_t   mask0  = nextMask_++;

  idx_t         icount;
  idx_t         jcount;
  idx_t         jnode;
  idx_t         n;


  mask[inode] = mask0;
  inodes[0]   = inode;
  icount      = 1;
  jcount      = 0;
  count       = 1;

  while ( icount )
  {
    jcount = 0;

    for ( idx_t i = 0; i < icount; i++ )
    {
      inode = inodes[i];
      n     = xadj[inode + 1];

      for ( idx_t j = xadj[inode]; j < n; j++ )
      {
        jnode = adjncy[j];

        if ( (colors[jnode] == color0) && (mask[jnode] != mask0) )
        {
          jnodes[jcount++] = jnode;
          mask  [jnode]    = mask0;
        }

        opCount_++;
      }
    }

    count += jcount;

    testCancelled_ ();

    jem::swap ( inodes, jnodes );
    jem::swap ( icount, jcount );
  }

  if ( ! jcount )
  {
    throw Error ( JEM_FUNC, "no fringe node found" );
  }

  return jnodes[jcount - 1];
}


//-----------------------------------------------------------------------
//   freeNode_
//-----------------------------------------------------------------------


inline void NDOrderer::freeNode_ ( Node_** node )
{
  JEM_ASSERT ( node && (*node) );

  (*node)->unlink ();
  allocator_.dealloc ( *node );
  *node = nullptr;
}


//-----------------------------------------------------------------------
//   testCancelled_
//-----------------------------------------------------------------------


inline void NDOrderer::testCancelled_ ()
{
  if ( opCount_ > 1000000_lint )
  {
    if ( Thread::cancelled() )
    {
      throw CancelledException (
        JEM_FUNC,
        "nested dissection ordering algorithm cancelled"
      );
    }

    opCount_ = 0;
  }
}


//=======================================================================
//   class NDOrderer::Node_
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline NDOrderer::Node_::Node_ ()
{
  next  = prev = this;
  inode = -1;
}


//-----------------------------------------------------------------------
//   unlink
//-----------------------------------------------------------------------


inline void NDOrderer::Node_::unlink ()
{
  next->prev  = prev;
  prev->next  = next;
  next = prev = this;
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


inline void NDOrderer::Node_::insert ( Node_* node )
{
  node->next = this;
  node->prev = prev;
  prev->next = node;
  prev       = node;
}


//=======================================================================
//   class Reorder
//=======================================================================


void                      Reorder::nd

  ( const Array<idx_t>&     iperm,
    const SparseStructure&  mstruc )

{
  JEM_PRECHECK2 ( iperm.size() == mstruc.size(0) &&
                  iperm.size() == mstruc.size(1),
                  "mismatch between SparseSructure shape "
                  "and permutation array size" );
  JEM_PRECHECK2 ( mstruc.isValid(), "invalid SparseStructure" );

  NDOrderer  nd;

  nd.reorder ( iperm, mstruc );
}


JEM_END_PACKAGE( numeric )
