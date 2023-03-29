
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
#include <jem/base/Thread.h>
#include <jem/base/Error.h>
#include <jem/base/CancelledException.h>
#include <jem/base/array/utilities.h>
#include <jem/util/Flex.h>
#include <jem/util/FastAllocator.h>
#include <jem/numeric/sparse/Reorder.h>
#include <jem/numeric/sparse/SparseStructure.h>


using jem::util::Flex;
using jem::util::FastAllocator;


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   class RCMOrderer
//=======================================================================


class RCMOrderer
{
 public:

  typedef RCMOrderer        Self;
  typedef SparseStructure   SparseStruct;


  inline                    RCMOrderer      ();

  inline                   ~RCMOrderer      ();

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
  idx_t                     findFringeNode_ ();

  inline void               freeNode_

    ( Node_**                 node );


 private:

  FastAllocator             allocator_;

  Array<idx_t>              xadj_;
  Array<idx_t>              adjncy_;

  Flex<Node_*>              nodes_;
  Array<idx_t>              inodes_;
  Array<idx_t>              jnodes_;
  Array<idx_t>              degrees_;
  Array<idx_t>              mask_;

  Node_                     freeList_;

  idx_t                     nodeCount_;
  idx_t                     maskValue_;
  lint                      opCount_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline RCMOrderer::RCMOrderer () :

  allocator_ ( sizeof(Node_) )

{
  nodeCount_ = 0;
  maskValue_ = 0;
  opCount_   = 0;
}


inline RCMOrderer::~RCMOrderer ()
{
  clear_ ();
}


//-----------------------------------------------------------------------
//   reorder
//-----------------------------------------------------------------------


void RCMOrderer::reorder

  ( const Array<idx_t>&  iperm,
    const SparseStruct&  mstruc )

{
  init_                 ( mstruc   );

  Array<idx_t>  xadj    ( xadj_    );
  Array<idx_t>  adjncy  ( adjncy_  );
  Array<idx_t>  inodes  ( inodes_  );
  Array<idx_t>  jnodes  ( jnodes_  );
  Array<idx_t>  degrees ( degrees_ );

  Node_**       nodes   = nodes_.addr ();

  idx_t         inodeCount;
  idx_t         jnodeCount;
  idx_t         inode;
  idx_t         jnode;
  idx_t         inext;
  idx_t         k;
  idx_t         n;


  inext = nodeCount_;

  while ( inext > 0 )
  {
    inode = findFringeNode_ ();

    if ( inode < 0 )
    {
      throw Error ( JEM_FUNC, "no fringe node found" );
    }

    freeNode_ ( & nodes[inode] );

    inodes[0]      = inode;
    inodeCount     = 1;
    iperm[--inext] = inode;
    n              = xadj[inode + 1];

    for ( idx_t j = xadj[inode + 1]; j < n; j++ )
    {
      degrees[adjncy[j]]--;
    }

    while ( inodeCount > 0 )
    {
      jnodeCount = 0;

      for ( idx_t i = 0; i < inodeCount; i++ )
      {
        inode = inodes[i];
        k     = jnodeCount;
        n     = xadj[inode + 1];

        for ( idx_t j = xadj[inode]; j < n; j++ )
        {
          jnode = adjncy[j];

          if ( nodes[jnode] )
          {
            freeNode_ ( & nodes[jnode] );

            jnodes[jnodeCount++] = jnode;
          }

          opCount_++;
        }

        if ( jnodeCount > k + 1 )
        {
          sort ( jnodes[slice(k,jnodeCount)], degrees );
        }

        for ( ; k < jnodeCount; k++ )
        {
          jnode          = jnodes[k];
          iperm[--inext] = jnode;
          n              = xadj[jnode + 1];

          for ( idx_t j = xadj[jnode]; j < n; j++ )
          {
            degrees[adjncy[j]]--;
            opCount_++;
          }
        }
      }

      if ( opCount_ > 1000000_lint )
      {
        if ( Thread::cancelled() )
        {
          throw CancelledException (
            JEM_FUNC,
            "RCM ordering algorithm cancelled"
          );
        }

        opCount_ = 0;
      }

      jem::swap ( inodes,     jnodes     );
      jem::swap ( inodeCount, jnodeCount );
    }
  }
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void RCMOrderer::init_ ( const SparseStruct&  mstruct )
{
  const idx_t  nodeCount = mstruct.size(0);

  Node_*       node;


  if ( nodeCount_ > 0 )
  {
    clear_ ();
  }

  xadj_   .ref    ( mstruct.getRowOffsets()    );
  adjncy_ .ref    ( mstruct.getColumnIndices() );

  nodes_  .resize ( nodeCount );
  inodes_ .resize ( nodeCount );
  jnodes_ .resize ( nodeCount );
  degrees_.resize ( nodeCount );
  mask_   .resize ( nodeCount );

  nodeCount_ = nodeCount;
  maskValue_ = 0_idx;
  opCount_   = 0;

  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    node          = (Node_*) allocator_.alloc ();
    node->inode   = inode;
    nodes_[inode] = node;

    freeList_.insert ( node );

    degrees_[inode] = xadj_[inode + 1] - xadj_[inode];
    mask_[inode]    = -1;
  }
}


//-----------------------------------------------------------------------
//   clear_
//-----------------------------------------------------------------------


void RCMOrderer::clear_ ()
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
//   findFringeNode_
//-----------------------------------------------------------------------


idx_t RCMOrderer::findFringeNode_ ()
{
  Array<idx_t>  xadj   ( xadj_   );
  Array<idx_t>  adjncy ( adjncy_ );
  Array<idx_t>  inodes ( inodes_ );
  Array<idx_t>  jnodes ( jnodes_ );
  Array<idx_t>  mask   ( mask_   );

  const idx_t   mvalue = maskValue_++;

  idx_t         minDeg;
  idx_t         inodeCount;
  idx_t         jnodeCount;
  idx_t         inode;
  idx_t         jnode;
  idx_t         n;


  inode = freeList_.next->inode;

  if ( inode < 0 )
  {
    return -1;
  }

  mask[inode] = mvalue;
  inodes[0]   = inode;
  inodeCount  = 1;

  while ( inodeCount > 0 )
  {
    jnodeCount = 0;

    for ( idx_t i = 0; i < inodeCount; i++ )
    {
      inode = inodes[i];
      n     = xadj[inode + 1];

      for ( idx_t j = xadj[inode]; j < n; j++ )
      {
        jnode = adjncy[j];

        if ( nodes_[jnode] && mask[jnode] != mvalue )
        {
          jnodes[jnodeCount++] = jnode;
          mask[jnode]          = mvalue;
        }

        opCount_++;
      }
    }

    jem::swap ( inodes,     jnodes     );
    jem::swap ( inodeCount, jnodeCount );
  }

  minDeg = maxOf ( minDeg );
  inode  = -1;

  for ( idx_t j = 0; j < jnodeCount; j++ )
  {
    jnode = jnodes[j];

    if ( degrees_[jnode] < minDeg )
    {
      minDeg = degrees_[jnode];
      inode  = jnode;
    }
  }

  if ( inode < 0 )
  {
    throw Error ( JEM_FUNC, "no node found" );
  }

  return inode;
}


//-----------------------------------------------------------------------
//   freeNode_
//-----------------------------------------------------------------------


inline void RCMOrderer::freeNode_ ( Node_** node )
{
  (*node)->unlink ();
  allocator_.dealloc ( *node );
  *node = nullptr;
}


//=======================================================================
//   class RCMOrderer::Node_
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline RCMOrderer::Node_::Node_ ()
{
  next  = prev = this;
  inode = -1;
}


//-----------------------------------------------------------------------
//   unlink
//-----------------------------------------------------------------------


inline void RCMOrderer::Node_::unlink ()
{
  next->prev  = prev;
  prev->next  = next;
  next = prev = this;
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


inline void RCMOrderer::Node_::insert ( Node_* node )
{
  node->next = this;
  node->prev = prev;
  prev->next = node;
  prev       = node;
}


//=======================================================================
//   class Reorder
//=======================================================================


void                      Reorder::rcm

  ( const Array<idx_t>&     iperm,
    const SparseStructure&  mstruc )

{
  JEM_PRECHECK2 ( iperm.size() == mstruc.size(0) &&
                  iperm.size() == mstruc.size(1),
                  "mismatch between SparseSructure shape "
                  "and permutation array size" );
  JEM_PRECHECK2 ( mstruc.isValid(), "invalid SparseStructure" );

  RCMOrderer  rcm;

  rcm.reorder ( iperm, mstruc );
}


JEM_END_PACKAGE( numeric )
