
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


#include <cmath>
#include <jem/base/assert.h>
#include <jem/base/Thread.h>
#include <jem/base/Error.h>
#include <jem/base/CancelledException.h>
#include <jem/util/Flex.h>
#include <jem/util/FastAllocator.h>
#include <jem/numeric/sparse/Reorder.h>
#include <jem/numeric/sparse/SparseStructure.h>


using jem::util::Flex;
using jem::util::FastAllocator;


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   class ColOrderer
//=======================================================================


class ColOrderer
{
 public:

  typedef ColOrderer        Self;
  typedef SparseStructure   SparseStruct;


                            ColOrderer ();
                           ~ColOrderer ();

  void                      reorder

    ( const Array<idx_t>&     jperm,
      const SparseStruct&     mstruc );


 private:

  class                     ColNode_
  {
   public:

    inline                    ColNode_ ();

    inline void               unlink   ();

    inline void               insertBefore

      ( ColNode_*               node );

    inline void               insertAfter

      ( ColNode_*               node );


   public:

    ColNode_*                 next;
    ColNode_*                 prev;

    union
    {
      idx_t                   irow;
      idx_t                   jcol;
    };

  };


  class                     RowNode_
  {
   public:

    inline                    RowNode_ ();

    inline void               insertAfter

      ( RowNode_*               node );


   public:

    RowNode_*                 next;
    ColNode_*                 cnode;

    union
    {
      idx_t                   size;
      idx_t                   jcol;
    };

  };


 private:

  void                      checkDegrees_ ();

  void                      init_

    ( const SparseStruct&     mstruc );

  void                      initDegrees_ ();
  void                      initHash_    ();
  void                      clear_       ();

  void                      clearLists_

    ( Flex<RowNode_>&         lists  );

  void                      clearLists_

    ( Flex<ColNode_>&         lists  );

  idx_t                     nextColumn_  ();

  void                      eraseColumn_

    ( idx_t                   jcol   );

  inline void               hashColumn_

    ( ColNode_*               node,
      double                  degree );

  void                      rehash_

    ( double                  maxDeg );

  inline void               setHashScale_

    ( double                  maxDeg );

  inline RowNode_*          newRowNode_

    ( idx_t                   jcol,
      ColNode_*               cnode  );

  inline ColNode_*          newColNode_

    ( idx_t                   irow   );

  inline void               freeNode_

    ( RowNode_*               node   );

  inline void               freeNode_

    ( ColNode_*               node   );


 private:

  FastAllocator             rowAllocator_;
  FastAllocator             colAllocator_;

  Flex<RowNode_>            rowLists_;
  Flex<ColNode_>            colLists_;

  Flex<bool>                colMask_;
  Flex<double>              colDegrees_;
  Flex<double>              newDegrees_;
  Flex<ColNode_*>           colHashNodes_;
  idx_t                     colCount_;

  Flex<ColNode_>            hashTable_;
  double                    hashScale_;
  idx_t                     hashFirst_;

  lint                      opCount_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ColOrderer::ColOrderer () :

  rowAllocator_ ( sizeof(RowNode_) ),
  colAllocator_ ( sizeof(ColNode_) )

{
  colCount_  = 0;
  hashScale_ = 0.0;
  hashFirst_ = 0;
  opCount_   = 0;
}


ColOrderer::~ColOrderer ()
{
  clear_ ();
}


//-----------------------------------------------------------------------
//   reorder
//-----------------------------------------------------------------------


void ColOrderer::reorder

  ( const Array<idx_t>&  jperm,
    const SparseStruct&  mstruc )

{
  idx_t  icol, jcol;


  init_ ( mstruc );

  for ( icol = 0; icol < colCount_; icol++ )
  {
    jcol = nextColumn_ ();

    eraseColumn_ ( jcol );

    jperm[icol] = jcol;

    if ( opCount_ > 1000000_lint )
    {
      if ( Thread::cancelled() )
      {
        throw CancelledException (
          JEM_FUNC,
          "column ordering algorithm cancelled"
        );
      }

      opCount_ = 0;
    }
  }
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void ColOrderer::init_ ( const SparseStruct& mstruc )
{
  const idx_t   rowCount   = mstruc.size(0);
  const idx_t   colCount   = mstruc.size(1);

  Array<idx_t>  rowOffsets = mstruc.getRowOffsets    ();
  Array<idx_t>  colIndices = mstruc.getColumnIndices ();

  RowNode_*     rnode;
  ColNode_*     cnode;

  idx_t         irow;
  idx_t         jcol;


  if ( colCount_ > 0 )
  {
    clear_ ();
  }

  rowLists_    .resize ( rowCount );
  colLists_    .resize ( colCount );
  colMask_     .resize ( colCount );
  colDegrees_  .resize ( colCount );
  newDegrees_  .resize ( colCount );
  colHashNodes_.resize ( colCount );
  hashTable_   .resize ( colCount );

  colCount_   = colCount;
  colMask_    = false;
  colDegrees_ = 0.0;
  newDegrees_ = 0.0;
  opCount_    = 0;

  for ( irow = 0; irow < rowCount; irow++ )
  {
    idx_t  n = rowOffsets[irow + 1];

    for ( idx_t i = rowOffsets[irow]; i < n; i++ )
    {
      jcol  = colIndices[i];
      cnode = newColNode_ ( irow );
      cnode->insertBefore ( & colLists_[jcol] );

      rnode = newRowNode_ ( jcol, cnode );
      rnode->insertAfter  ( & rowLists_[irow] );

      rowLists_[irow].size++;
    }
  }

  initDegrees_  ();
  initHash_     ();
}


//-----------------------------------------------------------------------
//   initDegrees_
//-----------------------------------------------------------------------


void ColOrderer::initDegrees_ ()
{
  const idx_t  n = colCount_;

  ColNode_*    list;
  ColNode_*    node;

  double       deg;


  for ( idx_t jcol = 0; jcol < n; jcol++ )
  {
    deg  = 0.0;
    list = colLists_.addr ( jcol );

    for ( node = list->next; node != list; node = node->next )
    {
      deg += (double) (rowLists_[node->irow].size - 1);
    }

    colDegrees_[jcol] = deg;
  }
}


//-----------------------------------------------------------------------
//   initHash_
//-----------------------------------------------------------------------


void ColOrderer::initHash_ ()
{
  const double* JEM_RESTRICT  degrees = colDegrees_.addr ();

  const idx_t  n = colCount_;

  ColNode_*    node;

  double       deg, maxDeg;


  maxDeg = 0.0;

  for ( idx_t jcol = 0; jcol < n; jcol++ )
  {
    deg = degrees[jcol];

    if ( deg > maxDeg )
    {
      maxDeg = deg;
    }
  }

  setHashScale_ ( maxDeg );

  hashFirst_ = n;

  for ( idx_t jcol = 0; jcol < n; jcol++ )
  {
    node = newColNode_ ( jcol );
    colHashNodes_[jcol] = node;

    hashColumn_ ( node, degrees[jcol] );
  }
}


//-----------------------------------------------------------------------
//   clear_
//-----------------------------------------------------------------------


void ColOrderer::clear_ ()
{
  clearLists_ ( rowLists_  );
  clearLists_ ( colLists_  );
  clearLists_ ( hashTable_ );
}


//-----------------------------------------------------------------------
//   clearLists_
//-----------------------------------------------------------------------


void ColOrderer::clearLists_ ( Flex<RowNode_>& lists )
{
  const idx_t  n = lists.size ();

  RowNode_*    node;
  RowNode_*    next;


  for ( idx_t i = 0; i < n; i++ )
  {
    next = lists[i].next;

    while ( next )
    {
      node = next;
      next = next->next;

      freeNode_ ( node );
    }

    lists[i].next = nullptr;
    lists[i].size = 0;
  }
}


void ColOrderer::clearLists_ ( Flex<ColNode_>& lists )
{
  const idx_t  n = lists.size ();

  ColNode_*    list;
  ColNode_*    node;


  for ( idx_t i = 0; i < n; i++ )
  {
    list = lists.addr ( i );
    node = list->next;

    while ( node != list )
    {
      node->unlink ();
      freeNode_ ( node );

      node = node->next;
    }
  }
}


//-----------------------------------------------------------------------
//   nextColumn_
//-----------------------------------------------------------------------


idx_t ColOrderer::nextColumn_ ()
{
  const idx_t  n = colCount_;

  ColNode_*    list;
  ColNode_*    node;

  idx_t        jcol;


  for ( ; hashFirst_ < n; hashFirst_++ )
  {
    list = hashTable_.addr ( hashFirst_ );
    node = list->next;

    if ( node != list )
    {
      jcol = node->jcol;

      node->unlink ();
      freeNode_ ( node );

      colHashNodes_[jcol] = nullptr;

      return jcol;
    }
  }

  throw Error ( JEM_FUNC, "no more columns left" );

  return -1;
}


//-----------------------------------------------------------------------
//   eraseColumn_
//-----------------------------------------------------------------------


void ColOrderer::eraseColumn_ ( idx_t jcol )
{
  double*    JEM_RESTRICT  degrees = colDegrees_.addr ();
  double*    JEM_RESTRICT  newDegs = newDegrees_.addr ();
  bool*      JEM_RESTRICT  colMask = colMask_   .addr ();

  RowNode_   newRow;
  RowNode_*  rlist;
  RowNode_*  rnode;
  ColNode_*  clist;
  ColNode_*  cnode;

  double     newDeg, rdeg;

  idx_t      irow;


  irow  = -1;
  clist = colLists_.addr ( jcol );
  cnode = clist->next;

  newDegs[jcol] = degrees[jcol];
  degrees[jcol] = 0.0;
  colMask[jcol] = true;

  while ( cnode != clist )
  {
    rlist     = rowLists_.addr ( cnode->irow );
    rnode     = rlist->next;
    opCount_ += (lint) rlist->size;
    rdeg      = (double) (rlist->size - 1);

    if ( irow < 0 )
    {
      irow = cnode->irow;
    }

    while ( rnode )
    {
      rlist->next = rnode->next;
      jcol        = rnode->jcol;

      if ( colMask[jcol] )
      {
        newDegs[jcol] -= rdeg;

        if ( rnode->cnode != cnode )
        {
          rnode->cnode->unlink ();
          freeNode_ ( rnode->cnode );
        }

        freeNode_ ( rnode );
      }
      else
      {
        newDegs[jcol]      = degrees[jcol] - rdeg;
        colMask[jcol]      = true;
        rnode->cnode->irow = irow;
        rnode->next        = newRow.next;
        newRow.next        = rnode;

        newRow.size++;
      }

      rnode = rlist->next;
    }

    rlist->size = 0;

    cnode->unlink ();
    freeNode_ ( cnode );

    cnode = clist->next;
  }

  if ( irow < 0 )
  {
    return;
  }

  rdeg = (double) (newRow.size - 1);

  for ( rnode = newRow.next; rnode; rnode = rnode->next )
  {
    jcol          = rnode->jcol;
    colMask[jcol] = false;
    newDeg        = jem::max ( 0.0, (newDegs[jcol] + rdeg) );

    if ( std::fabs( newDeg - degrees[jcol] ) < 0.01 )
    {
      continue;
    }

    degrees[jcol] = newDeg;

    if ( newDeg * hashScale_ > 0.99 )
    {
      rehash_ ( newDeg );
    }
    else
    {
      cnode = colHashNodes_[jcol];
      cnode->unlink ();

      hashColumn_ ( cnode, newDeg );
    }
  }

  if ( rowLists_[irow].next )
  {
    throw Error (
      JEM_FUNC,
      String::format ( "row list %d not empty", irow )
    );
  }

  rowLists_[irow] = newRow;
}


//-----------------------------------------------------------------------
//   hashColumn_
//-----------------------------------------------------------------------


inline void ColOrderer::hashColumn_

  ( ColNode_*  node,
    double     degree )

{
  double  x = (double) colCount_ * std::sqrt ( degree * hashScale_ );
  idx_t   i = (idx_t) x;

  x = x - (double) i;

  if ( x < 0.5 )
  {
    node->insertAfter  ( & hashTable_[i] );
  }
  else
  {
    node->insertBefore ( & hashTable_[i] );
  }

  if ( i < hashFirst_ )
  {
    hashFirst_ = i;
  }
}


//-----------------------------------------------------------------------
//   rehash_
//-----------------------------------------------------------------------


void ColOrderer::rehash_ ( double maxDeg )
{
  const double*  JEM_RESTRICT  degrees = colDegrees_.addr ();

  const idx_t    n = colCount_;

  ColNode_*      node;


  setHashScale_ ( maxDeg );

  hashFirst_ = n;

  for ( idx_t jcol = 0; jcol < n; jcol++ )
  {
    node = colHashNodes_[jcol];

    if ( node )
    {
      node->unlink ();
      hashColumn_  ( node, degrees[jcol] );
    }
  }
}


//-----------------------------------------------------------------------
//   setHashScale_
//-----------------------------------------------------------------------


inline void ColOrderer::setHashScale_ ( double maxDeg )
{
  if ( maxDeg > 0.0 )
  {
    hashScale_ = 1.0 / (4.0 * maxDeg);
  }
  else
  {
    hashScale_ = 0.0;
  }
}


//-----------------------------------------------------------------------
//   newRowNode_
//-----------------------------------------------------------------------


inline ColOrderer::RowNode_* ColOrderer::newRowNode_

  ( idx_t      jcol,
    ColNode_*  cnode )

{
  RowNode_*  rnode = (RowNode_*) rowAllocator_.alloc ();

  rnode->jcol  = jcol;
  rnode->cnode = cnode;

  return rnode;
}


//-----------------------------------------------------------------------
//   newColNode_
//-----------------------------------------------------------------------


inline ColOrderer::ColNode_* ColOrderer::newColNode_

  ( idx_t  irow )

{
  ColNode_*  cnode = (ColNode_*) colAllocator_.alloc ();

  cnode->irow = irow;

  return cnode;
}


//-----------------------------------------------------------------------
//   freeNode_
//-----------------------------------------------------------------------


inline void ColOrderer::freeNode_ ( RowNode_* node )
{
  rowAllocator_.dealloc ( node );
}


inline void ColOrderer::freeNode_ ( ColNode_* node )
{
  colAllocator_.dealloc ( node );
}


//=======================================================================
//   class ColOrderer::ColNode_
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline ColOrderer::ColNode_::ColNode_ ()
{
  next = prev = this;
  irow = 0;
}


//-----------------------------------------------------------------------
//   unlink
//-----------------------------------------------------------------------


inline void ColOrderer::ColNode_::unlink ()
{
  next->prev  = prev;
  prev->next  = next;
  next = prev = this;
}


//-----------------------------------------------------------------------
//   insertBefore
//-----------------------------------------------------------------------


inline void ColOrderer::ColNode_::insertBefore ( ColNode_* node )
{
  next       = node;
  prev       = node->prev;
  node->prev = this;
  prev->next = this;
}


//-----------------------------------------------------------------------
//   insertAfter
//-----------------------------------------------------------------------


inline void ColOrderer::ColNode_::insertAfter ( ColNode_* node )
{
  next       = node->next;
  prev       = node;
  node->next = this;
  next->prev = this;
}


//=======================================================================
//   class ColOrderer::RowNode_
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline ColOrderer::RowNode_::RowNode_ ()
{
  next  = nullptr;
  cnode = nullptr;
  size  = 0;
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


inline void ColOrderer::RowNode_::insertAfter ( RowNode_* node )
{
  next       = node->next;
  node->next = this;
}


//=======================================================================
//   class Reorder
//=======================================================================


void                      Reorder::colamd

  ( const Array<idx_t>&     jperm,
    const SparseStructure&  mstruc )

{
  JEM_PRECHECK2 ( jperm.size() == mstruc.size(1),
                  "permutation array has wrong length" );
  JEM_PRECHECK2 ( mstruc.isValid(), "invalid SparseStructure" );

  ColOrderer  colOrderer;

  colOrderer.reorder ( jperm, mstruc );
}


JEM_END_PACKAGE( numeric )
