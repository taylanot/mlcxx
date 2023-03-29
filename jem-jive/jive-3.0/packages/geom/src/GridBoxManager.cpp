
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


#include <cmath>
#include <cstring>
#include <jem/pragmas.h>
#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/xutil/IListArray.h>
#include <jive/log.h>
#include <jive/geom/MaskMatrix.h>
#include <jive/geom/BoxArray.h>
#include <jive/geom/GridBoxManager.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::GridBoxManager );


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;
using jem::IllegalArgumentException;
using jem::xutil::IListArray;


//=======================================================================
//   class GridBoxManager::Grid_
//=======================================================================


class GridBoxManager::Grid_
{
 public:

  explicit inline           Grid_

    ( idx_t                   rnk );

  inline void               getCellRange

    ( idx_t&                  ifirst,
      idx_t&                  ilast,
      idx_t                   idim,
      const Vector&           box )    const;


 public:

  const idx_t               rank;

  idx_t                     size;
  double*                   bounds;
  double*                   scales;


 private:

  Vector                    buffer_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


GridBoxManager::Grid_::Grid_ ( idx_t r ) :

  rank    ( r ),
  size    ( 0 ),
  buffer_ ( 3 * r )

{
  bounds  = buffer_.addr ();
  scales  = bounds + (2 * r);
  buffer_ = 0.0;
}


//-----------------------------------------------------------------------
//   getCellRange
//-----------------------------------------------------------------------


inline void GridBoxManager::Grid_::getCellRange

  ( idx_t&         ifirst,
    idx_t&         ilast,
    idx_t          idim,
    const Vector&  box ) const

{
  double  x = box[idim + 0];
  double  y = box[idim + rank];

  // Return an empty cell index range if the box is empty.

  if ( x > y )
  {
    ifirst = ilast = 0;
  }
  else
  {
    x -= bounds[idim];
    y -= bounds[idim];

    ifirst = (idx_t) (x * scales[idim]);
    ilast  = (idx_t) (y * scales[idim]) + 1;

    if      ( ifirst < 0 )
    {
      ifirst = 0;
    }
    else if ( ifirst >= size )
    {
      ifirst = size - 1;
    }

    if      ( ilast < ifirst )
    {
      ilast = ifirst;
    }
    else if ( ilast > size )
    {
      ilast = size;
    }
  }
}


//=======================================================================
//   class GridBoxManager::Utils_
//=======================================================================


class GridBoxManager::Utils_
{
 public:

  static const int          MAX_RANK = 32;


  static idx_t              findCells

    ( idx_t*                  icells,
      const Vector&           box,
      const Grid_&            grid );

  static idx_t              findCells1D

    ( idx_t*                  icells,
      const Vector&           box,
      const Grid_&            grid );

  static idx_t              findCells2D

    ( idx_t*                  icells,
      const Vector&           box,
      const Grid_&            grid );

  static idx_t              findCells3D

    ( idx_t*                  icells,
      const Vector&           box,
      const Grid_&            grid );

};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


#ifndef _MSC_EXTENSIONS
const int  GridBoxManager::Utils_::MAX_RANK;
#endif


//-----------------------------------------------------------------------
//   findCells
//-----------------------------------------------------------------------


idx_t GridBoxManager::Utils_::findCells

  ( idx_t*         icells,
    const Vector&  box,
    const Grid_&   grid )

{
  const idx_t  r = grid.rank;

  idx_t  lbounds [MAX_RANK];
  idx_t  ubounds [MAX_RANK];
  idx_t  index   [MAX_RANK];

  idx_t  cellCount;
  idx_t  stride;
  idx_t  icell;


  cellCount = 1;
  stride    = 1;
  icell     = 0;

  for ( idx_t i = 0; i < r; i++ )
  {
    grid.getCellRange ( lbounds[i], ubounds[i], i, box );

    index[i]   = lbounds[i];
    icell     += lbounds[i] * stride;
    stride    *= grid.size;
    cellCount *= (ubounds[i] - lbounds[i]);
  }

  if ( cellCount == 1 )
  {
    icells[0] = icell;
    return 1;
  }

  for ( idx_t i = 0; i < cellCount; i++ )
  {
    icells[i] = icell++;

    index[0]++;

    if ( index[0] >= ubounds[0] )
    {
      icell   -= (ubounds[0] - lbounds[0]);
      index[0] = lbounds[0];
      stride   = grid.size;

      for ( idx_t j = 1; j < r; j++ )
      {
        index[j]++;
        icell += stride;

        if ( index[j] < ubounds[j] )
        {
          break;
        }
        else
        {
          icell   -= stride * (ubounds[j] - lbounds[j]);
          index[j] = lbounds[j];
          stride  *= grid.size;
        }
      }
    }
  }

  return cellCount;
}


//-----------------------------------------------------------------------
//   findCells1D
//-----------------------------------------------------------------------


idx_t GridBoxManager::Utils_::findCells1D

  ( idx_t*         icells,
    const Vector&  box,
    const Grid_&   grid )

{
  idx_t  cellCount;
  idx_t  lbound;
  idx_t  ubound;


  grid.getCellRange ( lbound, ubound, 0, box );

  cellCount = ubound - lbound;

  for ( idx_t i = 0; i < cellCount; i++ )
  {
    icells[i] = i + lbound;
  }

  return cellCount;
}


//-----------------------------------------------------------------------
//   findCells2D
//-----------------------------------------------------------------------


idx_t GridBoxManager::Utils_::findCells2D

  ( idx_t*           icells,
    const Vector&  box,
    const Grid_&   grid )

{
  const idx_t gsz = grid.size;

  idx_t  lbounds[2];
  idx_t  ubounds[2];

  idx_t  cellCount;
  idx_t  icell;
  idx_t  ilast;
  idx_t  i, j, m, n;


  grid.getCellRange ( lbounds[0], ubounds[0], 0, box );
  grid.getCellRange ( lbounds[1], ubounds[1], 1, box );

  m = ubounds[0] - lbounds[0];
  n = ubounds[1] - lbounds[1];

  cellCount = m * n;
  icell     = lbounds[0] + gsz * lbounds[1];

  if ( cellCount == 1 )
  {
    icells[0] = icell;
    return cellCount;
  }

  ilast = 0;

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      icells[ilast++] = icell++;
    }

    icell = icell - m + gsz;
  }

  return cellCount;
}


//-----------------------------------------------------------------------
//   findCells3D
//-----------------------------------------------------------------------


idx_t GridBoxManager::Utils_::findCells3D

  ( idx_t*         icells,
    const Vector&  box,
    const Grid_&   grid )

{
  const idx_t gsz  = grid.size;
  const idx_t gsz2 = gsz * gsz;

  idx_t  lbounds[3];
  idx_t  ubounds[3];

  idx_t  cellCount;
  idx_t  icell;
  idx_t  ilast;
  idx_t  i, j, k, m, n, p;


  grid.getCellRange ( lbounds[0], ubounds[0], 0, box );
  grid.getCellRange ( lbounds[1], ubounds[1], 1, box );
  grid.getCellRange ( lbounds[2], ubounds[2], 2, box );

  m = ubounds[0] - lbounds[0];
  n = ubounds[1] - lbounds[1];
  p = ubounds[2] - lbounds[2];

  cellCount = m * n * p;
  icell     = lbounds[0] + gsz * lbounds[1] + gsz2 * lbounds[2];

  if ( cellCount == 1 )
  {
    icells[0] = icell;
    return 1;
  }

  ilast = 0;

  for ( k = 0; k < p; k++ )
  {
    for ( j = 0; j < n; j++ )
    {
      for ( i = 0; i < m; i++ )
      {
        icells[ilast++] = icell++;
      }

      icell = icell - m + gsz;
    }

    icell = icell - n * gsz + gsz2;
  }

  return cellCount;
}


//=======================================================================
//   class GridBoxManager::Scratch_
//=======================================================================


class GridBoxManager::Scratch_
{
 public:

  BoolVector                mask;
  IdxVector                 icells;
  IdxVector                 iboxes;
  IdxVector                 jboxes;

};


//=======================================================================
//   class GridBoxManager::Data_
//=======================================================================


class GridBoxManager::Data_ : public Collectable
{
 public:

  typedef idx_t           (*FindCellsFunc)

    ( idx_t*                  icells,
      const Vector&           box,
      const Grid_&            grid );


  inline                    Data_

    ( idx_t                   rank,
      idx_t                   boxesPerCell,
      FindCellsFunc           findCellsFunc );

  void                      update          ();

  void                      updateSome

    ( const IdxVector&        iboxes );

  inline bool               isUpToDate      () const;
  void                      invalidate      ();


 public:

  BoxArray                  boxes;

  const idx_t               boxesPerCell;
  const FindCellsFunc       findCells;

  IListArray                boxBoxMap;
  IListArray                boxCellMap;
  IListArray                cellBoxMap;

  Ref<MaskMatrix>           maskMatrix;

  Grid_                     grid;
  Scratch_                  scratch;


 private:

  void                      updateAll_      ();

  void                      updateSome_

    ( const IdxVector&        iboxes );

  void                      updateGrid_     ();

  inline void               setZLevel_

    ( int                     zlevel );

  void                      syncBoxCount_   ();


 private:

  idx_t                     updated_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline GridBoxManager::Data_::Data_

  ( idx_t          r,
    idx_t          n,
    FindCellsFunc  f ) :

  boxes        (  r ),
  boxesPerCell (  n ),
  findCells    (  f ),
  grid         (  r ),
  updated_     ( -1 )

{}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void GridBoxManager::Data_::update ()
{
  const idx_t  boxCount = boxes.size ();

  idx_t        todo;


  updateGrid_   ();
  syncBoxCount_ ();

  todo = boxCount - updated_;

  if      ( todo >= boxCount / 4 )
  {
    updateAll_ ();
  }
  else if ( todo > 0 )
  {
    IdxVector  iboxes ( todo );

    for ( idx_t i = 0; i < todo; i++ )
    {
      iboxes[i] = i + updated_;
    }

    updateSome_ ( iboxes );

    updated_ = boxCount;
  }
}


//-----------------------------------------------------------------------
//   updateSome
//-----------------------------------------------------------------------


void GridBoxManager::Data_::updateSome ( const IdxVector& iboxes )
{
  using jem::count;

  const idx_t  boxCount  = boxes .size ();
  const idx_t  iboxCount = iboxes.size ();

  IdxVector    jboxes;
  idx_t        todo;


  updateGrid_   ();
  syncBoxCount_ ();

  todo = boxCount - updated_;

  if ( todo >= boxCount / 4 )
  {
    invalidate ();

    return;
  }

  todo = count ( iboxes < updated_ );

  if ( todo >= boxCount / 4 )
  {
    invalidate ();

    return;
  }

  if ( todo == iboxCount )
  {
    jboxes.ref ( iboxes );
  }
  else
  {
    idx_t  ibox;
    idx_t  i, j;

    jboxes.resize ( todo );

    for ( i = j = 0; i < iboxCount; i++ )
    {
      ibox = iboxes[i];

      if ( ibox < updated_ )
      {
        jboxes[j++] = ibox;
      }
    }
  }

  updateSome_ ( jboxes );
}


//-----------------------------------------------------------------------
//   isUpToDate
//-----------------------------------------------------------------------


inline bool GridBoxManager::Data_::isUpToDate () const
{
  return (updated_ == boxes.size());
}


//-----------------------------------------------------------------------
//   invalidate
//-----------------------------------------------------------------------


void GridBoxManager::Data_::invalidate ()
{
  if ( updated_ >= 0 )
  {
    boxBoxMap .clear ();
    boxCellMap.clear ();
    cellBoxMap.clear ();

    updated_ = -1;
  }
}


//-----------------------------------------------------------------------
//   updateAll_
//-----------------------------------------------------------------------


void GridBoxManager::Data_::updateAll_ ()
{
  const idx_t   boxCount = boxes.size ();

  Vector        box      ( 2 * grid.rank );

  bool*         mask     = scratch.mask  .addr ();
  idx_t*        icells   = scratch.icells.addr ();
  idx_t*        iboxes   = scratch.iboxes.addr ();
  idx_t*        jboxes   = scratch.jboxes.addr ();

  const idx_t*  ilist;

  idx_t         icell;
  idx_t         ibox;
  idx_t         jbox;
  idx_t         ip;

  idx_t         i, j, k, n;


  boxes.checkBoxes ( CLASS_NAME );

  boxBoxMap .clear ();
  boxCellMap.clear ();
  cellBoxMap.clear ();

  setZLevel_ ( 1 );

  for ( ibox = 0; ibox < boxCount; ibox++ )
  {
    boxes.getBox  ( box, ibox );

    k = findCells ( icells, box, grid );

    // Skip empty boxes.

    if ( k == 0 )
    {
      continue;
    }

    j = 0;

    for ( i = 0; i < k; i++ )
    {
      icell = icells[i];
      ilist = cellBoxMap.getList ( n, icell );

      for ( ip = 0; ip < n; ip++ )
      {
        jbox = ilist[ip];

        if ( ! mask[jbox] )
        {
          mask [jbox] = true;
          jboxes[j++] = jbox;
        }
      }

      cellBoxMap.addToList ( icell, ibox  );
      boxCellMap.addToList ( ibox,  icell );
    }

JEM_IVDEP

    for ( i = 0; i < j; i++ )
    {
      mask[jboxes[i]] = false;
    }

    if ( ! maskMatrix )
    {
      k = boxes.findBoxNeighbors ( iboxes, box, jboxes, j );
    }
    else
    {
      k = boxes.findBoxNeighbors ( iboxes, box,
                                   boxes.getColor ( ibox ),
                                   * maskMatrix,
                                   jboxes, j );
    }

    for ( i = 0; i < k; i++ )
    {
      jbox = iboxes[i];

      boxBoxMap.addToList ( ibox, jbox );
      boxBoxMap.addToList ( jbox, ibox );
    }
  }

  updated_ = boxCount;

  setZLevel_ ( 2 );

  boxBoxMap .compress ();
  boxCellMap.compress ();
  cellBoxMap.compress ();
}


//-----------------------------------------------------------------------
//   updateSome_
//-----------------------------------------------------------------------


void GridBoxManager::Data_::updateSome_ ( const IdxVector& iboxes )
{
  const idx_t   iboxCount = iboxes.size ();

  bool*         mask      = scratch.mask  .addr ();
  idx_t*        icells    = scratch.icells.addr ();
  idx_t*        jboxes    = scratch.iboxes.addr ();
  idx_t*        kboxes    = scratch.jboxes.addr ();

  Vector        box       ( 2 * grid.rank );

  const idx_t*  ilist;

  idx_t         jboxCount;
  idx_t         cellCount;
  idx_t         icell;
  idx_t         ibox;
  idx_t         jbox;
  idx_t         ip;

  idx_t         i, j, n;


  // Remove all modified boxes from the box -> cell map and
  // from the cell -> box map.

  cellCount = 0;

  for ( i = 0; i < iboxCount; i++ )
  {
    ibox  = iboxes[i];
    ilist = boxCellMap.getList ( n, ibox );

    for ( ip = 0; ip < n; ip++ )
    {
      icell = ilist[ip];

      if ( ! mask[icell] )
      {
        mask  [icell]       = true;
        icells[cellCount++] = icell;
      }
    }
  }

JEM_IVDEP

  for ( i = 0; i < cellCount; i++ )
  {
    mask[icells[i]] = false;
  }

  for ( i = 0; i < iboxCount; i++ )
  {
    ibox       = iboxes[i];
    mask[ibox] = true;

    boxCellMap.clearList ( ibox );
  }

  for ( i = 0; i < cellCount; i++ )
  {
    cellBoxMap.pruneList ( icells[i], mask );
  }

  // Remove all modified boxes from the box -> box map.

  jboxCount = 0;

  for ( i = 0; i < iboxCount; i++ )
  {
    ibox  = iboxes[i];
    ilist = boxBoxMap.getList ( n, ibox );

    for ( ip = 0; ip < n; ip++ )
    {
      jbox = ilist[ip];

      if ( ! mask[jbox] )
      {
        mask  [jbox]        = true;
        jboxes[jboxCount++] = jbox;
      }
    }

    boxBoxMap.clearList ( ibox );
  }

JEM_IVDEP

  for ( j = 0; j < jboxCount; j++ )
  {
    mask[jboxes[j]] = false;
  }

  for ( j = 0; j < jboxCount; j++ )
  {
    boxBoxMap.pruneList ( jboxes[j], mask );
  }

JEM_IVDEP

  for ( i = 0; i < iboxCount; i++ )
  {
    mask[iboxes[i]] = false;
  }

  // Add the modified boxes to the cell -> box map, the
  // box -> cell map, and the box -> box map.

  for ( i = 0; i < iboxCount; i++ )
  {
    ibox = iboxes[i];

    boxes.getBox ( box, ibox );

    cellCount = findCells ( icells, box, grid );

    // Skip empty boxes.

    if ( cellCount == 0 )
    {
      continue;
    }

    jboxCount = 0;

    for ( j = 0; j < cellCount; j++ )
    {
      icell = icells[j];
      ilist = cellBoxMap.getList ( n, icell );

      for ( ip = 0; ip < n; ip++ )
      {
        jbox = ilist[ip];

        if ( ! mask[jbox] )
        {
          mask  [jbox]        = true;
          jboxes[jboxCount++] = jbox;
        }
      }

      cellBoxMap.addToList ( icell, ibox  );
      boxCellMap.addToList ( ibox,  icell );
    }

JEM_IVDEP

    for ( j = 0; j < jboxCount; j++ )
    {
      mask[jboxes[j]] = false;
    }

    if ( ! maskMatrix )
    {
      jboxCount = boxes.findBoxNeighbors ( kboxes, box,
                                           jboxes, jboxCount );
    }
    else
    {
      jboxCount = boxes.findBoxNeighbors ( kboxes, box,
                                           boxes.getColor ( ibox ),
                                           * maskMatrix,
                                           jboxes, jboxCount );
    }

    for ( j = 0; j < jboxCount; j++ )
    {
      jbox = kboxes[j];

      boxBoxMap.addToList ( ibox, jbox );
      boxBoxMap.addToList ( jbox, ibox );
    }
  }
}


//-----------------------------------------------------------------------
//   updateGrid_
//-----------------------------------------------------------------------


void GridBoxManager::Data_::updateGrid_ ()
{
  const idx_t  boxCount = boxes.size  ();
  const idx_t  r        = boxes.rank  ();
  const idx_t  r2       = boxes.rank2 ();

  Vector       b ( r2 );

  double       dx;
  double       dy;

  idx_t        cellCount;
  idx_t        i, k, n;


  k = jem::max ( 1_idx, boxCount / boxesPerCell );
  n = jem::max ( 1_idx, (idx_t)
                 std::pow( (double) k, 1.0 / (double) r ) );

  if ( n != grid.size )
  {
    invalidate ();

    // Update the number of cells.

    grid.size = n;
    cellCount = 1;

    for ( i = 0; i < r; i++ )
    {
      cellCount *= grid.size;
    }

    cellBoxMap.resize ( cellCount );

    if ( cellCount != scratch.icells.size() )
    {
      scratch.icells.resize ( cellCount );
    }

    if ( cellCount > scratch.mask.size() )
    {
      scratch.mask.resize ( cellCount );

      scratch.mask = false;
    }
  }

  boxes.getEnclosingBox ( b );

  // Check if the grid bounds must be updated.

  for ( i = 0; i < r; i++ )
  {
    // Check if the enclosing box is empty.

    if ( b[i] > b[i + r] )
    {
      b[i] = b[i + r] = 0.0;
    }

    if ( b[i + 0] < grid.bounds[i + 0] ||
         b[i + r] > grid.bounds[i + r] )
    {
      break;
    }

    dx =           b[i + r] -           b[i + 0];
    dy = grid.bounds[i + r] - grid.bounds[i + 0];

    if ( dx < 0.25 * dy )
    {
      break;
    }
  }

  if ( i < r )
  {
    invalidate ();

    // The grid bounds must be updated.

    for ( i = 0; i < r; i++ )
    {
      dx        = b[i + r] - b[i + 0];
      b[i + 0] -= 0.25 * dx;
      b[i + r] += 0.25 * dx;
      dx        = b[i + r] - b[i + 0];

      if ( dx > 0.0 )
      {
        grid.scales[i] = (double) grid.size * 1.0 / dx;
      }
      else
      {
        grid.scales[i] = 0.0;
      }

      grid.bounds[i + 0] = b[i + 0];
      grid.bounds[i + r] = b[i + r];
    }

    print ( JIVE_LOG_DEBUG, CLASS_NAME,
            " : new grid bounds: ", b, "\n" );
  }
}


//-----------------------------------------------------------------------
//   setZLevel_
//-----------------------------------------------------------------------


inline void GridBoxManager::Data_::setZLevel_ ( int zlevel )
{
  boxBoxMap .setZLevel ( zlevel );
  boxCellMap.setZLevel ( zlevel );
  cellBoxMap.setZLevel ( zlevel );
}


//-----------------------------------------------------------------------
//   syncBoxCount_
//-----------------------------------------------------------------------


void GridBoxManager::Data_::syncBoxCount_ ()
{
  const idx_t  cellCount = cellBoxMap.size ();
  const idx_t  boxCount  = boxes     .size ();

  if ( boxCount != boxBoxMap.size() )
  {
    Scratch_&  s = scratch;

    boxBoxMap .resize ( boxCount );
    boxCellMap.resize ( boxCount );

    s.iboxes  .resize ( boxCount );
    s.jboxes  .resize ( boxCount );
    s.mask    .resize ( jem::max( boxCount, cellCount ) );

    s.mask = false;
  }
}


//=======================================================================
//   class GridBoxManager
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


GridBoxManager::GridBoxManager ()
{}


GridBoxManager::GridBoxManager

  ( idx_t  rank,
    idx_t  boxesPerCell )

{
  JEM_PRECHECK2 ( rank > 0,
                  "invalid rank" );
  JEM_PRECHECK2 ( boxesPerCell > 0,
                  "invalid number of boxes per cell" );

  if ( rank > (idx_t) Utils_::MAX_RANK )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      String::format ( "rank too large; maximum value is %d",
                       Utils_::MAX_RANK )
    );
  }

  init_ ( rank, boxesPerCell );
}


GridBoxManager::~GridBoxManager ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void GridBoxManager::readFrom ( ObjectInput& in )
{
  uidx_t  r, n;

  decode ( in, r, n );

  init_  ( (idx_t) r, (idx_t) n );
  decode ( in, data_->boxes, data_->maskMatrix );

  data_->invalidate ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void GridBoxManager::writeTo ( ObjectOutput& out ) const
{
  const Data_& d = * data_;

  encode ( out, d.grid.rank, d.boxesPerCell,
           d.boxes, d.maskMatrix );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t GridBoxManager::size () const
{
  return data_->boxes.size ();
}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t GridBoxManager::rank () const
{
  return data_->boxes.rank ();
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void GridBoxManager::clear ()
{
  Data_&  d = * data_;

  d.boxes.clear ();
  d.invalidate  ();
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void GridBoxManager::reserve ( idx_t n )
{
  data_->boxes.reserve ( n );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void GridBoxManager::trimToSize ()
{
  Data_&  d = * data_;

  d.update ();

  d.boxes     .trimToSize ();
  d.boxBoxMap .compress   ();
  d.boxCellMap.compress   ();
  d.cellBoxMap.compress   ();
}


//-----------------------------------------------------------------------
//   flushChanges
//-----------------------------------------------------------------------


void GridBoxManager::flushChanges ()
{
  if ( ! data_->isUpToDate() )
  {
    data_->update ();
  }
}


//-----------------------------------------------------------------------
//   setMaskMatrix
//-----------------------------------------------------------------------


void GridBoxManager::setMaskMatrix ( Ref<MaskMatrix> mask )
{
  JEM_PRECHECK2 ( mask, "NULL MaskMatrix" );

  data_->maskMatrix = mask;

  data_->invalidate ();
}


//-----------------------------------------------------------------------
//   getEnclosingBox
//-----------------------------------------------------------------------


void GridBoxManager::getEnclosingBox

  ( const Vector&  box ) const

{
  data_->boxes.getEnclosingBox ( box );
}


//-----------------------------------------------------------------------
//   addBox
//-----------------------------------------------------------------------


idx_t GridBoxManager::addBox

  ( const Vector&  box,
    idx_t          color )

{
  return data_->boxes.addBox ( box, color );
}


//-----------------------------------------------------------------------
//   addBoxes
//-----------------------------------------------------------------------


idx_t GridBoxManager::addBoxes

  ( const Matrix&  boxes,
    idx_t          color )

{
  return data_->boxes.addBoxes ( boxes, color );
}


//-----------------------------------------------------------------------
//   reorderBoxes
//-----------------------------------------------------------------------


void GridBoxManager::reorderBoxes ( const Reordering& reord )
{
  data_->invalidate         ();
  data_->boxes.reorderBoxes ( reord );
}


//-----------------------------------------------------------------------
//   getBox
//-----------------------------------------------------------------------


void GridBoxManager::getBox

  ( const Vector&  box,
    idx_t          ibox ) const

{
  data_->boxes.getBox ( box, ibox );
}


//-----------------------------------------------------------------------
//   getBoxes
//-----------------------------------------------------------------------


void GridBoxManager::getBoxes ( const Matrix& boxes ) const

{
  data_->boxes.getBoxes ( boxes );
}


//-----------------------------------------------------------------------
//   getSomeBoxes
//-----------------------------------------------------------------------


void GridBoxManager::getSomeBoxes

  ( const Matrix&     boxes,
    const IdxVector&  iboxes ) const

{
  data_->boxes.getSomeBoxes ( boxes, iboxes );
}


//-----------------------------------------------------------------------
//   setColors
//-----------------------------------------------------------------------


void GridBoxManager::setColors ( const IdxVector& colors )
{
  data_->invalidate      ();
  data_->boxes.setColors ( colors );
}


//-----------------------------------------------------------------------
//   setSomeColors
//-----------------------------------------------------------------------


void GridBoxManager::setSomeColors

  ( const IdxVector&  iboxes,
    const IdxVector&  colors )

{
  Data_&  d = * data_;

  d.boxes.setSomeColors ( iboxes, colors );
  d.updateSome          ( iboxes );
}


//-----------------------------------------------------------------------
//   updateBoxes
//-----------------------------------------------------------------------


void GridBoxManager::updateBoxes ( const Matrix& boxes )
{
  data_->invalidate        ();
  data_->boxes.updateBoxes ( boxes );
}


//-----------------------------------------------------------------------
//   updateSomeBoxes
//-----------------------------------------------------------------------


void GridBoxManager::updateSomeBoxes

  ( const IdxVector&  iboxes,
    const Matrix&     boxes )

{
  Data_&  d = * data_;

  d.boxes.updateSomeBoxes ( iboxes, boxes );
  d.updateSome            ( iboxes );
}


//-----------------------------------------------------------------------
//   findChangedBoxes
//-----------------------------------------------------------------------


idx_t GridBoxManager::findChangedBoxes

  ( const IdxVector&  iboxes,
    const Matrix&     boxes ) const

{
  return data_->boxes.findChangedBoxes ( iboxes, boxes );
}


//-----------------------------------------------------------------------
//   findBoxNeighbors (given a box index)
//-----------------------------------------------------------------------


idx_t GridBoxManager::findBoxNeighbors

  ( const IdxVector&  iboxes,
    idx_t             jbox ) const

{
  const idx_t*  JEM_RESTRICT  ilist;

  Data_&        d = * data_;

  idx_t         i, n;


  d.boxes.checkIndex ( jbox );

  if ( ! d.isUpToDate() )
  {
    d.update ();
  }

  ilist = d.boxBoxMap.getList ( n, jbox );

  if ( iboxes.size() < n )
  {
    n = iboxes.size ();
  }

  if ( iboxes.isContiguous() )
  {
    idx_t* JEM_RESTRICT  ib = iboxes.addr ();

    if      ( n == 1_idx )
    {
      ib[0] = ilist[0];
    }
    else if ( n == 2_idx )
    {
      ib[0] = ilist[0];
      ib[1] = ilist[1];
    }
    else if ( n == 3_idx )
    {
      ib[0] = ilist[0];
      ib[1] = ilist[1];
      ib[2] = ilist[2];
    }
    else if ( n == 4_idx )
    {
      ib[0] = ilist[0];
      ib[1] = ilist[1];
      ib[2] = ilist[2];
      ib[3] = ilist[3];
    }
    else if ( n >= 5_idx )
    {
      std::memcpy ( ib, ilist, (size_t) n * sizeof(idx_t) );
    }
  }
  else
  {
    for ( i = 0; i < n; i++ )
    {
      iboxes[i] = ilist[i];
    }
  }

  return n;
}


//-----------------------------------------------------------------------
//   findBoxNeighbors (given a box)
//-----------------------------------------------------------------------


idx_t GridBoxManager::findBoxNeighbors

  ( const IdxVector&  iboxes,
    const Vector&     box,
    idx_t               color ) const

{
  Data_&        d = * data_;

  bool*         mask;
  idx_t*        icells;
  idx_t*        jboxes;

  const idx_t*  ilist;

  idx_t         icellCount;
  idx_t         jlast;
  idx_t         jbox;

  idx_t         i, j, n;


  // Note: this may change the addresses of the scratch arrays

  if ( ! d.isUpToDate() )
  {
    d.update ();
  }

  mask       = d.scratch.mask  .addr ();
  icells     = d.scratch.icells.addr ();
  jboxes     = d.scratch.jboxes.addr ();

  icellCount = d.findCells ( icells, box, d.grid );

  // An empty box has no neighbors.

  if ( icellCount == 0 )
  {
    return 0;
  }

  jlast = 0;

  for ( i = 0; i < icellCount; i++ )
  {
    ilist = d.cellBoxMap.getList ( n, icells[i] );

    for ( j = 0; j < n; j++ )
    {
      jbox = ilist[j];

      if ( ! mask[jbox] )
      {
        mask  [jbox]    = true;
        jboxes[jlast++] = jbox;
      }
    }
  }

JEM_IVDEP

  for ( j = 0; j < jlast; j++ )
  {
    mask[jboxes[j]] = false;
  }

  if ( iboxes.isContiguous() && iboxes.size() >= d.boxes.size() )
  {
    if ( ! d.maskMatrix )
    {
      n = d.boxes.findBoxNeighbors ( iboxes.addr(), box,
                                     jboxes, jlast );
    }
    else
    {
      n = d.boxes.findBoxNeighbors ( iboxes.addr(), box,
                                     color,
                                     * d.maskMatrix,
                                     jboxes, jlast );
    }
  }
  else
  {
    idx_t*  ib = d.scratch.iboxes.addr ();

    if ( ! d.maskMatrix )
    {
      n = d.boxes.findBoxNeighbors ( ib, box,
                                     jboxes, jlast );
    }
    else
    {
      n = d.boxes.findBoxNeighbors ( ib, box,
                                     color,
                                     * d.maskMatrix,
                                     jboxes, jlast );
    }

    if ( iboxes.size() < n )
    {
      n = iboxes.size ();
    }

    for ( i = 0; i < n; i++ )
    {
      iboxes[i] = ib[i];
    }
  }

  return n;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void GridBoxManager::init_

  ( idx_t  rank,
    idx_t  boxesPerCell )

{
  Data_::FindCellsFunc  func;


  if      ( rank == 1_idx )
  {
    func = & Utils_::findCells1D;
  }
  else if ( rank == 2_idx )
  {
    func = & Utils_::findCells2D;
  }
  else if ( rank == 3_idx )
  {
    func = & Utils_::findCells3D;
  }
  else
  {
    func = & Utils_::findCells;
  }

  data_ = newInstance<Data_> ( rank, boxesPerCell, func );
}


JIVE_END_PACKAGE( geom )
