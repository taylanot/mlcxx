
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
#include <jem/pragmas.h>
#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/xutil/IListArray.h>
#include <jive/geom/MaskMatrix.h>
#include <jive/geom/BoxArray.h>
#include <jive/geom/SimpleBoxManager.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::SimpleBoxManager );


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;
using jem::xutil::IListArray;


//=======================================================================
//   class SimpleBoxManager::Data_
//=======================================================================


class SimpleBoxManager::Data_ : public jem::Collectable
{
 public:

  explicit inline         Data_

    ( idx_t                 rank );

  void                    update        ();

  void                    updateSome

    ( const IdxVector&      iboxes );

  inline bool             isUpToDate    () const;
  void                    invalidate    ();


 public:

  BoxArray                boxes;
  IListArray              boxBoxMap;
  Ref<MaskMatrix>         maskMatrix;


 private:

  void                    updateAll_    ();

  void                    updateSome_

    ( const IdxVector&      iboxes );


 private:

  idx_t                   updated_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline SimpleBoxManager::Data_::Data_ ( idx_t rank ) :

  boxes ( rank )

{
  updated_ = -1;
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void SimpleBoxManager::Data_::update ()
{
  const idx_t  boxCount = boxes.size ();

  idx_t        todo;


  if ( boxCount != boxBoxMap.size() )
  {
    boxBoxMap.resize ( boxCount );
  }

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


void SimpleBoxManager::Data_::updateSome ( const IdxVector& iboxes )
{
  using jem::count;

  const idx_t  boxCount  = boxes .size ();
  const idx_t  iboxCount = iboxes.size ();

  IdxVector    jboxes;
  idx_t        todo;


  if ( boxCount != boxBoxMap.size() )
  {
    boxBoxMap.resize ( boxCount );
  }

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


inline bool SimpleBoxManager::Data_::isUpToDate () const
{
  return (updated_ == boxes.size());
}


//-----------------------------------------------------------------------
//   invalidate
//-----------------------------------------------------------------------


void SimpleBoxManager::Data_::invalidate ()
{
  if ( updated_ >= 0 )
  {
    boxBoxMap.clear ();

    updated_ = -1;
  }
}


//-----------------------------------------------------------------------
//   updateAll_
//-----------------------------------------------------------------------


void SimpleBoxManager::Data_::updateAll_ ()
{
  const idx_t  boxCount = boxes.size ();
  const idx_t  rank     = boxes.rank ();

  Vector       box      ( 2 * rank     );
  IdxVector    ibuf     ( 2 * boxCount );

  idx_t*       iboxes   = ibuf.addr ();
  idx_t*       jboxes   = iboxes + boxCount;

  idx_t        ibox;
  idx_t        jbox;
  idx_t        j, k;


  boxes.checkBoxes ( CLASS_NAME );

  boxBoxMap.clear     ();
  boxBoxMap.setZLevel ( 1 );

  for ( ibox = 0; ibox < boxCount; ibox++ )
  {
    iboxes[ibox] = ibox;
  }

  for ( ibox = 0; ibox < boxCount; ibox++ )
  {
    boxes.getBox ( box, ibox );

    if ( ! maskMatrix )
    {
      k = boxes.findBoxNeighbors ( jboxes, box,
                                   iboxes   + (ibox + 1),
                                   boxCount - (ibox + 1) );
    }
    else
    {
      k = boxes.findBoxNeighbors ( jboxes, box,
                                   boxes.getColor ( ibox ),
                                   * maskMatrix,
                                   iboxes   + (ibox + 1),
                                   boxCount - (ibox + 1) );
    }

    for ( j = 0; j < k; j++ )
    {
      jbox = jboxes[j];

      boxBoxMap.addToList ( ibox, jbox );
      boxBoxMap.addToList ( jbox, ibox );
    }
  }

  boxBoxMap.setZLevel ( 2 );
  boxBoxMap.compress  ();

  updated_ = boxCount;
}


//-----------------------------------------------------------------------
//   updateSome_
//-----------------------------------------------------------------------


void SimpleBoxManager::Data_::updateSome_ ( const IdxVector& iboxes )
{
  const idx_t   boxCount  = boxes.size  ();
  const idx_t   rank      = boxes.rank  ();
  const idx_t   iboxCount = iboxes.size ();

  Vector        box       ( 2 * rank     );
  IdxVector     ibuf      ( 2 * boxCount );
  BoolVector    mask      (     boxCount );

  idx_t*        jboxes    = ibuf.addr ();
  idx_t*        kboxes    = jboxes + boxCount;

  const idx_t*  ilist;

  idx_t         jboxCount;
  idx_t         ibox;
  idx_t         jbox;
  idx_t         i, j, n;


  // Remove all modified boxes from the box -> box map.

  jboxCount = 0;
  mask      = false;

  for ( i = 0; i < iboxCount; i++ )
  {
    mask[iboxes[i]] = true;
  }

  for ( i = 0; i < iboxCount; i++ )
  {
    ilist = boxBoxMap.getList ( n, iboxes[i] );

    for ( j = 0; j < n; j++ )
    {
      jbox = ilist[j];

      if ( ! mask[jbox] )
      {
        mask  [jbox]        = true;
        jboxes[jboxCount++] = jbox;
      }
    }
  }

JEM_IVDEP

  for ( j = 0; j < jboxCount; j++ )
  {
    mask[jboxes[j]] = false;
  }

  for ( j = 0; j < jboxCount; j++ )
  {
    boxBoxMap.pruneList ( jboxes[j], mask.addr() );
  }

  for ( i = 0; i < iboxCount; i++ )
  {
    ibox       = iboxes[i];
    mask[ibox] = false;

    boxBoxMap.clearList ( ibox );
  }

  for ( ibox = 0; ibox < boxCount; ibox++ )
  {
    kboxes[ibox] = ibox;
  }

  for ( i = 0; i < iboxCount; i++ )
  {
    ibox       = iboxes[i];
    mask[ibox] = true;

    boxes.getBox ( box, ibox );

    if ( ! maskMatrix )
    {
      jboxCount = boxes.findBoxNeighbors ( jboxes, box,
                                           kboxes, boxCount );
    }
    else
    {
      jboxCount = boxes.findBoxNeighbors ( jboxes, box,
                                           boxes.getColor ( ibox ),
                                           * maskMatrix,
                                           kboxes, boxCount );
    }

    for ( j = 0; j < jboxCount; j++ )
    {
      jbox = jboxes[j];

      if ( ! mask[jbox] )
      {
        boxBoxMap.addToList ( ibox, jbox );
        boxBoxMap.addToList ( jbox, ibox );
      }
    }
  }
}


//=======================================================================
//   class SimpleBoxManager
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SimpleBoxManager::SimpleBoxManager ()
{}


SimpleBoxManager::SimpleBoxManager ( idx_t rank )

{
  JEM_PRECHECK2 ( rank > 0, "invalid rank" );

  data_ = newInstance<Data_> ( rank );
}


SimpleBoxManager::~SimpleBoxManager ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void SimpleBoxManager::readFrom ( ObjectInput& in )
{
  idx_t  r;

  decode ( in, r );

  if ( r <= 0 )
  {
    jem::io::decodeError ( JEM_FUNC );
  }

  data_ = newInstance<Data_> ( r );

  decode ( in, data_->boxes, data_->maskMatrix );

  data_->invalidate ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void SimpleBoxManager::writeTo ( ObjectOutput& out ) const
{
  const Data_& d = * data_;

  encode ( out, d.boxes.rank(), d.boxes, d.maskMatrix );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t SimpleBoxManager::size () const
{
  return data_->boxes.size ();
}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t SimpleBoxManager::rank () const
{
  return data_->boxes.rank ();
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void SimpleBoxManager::clear ()
{
  Data_&  d = * data_;

  d.boxes.clear ();
  d.invalidate  ();
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void SimpleBoxManager::reserve ( idx_t n )
{
  data_->boxes.reserve ( n );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void SimpleBoxManager::trimToSize ()
{
  Data_& d = * data_;

  d.update ();

  d.boxes    .trimToSize ();
  d.boxBoxMap.compress   ();
}


//-----------------------------------------------------------------------
//   flushChanges
//-----------------------------------------------------------------------


void SimpleBoxManager::flushChanges ()
{
  if ( ! data_->isUpToDate() )
  {
    data_->update ();
  }
}


//-----------------------------------------------------------------------
//   setMaskMatrix
//-----------------------------------------------------------------------


void SimpleBoxManager::setMaskMatrix ( Ref<MaskMatrix> mask )
{
  JEM_PRECHECK2 ( mask, "NULL MaskMatrix" );

  data_->maskMatrix = mask;

  data_->invalidate ();
}


//-----------------------------------------------------------------------
//   getEnclosingBox
//-----------------------------------------------------------------------


void SimpleBoxManager::getEnclosingBox ( const Vector& box ) const
{
  data_->boxes.getEnclosingBox ( box );
}


//-----------------------------------------------------------------------
//   addBox
//-----------------------------------------------------------------------


idx_t SimpleBoxManager::addBox

  ( const Vector&  box,
    idx_t          color )

{
  return data_->boxes.addBox ( box, color );
}


//-----------------------------------------------------------------------
//   addBoxes
//-----------------------------------------------------------------------


idx_t SimpleBoxManager::addBoxes

  ( const Matrix&  boxes,
    idx_t          color )

{
  return data_->boxes.addBoxes ( boxes, color );
}


//-----------------------------------------------------------------------
//   reorderBoxes
//-----------------------------------------------------------------------


void SimpleBoxManager::reorderBoxes ( const Reordering& reord )
{
  data_->invalidate         ();
  data_->boxes.reorderBoxes ( reord );
}


//-----------------------------------------------------------------------
//   getBox
//-----------------------------------------------------------------------


void SimpleBoxManager::getBox

  ( const Vector&  box,
    idx_t          ibox ) const

{
  data_->boxes.getBox ( box, ibox );
}


//-----------------------------------------------------------------------
//   getBoxes
//-----------------------------------------------------------------------


void SimpleBoxManager::getBoxes

  ( const Matrix&  boxes ) const

{
  data_->boxes.getBoxes ( boxes );
}


//-----------------------------------------------------------------------
//   getSomeBoxes
//-----------------------------------------------------------------------


void SimpleBoxManager::getSomeBoxes

  ( const Matrix&     boxes,
    const IdxVector&  iboxes ) const

{
  data_->boxes.getSomeBoxes ( boxes, iboxes );
}


//-----------------------------------------------------------------------
//   setColors
//-----------------------------------------------------------------------


void SimpleBoxManager::setColors ( const IdxVector& colors )
{
  data_->invalidate      ();
  data_->boxes.setColors ( colors );
}


//-----------------------------------------------------------------------
//   setSomeColors
//-----------------------------------------------------------------------


void SimpleBoxManager::setSomeColors

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


void SimpleBoxManager::updateBoxes ( const Matrix&  boxes )
{
  data_->invalidate        ();
  data_->boxes.updateBoxes ( boxes );
}


//-----------------------------------------------------------------------
//   updateSomeBoxes
//-----------------------------------------------------------------------


void SimpleBoxManager::updateSomeBoxes

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


idx_t SimpleBoxManager::findChangedBoxes

  ( const IdxVector&  iboxes,
    const Matrix&     boxes ) const

{
  return data_->boxes.findChangedBoxes ( iboxes, boxes );
}


//-----------------------------------------------------------------------
//   findBoxNeighbors (given a box index)
//-----------------------------------------------------------------------


idx_t SimpleBoxManager::findBoxNeighbors

  ( const IdxVector&  iboxes,
    idx_t             jbox ) const

{
  JEM_PRECHECK2 ( iboxes.size() >= size(),
                  "output array is too small" );

  Data_&        d = * data_;

  const idx_t*  ilist;

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

    switch ( n )
    {
    case 0:

      break;

    case 1:

      ib[0] = ilist[0];

      break;

    case 2:

      ib[0] = ilist[0];
      ib[1] = ilist[1];

      break;

    case 3:

      ib[0] = ilist[0];
      ib[1] = ilist[1];
      ib[2] = ilist[2];

      break;

    case 4:

      ib[0] = ilist[0];
      ib[1] = ilist[1];
      ib[2] = ilist[2];
      ib[3] = ilist[3];

      break;

    default:

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


idx_t SimpleBoxManager::findBoxNeighbors

  ( const IdxVector&  iboxes,
    const Vector&     box,
    idx_t             color ) const

{
  JEM_PRECHECK2 ( iboxes.size() >= size(),
                  "output array is too small" );

  Data_&       d        = * data_;
  const idx_t  boxCount = d.boxes.size ();

  IdxVector    ibuf     ( 2 * boxCount );

  idx_t*       jboxes   = ibuf.addr ();

  idx_t        ibox;
  idx_t        i, n;


  if ( ! d.isUpToDate() )
  {
    d.update ();
  }

  for ( ibox = 0; ibox < boxCount; ibox++ )
  {
    jboxes[ibox] = ibox;
  }

  if ( iboxes.isContiguous() && iboxes.size() >= boxCount )
  {
    if ( ! d.maskMatrix )
    {
      n = d.boxes.findBoxNeighbors ( iboxes.addr(), box,
                                     jboxes, boxCount );
    }
    else
    {
      n = d.boxes.findBoxNeighbors ( iboxes.addr(), box,
                                     color,  * d.maskMatrix,
                                     jboxes, boxCount );
    }
  }
  else
  {
    idx_t* JEM_RESTRICT  ib = jboxes + boxCount;

    if ( ! d.maskMatrix )
    {
      n = d.boxes.findBoxNeighbors ( ib,     box,
                                     jboxes, boxCount );
    }
    else
    {
      n = d.boxes.findBoxNeighbors ( ib,     box,
                                     color,  * d.maskMatrix,
                                     jboxes, boxCount );
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


JIVE_END_PACKAGE( geom )
