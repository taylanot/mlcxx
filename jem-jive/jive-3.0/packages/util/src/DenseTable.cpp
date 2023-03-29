
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


#include <jem/base/assert.h>
#include <jem/base/Error.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jive/util/private/NameList.h>
#include <jive/util/error.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemMap.h>
#include <jive/util/EventMask.h>
#include <jive/util/Reordering.h>
#include <jive/util/DenseTable.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::DenseTable );


JIVE_BEGIN_PACKAGE( util )


using jem::min;
using jem::shape;
using jem::newInstance;
using jem::Array;


//=======================================================================
//   class DenseTable::Data_
//=======================================================================


class DenseTable::Data_ : public Collectable
{
 public:

  typedef Data_           Self;
  typedef Collectable     Super;

  static const int        NEW_VALUES = 1 << 0;
  static const int        NEW_STRUCT = 1 << 1;


  inline                  Data_           ();

  explicit                Data_

    ( const Ref<ItemSet>&   rows );

  inline void             readFrom

    ( ObjectInput&          in  );

  inline void             writeTo

    ( ObjectOutput&         out )            const;

  void                    resize          ();
  inline idx_t            colCount        () const;

  inline void             checkColIndex

    ( idx_t                 jcol );

  inline bool             checkColIndex

    ( idx_t                 jcol )           const;

  inline void             checkRowIndex

    ( idx_t                 irow );

  inline bool             checkRowIndex

    ( idx_t                 irow )           const;

  inline void             setValue

    ( idx_t                 irow,
      idx_t                 jcol,
      double                value );

  inline void             addValue

    ( idx_t                 irow,
      idx_t                 jcol,
      double                value );

  inline idx_t            eraseValue

    ( idx_t                 irow,
      idx_t                 jcol  );

  void                    colIndexError

    ( idx_t                 jcol )           const;

  void                    rowIndexError

    ( idx_t                 irow )           const;


 public:

  Matrix                  table;
  Array<byte,2>           mask;
  NameList                colNames;
  Ref<ItemSet>            rowItems;
  EventMask               events;

  idx_t                   entryCount;


 private:

  void                    checkRowIndex_

    ( idx_t                 irow );

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline DenseTable::Data_::Data_ ()
{}


DenseTable::Data_::Data_ ( const Ref<ItemSet>&  rows ) :

  rowItems ( rows )

{
  entryCount = 0;
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


inline void DenseTable::Data_::readFrom ( ObjectInput& in )
{
  uidx_t  count;

  decode ( in, rowItems, table, mask, colNames, count );

  entryCount = (idx_t) count;

  events.reset ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


inline void DenseTable::Data_::writeTo ( ObjectOutput& out ) const
{
  encode ( out, rowItems, table, mask, colNames, entryCount );
}


//-----------------------------------------------------------------------
//   resize
//-----------------------------------------------------------------------


void DenseTable::Data_::resize ()
{
  const idx_t  m = table.size (0);
  const idx_t  n = table.size (1);

  idx_t        p = rowItems->size ();
  idx_t        q = colNames .size ();


  if ( p != m || q != n )
  {
    Matrix         newTable ( p, q );
    Array<byte,2>  newMask  ( p, q );


    p = min ( p, m );
    q = min ( q, n );

    newTable =        0.0;
    newMask  = (byte) 0x0;

    newTable(slice(BEGIN,p),slice(BEGIN,q)) =
      table (slice(BEGIN,p),slice(BEGIN,q));

    newMask (slice(BEGIN,p),slice(BEGIN,q)) =
      mask  (slice(BEGIN,p),slice(BEGIN,q));

    table.swap ( newTable );
    mask .swap ( newMask  );
  }
}


//-----------------------------------------------------------------------
//   colCount
//-----------------------------------------------------------------------


inline idx_t DenseTable::Data_::colCount () const
{
  return colNames.size ();
}


//-----------------------------------------------------------------------
//   checkColIndex
//-----------------------------------------------------------------------


inline void DenseTable::Data_::checkColIndex ( idx_t jcol )
{
  if ( jcol < 0 || jcol >= colNames.size() )
  {
    colIndexError ( jcol );
  }

  if ( jcol >= table.size(1) )
  {
    resize ();
  }
}


inline bool DenseTable::Data_::checkColIndex ( idx_t jcol ) const
{
  if ( jcol < 0 || jcol >= colNames.size() )
  {
    colIndexError ( jcol );
  }

  return (jcol < table.size(1));
}


//-----------------------------------------------------------------------
//   checkRowIndex
//-----------------------------------------------------------------------


inline void DenseTable::Data_::checkRowIndex ( idx_t irow )
{
  if ( irow < 0 || irow >= table.size(0) )
  {
    checkRowIndex_ ( irow );
  }
}


inline bool DenseTable::Data_::checkRowIndex ( idx_t irow ) const
{
  if ( irow < 0 )
  {
    rowIndexError ( irow );
  }

  return (irow < table.size(0));
}


//-----------------------------------------------------------------------
//   setValue
//-----------------------------------------------------------------------


inline void DenseTable::Data_::setValue

  ( idx_t   irow,
    idx_t   jcol,
    double  value )

{
  table(irow,jcol) = value;

  if ( mask(irow,jcol) )
  {
    events.set ( NEW_VALUES );
  }
  else
  {
    mask(irow,jcol) = 0x1;
    entryCount++;

    events.set ( NEW_STRUCT );
  }
}


//-----------------------------------------------------------------------
//   addValue
//-----------------------------------------------------------------------


inline void DenseTable::Data_::addValue

  ( idx_t   irow,
    idx_t   jcol,
    double  value )

{
  table(irow,jcol) += value;

  if ( mask(irow,jcol) )
  {
    events.set ( NEW_VALUES );
  }
  else
  {
    mask(irow,jcol) =  0x1;
    entryCount++;

    events.set ( NEW_STRUCT );
  }
}


//-----------------------------------------------------------------------
//   eraseValue
//-----------------------------------------------------------------------


inline idx_t DenseTable::Data_::eraseValue

  ( idx_t  irow,
    idx_t  jcol  )

{
  if ( mask(irow,jcol) )
  {
    mask(irow,jcol) = false;
    entryCount--;

    events.set ( NEW_STRUCT );

    return 1;
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------
//   colIndexError
//-----------------------------------------------------------------------


void DenseTable::Data_::colIndexError ( idx_t jcol ) const
{
  indexError ( CLASS_NAME, "column", jcol, colCount() );
}


//-----------------------------------------------------------------------
//   rowIndexError
//-----------------------------------------------------------------------


void DenseTable::Data_::rowIndexError ( idx_t irow ) const
{
  indexError ( CLASS_NAME,
               rowItems->getItemName(), irow, rowItems->size() );
}


//-----------------------------------------------------------------------
//   checkRowIndex_
//-----------------------------------------------------------------------


void DenseTable::Data_::checkRowIndex_ ( idx_t irow )
{
  const idx_t  rowCount = rowItems->size ();

  if ( irow < 0 || irow >= rowCount )
  {
    rowIndexError ( irow );
  }

  if ( irow >= table.size(0) )
  {
    resize ();
  }
}


//=======================================================================
//   class DenseTable
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


DenseTable::DenseTable ()
{}


DenseTable::DenseTable

  ( const String&        name,
    const Ref<ItemSet>&  rowItems ) :

    Super ( name )

{
  data_ = newInstance<Data_> ( rowItems );

  connect_ ();
}


DenseTable::DenseTable ( const Self& rhs ) :

  Super ( rhs.myName_ )

{
  data_ = newInstance<Data_> ( * rhs.data_ );

  connect_ ();
}


DenseTable::~DenseTable ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void DenseTable::resetEvents ()
{
  data_->events   .reset       ();
  data_->rowItems->resetEvents ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void DenseTable::readFrom ( ObjectInput& in )
{
  decode ( in, myName_ );

  data_ = newInstance<Data_> ();

  data_->readFrom ( in );

  connect_ ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void DenseTable::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_ );

  data_->writeTo ( out );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> DenseTable::clone () const
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t DenseTable::size () const
{
  return data_->entryCount;
}


//-----------------------------------------------------------------------
//   columnCount
//-----------------------------------------------------------------------


idx_t DenseTable::columnCount () const
{
  return data_->colCount ();
}


//-----------------------------------------------------------------------
//   clearData
//-----------------------------------------------------------------------


void DenseTable::clearData ()
{
  Data_&  d = * data_;

  if ( d.entryCount > 0 )
  {
    d.events.reset ();

    d.table      =        0.0;
    d.mask       = (byte) 0x0;
    d.entryCount =        0;

    newStructEvent.emit ( *this );
  }
}


//-----------------------------------------------------------------------
//   clearAll
//-----------------------------------------------------------------------


void DenseTable::clearAll ()
{
  data_->colNames.clear ();

  clearData ();
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void DenseTable::trimToSize ()
{
  data_->resize ();
}


//-----------------------------------------------------------------------
//   addColumn
//-----------------------------------------------------------------------


idx_t DenseTable::addColumn ( const String& name )
{
  return data_->colNames.addName ( name );
}


//-----------------------------------------------------------------------
//   findColumn
//-----------------------------------------------------------------------


idx_t DenseTable::findColumn ( const String& name ) const
{
  return data_->colNames.findName ( name );
}


//-----------------------------------------------------------------------
//   getColumnName
//-----------------------------------------------------------------------


String DenseTable::getColumnName ( idx_t jcol ) const
{
  const Data_&  d = * data_;

  d.checkColIndex ( jcol );

  return d.colNames.getName ( jcol );
}


//-----------------------------------------------------------------------
//   getColumnNames
//-----------------------------------------------------------------------


StringVector DenseTable::getColumnNames () const
{
  return data_->colNames.toArray ();
}


//-----------------------------------------------------------------------
//   setData
//-----------------------------------------------------------------------


void DenseTable::setData

  ( const idx_t*   irows,
    idx_t          icount,
    const idx_t*   jcols,
    idx_t          jcount,
    const double*  values )

{
  Data_&  d = * data_;

  if      ( icount * jcount == 1_idx )
  {
    idx_t  irow = irows[0];
    idx_t  jcol = jcols[0];

    d.checkRowIndex ( irow );
    d.checkColIndex ( jcol );
    d.setValue      ( irow, jcol, *values );
  }
  else if ( icount == 1_idx )
  {
    idx_t  irow = irows[0];

    d.checkRowIndex ( irow );

    for ( idx_t j = 0; j < jcount; j++ )
    {
      idx_t  jcol = jcols[j];

      d.checkColIndex ( jcol );
      d.setValue      ( irow, jcol, values[j] );
    }
  }
  else if ( jcount == 1_idx )
  {
    idx_t  jcol = jcols[0];

    d.checkColIndex ( jcol );

    for ( idx_t i = 0; i < icount; i++ )
    {
      idx_t  irow = irows[i];

      d.checkRowIndex ( irow );
      d.setValue      ( irow, jcol, values[i] );
    }
  }
  else
  {
    double*  JEM_RESTRICT table;
    byte*    JEM_RESTRICT mask;


    for ( idx_t i = 0; i < icount; i++ )
    {
      d.checkRowIndex ( irows[i] );
    }

    idx_t  k = 0;

    for ( idx_t j = 0; j < jcount; j++, values += icount )
    {
      idx_t  jcol = jcols[j];

      d.checkColIndex ( jcol );

      table = d.table.addr() + (jcol * d.table.stride(1));
      mask  = d.mask .addr() + (jcol * d.mask .stride(1));

      for ( idx_t i = 0; i < icount; i++ )
      {
        idx_t  irow   = irows[i];

        table[irow] = values[i];
        k          += (idx_t) (0x1 - mask[irow]);
        mask [irow] = 0x1;
      }
    }

    d.entryCount += k;

    if ( k > 0 )
    {
      d.events.set ( Data_::NEW_STRUCT );
    }
    else
    {
      d.events.set ( Data_::NEW_VALUES );
    }
  }

  d.events.emit ( this );
}


//-----------------------------------------------------------------------
//   addData
//-----------------------------------------------------------------------


void DenseTable::addData

  ( const idx_t*   irows,
    idx_t          icount,
    const idx_t*   jcols,
    idx_t          jcount,
    const double*  values )

{
  Data_&  d = * data_;

  if      ( icount * jcount == 1_idx )
  {
    idx_t  irow = irows[0];
    idx_t  jcol = jcols[0];

    d.checkRowIndex ( irow );
    d.checkColIndex ( jcol );
    d.addValue      ( irow, jcol, *values );
  }
  else if ( icount == 1_idx )
  {
    idx_t  irow = irows[0];

    d.checkRowIndex ( irow );

    for ( idx_t j = 0; j < jcount; j++ )
    {
      idx_t  jcol = jcols[j];

      d.checkColIndex ( jcol );
      d.addValue      ( irow, jcol, values[j] );
    }
  }
  else if ( jcount == 1_idx )
  {
    idx_t  jcol = jcols[0];

    d.checkColIndex ( jcol );

    for ( idx_t i = 0; i < icount; i++ )
    {
      idx_t  irow = irows[i];

      d.checkRowIndex ( irow );
      d.addValue      ( irow, jcol, values[i] );
    }
  }
  else
  {
    double*  JEM_RESTRICT table;
    byte*    JEM_RESTRICT mask;


    for ( idx_t i = 0; i < icount; i++ )
    {
      d.checkRowIndex ( irows[i] );
    }

    idx_t  k = 0 ;

    for ( idx_t j = 0; j < jcount; j++, values += icount )
    {
      idx_t  jcol = jcols[j];

      d.checkColIndex ( jcol );

      table = d.table.addr() + (jcol * d.table.stride(1));
      mask  = d.mask .addr() + (jcol * d.mask .stride(1));

      for ( idx_t i = 0; i < icount; i++ )
      {
        idx_t  irow    = irows[i];

        table[irow] += values[i];
        k           += (idx_t) (0x1 - mask[irow]);
        mask [irow]  = 0x1;
      }
    }

    d.entryCount += k;

    if ( k > 0 )
    {
      d.events.set ( Data_::NEW_STRUCT );
    }
    else
    {
      d.events.set ( Data_::NEW_VALUES );
    }
  }

  d.events.emit ( this );
}


//-----------------------------------------------------------------------
//   eraseData
//-----------------------------------------------------------------------


idx_t DenseTable::eraseData

  ( const idx_t*  irows,
    idx_t         icount,
    const idx_t*  jcols,
    idx_t         jcount )

{
  Data_&  d = * data_;
  idx_t   k = 0;

  if      ( icount * jcount == 1_idx )
  {
    idx_t  irow = irows[0];
    idx_t  jcol = jcols[0];

    d.checkRowIndex ( irow );
    d.checkColIndex ( jcol );

    k = d.eraseValue ( irow, jcol );
  }
  else if ( icount == 1_idx )
  {
    idx_t  irow = irows[0];

    d.checkRowIndex ( irow );

    for ( idx_t j = 0; j < jcount; j++ )
    {
      idx_t  jcol = jcols[j];

      d.checkColIndex ( jcol );

      k += d.eraseValue ( irow, jcol );
    }
  }
  else if ( jcount == 1_idx )
  {
    idx_t  jcol = jcols[0];

    d.checkColIndex ( jcol );

    for ( idx_t i = 0; i < icount; i++ )
    {
      idx_t  irow = irows[i];

      d.checkRowIndex ( irow );

      k += d.eraseValue ( irow, jcol );
    }
  }
  else
  {
    for ( idx_t i = 0; i < icount; i++ )
    {
      d.checkRowIndex ( irows[i] );
    }

    for ( idx_t j = 0; j < jcount; j++ )
    {
      idx_t  jcol = jcols[j];

      d.checkColIndex ( jcol );

      for ( idx_t i = 0; i < icount; i++ )
      {
        k += d.eraseValue ( irows[i], jcol );
      }
    }
  }

  if ( k )
  {
    d.events.emit ( this );
  }

  return k;
}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


idx_t DenseTable::getData

  ( double*       buf,
    const idx_t*  irows,
    idx_t         icount,
    const idx_t*  jcols,
    idx_t         jcount ) const

{
  const Data_&  d = * data_;
  idx_t         k = 0;

  if      ( icount * jcount == 1_idx )
  {
    idx_t  irow = irows[0];
    idx_t  jcol = jcols[0];

    if ( d.checkRowIndex( irow ) &&
         d.checkColIndex( jcol ) && d.mask(irow,jcol) )
    {
      *buf = d.table(irow,jcol);
      k    = 1;
    }
  }
  else if ( icount == 1_idx )
  {
    idx_t  irow = irows[0];

    if ( d.checkRowIndex( irow ) )
    {
      for ( idx_t j = 0; j < jcount; j++ )
      {
        idx_t  jcol = jcols[j];

        if ( d.checkColIndex( jcol ) && d.mask(irow,jcol) )
        {
          buf[j] = d.table(irow,jcol);
          k++;
        }
      }
    }
  }
  else if ( jcount == 1_idx )
  {
    idx_t  jcol = jcols[0];

    if ( d.checkColIndex( jcol ) )
    {
      for ( idx_t i = 0; i < icount; i++ )
      {
        idx_t  irow = irows[i];

        if ( d.checkRowIndex( irow ) && d.mask(irow,jcol) )
        {
          buf[i] = d.table(irow,jcol);
          k++;
        }
      }
    }
  }
  else
  {
    const double*  JEM_RESTRICT table;
    const byte*    JEM_RESTRICT mask;

    const idx_t    lastRow = d.table.size(0) - 1;

    for ( idx_t i = 0; i < icount; i++ )
    {
      d.checkRowIndex ( irows[i] );
    }

    for ( idx_t j = 0; j < jcount; j++, buf += icount )
    {
      idx_t  jcol = jcols[j];

      if ( d.checkColIndex( jcol ) )
      {
        table = d.table.addr() + (jcol * d.table.stride(1));
        mask  = d.mask .addr() + (jcol * d.mask .stride(1));

        for ( idx_t i = 0; i < icount; i++ )
        {
          idx_t  irow = irows[i];

          if ( irow <= lastRow && mask[irow] )
          {
            buf[i] = table[irow];
            k++;
          }
        }
      }
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   getAllData
//-----------------------------------------------------------------------


idx_t DenseTable::getAllData

  ( double*  buf,
    idx_t    jst ) const

{
  const Data_&  d = * data_;

  const idx_t   colCount = d.colNames .size ();
  const idx_t   rowCount = d.rowItems->size ();

  idx_t  m = d.table.size (0);
  idx_t  n = min ( colCount, d.table.size(1) );


  if ( m            == rowCount &&
       jst          == rowCount &&
       d.entryCount == rowCount * n )
  {
    std::memcpy ( buf, d.table.addr(),
                  (size_t) d.entryCount * sizeof(double) );
  }
  else
  {
    const double*  table = d.table.addr ();
    const byte*    mask  = d.mask .addr ();

    m = min ( m, rowCount );

    for ( idx_t jcol = 0; jcol < n; jcol++ )
    {
      for ( idx_t irow = 0; irow < m; irow++ )
      {
        if ( mask[irow] )
        {
          buf[irow] = table[irow];
        }
      }
    }

    buf   += jst;
    table += d.table.stride (1);
    mask  += d.mask .stride (1);
  }

  return d.entryCount;
}


//-----------------------------------------------------------------------
//   scaleRows
//-----------------------------------------------------------------------


void DenseTable::scaleRows ( const Vector& scale )
{
  JEM_PRECHECK2 ( scale.size() == rowCount(),
                  "row-scale vector has wrong size" );

  Data_&       d = * data_;

  const byte*  mask  = d.mask .addr ();
  double*      table = d.table.addr ();

  idx_t  m = min ( d.table.size(0), scale.size() );
  idx_t  n = min ( d.table.size(1), d.colCount() );


  for ( idx_t jcol = 0; jcol < n; jcol++ )
  {
    for ( idx_t irow = 0; irow < m; irow++ )
    {
      if ( mask[irow] )
      {
        table[irow] *= scale[irow];
      }
    }

    table += d.table.stride (1);
    mask  += d.mask .stride (1);
  }

  d.events.emit ( Data_::NEW_VALUES, this );
}


//-----------------------------------------------------------------------
//   toMatrix
//-----------------------------------------------------------------------


SparseMatrix DenseTable::toMatrix () const
{
  const Data_&  d        = * data_;

  const idx_t   colCount = d.colNames .size ();
  const idx_t   rowCount = d.rowItems->size ();

  IdxVector     ibuf     ( rowCount + 1 + d.entryCount );

  IdxVector     offsets  ( ibuf[slice(BEGIN,rowCount + 1)] );
  IdxVector     indices  ( ibuf[slice(rowCount + 1,END)]   );

  Vector        values   ( d.entryCount );

  idx_t  m = min ( rowCount, d.table.size(0) );
  idx_t  n = min ( colCount, d.table.size(1) );
  idx_t  j = 0;


  for ( idx_t irow = 0; irow < m; irow++ )
  {
    offsets[irow] = j;

    for ( idx_t jcol = 0; jcol < n; jcol++ )
    {
      if ( d.mask(irow,jcol) )
      {
        indices[j] = jcol;
        values [j] = d.table(irow,jcol);
        j++;
      }
    }
  }

  JEM_ASSERT ( j == d.entryCount );

  offsets[slice(m,END)] = j;

  return SparseMatrix ( shape ( rowCount, colCount ),
                        offsets, indices, values );
}


//-----------------------------------------------------------------------
//   getRowItems
//-----------------------------------------------------------------------


ItemSet* DenseTable::getRowItems () const
{
  return data_->rowItems.get ();
}


//-----------------------------------------------------------------------
//   emitEvents_
//-----------------------------------------------------------------------


void DenseTable::emitEvents_ ( int events )
{
  if      ( events & Data_::NEW_STRUCT )
  {
    newStructEvent.emit ( *this );
  }
  else if ( events & Data_::NEW_VALUES )
  {
    newValuesEvent.emit ( *this );
  }
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void DenseTable::connect_ ()
{
  using jem::util::connect;

  Data_&  d = * data_;

  connect ( d.rowItems->newSizeEvent,  this, & Self::itemsResized_   );
  connect ( d.rowItems->newOrderEvent, this, & Self::itemsReordered_ );
}


//-----------------------------------------------------------------------
//   itemsResized_
//-----------------------------------------------------------------------


void DenseTable::itemsResized_ ( idx_t rowCount )
{
  if ( rowCount == 0 )
  {
    clearData ();
  }
}


//-----------------------------------------------------------------------
//   itemsReordered_
//-----------------------------------------------------------------------


void DenseTable::itemsReordered_ ( const Reordering& reord )
{
  const idx_t*     JEM_RESTRICT  rmap = reord.rmapPtr ();

  Data_&         d        = * data_;

  const idx_t    rowCount = reord.newSize ();
  const idx_t    colCount = d.colCount    ();

  Array<byte,2>  newMask  ( rowCount, colCount );
  Matrix         newTable ( rowCount, colCount );

  idx_t          start;
  idx_t          k, m, n;


  newMask  = (byte) 0x0;
  newTable = 0.0;
  k        = 0;
  m        = d.table.size (0);
  n        = min ( colCount, d.table.size(1) );
  start    = min ( m, reord.start() );

  for ( idx_t jcol = 0; jcol < n; jcol++ )
  {
    for ( idx_t irow = 0; irow < start; irow++ )
    {
      newMask (irow,jcol) = d.mask (irow,jcol);
      newTable(irow,jcol) = d.table(irow,jcol);

      k += (idx_t) newMask(irow,jcol);
    }
  }

  for ( idx_t irow = start; irow < rowCount; irow++ )
  {
    idx_t  jrow = rmap[irow];

    if ( jrow < m )
    {
      for ( idx_t jcol = 0; jcol < n; jcol++ )
      {
        newMask (irow,jcol) = d.mask (jrow,jcol);
        newTable(irow,jcol) = d.table(jrow,jcol);

        k += (idx_t) newMask(irow,jcol);
      }
    }
  }

  if ( k != d.entryCount )
  {
    d.entryCount = k;
    d.events.set ( Data_::NEW_STRUCT );
  }

  d.mask  .swap ( newMask  );
  d.table .swap ( newTable );
  d.events.emit ( this     );
}


JIVE_END_PACKAGE( util )
