
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
#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jem/util/SparseArray.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jive/util/private/NameList.h>
#include <jive/util/error.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemMap.h>
#include <jive/util/EventMask.h>
#include <jive/util/Reordering.h>
#include <jive/util/SparseTable.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::SparseTable );


JIVE_BEGIN_PACKAGE( util )


using jem::shape;
using jem::newInstance;
using jem::util::SparseArray;


//=======================================================================
//   class SparseTable::Data_
//=======================================================================


class SparseTable::Data_ : public jem::Collectable
{
 public:

  typedef Data_             Self;
  typedef
    SparseArray<double,2>   Table;

  typedef Table::Iterator   Iterator;
  typedef
    Table::ConstIterator    ConstIterator;

  static const int          NEW_VALUES = 1 << 0;
  static const int          NEW_STRUCT = 1 << 1;


  inline                    Data_     ();

  explicit                  Data_

    ( const Ref<ItemSet>&     rows );

  inline void               readFrom

    ( ObjectInput&            in  );

  inline void               writeTo

    ( ObjectOutput&           out )      const;

  void                      reshape   ();

  inline void               checkColIndex

    ( idx_t                   jcol )     const;

  inline void               checkRowIndex

    ( idx_t                   irow );

  inline void               checkRowIndex

    ( idx_t                   irow )     const;

  void                      rowIndexError

    ( idx_t                   irow )     const;

  inline void               reserve

    ( idx_t                   count );

  inline void               setValue

    ( idx_t                   irow,
      idx_t                   jcol,
      double                  value );

  inline void               addValue

    ( idx_t                   irow,
      idx_t                   jcol,
      double                  value );

  inline idx_t              eraseValue

    ( idx_t                   irow,
      idx_t                   jcol  );


 public:

  Table                     table;
  NameList                  colNames;
  Ref<ItemSet>              rowItems;
  EventMask                 events;


 private:

  void                      checkRowIndex_

    ( idx_t                   irow );

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline SparseTable::Data_::Data_ ()
{}


SparseTable::Data_::Data_ ( const Ref<ItemSet>& rows ) :

  rowItems ( rows )

{
  table.reshape ( shape( rows->size(), 4 ) );
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


inline void SparseTable::Data_::readFrom ( ObjectInput& in )
{
  decode ( in, rowItems, table, colNames );

  events.reset ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


inline void SparseTable::Data_::writeTo ( ObjectOutput& out ) const
{
  encode ( out, rowItems, table, colNames );
}


//-----------------------------------------------------------------------
//   reshape
//-----------------------------------------------------------------------


void SparseTable::Data_::reshape ()
{
  const idx_t  m = rowItems->size ();
  const idx_t  n = colNames .size ();

  if ( m != table.size(0) || n > table.size(1) )
  {
    table.reshape ( shape( m, n + 2 ) );
  }
}


//-----------------------------------------------------------------------
//   checkColIndex
//-----------------------------------------------------------------------


inline void SparseTable::Data_::checkColIndex ( idx_t jcol ) const
{
  if ( jcol < 0 || jcol >= colNames.size() )
  {
    indexError ( CLASS_NAME, "column", jcol, colNames.size() );
  }
}


//-----------------------------------------------------------------------
//   checkRowIndex
//-----------------------------------------------------------------------


inline void SparseTable::Data_::checkRowIndex ( idx_t irow )
{
  if ( irow < 0 || irow >= table.size(0) )
  {
    checkRowIndex_ ( irow );
  }
}


inline void SparseTable::Data_::checkRowIndex ( idx_t irow ) const
{
  if ( irow < 0 )
  {
    rowIndexError ( irow );
  }
}


//-----------------------------------------------------------------------
//   rowIndexError
//-----------------------------------------------------------------------


void SparseTable::Data_::rowIndexError ( idx_t irow ) const
{
  indexError ( CLASS_NAME,
               rowItems->getItemName(), irow, rowItems->size() );
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


inline void SparseTable::Data_::reserve ( idx_t n )
{
  table.reserve ( n );
  reshape       ();
}


//-----------------------------------------------------------------------
//   setValue
//-----------------------------------------------------------------------


inline void SparseTable::Data_::setValue

  ( idx_t   irow,
    idx_t   jcol,
    double  value )

{
  double*  xptr = table.find ( irow, jcol );

  if ( xptr )
  {
    *xptr = value;

    events.set ( NEW_VALUES );
  }
  else
  {
    table(irow,jcol) = value;

    events.set ( NEW_STRUCT );
  }
}


//-----------------------------------------------------------------------
//   addValue
//-----------------------------------------------------------------------


inline void SparseTable::Data_::addValue

  ( idx_t   irow,
    idx_t   jcol,
    double  value )

{
  double*  xptr = table.find ( irow, jcol );

  if ( xptr )
  {
    *xptr += value;

    events.set ( NEW_VALUES );
  }
  else
  {
    table(irow,jcol) = value;

    events.set ( NEW_STRUCT );
  }
}


//-----------------------------------------------------------------------
//   eraseValue
//-----------------------------------------------------------------------


inline idx_t SparseTable::Data_::eraseValue

  ( idx_t  irow,
    idx_t  jcol  )

{
  if ( table.erase( irow, jcol ) )
  {
    events.set ( NEW_STRUCT );

    return 1;
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------
//   checkRowIndex_
//-----------------------------------------------------------------------


void SparseTable::Data_::checkRowIndex_ ( idx_t irow )
{
  const idx_t  rowCount = rowItems->size ();

  if ( irow < 0 || irow >= rowCount )
  {
    rowIndexError ( irow );
  }

  if ( irow > table.size(0) )
  {
    reshape ();
  }
}


//=======================================================================
//   class SparseTable
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SparseTable::SparseTable ()
{}


SparseTable::SparseTable

  ( const String&        name,
    const Ref<ItemSet>&  rowItems ) :

    Super ( name )

{
  JEM_PRECHECK2 ( rowItems, "NULL ItemSet" );

  data_ = newInstance<Data_> ( rowItems );

  connect_ ();
}


SparseTable::SparseTable ( const Self& rhs ) :

  Super ( rhs.myName_ )

{
  data_ = newInstance<Data_> ( * rhs.data_ );

  connect_ ();
}


SparseTable::~SparseTable ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void SparseTable::resetEvents ()
{
  data_->events   .reset       ();
  data_->rowItems->resetEvents ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void SparseTable::readFrom ( ObjectInput& in )
{
  decode ( in, myName_ );

  data_ = newInstance<Data_> ();

  data_->readFrom ( in );

  connect_ ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void SparseTable::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_ );

  data_->writeTo ( out );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> SparseTable::clone () const
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t SparseTable::size () const
{
  return data_->table.nonZeroCount ();
}


//-----------------------------------------------------------------------
//   columnCount
//-----------------------------------------------------------------------


idx_t SparseTable::columnCount () const
{
  return data_->colNames.size ();
}


//-----------------------------------------------------------------------
//   clearData
//-----------------------------------------------------------------------


void SparseTable::clearData ()
{
  Data_&  d = * data_;

  if ( d.table.nonZeroCount() > 0 )
  {
    d.table.clear  ();
    d.reshape      ();
    d.events.reset ();

    newStructEvent.emit ( *this );
  }
}


//-----------------------------------------------------------------------
//   clearAll
//-----------------------------------------------------------------------


void SparseTable::clearAll ()
{
  data_->colNames.clear ();

  clearData ();
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void SparseTable::reserve ( idx_t n )
{
  data_->reserve ( n );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void SparseTable::trimToSize ()
{
  data_->table.trimToSize ();
}


//-----------------------------------------------------------------------
//   addColumn
//-----------------------------------------------------------------------


idx_t SparseTable::addColumn ( const String& name )
{
  return data_->colNames.addName ( name );
}


//-----------------------------------------------------------------------
//   findColumn
//-----------------------------------------------------------------------


idx_t SparseTable::findColumn ( const String& name ) const
{
  return data_->colNames.findName ( name );
}


//-----------------------------------------------------------------------
//   getColumnName
//-----------------------------------------------------------------------


String SparseTable::getColumnName ( idx_t jcol ) const
{
  const Data_&  d = * data_;

  d.checkColIndex ( jcol );

  return d.colNames.getName ( jcol );
}


//-----------------------------------------------------------------------
//   getColumnNames
//-----------------------------------------------------------------------


StringVector SparseTable::getColumnNames () const
{
  return data_->colNames.toArray ();
}


//-----------------------------------------------------------------------
//   setData
//-----------------------------------------------------------------------


void SparseTable::setData

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
    if ( icount * jcount > 1000_idx )
    {
      d.reserve ( d.table.nonZeroCount() + icount * jcount );
    }

    for ( idx_t i = 0; i < icount; i++ )
    {
      d.checkRowIndex ( irows[i] );
    }

    for ( idx_t j = 0; j < jcount; j++, values += icount )
    {
      idx_t  jcol = jcols[j];

      d.checkColIndex ( jcol );

      for ( idx_t i = 0; i < icount; i++ )
      {
        d.setValue ( irows[i], jcol, values[i] );
      }
    }
  }

  d.events.emit ( this );
}


//-----------------------------------------------------------------------
//   addData
//-----------------------------------------------------------------------


void SparseTable::addData

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
    if ( icount * jcount > 1000_idx )
    {
      d.reserve ( d.table.nonZeroCount() + icount * jcount );
    }

    for ( idx_t i = 0; i < icount; i++ )
    {
      d.checkRowIndex ( irows[i] );
    }

    for ( idx_t j = 0; j < jcount; j++, values += icount )
    {
      idx_t  jcol = jcols[j];

      d.checkColIndex ( jcol );

      for ( idx_t i = 0; i < icount; i++ )
      {
        d.addValue ( irows[i], jcol, values[i] );
      }
    }
  }

  d.events.emit ( this );
}


//-----------------------------------------------------------------------
//   eraseData
//-----------------------------------------------------------------------


idx_t SparseTable::eraseData

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

    k = 0;

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


idx_t SparseTable::getData

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

    d.checkRowIndex ( irow );
    d.checkColIndex ( jcol );

    const double*  xptr = d.table.find ( irow, jcol );

    if ( xptr )
    {
      *buf = *xptr;
      k    = 1;
    }
  }
  else if ( icount == 1_idx )
  {
    idx_t  irow = irows[0];

    d.checkRowIndex ( irow );

    for ( idx_t j = 0; j < jcount; j++ )
    {
      idx_t  jcol = jcols[j];

      d.checkColIndex ( jcol );

      const double*  xptr = d.table.find ( irow, jcol );

      if ( xptr )
      {
        buf[j] = *xptr;
        k++;
      }
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

      const double*  xptr = d.table.find ( irow, jcol );

      if ( xptr )
      {
        buf[i] = *xptr;
        k++;
      }
    }
  }
  else
  {
    for ( idx_t i = 0; i < icount; i++ )
    {
      d.checkRowIndex ( irows[i] );
    }

    for ( idx_t j = 0; j < jcount; j++, buf += icount )
    {
      idx_t  jcol = jcols[j];

      d.checkColIndex ( jcol );

      for ( idx_t i = 0; i < icount; i++ )
      {
        const double*  xptr = d.table.find ( irows[i], jcol );

        if ( xptr )
        {
          buf[i] = *xptr;
          k++;
        }
      }
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   getAllData
//-----------------------------------------------------------------------


idx_t SparseTable::getAllData

  ( double*  buf,
    idx_t    jst ) const

{
  const Data_&  d = * data_;

  Data_::ConstIterator  it  = d.table.begin ();
  Data_::ConstIterator  end = d.table.end   ();

  for ( ; it != end; ++it )
  {
    buf[it->index[0] + jst * it->index[1]] = it->value;
  }

  return d.table.nonZeroCount ();
}


//-----------------------------------------------------------------------
//   scaleRows
//-----------------------------------------------------------------------


void SparseTable::scaleRows ( const Vector& scale )
{
  JEM_PRECHECK2 ( scale.size() == rowCount(),
                  "row-scale vector has wrong size" );

  Data_&  d = * data_;

  Data_::Iterator  it  = d.table.begin ();
  Data_::Iterator  end = d.table.end   ();

  for ( ; it != end; ++it )
  {
    it->value *= scale[it->index[0]];
  }

  d.events.emit ( Data_::NEW_VALUES, this );
}


//-----------------------------------------------------------------------
//   toMatrix
//-----------------------------------------------------------------------


SparseMatrix SparseTable::toMatrix () const
{
  using jem::numeric::SparseUtils;

  const Data_&  d = * data_;

  Data_::ConstIterator  it   = d.table.begin ();
  Data_::ConstIterator  end  = d.table.end   ();

  const idx_t  rowCount = d.rowItems->size     ();
  const idx_t  colCount = d.colNames .size     ();
  const idx_t  nzCount  = d.table.nonZeroCount ();

  IdxVector    ibuf     ( rowCount + 1 + nzCount );

  IdxVector    offsets  ( ibuf[slice(BEGIN,rowCount + 1)] );
  IdxVector    indices  ( ibuf[slice(rowCount + 1,END)]   );

  Vector       values   ( nzCount );


  offsets = 0_idx;

  for ( ; it != end; ++it )
  {
    offsets[it->index[0]]++;
  }

  SparseUtils::sumOffsets ( offsets );

  for ( it = d.table.begin(); it != end; ++it )
  {
    idx_t  irow              = it->index[0];
    indices[offsets[irow]] = it->index[1];
    values [offsets[irow]] = it->value;

    offsets[irow]++;
  }

  SparseUtils::shiftOffsets ( offsets );

  return SparseMatrix ( shape( rowCount, colCount ),
                        offsets, indices, values );
}


//-----------------------------------------------------------------------
//   getRowItems
//-----------------------------------------------------------------------


ItemSet* SparseTable::getRowItems () const
{
  return data_->rowItems.get ();
}


//-----------------------------------------------------------------------
//   emitEvents_
//-----------------------------------------------------------------------


void SparseTable::emitEvents_ ( int events )
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


void SparseTable::connect_ ()
{
  using jem::util::connect;

  Data_&  d = * data_;

  connect ( d.rowItems->newSizeEvent,  this, & Self::itemsResized_   );
  connect ( d.rowItems->newOrderEvent, this, & Self::itemsReordered_ );
}


//-----------------------------------------------------------------------
//   itemsResized_
//-----------------------------------------------------------------------


void SparseTable::itemsResized_ ( idx_t rowCount )
{
  if ( rowCount == 0 )
  {
    clearData ();
  }
}


//-----------------------------------------------------------------------
//   itemsReordered_
//-----------------------------------------------------------------------


void SparseTable::itemsReordered_ ( const Reordering& reord )
{
  const idx_t*     JEM_RESTRICT  fmap = reord.fmapPtr ();

  Data_&           d        = * data_;

  Data_::Iterator  it       = d.table.begin ();
  Data_::Iterator  end      = d.table.end   ();

  const idx_t      rowCount = d.rowItems->size ();
  const idx_t      colCount = d.colNames .size ();

  Data_::Table     newTable;


  newTable.reserve ( d.table.nonZeroCount() );
  newTable.reshape ( shape( rowCount, colCount ) );

  for ( ; it != end; ++it )
  {
    idx_t  irow = it->index[0];
    idx_t  jcol = it->index[1];
    idx_t  jrow = fmap[irow];

    if ( jrow >= 0 )
    {
      newTable(jrow,jcol) = it->value;
    }
    else
    {
      d.events.set ( Data_::NEW_STRUCT );
    }
  }

  d.table .swap ( newTable );
  d.events.emit ( this     );
}


JIVE_END_PACKAGE( util )
