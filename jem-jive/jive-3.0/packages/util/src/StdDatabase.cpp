
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
#include <jem/base/MemCache.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/error.h>
#include <jem/util/Event.h>
#include <jem/util/ArrayBuffer.h>
#include <jem/util/SparseArray.h>
#include <jive/util/error.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemSet.h>
#include <jive/util/EventMask.h>
#include <jive/util/Reordering.h>
#include <jive/util/StdDatabase.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::StdDatabase );


JIVE_BEGIN_PACKAGE( util )


using jem::byte;
using jem::newInstance;
using jem::Array;
using jem::MemCache;
using jem::io::DataInput;
using jem::io::DataOutput;
using jem::util::ArrayBuffer;
using jem::util::SparseArray;


//=======================================================================
//   class StdDBRow
//=======================================================================


class StdDBRow
{
 public:

  inline                    StdDBRow    ();


 public:

  idx_t                     size;

  union
  {
    idx_t                   addr;
    idx_t                   ival;
    double                  fval;
  };

};


typedef
  SparseArray<StdDBRow>     StdDBRowArray;


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline StdDBRow::StdDBRow ()
{
  size = 0;
  addr = 0;
}


//=======================================================================
//   class StdDBData
//=======================================================================


class StdDBData
{
 public:

  static const int        NEW_VALUES = 1 << 0;
  static const int        NEW_STRUCT = 1 << 1;


  inline                  StdDBData

    ( Database*             dbase,
      const Ref<ItemSet>&   items );

  inline void             emitEvents  ();


 public:

  Database*               dbase;
  Ref<ItemSet>            rowItems;
  EventMask               events;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline StdDBData::StdDBData

  ( Database*            db,
    const Ref<ItemSet>&  it ) :

    dbase    ( db ),
    rowItems ( it )

{}


//-----------------------------------------------------------------------
//   emitEvents
//-----------------------------------------------------------------------


inline void StdDBData::emitEvents ()
{
  events.emit ( dbase );
}


//=======================================================================
//   class StdDBColumn
//=======================================================================


class StdDBColumn : virtual public DBColumn
{
 public:

  typedef StdDBColumn       Self;
  typedef StdDBRow          Row;
  typedef StdDBData         DBData;
  typedef StdDBRowArray     RowArray;

  enum                      DataType
  {
                              INT,
                              FLOAT
  };


  inline                    StdDBColumn

    ( DBData&                 dbase,
      const String&           name,
      DataType                type );

  virtual                  ~StdDBColumn     ();

  virtual String            getName         () const override;
  virtual Database*         getDatabase     () const override;
  virtual ItemSet*          getRowItems     () const override;

  virtual idx_t             rowSize

    ( idx_t                   irow )           const override;

  virtual idx_t             maxRowSize      () const override;
  virtual IdxVector         getNonEmptyRows () const override;
  virtual void              clear           ()       override;

  virtual void              clearRow

    ( idx_t                   irow )                 override;

  virtual void              trimToSize      ()       override;

  virtual void              printTo

    ( PrintWriter&            out )            const override;

  void                      readFrom

    ( DataInput&              in );

  void                      writeTo

    ( DataOutput&             out )            const;

  void                      copy

    ( const Self&             rhs );

  void                      reorder

    ( RowArray&               tmp,
      const Reordering&       reord );

  inline void               checkRowIndex

    ( idx_t                   irow );

  inline void               checkRowIndex

    ( idx_t                   irow )           const;

  void                      rowIndexError

    ( idx_t                   irow )           const;


 public:

  const String              name;


 protected:

  idx_t                     alloc_

    ( idx_t                   count );

  void                      resize_

    ( Row&                    row,
      idx_t                   count );

  void                      compress_

    ( idx_t                   size );

  void                      checkRowIndex_

    ( idx_t                   irow );


 protected:

  struct                    Heap_
  {
    byte*                     data;
    idx_t                     size;
    idx_t                     last;
    idx_t                     used;
  };


  DBData&                   data_;
  RowArray                  rows_;
  Heap_                     heap_;

  DataType                  dataType_;
  idx_t                     dataSize_;
  idx_t                     maxRowSize_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline StdDBColumn::StdDBColumn

  ( DBData&        db,
    const String&  nm,
    DataType       dt ) :

    name  ( nm ),
    data_ ( db ),
    rows_ ( Row() )

{
  heap_.data  = nullptr;
  heap_.size  = 0;
  heap_.last  = 0;
  heap_.used  = 0;

  dataType_   = dt;
  dataSize_   = (idx_t) ((dt == INT) ? sizeof(idx_t) : sizeof(double));
  maxRowSize_ = 0;
}


StdDBColumn::~StdDBColumn ()
{
  if ( heap_.data )
  {
    MemCache::dealloc ( heap_.data,
                        (size_t) (heap_.size * dataSize_) );

    heap_.data = nullptr;
  }
}


//-----------------------------------------------------------------------
//   getName
//-----------------------------------------------------------------------


String StdDBColumn::getName () const
{
  return name;
}


//-----------------------------------------------------------------------
//   getDatabase
//-----------------------------------------------------------------------


Database* StdDBColumn::getDatabase () const
{
  return data_.dbase;
}


//-----------------------------------------------------------------------
//   getRowItems
//-----------------------------------------------------------------------


ItemSet* StdDBColumn::getRowItems () const
{
  return data_.rowItems.get ();
}


//-----------------------------------------------------------------------
//   rowSize
//-----------------------------------------------------------------------


idx_t StdDBColumn::rowSize ( idx_t irow ) const
{
  checkRowIndex ( irow );

  const Row*  row = rows_.find ( irow );

  if ( row )
  {
    return row->size;
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------
//   maxRowSize
//-----------------------------------------------------------------------


idx_t StdDBColumn::maxRowSize () const
{
  return maxRowSize_;
}


//-----------------------------------------------------------------------
//   getNonEmptyRows
//-----------------------------------------------------------------------


IdxVector StdDBColumn::getNonEmptyRows () const
{
  RowArray::ConstIterator  first = rows_.begin ();
  RowArray::ConstIterator  last  = rows_.end   ();

  IdxVector  irows ( rows_.nonZeroCount() );

  for ( idx_t i = 0; first != last; ++first, i++ )
  {
    irows[i] = first->index[0];
  }

  return irows;
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void StdDBColumn::clear ()
{
  if ( rows_.nonZeroCount() > 0 )
  {
    data_.events.set ( DBData::NEW_STRUCT );
  }

  rows_.clear ();

  heap_.last  = 0;
  heap_.used  = 0;
  maxRowSize_ = 0;

  data_.emitEvents ();
}


//-----------------------------------------------------------------------
//   clearRow
//-----------------------------------------------------------------------


void StdDBColumn::clearRow ( idx_t irow )
{
  checkRowIndex ( irow );

  Row*  row = rows_.find ( irow );

  if ( row )
  {
    if ( row->size > 1 )
    {
      heap_.used -= row->size;

      JEM_ASSERT ( heap_.used >= 0 );
    }

    rows_.erase      ( irow );
    data_.events.set ( DBData::NEW_STRUCT );

    if ( rows_.nonZeroCount() == 0 )
    {
      maxRowSize_ = 0;
    }

    data_.emitEvents ();
  }
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void StdDBColumn::trimToSize ()
{
  rows_.trimToSize ();

  if ( heap_.size > heap_.used )
  {
    compress_ ( heap_.used );
  }
}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void StdDBColumn::printTo ( PrintWriter& out ) const
{
  using jem::io::endl;
  using jem::io::space;

  const ItemMap&  imap = * data_.rowItems->getItemMap ();

  RowArray::ConstIterator  first = rows_.begin ();
  RowArray::ConstIterator  last  = rows_.end   ();


  for ( ; first != last; ++first )
  {
    idx_t  irow = first->index[0];
    Row    row  = first->value;

    if ( row.size == 0 )
    {
      continue;
    }

    print ( out, endl, imap.getItemID( irow ) );

    if ( row.size == 1 )
    {
      if ( dataType_ == INT )
      {
        print ( out, space, row.ival );
      }
      else
      {
        print ( out, space, row.fval );
      }
    }
    else
    {
      if ( dataType_ == INT )
      {
        const idx_t*  rowData = (const idx_t*)

          (heap_.data + row.addr);

        for ( idx_t i = 0; i < row.size; i++ )
        {
          print ( out, space, rowData[i] );
        }
      }
      else
      {
        const double*  rowData = (const double*)

          (heap_.data + row.addr);

        for ( idx_t i = 0; i < row.size; i++ )
        {
          print ( out, space, rowData[i] );
        }
      }
    }

    print ( out, ';' );
  }
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void StdDBColumn::readFrom ( DataInput& in )
{
  idx_t  nnzCount;

  decode ( in, nnzCount );

  rows_.clear ();

  heap_.last  = 0;
  heap_.used  = 0;

  maxRowSize_ = 0;

  if ( nnzCount > 0 )
  {
    idx_t  dataCount;
    idx_t  rowAddr;

    decode ( in, dataCount );

    if ( heap_.size < dataCount )
    {
      heap_.data = (byte*)

        MemCache::realloc ( heap_.data,
                            (size_t) (heap_.size * dataSize_),
                            (size_t) (dataCount  * dataSize_) );

      heap_.size = dataCount;
    }

    rows_.reserve ( nnzCount );

    rowAddr = 0;

    for ( idx_t i = 0; i < nnzCount; i++ )
    {
      idx_t  irow, rowSize;

      decode ( in, irow, rowSize );

      Row&  row = rows_[irow];

      row.size  = rowSize;

      if ( rowSize == 1 )
      {
        if ( dataType_ == INT )
        {
          decode ( in, row.ival );
        }
        else
        {
          decode ( in, row.fval );
        }
      }
      else
      {
        row.addr = rowAddr;

        if ( dataType_ == INT )
        {
          in.decode ( (idx_t*)    (heap_.data + rowAddr), rowSize );
        }
        else
        {
          in.decode ( (double*) (heap_.data + rowAddr), rowSize );
        }

        rowAddr    += rowSize * dataSize_;
        heap_.last += rowSize;
        heap_.used += rowSize;
      }

      if ( rowSize > maxRowSize_ )
      {
        maxRowSize_ = rowSize;
      }
    }

    JEM_ASSERT ( heap_.used == dataCount );
  }
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void StdDBColumn::writeTo ( DataOutput& out ) const
{
  idx_t  nnzCount = rows_.nonZeroCount ();

  encode ( out, nnzCount );

  if ( nnzCount > 0 )
  {
    RowArray::ConstIterator  first = rows_.begin ();
    RowArray::ConstIterator  last  = rows_.end   ();

    idx_t  dataCount = heap_.used;

    encode ( out, dataCount );

    for ( ; first != last; ++first )
    {
      idx_t  irow = first->index[0];
      Row    row  = first->value;

      encode ( out, irow, row.size );

      if ( row.size == 1 )
      {
        if ( dataType_ == INT )
        {
          encode ( out, row.ival );
        }
        else
        {
          encode ( out, row.fval );
        }
      }
      else
      {
        if ( dataType_ == INT )
        {
          const idx_t*  rowData = (const idx_t*)

            (heap_.data + row.addr);

          out.encode ( rowData, row.size );
        }
        else
        {
          const double*  rowData = (const double*)

            (heap_.data + row.addr);

          out.encode ( rowData, row.size );
        }
      }
    }
  }
}


//-----------------------------------------------------------------------
//   copy
//-----------------------------------------------------------------------


void StdDBColumn::copy ( const Self& rhs )
{
  JEM_ASSERT2 ( dataType_ == rhs.dataType_, "data type mismatch" );

  RowArray::ConstIterator  first = rhs.rows_.begin ();
  RowArray::ConstIterator  last  = rhs.rows_.end   ();

  idx_t  rowAddr;


  rows_.clear ();

  heap_.last  = 0;
  heap_.used  = 0;

  maxRowSize_ = 0;

  rows_.reserve ( rhs.rows_.nonZeroCount() );

  if ( heap_.size < rhs.heap_.used )
  {
    heap_.data = (byte*)

      MemCache::realloc ( heap_.data,
                          (size_t) (heap_.size     * dataSize_),
                          (size_t) (rhs.heap_.used * dataSize_) );

    heap_.size = rhs.heap_.used;
  }

  rowAddr = 0;

  for ( ; first != last; ++first )
  {
    Row&  row = rows_[first->index[0]];

    row = first->value;

    if ( row.size > 1 )
    {
      row.addr = rowAddr;

      std::memcpy ( heap_.data + rowAddr,
                    rhs.heap_.data + first->value.addr,
                    (size_t) (row.size * dataSize_) );

      rowAddr    += row.size * dataSize_;
      heap_.last += row.size;
      heap_.used += row.size;
    }

    if ( row.size > maxRowSize_ )
    {
      maxRowSize_ = row.size;
    }
  }

  JEM_ASSERT ( heap_.used  == rhs.heap_.used  );
  JEM_ASSERT ( maxRowSize_ <= rhs.maxRowSize_ );
}


//-----------------------------------------------------------------------
//   reorder
//-----------------------------------------------------------------------


void StdDBColumn::reorder

  ( RowArray&          tmp,
    const Reordering&  reord )

{
  const idx_t*  JEM_RESTRICT  fmap  = reord.fmapPtr ();

  RowArray::Iterator  first = rows_.begin ();
  RowArray::Iterator  last  = rows_.end   ();

  idx_t  maxSize = 0;


  tmp.clear   ();
  tmp.reserve ( rows_.nonZeroCount() );

  for ( ; first != last; ++first )
  {
    idx_t  irow = first->index[0];
    idx_t  jrow = fmap[irow];
    Row&   row  = first->value;

    if      ( jrow >= 0 )
    {
      tmp[jrow] = row;

      if ( row.size > maxSize )
      {
        maxSize = row.size;
      }
    }
    else if ( row.size > 1 )
    {
      heap_.used -= row.size;
    }
  }

  JEM_ASSERT ( heap_.used >= 0 );

  maxRowSize_ = maxSize;

  if ( tmp.nonZeroCount() != rows_.nonZeroCount() )
  {
    data_.events.set ( DBData::NEW_STRUCT );
  }

  rows_.clear ();
  rows_.swap  ( tmp );

  idx_t  n = heap_.size - heap_.used;

  if ( n > heap_.used / 4_idx )
  {
    compress_ ( heap_.used + heap_.used / 4_idx );
  }

  data_.emitEvents ();
}


//-----------------------------------------------------------------------
//   checkRowIndex
//-----------------------------------------------------------------------


inline void StdDBColumn::checkRowIndex ( idx_t irow )
{
  if ( irow < 0 || irow >= rows_.size() )
  {
    checkRowIndex_ ( irow );
  }
}


inline void StdDBColumn::checkRowIndex ( idx_t irow ) const
{
  if ( irow < 0 )
  {
    rowIndexError ( irow );
  }
}


//-----------------------------------------------------------------------
//   rowIndexError
//-----------------------------------------------------------------------


void StdDBColumn::rowIndexError ( idx_t irow ) const
{
  indexError ( StdDatabase::CLASS_NAME,
               data_.rowItems->getItemName(), irow,
               data_.rowItems->size() );
}


//-----------------------------------------------------------------------
//   alloc_
//-----------------------------------------------------------------------


idx_t StdDBColumn::alloc_ ( idx_t count )
{
  using jem::util::overflowError;

  JEM_ASSERT ( count > 1 );

  idx_t  n = heap_.last + count;

  if ( n < 0 )
  {
    overflowError ( JEM_FUNC, n );
  }

  if ( n > heap_.size )
  {
    idx_t  n = heap_.last - heap_.used;

    if ( n > heap_.used / 4_idx )
    {
      n = heap_.used + heap_.used / 4_idx + count;

      compress_ ( n );
    }
    else
    {
      n = heap_.last + heap_.size / 4_idx + count;

      if ( n < 0 )
      {
        overflowError ( JEM_FUNC, n );
      }

      heap_.data = (byte*)

        MemCache::realloc ( heap_.data,
                            (size_t) (heap_.size * dataSize_),
                            (size_t) (n          * dataSize_) );

      heap_.size = n;
    }
  }

  idx_t  rowAddr = heap_.last * dataSize_;

  heap_.last += count;

  JEM_ASSERT ( heap_.last <= heap_.size );

  return rowAddr;
}


//-----------------------------------------------------------------------
//   resize_
//-----------------------------------------------------------------------


void StdDBColumn::resize_

  ( Row&   row,
    idx_t  count )

{
  if ( count < row.size )
  {
    if ( count <= 1 )
    {
      heap_.used -= row.size;
    }
    else
    {
      heap_.used -= (row.size - count);
    }
  }
  else
  {
    if ( count > 1 )
    {
      row.addr = alloc_ ( count );

      if ( row.size <= 1 )
      {
        heap_.used += count;
      }
      else
      {
        heap_.used += (count - row.size);
      }
    }

    if ( count > maxRowSize_ )
    {
      maxRowSize_ = count;
    }
  }

  row.size = count;
}


//-----------------------------------------------------------------------
//   compress_
//-----------------------------------------------------------------------


void StdDBColumn::compress_ ( idx_t size )
{
  using jem::util::overflowError;

  if ( size < 0 )
  {
    overflowError ( JEM_FUNC, size );
  }

  if ( size < heap_.used )
  {
    size = heap_.used;
  }

  if ( heap_.last == heap_.used )
  {
    heap_.data = (byte*)

      MemCache::realloc ( heap_.data,
                          (size_t) (heap_.size * dataSize_),
                          (size_t) (size       * dataSize_) );

    heap_.size = size;
  }
  else
  {
    RowArray::Iterator  first = rows_.begin ();
    RowArray::Iterator  last  = rows_.end   ();

    byte*  newHeap = (byte*)

      MemCache::alloc ( (size_t) (size * dataSize_) );

    idx_t  rowAddr = 0;
    idx_t  maxSize = 0;


    for ( ; first != last; ++first )
    {
      Row&  row = first->value;

      if ( row.size > 1 )
      {
        idx_t  len = row.size * dataSize_;

        std::memcpy ( newHeap    + rowAddr,
                      heap_.data + row.addr, (size_t) len );

        row.addr = rowAddr;
        rowAddr += len;
      }

      if ( row.size > maxSize )
      {
        maxSize = row.size;
      }
    }

    if ( heap_.data )
    {
      MemCache::dealloc ( heap_.data,
                          (size_t) (heap_.size * dataSize_) );
    }

    heap_.data  = newHeap;
    heap_.last  = heap_.used;
    heap_.size  = size;

    maxRowSize_ = maxSize;
  }
}


//-----------------------------------------------------------------------
//   checkRowIndex_
//-----------------------------------------------------------------------


void StdDBColumn::checkRowIndex_ ( idx_t irow )
{
  const idx_t  rowCount = data_.rowItems->size ();

  if ( irow < 0 || irow >= rowCount )
  {
    rowIndexError ( irow );
  }

  if ( rows_.size() == 0 )
  {
    // Reshape the row array on the first time data is inserted.

    rows_.reshape ( jem::shape( rowCount ) );
  }
}


//=======================================================================
//   class StdIntDBColumn
//=======================================================================


class StdIntDBColumn : public StdDBColumn, public IntDBColumn
{
 public:

  typedef StdIntDBColumn    Self;
  typedef StdDBColumn       Base;
  typedef IntDBColumn       ColumnType;


  inline                    StdIntDBColumn

    ( DBData&                 data,
      const String&           name );

  virtual idx_t             getData

    ( idx_t*                  buf,
      idx_t                   size,
      idx_t                   irow )       const override;

  virtual void              setData

    ( idx_t                   irow,
      const idx_t*            buf,
      idx_t                   count )            override;

  virtual void              printTo

    ( PrintWriter&            out )        const override;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline StdIntDBColumn::StdIntDBColumn

  ( DBData&        data,
    const String&  name ) :

    Base ( data, name, INT )

{}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


idx_t StdIntDBColumn::getData

  ( idx_t*  buf,
    idx_t   size,
    idx_t   irow ) const

{
  checkRowIndex ( irow );

  if ( size <= 0 )
  {
    return 0;
  }

  const Row*  row = rows_.find ( irow );

  if ( ! row )
  {
    return 0;
  }

  if ( row->size == 1 )
  {
    buf[0] = row->ival;

    return 1;
  }

  if ( row->size < size )
  {
    size = row->size;
  }

  const idx_t* JEM_RESTRICT  rowData =

    (const idx_t*) (heap_.data + row->addr);

  if      ( size == 1_idx )
  {
    buf[0] = rowData[0];
  }
  else if ( size == 2_idx )
  {
    buf[0] = rowData[0];
    buf[1] = rowData[1];
  }
  else if ( size == 3_idx )
  {
    buf[0] = rowData[0];
    buf[1] = rowData[1];
    buf[2] = rowData[2];
  }
  else if ( size == 4_idx )
  {
    buf[0] = rowData[0];
    buf[1] = rowData[1];
    buf[2] = rowData[2];
    buf[3] = rowData[3];
  }
  else
  {
    std::memcpy ( buf, rowData, (size_t) size * sizeof(*rowData) );
  }

  return size;
}


//-----------------------------------------------------------------------
//   setData
//-----------------------------------------------------------------------


void StdIntDBColumn::setData

  ( idx_t         irow,
    const idx_t*  buf,
    idx_t         count )

{
  if ( count <= 0 )
  {
    clearRow ( irow );
  }
  else
  {
    checkRowIndex ( irow );

    Row&  row = rows_[irow];

    if ( count == row.size )
    {
      data_.events.set ( DBData::NEW_VALUES );
    }
    else
    {
      data_.events.set ( DBData::NEW_STRUCT );

      resize_ ( row, count );
    }

    if ( count == 1 )
    {
      row.ival = buf[0];
    }
    else
    {
      idx_t* JEM_RESTRICT  rowData = (idx_t*)

        (heap_.data + row.addr);

      if      ( count == 2_idx )
      {
        rowData[0] = buf[0];
        rowData[1] = buf[1];
      }
      else if ( count == 3_idx )
      {
        rowData[0] = buf[0];
        rowData[1] = buf[1];
        rowData[2] = buf[2];
      }
      else if ( count == 4_idx )
      {
        rowData[0] = buf[0];
        rowData[1] = buf[1];
        rowData[2] = buf[2];
        rowData[3] = buf[3];
      }
      else
      {
        std::memcpy ( rowData, buf,
                      (size_t) count * sizeof(*rowData) );
      }
    }

    data_.emitEvents ();
  }
}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void StdIntDBColumn::printTo ( PrintWriter& out ) const
{
  Base::printTo ( out );
}


//=======================================================================
//   class StdFloatDBColumn
//=======================================================================


class StdFloatDBColumn : public StdDBColumn, public FloatDBColumn
{
 public:

  typedef StdFloatDBColumn  Self;
  typedef StdDBColumn       Base;
  typedef FloatDBColumn     ColumnType;


  inline                    StdFloatDBColumn

    ( DBData&                 data,
      const String&           name );

  virtual idx_t             getData

    ( double*                 buf,
      idx_t                   size,
      idx_t                   irow )       const override;

  virtual void              setData

    ( idx_t                   irow,
      const double*           buf,
      idx_t                   count )            override;

  virtual void              printTo

    ( PrintWriter&            out )        const override;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline StdFloatDBColumn::StdFloatDBColumn

  ( DBData&        data,
    const String&  name ) :

    Base ( data, name, FLOAT )

{}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


idx_t StdFloatDBColumn::getData

  ( double*  buf,
    idx_t    size,
    idx_t    irow ) const

{
  checkRowIndex ( irow );

  if ( size <= 0 )
  {
    return 0;
  }

  const Row*  row = rows_.find ( irow );

  if ( ! row )
  {
    return 0;
  }

  if ( row->size == 1 )
  {
    buf[0] = row->fval;

    return 1;
  }

  if ( row->size < size )
  {
    size = row->size;
  }

  const double* JEM_RESTRICT  rowData =

    (const double*) (heap_.data + row->addr);

  if      ( size == 1_idx )
  {
    buf[0] = rowData[0];
  }
  else if ( size == 2_idx )
  {
    buf[0] = rowData[0];
    buf[1] = rowData[1];
  }
  else if ( size == 3_idx )
  {
    buf[0] = rowData[0];
    buf[1] = rowData[1];
    buf[2] = rowData[2];
  }
  else if ( size == 4_idx )
  {
    buf[0] = rowData[0];
    buf[1] = rowData[1];
    buf[2] = rowData[2];
    buf[3] = rowData[3];
  }
  else
  {
    std::memcpy ( buf, rowData, (size_t) size * sizeof(*rowData) );
  }

  return size;
}


//-----------------------------------------------------------------------
//   setData
//-----------------------------------------------------------------------


void StdFloatDBColumn::setData

  ( idx_t          irow,
    const double*  buf,
    idx_t          count )

{
  if ( count <= 0 )
  {
    clearRow ( irow );
  }
  else
  {
    checkRowIndex ( irow );

    Row&  row = rows_[irow];

    if ( count == row.size )
    {
      data_.events.set ( DBData::NEW_VALUES );
    }
    else
    {
      data_.events.set ( DBData::NEW_STRUCT );

      resize_ ( row, count );
    }

    if ( count == 1 )
    {
      row.fval = buf[0];
    }
    else
    {
      double* JEM_RESTRICT  rowData =

        (double*) (heap_.data + row.addr);

      if      ( count == 2_idx )
      {
        rowData[0] = buf[0];
        rowData[1] = buf[1];
      }
      else if ( count == 3_idx )
      {
        rowData[0] = buf[0];
        rowData[1] = buf[1];
        rowData[2] = buf[2];
      }
      else if ( count == 4_idx )
      {
        rowData[0] = buf[0];
        rowData[1] = buf[1];
        rowData[2] = buf[2];
        rowData[3] = buf[3];
      }
      else
      {
        std::memcpy ( rowData, buf,
                      (size_t) count * sizeof(*rowData) );
      }
    }

    data_.emitEvents ();
  }
}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void StdFloatDBColumn::printTo ( PrintWriter& out ) const
{
  Base::printTo ( out );
}


//=======================================================================
//   class StdDBColumnList
//=======================================================================


template <class DBCol>

  class StdDBColumnList

{
 public:

  typedef StdDBColumnList   Self;
  typedef StdDBData         DBData;
  typedef StdDBRowArray     RowArray;
  typedef typename
    DBCol::ColumnType       ColumnType;
  typedef
    Array<ColumnType*>      ColumnArray;


  inline                    StdDBColumnList ();
  inline                   ~StdDBColumnList ();

  void                      readFrom

    ( DBData&                 data,
      DataInput&              in );

  void                      writeTo

    ( DataOutput&             out )            const;

  void                      copy

    ( DBData&                 data,
      const Self&             rhs );

  inline DBCol*             find

    ( const String&           name )           const;

  inline DBCol*             pushBack

    ( DBData&                 data,
      const String&           name );

  inline void               clear           ();

  inline void               reorder

    ( RowArray&               tmp,
      const Reordering&       reord );

  inline ColumnArray        toArray         () const;


 private:

  ArrayBuffer<ColumnType*>  columns_;

};


typedef StdDBColumnList
  <StdIntDBColumn>          StdIntDBColumnList;

typedef StdDBColumnList
  <StdFloatDBColumn>        StdFloatDBColumnList;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


template <class C>

 inline StdDBColumnList<C>::StdDBColumnList ()

{}


template <class C>

  inline StdDBColumnList<C>::~StdDBColumnList ()

{
  clear ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


template <class C>

  void StdDBColumnList<C>::readFrom

  ( DBData&     data,
    DataInput&  in )

{
  uidx_t  sz;

  decode ( in, sz );

  idx_t  n = (idx_t) sz;

  clear ();
  columns_.reserve ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    String  name;

    decode ( in, name );

    C*  col = pushBack ( data, name );

    col->readFrom ( in );
  }
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


template <class C>

  void StdDBColumnList<C>::writeTo ( DataOutput& out ) const

{
  const idx_t  n = columns_.size ();

  encode ( out, n );

  for ( idx_t i = 0; i < n; i++ )
  {
    C*  col = static_cast<C*> ( columns_[i] );

    encode ( out, col->name );

    col->writeTo ( out );
  }
}


//-----------------------------------------------------------------------
//   copy
//-----------------------------------------------------------------------


template <class C>

  void StdDBColumnList<C>::copy

  ( DBData&      data,
    const Self&  rhs )

{
  const idx_t  n = rhs.columns_.size ();

  if ( columns_.size() > 0 )
  {
    clear ();
  }

  columns_.reserve ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    C*  rhsCol = static_cast<C*> ( rhs.columns_[i] );
    C*  newCol = pushBack ( data, rhsCol->name );

    newCol->copy ( *rhsCol );
  }
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


template <class C>

  inline C* StdDBColumnList<C>::find ( const String& name ) const

{
  const idx_t  n = columns_.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    C*  col = static_cast<C*> ( columns_[i] );

    if ( col->name == name )
    {
      return col;
    }
  }

  return nullptr;
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


template <class C>

  inline C* StdDBColumnList<C>::pushBack

  ( DBData&        data,
    const String&  name )

{
  C*  col = new C ( data, name );

  try
  {
    columns_.pushBack ( col );
  }
  catch ( ... )
  {
    delete col;
    throw;
  }

  return col;
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


template <class C>

  inline void StdDBColumnList<C>::clear ()

{
  const idx_t  n = columns_.size ();

  for ( idx_t i = n - 1; i >= 0; i-- )
  {
    delete static_cast<C*> ( columns_[i] );
  }

  columns_.clear ();
}


//-----------------------------------------------------------------------
//   reorder
//-----------------------------------------------------------------------


template <class C>

  inline void StdDBColumnList<C>::reorder

  ( RowArray&          tmp,
    const Reordering&  reord )

{
  const idx_t  n = columns_.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    static_cast<C*> ( columns_[i] ) -> reorder ( tmp, reord );
  }
}


//-----------------------------------------------------------------------
//   toArray
//-----------------------------------------------------------------------


template <class C>

  inline typename StdDBColumnList<C>::ColumnArray
  StdDBColumnList<C>::toArray () const

{
  return columns_.toArray ();
}


//=======================================================================
//   class StdDatabase::Data_
//=======================================================================


class StdDatabase::Data_ : public Collectable,
                           public StdDBData
{
 public:

  typedef Data_             Self;
  typedef StdDBData         DBData;


  explicit inline           Data_

    ( Database*               dbase,
      const Ref<ItemSet>&     items = nullptr );

  inline                    Data_

    ( const Self&             rhs );

  inline void               readFrom

    ( ObjectInput&            in );

  inline void               writeTo

    ( ObjectOutput&           out )    const;


 public:

  StdIntDBColumnList        icolumns;
  StdFloatDBColumnList      fcolumns;

};


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


inline StdDatabase::Data_::Data_

  ( Database*            dbase,
    const Ref<ItemSet>&  items ) :

    DBData ( dbase, items )

{}


inline StdDatabase::Data_::Data_ ( const Self& rhs ) :

  DBData ( rhs.dbase, rhs.rowItems )

{
  icolumns.copy ( *this, rhs.icolumns );
  fcolumns.copy ( *this, rhs.fcolumns );
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


inline void StdDatabase::Data_::readFrom ( ObjectInput& in )
{
  decode ( in, rowItems );

  icolumns.readFrom ( *this, in );
  fcolumns.readFrom ( *this, in );

  events.reset ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void StdDatabase::Data_::writeTo ( ObjectOutput& out ) const
{
  encode ( out, rowItems );

  icolumns.writeTo ( out );
  fcolumns.writeTo ( out );
}


//=======================================================================
//   class StdDatabase
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


StdDatabase::StdDatabase ()
{}


StdDatabase::StdDatabase

  ( const String&        name,
    const Ref<ItemSet>&  items ) :

    Super ( name )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  data_ = newInstance<Data_> ( this, items );

  connect_ ();
}


StdDatabase::StdDatabase ( const Self& rhs ) :

  Super ( rhs.myName_ )

{
  data_ = newInstance<Data_> ( * rhs.data_ );

  connect_ ();
}


StdDatabase::~StdDatabase ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void StdDatabase::resetEvents ()
{
  data_->events.reset ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void StdDatabase::readFrom ( ObjectInput& in )
{
  data_ = newInstance<Data_> ( this );

  decode ( in, myName_ );
  data_->readFrom ( in );

  connect_ ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void StdDatabase::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_ );
  data_->writeTo  ( out );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> StdDatabase::clone () const
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   getRowItems
//-----------------------------------------------------------------------


ItemSet* StdDatabase::getRowItems () const
{
  return data_->rowItems.get ();
}


//-----------------------------------------------------------------------
//   clearAll
//-----------------------------------------------------------------------


void StdDatabase::clearAll ()
{
  Data_&  d = * data_;

  d.icolumns.clear ();
  d.fcolumns.clear ();
  d.events  .reset ();
}


//-----------------------------------------------------------------------
//   addIntColumn
//-----------------------------------------------------------------------


StdDatabase::IntColumn* StdDatabase::addIntColumn

  ( const String& name )

{
  Data_&      d   = * data_;
  IntColumn*  col = d.icolumns.find ( name );

  if ( ! col )
  {
    col = d.icolumns.pushBack ( d, name );
  }

  return col;
}


//-----------------------------------------------------------------------
//   findIntColumn
//-----------------------------------------------------------------------


StdDatabase::IntColumn* StdDatabase::findIntColumn

  ( const String& name ) const

{
  return data_->icolumns.find ( name );
}


//-----------------------------------------------------------------------
//   getIntColumns
//-----------------------------------------------------------------------


Array<IntDBColumn*> StdDatabase::getIntColumns () const
{
  return data_->icolumns.toArray ();
}


//-----------------------------------------------------------------------
//   addFloatColumn
//-----------------------------------------------------------------------


StdDatabase::FloatColumn* StdDatabase::addFloatColumn

  ( const String&  name )

{
  Data_&        d   = * data_;
  FloatColumn*  col = findFloatColumn ( name );

  if ( ! col )
  {
    col = d.fcolumns.pushBack ( d, name );
  }

  return col;
}


//-----------------------------------------------------------------------
//   findFloatColumn
//-----------------------------------------------------------------------


StdDatabase::FloatColumn* StdDatabase::findFloatColumn

  ( const String&  name ) const

{
  return data_->fcolumns.find ( name );
}


//-----------------------------------------------------------------------
//   getFloatColumns
//-----------------------------------------------------------------------


Array<FloatDBColumn*> StdDatabase::getFloatColumns () const
{
  return data_->fcolumns.toArray ();
}


//-----------------------------------------------------------------------
//   emitEvents_
//-----------------------------------------------------------------------


void StdDatabase::emitEvents_ ( int events )
{
  if      ( events & StdDBData::NEW_STRUCT )
  {
    newStructEvent.emit ( *this );
  }
  else if ( events & StdDBData::NEW_VALUES )
  {
    newValuesEvent.emit ( *this );
  }
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void StdDatabase::connect_ ()
{
  using jem::util::connect;

  Data_&  d = * data_;

  connect ( d.rowItems->newSizeEvent,  this, & Self::itemsResized_   );
  connect ( d.rowItems->newOrderEvent, this, & Self::itemsReordered_ );
}


//-----------------------------------------------------------------------
//   itemsResized_
//-----------------------------------------------------------------------


void StdDatabase::itemsResized_ ( idx_t size )
{
  if ( size == 0 )
  {
    clearData ();
  }
}


//-----------------------------------------------------------------------
//   itemsReordered_
//-----------------------------------------------------------------------


void StdDatabase::itemsReordered_ ( const Reordering& reord )
{
  Data_&    d = * data_;

  StdDBRow       zero;
  StdDBRowArray  tmp ( zero );

  d.icolumns.reorder ( tmp, reord );
  d.fcolumns.reorder ( tmp, reord );

  d.emitEvents   ();
  d.events.reset ();
}


JIVE_END_PACKAGE( util )
