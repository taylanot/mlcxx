
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
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/array/utilities.h>
#include <jem/base/array/intrinsics.h>
#include <jem/mp/Buffer.h>
#include <jem/mp/Context.h>
#include <jem/mp/RequestList.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jive/log.h>
#include <jive/util/ItemSet.h>
#include <jive/mp/Globdat.h>
#include <jive/mp/BorderXData.h>
#include <jive/mp/ExchangeTags.h>
#include <jive/mp/DataExchanger.h>


JEM_DEFINE_CLASS( jive::mp::DataExchanger );


JIVE_BEGIN_PACKAGE( mp )


using jem::castTo;
using jem::newInstance;
using jem::Array;
using jem::mp::RecvBuffer;
using jem::mp::SendBuffer;
using jem::mp::RequestList;
using jive::util::DBColumn;


//=======================================================================
//   class DataExchanger::XData_
//=======================================================================


class DataExchanger::XData_ : public BorderXData
{
 public:

  typedef BorderXData       Super;


  explicit inline           XData_

    ( const BorderSet&        borders );


 public:

  IdxVector                 rowSizes;
  IdxVector                 intBuffer;
  Vector                    floatBuffer;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline DataExchanger::XData_::XData_ ( const BorderSet& borders ) :

  Super ( borders )

{}


//=======================================================================
//   class DataExchanger::RecvData_
//=======================================================================


class DataExchanger::RecvData_ : public XData_
{
 public:

  typedef XData_            Super;


  inline                    RecvData_

    ( const Ref<Context>&     mpx,
      const BorderSet&        borders );


 public:

  Ref<RequestList>          requests;
  idx_t                     rbufSize;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline DataExchanger::RecvData_::RecvData_

  ( const Ref<Context>&  mpx,
    const BorderSet&     borders ) :

    Super ( borders )

{
  requests = mpx->newRequestList ();
  rbufSize = 0;
}


//=======================================================================
//   class DataExchanger::SendData_
//=======================================================================


class DataExchanger::SendData_ : public XData_
{
 public:

  typedef XData_            Super;


  inline                    SendData_

    ( const Ref<Context>&     mpx,
      const BorderSet&        borders );


 public:

  Ref<Context>              mpContext;
  idx_t                     maxMsgSize;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline DataExchanger::SendData_::SendData_

  ( const Ref<Context>&  mpx,
    const BorderSet&     borders ) :

    Super      ( borders ),
    mpContext  (     mpx )

{
  maxMsgSize = 0;
}


//=======================================================================
//   class DataExchanger::Utils_
//=======================================================================


class DataExchanger::Utils_
{
 public:

  typedef Array<DBColumn*>  ColumnArray;


  static void               exchangeRowSizes

    ( RecvData_&              rd,
      SendData_&              sd,
      const ColumnArray&      cols );

  template <class T, class Column>

    static void             exchangeData

    ( RecvData_&              rd,
      SendData_&              sd,
      Array<T>&               rbuf,
      Array<T>&               sbuf,
      const Array<Column*>&   cols );

  template <class T, class Column>

    static void             addToColumns

    ( const RecvData_&        rd,
      const Array<T>&         rbuf,
      const Array<Column*>&   cols );

  template <class T, class Column>

    static void             updateColumns

    ( const RecvData_&        rd,
      const Array<T>&         rbuf,
      const Array<Column*>&   cols );

};


//-----------------------------------------------------------------------
//   exchangeRowSizes
//-----------------------------------------------------------------------


void DataExchanger::Utils_::exchangeRowSizes

  ( RecvData_&          rd,
    SendData_&          sd,
    const ColumnArray&  cols )

{
  const idx_t  colCount = cols.size ();

  DBColumn*    col;
  idx_t*       ptr;

  int          rank;
  idx_t        msgSize;
  idx_t        ifirst;
  idx_t        ilast;
  idx_t        irow;
  idx_t        jcol;
  idx_t        ib;

  idx_t        i, j, n;


  rd.rowSizes.resize ( colCount * rd.borderItems.size() );
  sd.rowSizes.resize ( colCount * sd.borderItems.size() );

  rd.requests->clear ();

  for ( ib = 0; ib < rd.borderCount; ib++ )
  {
    rank   = (int) rd.borderIDs[ib];
    ifirst = rd.borderOffsets[ib];
    ilast  = rd.borderOffsets[ib + 1];
    n      = colCount * (ilast - ifirst);
    ptr    = rd.rowSizes.addr ( ifirst * colCount );

    rd.requests->addRequest ( RecvBuffer( ptr, n ), rank, XTAG_ );
  }

  rd.requests->startAll ();

  sd.maxMsgSize = 0;

  for ( ib = 0; ib < sd.borderCount; ib++ )
  {
    rank    = (int) sd.borderIDs[ib];
    ifirst  = sd.borderOffsets[ib];
    ilast   = sd.borderOffsets[ib + 1];
    ptr     = sd.rowSizes.addr ( ifirst * colCount );
    msgSize = 0;

    for ( jcol = j = 0; jcol < colCount; jcol++ )
    {
      col = cols[jcol];

      for ( i = ifirst; i < ilast; i++, j++ )
      {
        irow     = sd.borderItems[i];
        ptr[j]   = col->rowSize ( irow );
        msgSize += ptr[j];
      }
    }

    n = colCount * (ilast - ifirst);

    sd.mpContext->send ( SendBuffer( ptr, n ), rank, XTAG_ );

    if ( msgSize > sd.maxMsgSize )
    {
      sd.maxMsgSize = msgSize;
    }
  }

  rd.requests->waitAll ();

  if ( rd.rowSizes.size() > 0 )
  {
    rd.rbufSize = jem::sum ( rd.rowSizes );
  }
  else
  {
    rd.rbufSize = 0;
  }
}


//-----------------------------------------------------------------------
//   exchangeData
//-----------------------------------------------------------------------


template <class T, class Column>

  void DataExchanger::Utils_::exchangeData

  ( RecvData_&             rd,
    SendData_&             sd,
    Array<T>&              rbuf,
    Array<T>&              sbuf,
    const Array<Column*>&  cols )

{
  const idx_t  colCount = cols.size ();

  Column*      col;
  T*           ptr;

  int          rank;
  idx_t        ifirst;
  idx_t        ilast;
  idx_t        irow;
  idx_t        jcol;
  idx_t        ib;

  idx_t        i, j, k, n;


  rd.requests->clear ();
  rbuf.resize ( rd.rbufSize );

  j = 0;

  for ( ib = 0; ib < rd.borderCount; ib++ )
  {
    rank   = (int) rd.borderIDs[ib];
    ifirst = colCount * rd.borderOffsets[ib];
    ilast  = colCount * rd.borderOffsets[ib + 1];
    ptr    = rbuf.addr ( j );
    n      = 0;

    for ( i = ifirst; i < ilast; i++ )
    {
      n += rd.rowSizes[i];
    }

    rd.requests->addRequest ( RecvBuffer( ptr, n ), rank, XTAG_ );

    j += n;
  }

  rd.requests->startAll ();

  sbuf.resize ( sd.maxMsgSize );

  k = 0;

  for ( ib = 0; ib < sd.borderCount; ib++ )
  {
    rank   = (int) sd.borderIDs[ib];
    ifirst = sd.borderOffsets[ib];
    ilast  = sd.borderOffsets[ib + 1];
    j      = 0;

    for ( jcol = 0; jcol < colCount; jcol++ )
    {
      col = cols[jcol];

      for ( i = ifirst; i < ilast; i++ )
      {
        irow = sd.borderItems[i];
        n    = sd.rowSizes[k++];

        col->getData ( sbuf.addr(j), n, irow );

        j   += n;
      }
    }

    sd.mpContext->send ( SendBuffer( sbuf.addr(), j ), rank, XTAG_ );
  }

  rd.requests->waitAll ();
}


//-----------------------------------------------------------------------
//   addToColumns
//-----------------------------------------------------------------------


template <class T, class Column>

  void DataExchanger::Utils_::addToColumns

  ( const RecvData_&       rd,
    const Array<T>&        rbuf,
    const Array<Column*>&  cols )

{
  const idx_t  colCount = cols.size ();

  Column*      col;

  Array<T>     tbuf;

  idx_t        ifirst;
  idx_t        ilast;
  idx_t        irow;
  idx_t        jcol;
  idx_t        ib;

  idx_t        i, j, k, m, n, p;


  n = 0;

  for ( jcol = 0; jcol < colCount; jcol++ )
  {
    n = jem::max ( n, cols[jcol]->maxRowSize() );
  }

  tbuf.resize ( n );

  j = k = 0;

  for ( ib = 0; ib < rd.borderCount; ib++ )
  {
    ifirst = rd.borderOffsets[ib];
    ilast  = rd.borderOffsets[ib + 1];

    for ( jcol = 0; jcol < colCount; jcol++ )
    {
      col = cols[jcol];

      for ( i = ifirst; i < ilast; i++ )
      {
        irow = rd.borderItems[i];
        n    = rd.rowSizes[k++];
        m    = col->getData ( tbuf.addr(), tbuf.size(), irow );

        for ( p = 0; p < m && p < n; p++ )
        {
          tbuf[p] += rbuf[j + p];
        }

        col->setData ( irow, tbuf.addr(), m );

        j += n;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   updateColumns
//-----------------------------------------------------------------------


template <class T, class Column>

  void DataExchanger::Utils_::updateColumns

  ( const RecvData_&       rd,
    const Array<T>&        rbuf,
    const Array<Column*>&  cols )

{
  const idx_t  colCount = cols.size ();

  Column*      col;

  idx_t        ifirst;
  idx_t        ilast;
  idx_t        irow;
  idx_t        jcol;
  idx_t        ib;

  idx_t        i, j, k, n;


  j = k = 0;

  for ( ib = 0; ib < rd.borderCount; ib++ )
  {
    ifirst = rd.borderOffsets[ib];
    ilast  = rd.borderOffsets[ib + 1];

    for ( jcol = 0; jcol < colCount; jcol++ )
    {
      col = cols[jcol];

      for ( i = ifirst; i < ilast; i++ )
      {
        irow = rd.borderItems[i];
        n    = rd.rowSizes[k++];

        if      ( n == 0 )
        {
          col->clearRow  ( irow );
        }
        else if ( n == 1 )
        {
          col->setValue  ( irow, rbuf[j] );
        }
        else
        {
          col->setValues ( irow, rbuf[slice(j,j + n)], n );
        }

        j += n;
      }
    }
  }
}


//=======================================================================
//   class DataExchanger
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  DataExchanger::XTAG_ = ExchangeTags::DBASE;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


DataExchanger::DataExchanger

  ( const Ref<Context>&  mpx,
    const BorderSet&     borders )

{
  JEM_PRECHECK ( mpx && borders );

  init_ ( mpx, borders, borders );
}


DataExchanger::DataExchanger

  ( const Ref<Context>&  mpx,
    const BorderSet&     recvBorders,
    const BorderSet&     sendBorders )

{
  JEM_PRECHECK ( mpx         &&
                 recvBorders &&
                 sendBorders );

  JEM_PRECHECK ( recvBorders.getItems() ==
                 sendBorders.getItems() );

  init_ ( mpx, recvBorders, sendBorders );
}


DataExchanger::~DataExchanger ()
{}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String DataExchanger::toString () const
{
  String  isetName = recvData_->borders.getItems()->getName ();

  return String::format ( "%s(%s)", CLASS_NAME, isetName );
}


//-----------------------------------------------------------------------
//   exchange (database)
//-----------------------------------------------------------------------


void DataExchanger::exchange ( Database& dbase )
{
  ItemSet*  myItems = recvData_->borders.getItems ();

  if ( dbase.getRowItems() != myItems )
  {
    String  dbName   = dbase   .getName     ();
    String  isetName = myItems->getName     ();
    String  itemName = myItems->getItemName ();

    throw jem::IllegalArgumentException (
      CLASS_NAME,
      String::format (
        "database `%s\' is not associated with %s set `%s\'",
        dbName,
        itemName,
        isetName
      )
    );
  }

  exchange ( dbase.getIntColumns()   );
  exchange ( dbase.getFloatColumns() );
}


//-----------------------------------------------------------------------
//   exchange (idx_t columns)
//-----------------------------------------------------------------------


void DataExchanger::exchange ( const IntColArray& cols )
{
  if ( cols.size() > 0 )
  {
    Array<DBColumn*>  dbCols ( castTo<DBColumn*>( cols ) );

    RecvData_&        rd = *recvData_;
    SendData_&        sd = *sendData_;

    if ( ! updated_ )
    {
      update_ ();
    }

    if ( rd.borderCount + sd.borderCount > 0 )
    {
      Utils_::exchangeRowSizes ( rd, sd, dbCols );

      Utils_::exchangeData     ( rd, sd,
                                 rd.intBuffer,
                                 sd.intBuffer,
                                 cols );

      if ( rd.borders == sd.borders )
      {
        Utils_::addToColumns   ( rd, rd.intBuffer, cols );
      }
      else
      {
        Utils_::updateColumns  ( rd, rd.intBuffer, cols );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   exchange (float columns)
//-----------------------------------------------------------------------


void DataExchanger::exchange ( const FloatColArray& cols )
{
  if ( cols.size() > 0 )
  {
    Array<DBColumn*>  dbCols ( castTo<DBColumn*>( cols ) );

    RecvData_&        rd = *recvData_;
    SendData_&        sd = *sendData_;

    if ( ! updated_ )
    {
      update_ ();
    }

    if ( rd.borderCount + sd.borderCount > 0 )
    {
      Utils_::exchangeRowSizes ( rd, sd, dbCols );

      Utils_::exchangeData     ( rd, sd,
                                 rd.floatBuffer,
                                 sd.floatBuffer,
                                 cols );

      if ( rd.borders == sd.borders )
      {
        Utils_::addToColumns   ( rd, rd.floatBuffer, cols );
      }
      else
      {
        Utils_::updateColumns  ( rd, rd.floatBuffer, cols );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void DataExchanger::store ( const Properties& globdat ) const
{
  Ref<ItemSet>  items = recvData_->borders.getItems ();
  Ref<Self>     self  = const_cast<Self*> ( this );

  Globdat::storeFor ( Globdat::DATA_EXCHANGER, self, items, globdat );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


Ref<DataExchanger> DataExchanger::find

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  Ref<Self>  self;

  Globdat::findFor ( self, Globdat::DATA_EXCHANGER, items, globdat );

  return self;
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<DataExchanger> DataExchanger::get

    ( const Ref<ItemSet>&  items,
      const Properties&    globdat )

{
  Properties  itemdat = Globdat::getDataFor ( items, globdat );
  Ref<Self>   self;

  if ( ! itemdat.find( self, Globdat::DATA_EXCHANGER ) )
  {
    self = newInstance<Self> (
      Globdat  ::getMPContext ( globdat ),
      BorderSet::get ( RECV_BORDERS, items, globdat ),
      BorderSet::get ( SEND_BORDERS, items, globdat )
    );

    itemdat.set ( Globdat::DATA_EXCHANGER, self );
  }

  return self;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void DataExchanger::init_

  ( const Ref<Context>&  mpx,
    const BorderSet&     rb,
    const BorderSet&     sb )

{
  using jem::util::connect;

  ItemSet*  items = rb.getItems ();


  recvData_ = newInstance<RecvData_> ( mpx, rb );
  sendData_ = newInstance<SendData_> ( mpx, sb );

  connect ( items->newSizeEvent,  this, & Self::invalidate_ );
  connect ( items->newOrderEvent, this, & Self::invalidate_ );
  connect ( rb.newSizeEvent (),   this, & Self::invalidate_ );
  connect ( rb.newTopoEvent (),   this, & Self::invalidate_ );
  connect ( rb.newOrderEvent(),   this, & Self::invalidate_ );

  if ( rb != sb )
  {
    connect ( sb.newSizeEvent (), this, & Self::invalidate_ );
    connect ( sb.newTopoEvent (), this, & Self::invalidate_ );
    connect ( sb.newOrderEvent(), this, & Self::invalidate_ );
  }

  updated_ = false;
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void DataExchanger::update_ ()
{
  ItemSet*  items = recvData_->borders.getItems ();

  print ( JIVE_LOG_DEBUG, toString (),
          " : updating exchange data structures ...\n" );

  BorderXData::init2 ( *recvData_, *sendData_ );

  items    ->        resetEvents ();
  recvData_->borders.resetEvents ();
  sendData_->borders.resetEvents ();

  updated_ = true;
}


//-----------------------------------------------------------------------
//   invalidate_
//-----------------------------------------------------------------------


void DataExchanger::invalidate_ ()
{
  updated_ = false;
}


JIVE_END_PACKAGE( mp )
