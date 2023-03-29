
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
#include <jem/mp/Buffer.h>
#include <jem/mp/Context.h>
#include <jem/mp/RequestList.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jive/log.h>
#include <jive/util/XTable.h>
#include <jive/util/ItemSet.h>
#include <jive/mp/Globdat.h>
#include <jive/mp/BorderXData.h>
#include <jive/mp/ExchangeTags.h>
#include <jive/mp/TableExchanger.h>


JEM_DEFINE_CLASS( jive::mp::TableExchanger );


JIVE_BEGIN_PACKAGE( mp )


using jem::newInstance;
using jem::mp::RecvBuffer;
using jem::mp::SendBuffer;
using jem::mp::RequestList;


//=======================================================================
//   class TableExchanger::XData_
//=======================================================================


class TableExchanger::XData_ : public BorderXData
{
 public:

  typedef BorderXData     Super;


  explicit inline         XData_

    ( const BorderSet&      borders );


 public:

  IdxVector               rowSizes;
  IdxVector               colIndices;
  Vector                  rowValues;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline TableExchanger::XData_::XData_ ( const BorderSet& borders ) :

  Super ( borders )

{}


//=======================================================================
//   class TableExchanger::RecvData_
//=======================================================================


class TableExchanger::RecvData_ : public XData_
{
 public:

  typedef XData_          Super;


  inline                  RecvData_

    ( const Ref<Context>&   mpx,
      const BorderSet&      borders );


 public:

  Ref<RequestList>        requests;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline TableExchanger::RecvData_::RecvData_

  ( const Ref<Context>&  mpx,
    const BorderSet&     borders ) :

    Super ( borders )

{
  requests = mpx->newRequestList ();
}


//=======================================================================
//   class TableExchanger::SendData_
//=======================================================================


class TableExchanger::SendData_ : public XData_
{
 public:

  typedef XData_          Super;


  inline                  SendData_

    ( const Ref<Context>&   mpx,
      const BorderSet&      borders );


 public:

  Ref<Context>            mpContext;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline TableExchanger::SendData_::SendData_

  ( const Ref<Context>&  mpx,
    const BorderSet&     borders ) :

    Super     ( borders ),
    mpContext (     mpx )

{}


//=======================================================================
//   class TableExchanger::Utils_
//=======================================================================


class TableExchanger::Utils_
{
 public:

  static void             exchangeRowSizes

    ( RecvData_&            rdata,
      SendData_&            sdata,
      const SparseMatrix&   matrix );

  static void             exchangeColIndices

    ( RecvData_&            rdata,
      SendData_&            sdata,
      const SparseMatrix&   matrix );

  static void             exchangeData

    ( RecvData_&            rdata,
      SendData_&            sdata,
      const SparseMatrix&   matrix );

  static void             addToTable

    ( XTable&               table,
      const RecvData_&      rdata );

  static void             updateTable

    ( XTable&               table,
      const RecvData_&      rdata );

  static void             exchangeWeights

    ( RecvData_&            rdata,
      SendData_&            sdata,
      const Vector&         weights );

};


//-----------------------------------------------------------------------
//   exchangeRowSizes
//-----------------------------------------------------------------------


void TableExchanger::Utils_::exchangeRowSizes

  ( RecvData_&           rd,
    SendData_&           sd,
    const SparseMatrix&  mtx )

{
  JEM_ASSERT ( mtx.isContiguous() );

  const idx_t*  rowOffsets = mtx.getOffsetPtr ();

  idx_t*        ptr;

  int           rank;
  idx_t         msgSize;
  idx_t         maxMsgSize;
  idx_t         ifirst;
  idx_t         ilast;
  idx_t         irow;

  idx_t         i, j, n;


  rd.requests->clear ();
  rd.rowSizes.resize ( rd.borderItems.size() );
  sd.rowSizes.resize ( sd.borderItems.size() );

  for ( idx_t ib = 0; ib < rd.borderCount; ib++ )
  {
    rank   = (int) rd.borderIDs[ib];
    ifirst = rd.borderOffsets[ib];
    ilast  = rd.borderOffsets[ib + 1];
    ptr    = rd.rowSizes.addr ( ifirst );
    n      = ilast - ifirst;

    rd.requests->addRequest ( RecvBuffer( ptr, n ), rank, XTAG_ );
  }

  rd.requests->startAll ();

  maxMsgSize = 0;

  for ( idx_t ib = 0; ib < sd.borderCount; ib++ )
  {
    rank    = (int) sd.borderIDs[ib];
    ifirst  = sd.borderOffsets[ib];
    ilast   = sd.borderOffsets[ib + 1];
    ptr     = sd.rowSizes.addr ( ifirst );
    msgSize = 0;

    for ( i = ifirst, j = 0; i < ilast; i++, j++ )
    {
      irow     = sd.borderItems[i];
      ptr[j]   = rowOffsets[irow + 1] - rowOffsets[irow];
      msgSize += ptr[j];
    }

    n = ilast - ifirst;

    sd.mpContext->send ( SendBuffer( ptr, n ), rank, XTAG_ );

    if ( msgSize > maxMsgSize )
    {
      maxMsgSize = msgSize;
    }
  }

  rd.requests->waitAll ();

  if ( rd.rowSizes.size() > 0 )
  {
    n = jem::sum ( rd.rowSizes );
  }
  else
  {
    n = 0;
  }

  rd.colIndices.resize ( n );
  rd.rowValues .resize ( n );
  sd.colIndices.resize ( maxMsgSize );
  sd.rowValues .resize ( maxMsgSize );
}


//-----------------------------------------------------------------------
//   exchangeColIndices
//-----------------------------------------------------------------------


void TableExchanger::Utils_::exchangeColIndices

  ( RecvData_&           rd,
    SendData_&           sd,
    const SparseMatrix&  mtx )

{
  JEM_ASSERT ( mtx.isContiguous() );

  const idx_t*  rowOffsets = mtx.getOffsetPtr ();
  const idx_t*  colIndices = mtx.getIndexPtr  ();

  idx_t*        ptr;

  int           rank;
  idx_t         ifirst;
  idx_t         ilast;
  idx_t         irow;

  idx_t         i, j, k, n;


  rd.requests->clear ();

  j = 0;

  for ( idx_t ib = 0; ib < rd.borderCount; ib++ )
  {
    rank   = (int) rd.borderIDs[ib];
    ifirst = rd.borderOffsets[ib];
    ilast  = rd.borderOffsets[ib + 1];
    ptr    = rd.colIndices.addr ( j );
    n      = 0;

    for ( i = ifirst; i < ilast; i++ )
    {
      n += rd.rowSizes[i];
    }

    rd.requests->addRequest ( RecvBuffer( ptr, n ), rank, XTAG_ );

    j += n;
  }

  rd.requests->startAll ();

  ptr = sd.colIndices.addr ();

  for ( idx_t ib = 0; ib < sd.borderCount; ib++ )
  {
    rank   = (int) sd.borderIDs[ib];
    ifirst = sd.borderOffsets[ib];
    ilast  = sd.borderOffsets[ib + 1];
    k      = 0;

    for ( i = ifirst; i < ilast; i++ )
    {
      irow = sd.borderItems[i];
      n    = rowOffsets[irow + 1];

      for ( j = rowOffsets[irow]; j < n; j++, k++ )
      {
        ptr[k] = colIndices[j];
      }
    }

    sd.mpContext->send ( SendBuffer( ptr, k ), rank, XTAG_ );
  }

  rd.requests->waitAll ();
}


//-----------------------------------------------------------------------
//   exchangeData
//-----------------------------------------------------------------------


void TableExchanger::Utils_::exchangeData

  ( RecvData_&           rd,
    SendData_&           sd,
    const SparseMatrix&  mtx )

{
  JEM_ASSERT ( mtx.isContiguous() );

  const idx_t*   rowOffsets = mtx.getOffsetPtr ();
  const double*  tabValues  = mtx.getValuePtr  ();

  double*        ptr;

  int            rank;
  idx_t          ifirst;
  idx_t          ilast;
  idx_t          irow;

  idx_t          i, j, k, n;


  rd.requests->clear ();

  j = 0;

  for ( idx_t ib = 0; ib < rd.borderCount; ib++ )
  {
    rank   = (int) rd.borderIDs[ib];
    ifirst = rd.borderOffsets[ib];
    ilast  = rd.borderOffsets[ib + 1];
    ptr    = rd.rowValues.addr ( j );
    n      = 0;

    for ( i = ifirst; i < ilast; i++ )
    {
      n += rd.rowSizes[i];
    }

    rd.requests->addRequest ( RecvBuffer( ptr, n ), rank, XTAG_ );

    j += n;
  }

  rd.requests->startAll ();

  ptr = sd.rowValues.addr ();

  for ( idx_t ib = 0; ib < sd.borderCount; ib++ )
  {
    rank   = (int) sd.borderIDs[ib];
    ifirst = sd.borderOffsets[ib];
    ilast  = sd.borderOffsets[ib + 1];
    k      = 0;

    for ( i = ifirst; i < ilast; i++ )
    {
      irow = sd.borderItems[i];
      n    = rowOffsets[irow + 1];

      for ( j = rowOffsets[irow]; j < n; j++, k++ )
      {
        ptr[k] = tabValues[j];
      }
    }

    sd.mpContext->send ( SendBuffer( ptr, k ), rank, XTAG_ );
  }

  rd.requests->waitAll ();
}


//-----------------------------------------------------------------------
//   addToTable
//-----------------------------------------------------------------------


void TableExchanger::Utils_::addToTable

  ( XTable&           tab,
    const RecvData_&  rd )

{
  const double*  rowValues  = rd.rowValues .addr ();
  const idx_t*   colIndices = rd.colIndices.addr ();

  idx_t          ifirst;
  idx_t          ilast;
  idx_t          irow;

  idx_t          i, n;


  for ( idx_t ib = 0; ib < rd.borderCount; ib++ )
  {
    ifirst = rd.borderOffsets[ib];
    ilast  = rd.borderOffsets[ib + 1];

    for ( i = ifirst; i < ilast; i++ )
    {
      n    = rd.rowSizes[i];
      irow = rd.borderItems[i];

      tab.addData ( &irow, 1, colIndices, n, rowValues );

      colIndices += n;
      rowValues  += n;
    }
  }
}


//-----------------------------------------------------------------------
//   updateTable
//-----------------------------------------------------------------------


void TableExchanger::Utils_::updateTable

  ( XTable&           tab,
    const RecvData_&  rd )

{
  const double*  rowValues  = rd.rowValues .addr ();
  const idx_t*   colIndices = rd.colIndices.addr ();

  idx_t          ifirst;
  idx_t          ilast;
  idx_t          irow;

  idx_t          i, n;


  for ( idx_t ib = 0; ib < rd.borderCount; ib++ )
  {
    ifirst = rd.borderOffsets[ib];
    ilast  = rd.borderOffsets[ib + 1];

    for ( i = ifirst; i < ilast; i++ )
    {
      n    = rd.rowSizes[i];
      irow = rd.borderItems[i];

      tab.setData ( &irow, 1, colIndices, n, rowValues );

      colIndices += n;
      rowValues  += n;
    }
  }
}


//-----------------------------------------------------------------------
//   exchangeWeights
//-----------------------------------------------------------------------


void TableExchanger::Utils_::exchangeWeights

  ( RecvData_&     rd,
    SendData_&     sd,
    const Vector&  w )

{
  double*  ptr;

  int      rank;
  idx_t    ifirst;
  idx_t    ilast;

  idx_t    i, j, n;


  rd.requests->clear ();

  rd.rowValues.resize ( rd.borderItems.size()      );
  sd.rowValues.resize ( sd.borders.maxBorderSize() );

  for ( idx_t ib = 0; ib < rd.borderCount; ib++ )
  {
    rank   = (int) rd.borderIDs[ib];
    ifirst = rd.borderOffsets[ib];
    ilast  = rd.borderOffsets[ib + 1];
    ptr    = rd.rowValues.addr ( ifirst );
    n      = ilast - ifirst;

    rd.requests->addRequest ( RecvBuffer( ptr, n ), rank, XTAG_ );
  }

  rd.requests->startAll ();

  ptr = sd.rowValues.addr ();

  for ( idx_t ib = 0; ib < sd.borderCount; ib++ )
  {
    rank   = (int) sd.borderIDs[ib];
    ifirst = sd.borderOffsets[ib];
    ilast  = sd.borderOffsets[ib + 1];

    for ( i = ifirst, j = 0; i < ilast; i++, j++ )
    {
      ptr[j] = w[sd.borderItems[i]];
    }

    n = ilast - ifirst;

    sd.mpContext->send ( SendBuffer( ptr, n ), rank, XTAG_ );
  }

  rd.requests->waitAll ();

  if ( rd.borders == sd.borders )
  {
    for ( idx_t ib = 0; ib < rd.borderCount; ib++ )
    {
      ifirst = rd.borderOffsets[ib];
      ilast  = rd.borderOffsets[ib + 1];

      for ( i = ifirst; i < ilast; i++ )
      {
        w[rd.borderItems[i]] += rd.rowValues[i];
      }
    }
  }
  else
  {
    for ( idx_t ib = 0; ib < rd.borderCount; ib++ )
    {
      ifirst = rd.borderOffsets[ib];
      ilast  = rd.borderOffsets[ib + 1];

      for ( i = ifirst; i < ilast; i++ )
      {
        w[rd.borderItems[i]] = rd.rowValues[i];
      }
    }
  }
}


//=======================================================================
//   class TableExchanger
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  TableExchanger::XTAG_ = ExchangeTags::TABLE;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


TableExchanger::TableExchanger

  ( const Ref<Context>&  mpx,
    const BorderSet&     borders )

{
  JEM_PRECHECK ( mpx && borders );

  init_ ( mpx, borders, borders );
}


TableExchanger::TableExchanger

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


TableExchanger::~TableExchanger ()
{}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String TableExchanger::toString () const
{
  String  isetName = recvData_->borders.getItems()->getName ();

  return String::format ( "%s(%s)", CLASS_NAME, isetName );
}


//-----------------------------------------------------------------------
//   exchange
//-----------------------------------------------------------------------


void TableExchanger::exchange ( XTable& table )
{
  RecvData_&  rd = * recvData_;
  SendData_&  sd = * sendData_;

  ItemSet*    myItems = rd.borders.getItems ();


  if ( table.getRowItems() != myItems )
  {
    String  tableName = table   .getName     ();
    String  isetName  = myItems->getName     ();
    String  itemName  = myItems->getItemName ();

    throw jem::IllegalArgumentException (
      CLASS_NAME,
      String::format (
        "table `%s\' is not associated with %s set `%s\'",
        tableName,
        itemName,
        isetName
      )
    );
  }

  if ( ! updated_ )
  {
    update_ ();
  }

  if ( rd.borderCount + sd.borderCount > 0 )
  {
    SparseMatrix  mtx = table.toMatrix ();

    if ( ! mtx.isContiguous() )
    {
      mtx = mtx.clone ();
    }

    Utils_::exchangeRowSizes   ( rd, sd, mtx );
    Utils_::exchangeColIndices ( rd, sd, mtx );
    Utils_::exchangeData       ( rd, sd, mtx );

    if ( rd.borders == sd.borders )
    {
      Utils_::addToTable       ( table, rd );
    }
    else
    {
      Utils_::updateTable      ( table, rd );
    }
  }
}


//-----------------------------------------------------------------------
//   exchange (with row weights)
//-----------------------------------------------------------------------


void TableExchanger::exchange

  ( XTable&        table,
    const Vector&  weights )

{
  JEM_PRECHECK ( table.rowCount() == weights.size() );

  exchange ( table );

  if ( recvData_->borderCount + sendData_->borderCount > 0 )
  {
    Utils_::exchangeWeights ( *recvData_, *sendData_, weights );
  }
}


//-----------------------------------------------------------------------
//   getItems
//-----------------------------------------------------------------------


ItemSet* TableExchanger::getItems () const
{
  return recvData_->borders.getItems ();
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void TableExchanger::store ( const Properties& globdat ) const
{
  Ref<ItemSet>  items = recvData_->borders.getItems ();
  Ref<Self>     self  = const_cast<Self*> ( this );

  Globdat::storeFor ( Globdat::TABLE_EXCHANGER,
                      self, items, globdat );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


Ref<TableExchanger> TableExchanger::find

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  Ref<Self>  tex;

  Globdat::findFor ( tex, Globdat::TABLE_EXCHANGER, items, globdat );

  return tex;
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<TableExchanger> TableExchanger::get

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  Properties  itemdat = Globdat::getDataFor ( items, globdat );
  Ref<Self>   tex;

  if ( ! itemdat.find( tex, Globdat::TABLE_EXCHANGER ) )
  {
    tex = newInstance<Self> (
      Globdat::getMPContext ( globdat ),
      BorderSet::get ( RECV_BORDERS, items, globdat ),
      BorderSet::get ( SEND_BORDERS, items, globdat )
    );

    itemdat.set ( Globdat::TABLE_EXCHANGER, tex );
  }

  return tex;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void TableExchanger::init_

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

  if ( sb != rb )
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


void TableExchanger::update_ ()
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


void TableExchanger::invalidate_ ()
{
  updated_ = false;
}


JIVE_END_PACKAGE( mp )
