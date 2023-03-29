
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
#include <jem/base/Array.h>
#include <jem/base/RuntimeException.h>
#include <jem/io/DataInputStream.h>
#include <jem/io/DataOutputStream.h>
#include <jem/io/ArrayInputStream.h>
#include <jem/io/ArrayOutputStream.h>
#include <jem/util/Event.h>
#include <jem/mp/utilities.h>
#include <jem/mp/BcastStream.h>
#include <jem/mp/RequestList.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemSet.h>
#include <jive/util/XDofSpace.h>
#include <jive/mp/Globdat.h>
#include <jive/mp/BorderSet.h>
#include <jive/mp/ExchangeTags.h>
#include <jive/mp/utilities.h>


JIVE_BEGIN_PACKAGE( mp )


//=======================================================================
//   class DofsTracker
//=======================================================================


class DofsTracker : public Object
{
 public:

  typedef Object          Super;
  typedef DofsTracker     Self;

  explicit                DofsTracker

    ( const Ref<DofSpace>&  dofs );

  inline void             reset        ();
  inline bool             hasChanged   () const noexcept;
  void                    markChanged  ();

  static Ref<Self>        get

    ( const Ref<DofSpace>&  dofs,
      const Properties&     globdat );


 protected:

  virtual                ~DofsTracker  ();


 private:

  Ref<DofSpace>           dofs_;
  bool                    changed_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


DofsTracker::DofsTracker ( const Ref<DofSpace>& dofs )
{
  using jem::util::connect;

  JEM_PRECHECK ( dofs );

  dofs_    = dofs;
  changed_ = true;

  connect ( dofs->newSizeEvent,  this, & Self::markChanged );
  connect ( dofs->newOrderEvent, this, & Self::markChanged );
}


DofsTracker::~DofsTracker ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


inline void DofsTracker::reset ()
{
  changed_ = false;

  dofs_->resetEvents ();
}


//-----------------------------------------------------------------------
//   hasChanged
//-----------------------------------------------------------------------


inline bool DofsTracker::hasChanged () const noexcept
{
  return changed_;
}


//-----------------------------------------------------------------------
//   markChanged
//-----------------------------------------------------------------------


void DofsTracker::markChanged ()
{
  changed_ = true;
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<DofsTracker> DofsTracker::get

  ( const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Properties  dofsdat = Globdat::getDataFor ( dofs, globdat );
  Ref<Self>   tracker;

  if ( ! dofsdat.find( tracker, Globdat::DOFS_TRACKER ) )
  {
    tracker = jem::newInstance<Self> ( dofs );

    dofsdat.set ( Globdat::DOFS_TRACKER, tracker );
  }

  return tracker;
}


//=======================================================================
//   non-member functions
//=======================================================================

//-----------------------------------------------------------------------
//   checkDofTypes
//-----------------------------------------------------------------------


static void           checkDofTypes

  ( const DofSpace&     dofs,
    Context&            mpx )

{
  using jem::newInstance;
  using jem::RuntimeException;
  using jem::io::InputStream;
  using jem::io::DataInputStream;
  using jem::io::DataOutputStream;
  using jem::io::ArrayInputStream;
  using jem::io::ArrayOutputStream;
  using jem::mp::allsum;
  using jem::mp::BcastStream;

  const idx_t       typeCount = dofs.typeCount ();

  Ref<InputStream>  inbuf;


  if ( mpx.myRank() == 0 )
  {
    Ref<ArrayOutputStream>  outbuf =

      newInstance<ArrayOutputStream> ();

    Ref<DataOutputStream>   output =

      newInstance<DataOutputStream>  ( outbuf );

    encode ( *output, typeCount );

    for ( idx_t i = 0; i < typeCount; i++ )
    {
      encode ( *output, dofs.getTypeName( i ) );
    }

    inbuf = newInstance<ArrayInputStream> ( outbuf->toArray() );
  }

  Ref<DataInputStream>  input =

    newInstance<DataInputStream> (
      newInstance<BcastStream> ( 0, &mpx, inbuf )
    );

  idx_t  errCount = 0;
  idx_t  n;

  decode ( *input, n );

  if ( n != typeCount )
  {
    errCount++;
  }

  for ( idx_t i = 0; i < n; i++ )
  {
    String  typeName;

    decode ( *input, typeName );

    if ( typeName != dofs.getTypeName( i ) )
    {
      errCount++;
    }
  }

  errCount = allsum ( mpx, errCount );

  if ( errCount > 0 )
  {
    String  what = "processes";

    if ( mpx.isShared() )
    {
      what = "threads";
    }

    throw RuntimeException (
      dofs.getContext (),
      String::format  (
        "inconsistent DOF spaces; different %s have "
        "defined different DOF types",
        what
      )
    );
  }
}


//-----------------------------------------------------------------------
//   syncDofs (with global data)
//-----------------------------------------------------------------------


void                  syncDofs

  ( XDofSpace&          dofs,
    const Properties&   globdat )

{
  using jem::mp::allsum;

  Ref<ItemSet>  items = dofs.getItems ();

  if ( BorderSet::hasBorders( items, globdat ) )
  {
    Ref<DofsTracker>  tracker = DofsTracker::get ( &dofs, globdat );
    Ref<MPContext>    mpx     = Globdat::getMPContext   ( globdat );
    int               status  = 0;

    syncDofs (
      dofs,
      *mpx,
      BorderSet::get ( RECV_BORDERS, items, globdat ),
      BorderSet::get ( SEND_BORDERS, items, globdat )
    );

    if ( tracker->hasChanged() )
    {
      status = 1;
    }

    if ( allsum( *mpx, status ) )
    {
      // The DofSpace has been modified in at least one process.
      // Trigger a dummy event to make sure that any data exchange
      // structures are properly rebuild.

      dofs.newSizeEvent.emit ( dofs.dofCount(), dofs );

      tracker->reset ();
    }
  }
}


//-----------------------------------------------------------------------
//   syncDofs (with one border set)
//-----------------------------------------------------------------------


void                  syncDofs

  ( XDofSpace&          dofs,
    Context&            mpx,
    const BorderSet&    borders )

{
  syncDofs ( dofs, mpx, borders, borders );
}


//-----------------------------------------------------------------------
//   syncDofs (with two border sets)
//-----------------------------------------------------------------------


void                  syncDofs

  ( XDofSpace&          dofs,
    Context&            mpx,
    const BorderSet&    recvBorders,
    const BorderSet&    sendBorders )

{
  using jem::RuntimeException;
  using jem::mp::SendBuffer;
  using jem::mp::RecvBuffer;
  using jem::mp::RequestList;
  using jive::util::ItemMap;

  JEM_PRECHECK ( recvBorders.getItems() == dofs.getItems() &&
                 sendBorders.getItems() == dofs.getItems() );

  const ItemSet&    items   = * dofs .getItems   ();
  const ItemMap&    imap    = * items.getItemMap ();

  const idx_t       rbCount = recvBorders.size ();
  const idx_t       sbCount = sendBorders.size ();

  const int         xtag    = ExchangeTags::DOFS;

  Ref<RequestList>  reqs;

  IdxVector         rsizes;
  IdxVector         offsets;
  IdxVector         idofs;
  IdxVector         iitems;
  IdxVector         itemIDs;
  IdxVector         itypes;
  IdxVector         rbuffer;
  IdxVector         sbuffer;

  int               rank;
  idx_t             iitem;
  idx_t             ib;
  idx_t             i, j, n;


  // Make sure that all processes/threads have defined the same DOF
  // types.

  checkDofTypes ( dofs, mpx );

  reqs = mpx.newRequestList ();
  rsizes.resize ( rbCount );

  for ( ib = 0; ib < rbCount; ib++ )
  {
    rank = (int) recvBorders.getBorderID ( ib );

    reqs->addRequest ( RecvBuffer( & rsizes[ib], 1 ), rank, xtag );
  }

  reqs->startAll ();

  getBorderDofs ( idofs, offsets, sendBorders, dofs );

  offsets *= 2;

  for ( ib = 0; ib < sbCount; ib++ )
  {
    rank = (int) sendBorders.getBorderID ( ib );
    n    = offsets[ib + 1] - offsets[ib];

    mpx.send ( SendBuffer( &n, 1 ), rank, xtag );
  }

  reqs->waitAll ();
  reqs->clear   ();

  if ( rbCount > 0 )
  {
    n = sum ( rsizes );
  }
  else
  {
    n = 0;
  }

  rbuffer.resize ( n );

  for ( ib = j = 0; ib < rbCount; ib++ )
  {
    rank = (int) recvBorders.getBorderID ( ib );
    n    = rsizes[ib];

    reqs->addRequest ( RecvBuffer( rbuffer.addr(j), n ),
                       rank, xtag );

    j += n;
  }

  reqs->startAll ();

  n = idofs.size ();

  sbuffer.resize ( 2 * n );
  iitems .resize ( n );

  if ( n > 0 )
  {
    itemIDs.ref ( sbuffer[slice(BEGIN,2 * n - 1,2)] );
    itypes .ref ( sbuffer[slice(1,END,2)] );

    dofs.decodeDofIndices ( iitems,  itypes, idofs );
    imap.getItemIDs       ( itemIDs, iitems );
  }

  for ( ib = 0; ib < sbCount; ib++ )
  {
    rank = (int) sendBorders.getBorderID ( ib );
    j    = offsets[ib];
    n    = offsets[ib + 1] - j;

    mpx.send ( SendBuffer( sbuffer.addr(j), n ), rank, xtag );
  }

  reqs->waitAll ();

  n = rbuffer.size() / 2;

  if ( n == 0 )
  {
    return;
  }

  itemIDs.ref ( rbuffer[slice(BEGIN,2 * n - 1,2)] );
  itypes .ref ( rbuffer[slice(1,END,2)] );

  iitems.resize ( n );

  imap.findItems ( iitems, itemIDs );

  for ( i = 0; i < n; i++ )
  {
    iitem = iitems[i];

    if ( iitem < 0 )
    {
      String  itemName = items.getItemName ();

      throw RuntimeException (
        JEM_FUNC,
        String::format (
          "invalid %s: %d (invalid border set?)",
          itemName,
          itemIDs[i]
        )
      );
    }

    dofs.addDof ( iitem, itypes[i] );
  }
}


JIVE_END_PACKAGE( mp )
