
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
#include <jem/base/Error.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/RuntimeException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/mp/Buffer.h>
#include <jem/mp/Context.h>
#include <jem/mp/RequestList.h>
#include <jem/util/Event.h>
#include <jem/util/HashMap.h>
#include <jem/util/FastAllocator.h>
#include <jem/util/Properties.h>
#include <jive/log.h>
#include <jive/util/error.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DofSpace.h>
#include <jive/mp/utilities.h>
#include <jive/mp/Globdat.h>
#include <jive/mp/BorderSet.h>
#include <jive/mp/ExchangeTags.h>
#include <jive/mp/VectorExchanger.h>


JEM_DEFINE_CLASS( jive::mp::VectorExchanger );


JIVE_BEGIN_PACKAGE( mp )


using jem::newInstance;
using jem::IllegalArgumentException;
using jem::Tuple;
using jem::mp::SendBuffer;
using jem::mp::RecvBuffer;
using jem::mp::RequestList;
using jem::util::HashMap;
using jem::util::FastAllocator;
using jive::util::sizeError;

typedef HashMap<idx_t,idx_t,FastAllocator>  Idx2Map;


//=======================================================================
//   class VectorExchanger::XData_
//=======================================================================


class VectorExchanger::XData_ : public Collectable
{
 public:

  static const int          INFO_TAG = ExchangeTags::VECTOR_INFO;
  static const int          DATA_TAG = ExchangeTags::VECTOR_DATA;


  explicit inline           XData_

    ( const Ref<Context>&     mpx );

  void                      update

    ( const DofSpace&         dofs,
      const BorderSet&        borders );

  void                      update

    ( const DofSpace&         dofs,
      const BorderSet&        recvBorders,
      const BorderSet&        sendBorders );


 public:

  idx_t                     sendInfo;
  IdxVector                 recvInfo;
  IdxVector                 recvTags;
  IdxVector                 sendDofs;
  IdxVector                 recvDofs;
  IntVector                 sendProcs;
  IntVector                 recvProcs;
  IdxVector                 sendOffsets;
  IdxVector                 recvOffsets;
  Vector                    sendVector;
  Vector                    recvVector;
  Vector                    sendBuffer;
  Vector                    recvBuffer;

  Ref<RequestList>          recvReqs;
  Ref<RequestList>          sendInfoReqs;
  Ref<RequestList>          recvInfoReqs;
  Ref<RequestList>          sendVecReqs;
  Ref<RequestList>          recvVecReqs;

  Idx2Map                   tagMap;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline VectorExchanger::XData_::XData_

  ( const Ref<Context>&  mpx )

{
  sendInfo     = 0;

  recvReqs     = mpx->newRequestList ();
  sendInfoReqs = mpx->newRequestList ();
  recvInfoReqs = mpx->newRequestList ();
  sendVecReqs  = mpx->newRequestList ();
  recvVecReqs  = mpx->newRequestList ();
}


//-----------------------------------------------------------------------
//   update (single border set)
//-----------------------------------------------------------------------


void VectorExchanger::XData_::update

  ( const DofSpace&   dofs,
    const BorderSet&  borders )

{
  const idx_t  borderCount = borders.size ();

  recvReqs    ->clear ();
  sendInfoReqs->clear ();
  recvInfoReqs->clear ();
  sendVecReqs ->clear ();
  recvVecReqs ->clear ();

  getBorderDofs      ( recvDofs, recvOffsets, borders, dofs );

  recvTags   .resize ( 0 );
  sendDofs   .ref    ( recvDofs );
  recvInfo   .resize ( borderCount );
  sendProcs  .resize ( borderCount );
  recvProcs  .ref    ( sendProcs   );
  sendOffsets.ref    ( recvOffsets );
  sendVector .resize ( sendDofs.size() );
  recvVector .resize ( recvDofs.size() );
  sendBuffer .resize ( 0 );
  recvBuffer .resize ( 0 );

  for ( idx_t ib = 0; ib < borderCount; ib++ )
  {
    idx_t  i  = recvOffsets[ib];
    idx_t  n  = recvOffsets[ib + 1] - i;

    int  rank = (int) borders.getBorderID ( ib );

    sendProcs[ib] = rank;

    sendInfoReqs->addRequest

      ( SendBuffer( &sendInfo, 1 ),          rank, INFO_TAG );

    recvInfoReqs->addRequest

      ( RecvBuffer( recvInfo.addr(ib), 1 ),  rank, INFO_TAG );

    sendVecReqs->addRequest

      ( SendBuffer( sendVector.addr(i), n ), rank, DATA_TAG );

    recvVecReqs->addRequest

      ( RecvBuffer( recvVector.addr(i), n ), rank, DATA_TAG );
  }

  sendInfo = 0;
  recvInfo = 0;

  tagMap.clear   ();
  tagMap.reserve ( borderCount );
}


//-----------------------------------------------------------------------
//   update (overlapping border sets)
//-----------------------------------------------------------------------


void VectorExchanger::XData_::update

  ( const DofSpace&   dofs,
    const BorderSet&  recvBorders,
    const BorderSet&  sendBorders )

{
  const idx_t  rbCount = recvBorders.size ();
  const idx_t  sbCount = sendBorders.size ();

  // Setup the send data structures.

  sendInfoReqs->clear ();
  sendVecReqs ->clear ();

  getBorderDofs     ( sendDofs, sendOffsets, sendBorders, dofs );

  sendProcs .resize ( sbCount );
  sendVector.resize ( sendDofs.size() );
  sendBuffer.resize ( 0 );

  for ( idx_t ib = 0; ib < sbCount; ib++ )
  {
    idx_t  i  = sendOffsets[ib];
    idx_t  n  = sendOffsets[ib + 1] - i;

    int  rank = (int) sendBorders.getBorderID ( ib );

    sendProcs[ib] = rank;

    sendInfoReqs->addRequest

      ( SendBuffer( &sendInfo, 1 ),          rank, INFO_TAG );

    sendVecReqs->addRequest

      ( SendBuffer( sendVector.addr(i), n ), rank, DATA_TAG );
  }

  // Setup the recv data structures.

  recvReqs    ->clear ();
  recvInfoReqs->clear ();
  recvVecReqs ->clear ();

  getBorderDofs     ( recvDofs, recvOffsets, recvBorders, dofs );

  recvTags  .resize ( 0 );
  recvInfo  .resize ( rbCount );
  recvProcs .resize ( rbCount );
  recvVector.resize ( recvDofs.size() );
  recvBuffer.resize ( 0 );

  for ( idx_t ib = 0; ib < rbCount; ib++ )
  {
    idx_t  i  = recvOffsets[ib];
    idx_t  n  = recvOffsets[ib + 1] - i;

    int  rank = (int) recvBorders.getBorderID ( ib );

    recvProcs[ib] = rank;

    recvInfoReqs->addRequest

      ( RecvBuffer( recvInfo.addr(ib), 1 ),  rank, INFO_TAG );

    recvVecReqs->addRequest

      ( RecvBuffer( recvVector.addr(i), n ), rank, DATA_TAG );
  }

  sendInfo = 0;
  recvInfo = 0;

  tagMap.clear   ();
  tagMap.reserve ( rbCount );
}


//=======================================================================
//   class VectorExchanger
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  VectorExchanger::INACTIVE_ = -1;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


VectorExchanger::VectorExchanger

  ( const Ref<Context>&   mpx,
    const Ref<DofSpace>&  dofs,
    const BorderSet&      borders ) :

    mpx_         ( mpx ),
    dofs_        ( dofs ),
    recvBorders_ ( borders ),
    sendBorders_ ( borders )

{
  JEM_PRECHECK ( mpx && dofs && borders );

  init_ ();
}


VectorExchanger::VectorExchanger

  ( const Ref<Context>&   mpx,
    const Ref<DofSpace>&  dofs,
    const BorderSet&      recvBorders,
    const BorderSet&      sendBorders ) :

    mpx_         ( mpx ),
    dofs_        ( dofs ),
    recvBorders_ ( recvBorders ),
    sendBorders_ ( sendBorders )

{
  JEM_PRECHECK ( mpx         &&
                 dofs        &&
                 recvBorders &&
                 sendBorders );

  init_ ();
}


VectorExchanger::~VectorExchanger ()
{}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String VectorExchanger::toString () const
{
  String  isetName = recvBorders_.getItems()->getName ();

  return String::format ( "%s(%s)", CLASS_NAME, isetName );
}


//-----------------------------------------------------------------------
//   barrier
//-----------------------------------------------------------------------


void VectorExchanger::barrier ()
{
  if ( xmode_ != INACTIVE_ )
  {
    xmodeError_ ( JEM_FUNC );
  }

  if ( ! updated_ )
  {
    update_ ();
  }

  XData_&  xd = *(xdata_[EXCHANGE]);

  xd.sendInfo = 42;

  xd.recvInfoReqs->startAll ();
  xd.sendInfoReqs->startAll ();
  xd.recvInfoReqs->waitAll  ();
  xd.sendInfoReqs->waitAll  ();

  if ( xd.recvInfo.size() > 0 )
  {
    if ( ! jem::testall( xd.recvInfo == xd.sendInfo ) )
    {
      throw jem::RuntimeException (
        JEM_FUNC,
        "received unexpected data; barrier failed"
      );
    }
  }
}


//-----------------------------------------------------------------------
//   startOne
//-----------------------------------------------------------------------


void VectorExchanger::startOne ( int mode )
{
  JEM_PRECHECK ( mode == EXCHANGE || mode == SCATTER );

  if ( xmode_ != INACTIVE_ )
  {
    xmodeError_ ( JEM_FUNC );
  }

  if ( ! updated_ )
  {
    update_ ();
  }

  xdata_[mode]->recvVecReqs->startAll ();

  xmode_ = mode;
}


//-----------------------------------------------------------------------
//   endOne
//-----------------------------------------------------------------------


void VectorExchanger::endOne

  ( const Vector&  recvVec,
    const Vector&  sendVec )

{
  JEM_PRECHECK ( recvVec.size() == sendVec.size() );

  if ( ! updated_ )
  {
    invalidError_ ( JEM_FUNC );
  }

  if ( xmode_ == INACTIVE_ )
  {
    xmodeError_   ( JEM_FUNC );
  }

  if ( sendVec.size() != dofCount_ )
  {
    sizeError ( JEM_FUNC, "send vector", sendVec.size(), dofCount_ );
  }

  XData_&  xd = *(xdata_[xmode_]);

  xd.sendVector = sendVec[xd.sendDofs];

  xd.sendVecReqs->startAll ();
  xd.recvVecReqs->waitAll  ();

  if ( overlap_ && xmode_ == EXCHANGE )
  {
    recvVec[xd.recvDofs]  = xd.recvVector;
  }
  else
  {
    recvVec[xd.recvDofs] += xd.recvVector;
  }

  xd.sendVecReqs->waitAll ();

  xmode_ = INACTIVE_;
}


//-----------------------------------------------------------------------
//   startSome
//-----------------------------------------------------------------------


void VectorExchanger::startSome

  ( int               mode,
    const IdxVector&  sendTags )

{
  JEM_PRECHECK ( mode == EXCHANGE || mode == SCATTER );

  const idx_t  sendCount = sendTags.size ();

  if ( xmode_ != INACTIVE_ )
  {
    xmodeError_ ( JEM_FUNC );
  }

  if ( ! updated_ )
  {
    update_ ();
  }

  XData_&  xd = *(xdata_[mode]);

  idx_t    ibuf;
  idx_t    itag;
  idx_t    k, n;


  // Exchange the number of vectors to be send,

  xd.sendInfo = sendCount;

  xd.recvInfoReqs->startAll ();
  xd.sendInfoReqs->startAll ();
  xd.recvInfoReqs->waitAll  ();
  xd.sendInfoReqs->waitAll  ();

  // Allocate space for the incoming tags and vectors.

  k = n = 0;

  for ( idx_t ip = 0; ip < xd.recvProcs.size(); ip++ )
  {
    idx_t  recvCount = xd.recvInfo[ip];

    k += recvCount;
    n += recvCount * (xd.recvOffsets[ip + 1] - xd.recvOffsets[ip]);
  }

  xd.recvTags  .resize ( k );
  xd.recvBuffer.resize ( n );

  // Start the receive requests for the incoming tags.

  xd.recvReqs->clear ();

  itag = 0;

  for ( idx_t ip = 0; ip < xd.recvProcs.size(); ip++ )
  {
    idx_t  recvCount = xd.recvInfo[ip];

    if ( recvCount > 0 )
    {
      xd.recvReqs->addRequest

        ( RecvBuffer( xd.recvTags.addr(itag), recvCount ),
          xd.recvProcs[ip], XData_::INFO_TAG );

      itag += recvCount;
    }
  }

  JEM_ASSERT ( itag == xd.recvTags.size() );

  xd.recvReqs->startAll ();

  // Send the tags to the neighbor processes.

  if ( sendCount > 0 )
  {
    IdxVector  sbuf = sendTags;

    if ( ! sbuf.isContiguous() )
    {
      sbuf.ref ( sendTags.clone() );
    }

    for ( idx_t ip = 0; ip < xd.sendProcs.size(); ip++ )
    {
      mpx_->send ( SendBuffer( sbuf.addr(), sendCount ),
                   xd.sendProcs[ip], XData_::INFO_TAG );
    }
  }

  xd.recvReqs->waitAll ();
  xd.recvReqs->clear   ();

  // Initialize the tag map. First, add the send tags.

  xd.tagMap.clear ();

  for ( n = itag = 0; itag < sendCount; itag++ )
  {
    idx_t  tag = sendTags[itag];

    if ( ! xd.tagMap.contains( tag ) )
    {
      xd.tagMap.insert ( tag, n++ );
    }
  }

  // Next, add the received tags to the tag map.

  k = xd.recvTags.size ();

  for ( itag = 0; itag < k; itag++ )
  {
    idx_t  tag = xd.recvTags[itag];

    if ( ! xd.tagMap.contains( tag ) )
    {
      xd.tagMap.insert ( tag, n++ );
    }
  }

  // Start the receive requests for the incoming vectors.

  ibuf = 0;

  for ( idx_t ip = 0; ip < xd.recvProcs.size(); ip++ )
  {
    idx_t  recvCount = xd.recvInfo[ip];

    if ( recvCount > 0 )
    {
      n = recvCount * (xd.recvOffsets[ip + 1] - xd.recvOffsets[ip]);

      xd.recvReqs->addRequest

        ( RecvBuffer( xd.recvBuffer.addr(ibuf), n ),
          xd.recvProcs[ip], XData_::DATA_TAG );

      ibuf += n;
    }
  }

  JEM_ASSERT ( ibuf == xd.recvBuffer.size() );

  xd.recvReqs->startAll ();

  xmode_ = mode;
}


//-----------------------------------------------------------------------
//   endSome
//-----------------------------------------------------------------------


void VectorExchanger::endSome

  ( Matrix&           recvVecs,
    IdxVector&        recvTags,
    const Matrix&     sendVecs,
    const IdxVector&  sendTags )

{
  JEM_PRECHECK ( sendTags.size() == sendVecs.size(1) );

  const idx_t  sendCount = sendTags.size ();

  if ( ! updated_ )
  {
    invalidError_ ( JEM_FUNC );
  }

  if ( xmode_ == INACTIVE_ )
  {
    xmodeError_   ( JEM_FUNC );
  }

  XData_&  xd = *(xdata_[xmode_]);

  idx_t    ibuf;
  idx_t    itag;
  idx_t    n;


  if ( sendCount != xd.sendInfo )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      String::format (
        "invalid number of send vectors: %d; expected %d",
        sendCount,
        xd.sendInfo
      )
    );
  }

  if ( sendCount > 0 && sendVecs.size(0) != dofCount_ )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      String::format (
        "send vectors have wrong size: %d; expected %d",
        sendVecs.size(0),
        dofCount_
      )
    );
  }

  // Send the vectors to the neighboring processes.

  if ( sendCount > 0 )
  {
    for ( idx_t ip = 0; ip < xd.sendProcs.size(); ip++ )
    {
      IdxVector  idofs = xd.sendDofs[slice(xd.sendOffsets[ip],
                                           xd.sendOffsets[ip + 1])];

      ibuf = 0;
      n    = idofs.size ();

      xd.sendBuffer.resize ( sendCount * n );

      for ( idx_t j = 0; j < sendCount; j++, ibuf += n )
      {
        xd.sendBuffer[slice(ibuf,ibuf + n)] = sendVecs(idofs,j);
      }

      mpx_->send ( SendBuffer( xd.sendBuffer.addr(),
                               xd.sendBuffer.size() ),
                   xd.sendProcs[ip], XData_::DATA_TAG );
    }
  }

  // Allocate space for the received tags and vectors.

  n = xd.tagMap.size ();

  recvVecs.resize ( dofCount_, n );
  recvTags.resize ( n );

  recvVecs = 0.0;

  // Copy the send vectors to the recv vectors first.

  for ( idx_t j = 0; j < sendCount; j++ )
  {
    idx_t  tag = sendTags [j];
    idx_t  jj  = xd.tagMap[tag];

    recvVecs[jj] = sendVecs[j];

    if ( overlap_ && xmode_ == EXCHANGE )
    {
      recvVecs(xd.recvDofs,jj) = 0.0;
    }

    recvTags[jj] = tag;
  }

  xd.recvReqs->waitAll ();

  // Collect the received tags and vectors.

  itag = ibuf = 0;

  for ( idx_t ip = 0; ip < xd.recvProcs.size(); ip++ )
  {
    IdxVector  idofs = xd.recvDofs[slice(xd.recvOffsets[ip],
                                         xd.recvOffsets[ip + 1])];
    idx_t    recvCount = xd.recvInfo[ip];

    n = idofs.size ();

    for ( idx_t j = 0; j < recvCount; j++, itag++, ibuf += n )
    {
      Vector  rbuf = xd.recvBuffer[slice(ibuf,ibuf + n)];
      idx_t   tag  = xd.recvTags  [itag];
      idx_t   jj   = xd.tagMap    [tag];

      recvTags[jj] = tag;

      if ( overlap_ && xmode_ == EXCHANGE )
      {
        recvVecs(idofs,jj)  = rbuf;
      }
      else
      {
        recvVecs(idofs,jj) += rbuf;
      }
    }
  }

  xmode_ = INACTIVE_;
}


//-----------------------------------------------------------------------
//   isDistributed
//-----------------------------------------------------------------------


bool VectorExchanger::isDistributed () const
{
  return (mpx_->size() > 1);
}


//-----------------------------------------------------------------------
//   getRecvDofs
//-----------------------------------------------------------------------


IdxVector VectorExchanger::getRecvDofs ( int mode ) const
{
  JEM_PRECHECK ( mode == EXCHANGE || mode == SCATTER );

  if ( ! updated_ )
  {
    const_cast<Self*>( this )->update_ ();
  }

  return xdata_[mode]->recvDofs;
}


//-----------------------------------------------------------------------
//   getSendDofs
//-----------------------------------------------------------------------


IdxVector VectorExchanger::getSendDofs ( int mode ) const
{
  JEM_PRECHECK ( mode == EXCHANGE || mode == SCATTER );

  if ( ! updated_ )
  {
    const_cast<Self*>( this )->update_ ();
  }

  return xdata_[mode]->sendDofs;
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void VectorExchanger::store ( const Properties& globdat ) const
{
  Ref<Self>  self ( const_cast<Self*>( this ) );

  Globdat::storeFor ( Globdat::VECTOR_EXCHANGER,
                      self, dofs_, globdat );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


Ref<VectorExchanger> VectorExchanger::find

  ( const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Ref<Self>  vex;

  Globdat::findFor ( vex, Globdat::VECTOR_EXCHANGER, dofs, globdat );

  return vex;
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<VectorExchanger> VectorExchanger::get

  ( const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Properties  dofsdat = Globdat::getDataFor ( dofs, globdat );
  Ref<Self>   vex;


  if ( ! dofsdat.find( vex, Globdat::VECTOR_EXCHANGER ) )
  {
    Ref<ItemSet>  items = dofs->getItems ();

    vex = newInstance<Self> (
      Globdat::getMPContext ( globdat ),
      dofs,
      BorderSet::get ( RECV_BORDERS, items, globdat ),
      BorderSet::get ( SEND_BORDERS, items, globdat )
    );

    dofsdat.set ( Globdat::VECTOR_EXCHANGER, vex );
  }

  return vex;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void VectorExchanger::init_ ()
{
  using jem::util::connect;

  ItemSet*  items = dofs_->getItems ();

  if ( recvBorders_.getItems() != items ||
       sendBorders_.getItems() != items )
  {
    throw IllegalArgumentException (
      CLASS_NAME,
      "incompatible exchange borders and dof space "
      "(different item sets)"
    );
  }

  dofCount_ = dofs_->dofCount ();
  overlap_  = (recvBorders_ != sendBorders_);
  xmode_    = INACTIVE_;
  updated_  = false;

  xdata_[EXCHANGE] = newInstance<XData_> ( mpx_ );

  if ( overlap_ )
  {
    xdata_[SCATTER] = newInstance<XData_> ( mpx_ );
  }
  else
  {
    xdata_[SCATTER] = xdata_[EXCHANGE];
  }

  connect ( dofs_->newSizeEvent,
            this, & Self::invalidate_ );

  connect ( dofs_->newOrderEvent,
            this, & Self::invalidate_ );

  connect ( items->newSizeEvent,
            this, & Self::invalidate_ );

  connect ( items->newOrderEvent,
            this, & Self::invalidate_ );

  connect ( recvBorders_.newSizeEvent(),
            this, & Self::invalidate_ );

  connect ( recvBorders_.newTopoEvent(),
            this, & Self::invalidate_ );

  connect ( recvBorders_.newOrderEvent(),
            this, & Self::invalidate_ );

  if ( overlap_ )
  {
    connect ( sendBorders_.newSizeEvent(),
              this, & Self::invalidate_ );

    connect ( sendBorders_.newTopoEvent(),
              this, & Self::invalidate_ );

    connect ( sendBorders_.newOrderEvent(),
              this, & Self::invalidate_ );
  }
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void VectorExchanger::update_ ()
{
  ItemSet*  items = dofs_->getItems ();


  print ( JIVE_LOG_DEBUG, toString (),
          " : updating exchange data structures ...\n" );

  if ( overlap_ )
  {
    xdata_[EXCHANGE]->update

      ( *dofs_, recvBorders_, sendBorders_ );

    xdata_[SCATTER] ->update

      ( *dofs_, mergeBorders( *mpx_, recvBorders_, sendBorders_ ) );
  }
  else
  {
    xdata_[EXCHANGE]->update ( *dofs_, recvBorders_ );

    xdata_[SCATTER] = xdata_[EXCHANGE];
  }

  items      ->resetEvents ();
  dofs_      ->resetEvents ();
  recvBorders_.resetEvents ();
  sendBorders_.resetEvents ();

  dofCount_ = dofs_->dofCount ();
  updated_  = true;
}


//-----------------------------------------------------------------------
//   invalidate_
//-----------------------------------------------------------------------


void VectorExchanger::invalidate_ ()
{
  updated_ = false;
}


//-----------------------------------------------------------------------
//   xmodeError_
//-----------------------------------------------------------------------


void VectorExchanger::xmodeError_ ( const String& where ) const
{
  String  what;

  if ( xmode_ == INACTIVE_ )
  {
    what = "no data exchange has been started";
  }
  else
  {
    what =

      "data exchange started while the previous "
      "exchange has not been ended";
  }

  throw jem::RuntimeException ( where, what );
}


//-----------------------------------------------------------------------
//   invalidError_
//-----------------------------------------------------------------------


void VectorExchanger::invalidError_ ( const String& where )
{
  throw jem::RuntimeException (
    where,
    "invalid exchange data "
    "(borders and/or dof space have been modified)"
  );
}


JIVE_END_PACKAGE( mp )
