
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

#ifndef JEM_MP_EXCHANGE_H
#define JEM_MP_EXCHANGE_H

#include <utility>
#include <jem/base/array/Array.h>


JEM_BEGIN_PACKAGE( mp )


class Context;


//-----------------------------------------------------------------------
//   class XPacket
//-----------------------------------------------------------------------


class XPacket
{
 public:

  inline                  XPacket ()       noexcept;

  inline                  XPacket

    ( const XPacket&        rhs )          noexcept;

  inline                  XPacket

    ( XPacket&&             rhs )          noexcept;

  inline XPacket&         operator =

    ( const XPacket&        rhs )          noexcept;

  inline XPacket&         operator =

    ( XPacket&&             rhs )          noexcept;

  inline void             swap

    ( XPacket&              rhs )          noexcept;

  inline bool             hasData () const noexcept;


 public:

  int                     rank;

  Array<byte>             binData;
  Array<lint>             intData;
  Array<double>           dblData;

};


//-----------------------------------------------------------------------
//   public functions
//-----------------------------------------------------------------------


inline void               swap

  ( XPacket&                lhs,
    XPacket&                rhs )          noexcept;

void                      exchange

  ( Context&                ctx,
    Array<XPacket>&         recvPackets,
    const Array<XPacket>&   sendPackets );




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class XPacket
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline XPacket::XPacket () noexcept : rank ( -1 )
{}


inline XPacket::XPacket ( const XPacket& rhs ) noexcept :

  rank    ( rhs.rank    ),
  binData ( rhs.binData ),
  intData ( rhs.intData ),
  dblData ( rhs.dblData )

{}

inline XPacket::XPacket ( XPacket&& rhs ) noexcept :

  rank    ( rhs.rank ),
  binData ( std::move( rhs.binData ) ),
  intData ( std::move( rhs.intData ) ),
  dblData ( std::move( rhs.dblData ) )

{}


//-----------------------------------------------------------------------
//   assignment operator
//-----------------------------------------------------------------------


inline XPacket& XPacket::operator = ( const XPacket& rhs ) noexcept
{
  rank = rhs.rank;

  binData.ref ( rhs.binData );
  intData.ref ( rhs.intData );
  dblData.ref ( rhs.dblData );

  return *this;
}


inline XPacket& XPacket::operator = ( XPacket&& rhs ) noexcept
{
  rank = rhs.rank;

  binData.ref ( std::move( rhs.binData ) );
  intData.ref ( std::move( rhs.intData ) );
  dblData.ref ( std::move( rhs.dblData ) );

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void XPacket::swap ( XPacket& rhs ) noexcept
{
  jem   ::swap ( rank, rhs.rank );

  binData.swap ( rhs.binData );
  intData.swap ( rhs.intData );
  dblData.swap ( rhs.dblData );
}


//-----------------------------------------------------------------------
//   hasData
//-----------------------------------------------------------------------


inline bool XPacket::hasData () const noexcept
{
  return ((binData.size() + intData.size() + dblData.size()) > 0);
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void swap ( XPacket& lhs, XPacket& rhs ) noexcept
{
  lhs.swap ( rhs );
}


JEM_END_PACKAGE( mp )

#endif
