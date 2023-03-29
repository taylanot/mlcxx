
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

#ifndef JIVE_MP_SCATTER_H
#define JIVE_MP_SCATTER_H

#include <jem/base/Interface.h>
#include <jem/base/Collectable.h>
#include <jive/mp/import.h>


JIVE_BEGIN_PACKAGE( mp )


//-----------------------------------------------------------------------
//   interface ScatterCodec
//-----------------------------------------------------------------------


class ScatterCodec : public jem::Interface
{
 public:

  virtual void              writeObject

    ( DataOutput&             out,
      const Ref<Object>&      obj )          = 0;

  virtual Ref<Object>       readObject

    ( DataInput&              in,
      const Ref<ItemSet>&     items )        = 0;


 protected:

  virtual                  ~ScatterCodec ();

};


//-----------------------------------------------------------------------
//   public functions
//-----------------------------------------------------------------------


void                        recvObjects

  ( Context&                  mpx,
    ScatterCodec&             codec,
    Dictionary&               recvDict,
    const Ref<ItemSet>&       myItems,
    int                       root = 0 );

void                        scatterObjects

  ( Context&                  mpx,
    ScatterCodec&             codec,
    Dictionary&               recvDict,
    Dictionary&               sendDict,
    const Ref<ItemSet>&       myItems );

void                        recvGroups

  ( Context&                  mpx,
    Dictionary&               recvDict,
    const Ref<ItemSet>&       myItems,
    int                       root = 0 );

void                        scatterGroups

  ( Context&                  mpx,
    Dictionary&               recvDict,
    Dictionary&               sendDict,
    const Ref<ItemSet>&       myItems );

void                        recvTables

  ( Context&                  mpx,
    Dictionary&               recvDict,
    const Ref<ItemSet>&       myItems,
    int                       root = 0 );

void                        scatterTables

  ( Context&                  mpx,
    Dictionary&               recvDict,
    Dictionary&               sendDict,
    const Ref<ItemSet>&       myItems );

void                        recvDBases

  ( Context&                  mpx,
    Dictionary&               recvDict,
    const Ref<ItemSet>&       myItems,
    int                       root = 0 );

void                        scatterDBases

  ( Context&                  mpx,
    Dictionary&               recvDict,
    Dictionary&               sendDict,
    const Ref<ItemSet>&       myItems );


void                        recvMembers

  ( Context&                  mpx,
    XMemberSet&               mbrs,
    int                       root = 0 );

void                        scatterMembers

  ( Context&                  mpx,
    XMemberSet&               recvMbrs,
    const MemberSet&          sendMbrs );


JIVE_END_PACKAGE( mp )

#endif
