
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


#include <jem/io/ObjectInputStream.h>
#include <jem/io/ObjectOutputStream.h>
#include <jem/mp/Context.h>
#include <jem/mp/InputChannel.h>
#include <jem/mp/OutputChannel.h>
#include <jive/fem/MeshBlock.h>


using jem::checkedCast;
using jem::newInstance;
using jem::io::ObjectInputStream;
using jem::io::ObjectOutputStream;
using jem::mp::Port;
using jem::mp::InputChannel;
using jem::mp::OutputChannel;


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   sendMeshBlock
//-----------------------------------------------------------------------


void                  sendMeshBlock

  ( MPContext&          mpx,
    int                 rank,
    const MeshBlock&    block )

{
  Ref<ObjectOutputStream>  output;
  Ref<OutputChannel>       channel;


  channel = newInstance<OutputChannel> ();

  channel->open ( Port( &mpx, rank ) );

  output  = newInstance<ObjectOutputStream> ( channel );

  output->encodeObject ( const_cast<MeshBlock*>( &block ) );
  output->close        ();
}


//-----------------------------------------------------------------------
//   recvMeshBlock
//-----------------------------------------------------------------------


Ref<MeshBlock>        recvMeshBlock

  ( MPContext&          mpx,
    int                 rank )

{
  Ref<ObjectInputStream>  input;
  Ref<InputChannel>       channel;
  Ref<MeshBlock>          block;


  channel = newInstance<InputChannel> ();

  channel->open ( Port( &mpx, 0 ) );

  input = newInstance<ObjectInputStream> ( channel );
  block = checkedCast<MeshBlock>         ( input->decodeObject() );

  return block;
}


JIVE_END_PACKAGE( fem )
