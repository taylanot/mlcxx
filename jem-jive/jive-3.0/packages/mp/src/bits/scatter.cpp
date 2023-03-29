
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


#include <jem/io/DataInputStream.h>
#include <jem/io/DataOutputStream.h>
#include <jem/io/ArrayInputStream.h>
#include <jem/io/ArrayOutputStream.h>
#include <jem/io/SerializationException.h>
#include <jem/mp/Context.h>
#include <jem/mp/BcastStream.h>
#include <jem/util/Dictionary.h>
#include <jem/util/DictionaryEnumerator.h>
#include <jive/util/ItemSet.h>
#include <jive/mp/scatter.h>


JIVE_BEGIN_PACKAGE( mp )


using jem::newInstance;
using jem::io::DataInputStream;
using jem::io::DataOutputStream;
using jem::io::ArrayInputStream;
using jem::io::ArrayOutputStream;
using jem::io::SerializationException;
using jem::mp::BcastStream;
using jem::util::ArrayBuffer;
using jem::util::DictEnum;


//-----------------------------------------------------------------------
//   class ScatterCodec
//-----------------------------------------------------------------------


ScatterCodec::~ScatterCodec ()
{}


//-----------------------------------------------------------------------
//   constants
//-----------------------------------------------------------------------


static const int  SCATTER_MAGIC = 1234;


//-----------------------------------------------------------------------
//   readObjects
//-----------------------------------------------------------------------


static void             readObjects

  ( DataInput&            in,
    ScatterCodec&         codec,
    Dictionary&           dict,
    const Ref<ItemSet>&   items )

{
  Ref<Object>  obj;
  String       name;

  uidx_t       sz;
  int          magic;


  decode ( in, sz );

  idx_t  n = (idx_t) sz;

  for ( idx_t i = 0; i < n; i++ )
  {
    decode ( in, name, magic );

    if ( magic != SCATTER_MAGIC )
    {
      throw SerializationException (
        JEM_FUNC,
        "error reading data stream (wrong magic value)"
      );
    }

    obj = codec.readObject ( in, items );

    if ( obj )
    {
      dict.insert ( name, obj );
    }
  }
}


//-----------------------------------------------------------------------
//   recvObjects
//-----------------------------------------------------------------------


void                    recvObjects

  ( Context&              mpx,
    ScatterCodec&         codec,
    Dictionary&           recvDict,
    const Ref<ItemSet>&   myItems,
    int                   root )

{
  Ref<DataInputStream>  input  =

    newInstance<DataInputStream> (
      newInstance<BcastStream> ( root, & mpx, nullptr )
    );

  readObjects ( *input, codec, recvDict, myItems );
}


//-----------------------------------------------------------------------
//   scatterObjects
//-----------------------------------------------------------------------


void                    scatterObjects

  ( Context&              mpx,
    ScatterCodec&         codec,
    Dictionary&           recvDict,
    Dictionary&           sendDict,
    const Ref<ItemSet>&   myItems )

{
  Ref<ArrayInputStream>   inBuf;
  Ref<ArrayOutputStream>  outBuf;
  Ref<DataInputStream>    input;
  Ref<DataOutputStream>   output;

  Ref<Object>             obj;
  Ref<DictEnum>           e;

  StringVector            names ( sendDict.size() );


  idx_t  i = 0;

  for ( e = sendDict.enumerate(); ! e->atEnd(); e->toNext() )
  {
    names[i++] = e->getKey ();
  }

  e      = nullptr;
  outBuf = newInstance<ArrayOutputStream> ();
  output = newInstance<DataOutputStream>  ( outBuf );

  idx_t  n = names.size ();

  encode ( *output, n );

  for ( i = 0; i < n; i++ )
  {
    obj = sendDict.get ( names[i] );

    sendDict.erase ( names[i] );

    encode ( *output, names[i], SCATTER_MAGIC );

    codec.writeObject ( *output, obj );
  }

  inBuf = newInstance<ArrayInputStream> ( outBuf->toArray() );

  input = newInstance<DataInputStream> (
    newInstance<BcastStream> ( mpx.myRank(), & mpx, inBuf )
  );

  readObjects ( *input, codec, recvDict, myItems );
}


JIVE_END_PACKAGE( mp )
