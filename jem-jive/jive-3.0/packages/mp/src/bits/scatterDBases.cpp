
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


#include <jem/base/array/utilities.h>
#include <jem/io/DataInput.h>
#include <jem/io/DataOutput.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemMap.h>
#include <jive/util/StdDatabase.h>
#include <jive/util/DenseTable.h>
#include <jive/mp/scatter.h>


JIVE_BEGIN_PACKAGE( mp )


using jem::dynamicCast;
using jem::newInstance;
using jem::Array;
using jive::util::ItemMap;
using jive::util::StdDatabase;
using jive::util::IntDBColumn;
using jive::util::FloatDBColumn;


//=======================================================================
//   helper functions
//=======================================================================

//-----------------------------------------------------------------------
//   writeDBColumn
//-----------------------------------------------------------------------


template <class DBCol>

  static void       writeDBColumn

  ( DataOutput&       out,
    const DBCol&      col )

{
  typedef typename DBCol::ValueType  T;

  const ItemMap*  imap      = col.getRowItems()->getItemMap ();

  IdxVector       irows     = col.getNonEmptyRows ();

  const idx_t     irowCount = irows.size ();

  Array<T>        rbuf      ( col.maxRowSize() );


  encode ( out, irowCount, rbuf.size() );

  for ( idx_t i = 0; i < irowCount; i++ )
  {
    idx_t  irow  = irows[i];
    idx_t  rowID = imap->getItemID ( irow );
    idx_t  n     = col.getData     ( rbuf.addr(),
                                     rbuf.size(), irow );

    encode     ( out, rowID,  n );
    out.encode ( rbuf.addr(), n );
  }
}


//-----------------------------------------------------------------------
//   readDBColumn
//-----------------------------------------------------------------------


template <class DBCol>

  static void       readDBColumn

  ( DataInput&        in,
    DBCol&            col )

{
  typedef typename DBCol::ValueType  T;

  const ItemMap*  imap = col.getRowItems()->getItemMap ();

  Array<T>        rbuf;

  idx_t           irowCount;
  idx_t           rowID;
  idx_t           irow;
  idx_t           n;


  decode ( in, irowCount, n );

  rbuf.resize ( n );

  for ( idx_t i = 0; i < irowCount; i++ )
  {
    decode    ( in, rowID,   n );
    in.decode ( rbuf.addr(), n );

    irow = imap->findItem ( rowID );

    if ( irow >= 0 )
    {
      col.setData ( irow, rbuf.addr(), n );
    }
  }

  col.trimToSize ();
}


//=======================================================================
//   class DBaseCodec
//=======================================================================


class DBaseCodec : public ScatterCodec
{
 public:

  enum                    TableTag
  {
                            NIL_DBASE_TAG,
                            STD_DBASE_TAG,
  };


  virtual void            writeObject

    ( DataOutput&           out,
      const Ref<Object>&    obj )   override;

  virtual Ref<Object>     readObject

    ( DataInput&            in,
      const Ref<ItemSet>&   items ) override;

};


//-----------------------------------------------------------------------
//   writeObject
//-----------------------------------------------------------------------


void DBaseCodec::writeObject

  ( DataOutput&         out,
    const Ref<Object>&  obj )

{
  Ref<Database>  dbase = dynamicCast<Database> ( obj );


  if ( ! dbase )
  {
    encode ( out, (int) NIL_DBASE_TAG );
  }
  else
  {
    Array<IntDBColumn*>    icols = dbase->getIntColumns   ();
    Array<FloatDBColumn*>  fcols = dbase->getFloatColumns ();

    idx_t  n;

    encode ( out, (int) STD_DBASE_TAG, dbase->getName() );

    n = icols.size ();

    encode ( out, n );

    for ( idx_t i = 0; i < n; i++ )
    {
      encode ( out, icols[i]->getName() );

      writeDBColumn ( out, *icols[i] );
    }

    n = fcols.size ();

    encode ( out, n );

    for ( idx_t i = 0; i < n; i++ )
    {
      encode ( out, fcols[i]->getName() );

      writeDBColumn ( out, *fcols[i] );
    }
  }
}


//-----------------------------------------------------------------------
//   readObject
//-----------------------------------------------------------------------


Ref<Object> DBaseCodec::readObject

  ( DataInput&           in,
    const Ref<ItemSet>&  items )

{
  Ref<Database>  dbase;
  int            tag;


  decode ( in, tag );

  if ( tag != NIL_DBASE_TAG )
  {
    IntDBColumn*    icol;
    FloatDBColumn*  fcol;
    String          name;
    idx_t           n;

    decode ( in, name );

    dbase = newInstance<StdDatabase> ( name, items );

    decode ( in, n );

    for ( idx_t i = 0; i < n; i++ )
    {
      decode ( in, name );

      icol = dbase->addIntColumn ( name );

      readDBColumn ( in, *icol );
    }

    decode ( in, n );

    for ( idx_t i = 0; i < n; i++ )
    {
      decode ( in, name );

      fcol = dbase->addFloatColumn ( name );

      readDBColumn ( in, *fcol );
    }
  }

  return dbase;
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   recvDBases
//-----------------------------------------------------------------------


void                    recvDBases

  ( Context&              mpx,
    Dictionary&           recvDict,
    const Ref<ItemSet>&   myItems,
    int                   root )

{
  DBaseCodec  codec;

  recvObjects ( mpx, codec, recvDict, myItems, root );
}


//-----------------------------------------------------------------------
//   scatterDBases
//-----------------------------------------------------------------------


void                    scatterDBases

  ( Context&              mpx,
    Dictionary&           recvDict,
    Dictionary&           sendDict,
    const Ref<ItemSet>&   myItems )

{
  DBaseCodec  codec;

  scatterObjects ( mpx, codec, recvDict, sendDict, myItems );
}


JIVE_END_PACKAGE( mp )
