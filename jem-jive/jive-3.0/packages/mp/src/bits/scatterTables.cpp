
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


#include <jem/io/DataInput.h>
#include <jem/io/DataOutput.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemMap.h>
#include <jive/util/SparseTable.h>
#include <jive/util/DenseTable.h>
#include <jive/mp/scatter.h>


JIVE_BEGIN_PACKAGE( mp )


using jem::dynamicCast;
using jem::newInstance;
using jem::io::DataInput;
using jem::io::DataOutput;
using jive::util::ItemMap;
using jive::util::Table;
using jive::util::SparseTable;
using jive::util::DenseTable;


//=======================================================================
//   class TableCodec
//=======================================================================


class TableCodec : public ScatterCodec
{
 public:

  enum                    TableTag
  {
                            NIL_TABLE_TAG,
                            DENSE_TABLE_TAG,
                            SPARSE_TABLE_TAG
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


void TableCodec::writeObject

  ( DataOutput&         out,
    const Ref<Object>&  obj )

{
  Ref<Table>  table = dynamicCast<Table> ( obj );


  if ( ! table )
  {
    encode ( out, (int) NIL_TABLE_TAG );
  }
  else
  {
    const ItemSet*  items = table->getRowItems ();
    const ItemMap*  imap  = items->getItemMap  ();

    SparseMatrix    tmat  = table->toMatrix    ();

    const idx_t     rowCount = tmat.size (0);
    const idx_t     colCount = tmat.size (1);

    IdxVector       offsets  = tmat.getRowOffsets    ();
    IdxVector       indices  = tmat.getColumnIndices ();
    Vector          values   = tmat.getValues        ();

    idx_t           n;


    if ( ! indices.isContiguous() )
    {
      indices.ref ( indices.clone() );
    }
    if ( ! values.isContiguous() )
    {
      values.ref ( values.clone() );
    }

    n = values.size ();

    if ( 2 * n > rowCount * colCount )
    {
      encode ( out, (int) DENSE_TABLE_TAG );
    }
    else
    {
      encode ( out, (int) SPARSE_TABLE_TAG );
    }

    encode ( out, table->getName(), table->getColumnNames(),
             rowCount );

    for ( idx_t irow = 0; irow < rowCount; irow++ )
    {
      idx_t  rowID = imap->getItemID ( irow );

      idx_t  i     = offsets[irow];
      idx_t  j     = offsets[irow + 1];
      idx_t  n     = j - i;

      encode     ( out, rowID, n );
      out.encode ( indices.addr(i), n );
      out.encode ( values .addr(i), n );
    }
  }
}


//-----------------------------------------------------------------------
//   readObject
//-----------------------------------------------------------------------


Ref<Object> TableCodec::readObject

  ( DataInput&           in,
    const Ref<ItemSet>&  items )

{
  Ref<XTable>  table;
  int          tag;


  decode ( in, tag );

  if ( tag != NIL_TABLE_TAG )
  {
    const ItemMap*  imap  = items->getItemMap ();

    StringVector    colNames;

    IdxVector       irows ( 1 );
    IdxVector       jcols;
    Matrix          block;

    String          name;

    idx_t           rowCount;
    idx_t           rowID;
    idx_t           irow;
    idx_t           n;


    decode ( in, name, colNames, rowCount );

    if ( tag == DENSE_TABLE_TAG )
    {
      table = newInstance<DenseTable>  ( name, items );
    }
    else
    {
      table = newInstance<SparseTable> ( name, items );
    }

    n = colNames.size ();

    jcols.resize (    n );
    block.resize ( 1, n );

    table->addColumns ( colNames );

    for ( idx_t i = 0; i < rowCount; i++ )
    {
      decode ( in, rowID, n );

      in.decode ( jcols.addr(), n );
      in.decode ( block.addr(), n );

      irow = imap->findItem ( rowID );

      if ( irow >= 0 )
      {
        irows[0] = irow;

        if ( n == jcols.size() )
        {
          table->setBlock ( irows, jcols, block );
        }
        else
        {
          table->setBlock ( irows,
                            jcols[slice(BEGIN,n)],
                            block(ALL,slice(BEGIN,n)) );
        }
      }
    }

    table->trimToSize ();
  }

  return table;
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   recvTables
//-----------------------------------------------------------------------


void                    recvTables

  ( Context&              mpx,
    Dictionary&           recvDict,
    const Ref<ItemSet>&   myItems,
    int                   root )

{
  TableCodec  codec;

  recvObjects ( mpx, codec, recvDict, myItems, root );
}


//-----------------------------------------------------------------------
//   scatterTables
//-----------------------------------------------------------------------


void                    scatterTables

  ( Context&              mpx,
    Dictionary&           recvDict,
    Dictionary&           sendDict,
    const Ref<ItemSet>&   myItems )

{
  TableCodec  codec;

  scatterObjects ( mpx, codec, recvDict, sendDict, myItems );
}


JIVE_END_PACKAGE( mp )
