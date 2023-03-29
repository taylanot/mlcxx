
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
#include <jem/base/array/utilities.h>
#include <jive/util/ItemSet.h>
#include <jive/gl/DataSet.h>


JEM_DEFINE_CLASS( jive::gl::DataSet );


JIVE_BEGIN_PACKAGE( gl )


using jem::shape;
using jem::iarray;
using jem::reshape;


//=======================================================================
//   class DataSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


DataSet::DataSet ( const String& name ) :

  Super ( name )

{}


DataSet::~DataSet ()
{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void DataSet::configure

  ( const Properties&  props,
    const Properties&  globdat,
    const AppContext&  apx )

{}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void DataSet::getConfig ( const Properties& props ) const
{}


//-----------------------------------------------------------------------
//   rowCount
//-----------------------------------------------------------------------


idx_t DataSet::rowCount ( const AppContext& apx ) const
{
  ItemSet*  items = getItems ( apx );

  if ( items )
  {
    return items->size ();
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------
//   getColData
//-----------------------------------------------------------------------


void DataSet::getColData

  ( const Vector&      buf,
    idx_t              jcol,
    const AppContext&  apx ) const

{
  const idx_t  n     = rowCount ( apx );

  IdxVector    irows ( n );
  IdxVector    jcols ( 1 );


  irows    = iarray ( n );
  jcols[0] = jcol;

  getData ( reshape( buf, shape( n, 1 ) ), irows, jcols, apx );
}


void DataSet::getColData

  ( const Vector&      buf,
    const IdxVector&   irows,
    idx_t              jcol,
    const AppContext&  apx ) const

{
  JEM_PRECHECK ( buf.size() == irows.size() );

  const idx_t  n      = irows.size ();

  IdxVector    jcols ( 1 );


  jcols[0] = jcol;

  getData ( reshape( buf, shape( n, 1 ) ), irows, jcols, apx );
}


//-----------------------------------------------------------------------
//   getRowData
//-----------------------------------------------------------------------


void DataSet::getRowData

  ( const Vector&      buf,
    idx_t              irow,
    const AppContext&  apx ) const

{
  const idx_t  n     = columnCount ( apx );

  IdxVector    ibuf  ( n + 1 );

  IdxVector    irows ( ibuf[slice(BEGIN,1)] );
  IdxVector    jcols ( ibuf[slice(1,END)]   );


  irows[0] = irow;
  jcols    = iarray ( n );

  getData ( reshape( buf, shape( 1, n ) ), irows, jcols, apx );
}


void DataSet::getRowData

  ( const Vector&      buf,
    idx_t              irow,
    const IdxVector&   jcols,
    const AppContext&  apx ) const

{
  JEM_PRECHECK ( buf.size() == jcols.size() );

  const idx_t  n     = jcols.size ();

  IdxVector    irows ( 1 );


  irows[0] = irow;

  getData ( reshape( buf, shape( 1, n ) ), irows, jcols, apx );
}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String DataSet::getContext () const
{
  return NamedObject::makeContext ( "data set", myName_ );
}


JIVE_END_PACKAGE( gl )
