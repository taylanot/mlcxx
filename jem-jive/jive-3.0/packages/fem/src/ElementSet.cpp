
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
#include <jem/base/Class.h>
#include <jem/base/array/utilities.h>
#include <jem/base/IllegalInputException.h>
#include <jive/util/error.h>
#include <jive/fem/error.h>
#include <jive/fem/Globdat.h>
#include <jive/fem/ElementSet.h>


JIVE_BEGIN_PACKAGE( fem )


//=======================================================================
//   class ElementSet
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  ElementSet::CLASS_NAME_ = "jive::fem::ElementSet";


//-----------------------------------------------------------------------
//   checkElements
//-----------------------------------------------------------------------


void ElementSet::checkElements

  ( const String&  context,
    idx_t          nodeCount ) const

{
  JEM_PRECHECK ( *this );

  IdxVector  ielems ( jem::iarray( size() ) );

  checkSomeElements ( context, ielems, nodeCount );
}


//-----------------------------------------------------------------------
//   checkSomeElements
//-----------------------------------------------------------------------


void ElementSet::checkSomeElements

  ( const String&     context,
    const IdxVector&  ielems,
    idx_t             nodeCount ) const

{
  JEM_PRECHECK ( *this );

  const idx_t  ielemCount = ielems.size ();


  for ( idx_t i = 0; i < ielemCount; i++ )
  {
    idx_t  ielem = ielems[i];
    idx_t  n     = getElemNodeCount ( ielem );

    if ( n != nodeCount )
    {
      elemNodeCountError (
        context,
        getElemID ( ielem ),
        n,
        nodeCount
      );
    }
  }
}


//-----------------------------------------------------------------------
//   getElemBoxes
//-----------------------------------------------------------------------


void ElementSet::getElemBoxes ( const Matrix& boxes ) const
{
  IdxVector  ielems ( jem::iarray( size() ) );

  getSomeElemBoxes ( boxes, ielems );
}


//-----------------------------------------------------------------------
//   getSomeElemBoxes
//-----------------------------------------------------------------------


void ElementSet::getSomeElemBoxes

  ( const Matrix&     boxes,
    const IdxVector&  ielems ) const

{
  JEM_PRECHECK ( *this );

  NodeSet      nodes      = getNodes         ();

  const idx_t  ielemCount = ielems.size      ();
  const idx_t  maxNodes   = maxElemNodeCount ();
  const idx_t  rank       = nodes .rank      ();

  IdxVector    inodes (       maxNodes );
  Matrix       coords ( rank, maxNodes );


  if ( boxes.size(0) != 2 * rank || boxes.size(1) < ielemCount )
  {
    util::shapeError ( JEM_FUNC, "bounding box matrix",
                       boxes.shape(),
                       jem::shape( 2 * rank, ielemCount ) );
  }

  for ( idx_t ie = 0; ie < ielemCount; ie++ )
  {
    idx_t  ielem = ielems[ie];
    idx_t  n     = getElemNodes ( inodes, ielem );

    if ( n == 0 )
    {
      for ( idx_t i = 0; i < rank; i += 2 )
      {
        boxes(i + 0,ie) = 0.0;
        boxes(i + 1,ie) = 0.0;
      }

      continue;
    }

    if ( n == maxNodes )
    {
      nodes.getSomeCoords ( coords, inodes );
    }
    else
    {
      nodes.getSomeCoords ( coords[slice(BEGIN,n)],
                            inodes[slice(BEGIN,n)] );
    }

    for ( idx_t i = 0; i < rank; i++ )
    {
      boxes(i +    0,ie) = coords(i,0);
      boxes(i + rank,ie) = coords(i,0);
    }

    for ( idx_t j = 1; j < n; j++ )
    {
      for ( idx_t i = 0; i < rank; i++ )
      {
        double  x = coords(i,j);

        if ( x < boxes(i,ie) )
        {
          boxes(i,ie) = x;
        }
        else if ( x > boxes(i + rank,ie) )
        {
          boxes(i + rank,ie) = x;
        }
      }
    }
  }
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void ElementSet::store

  ( const Properties&  globdat ) const

{
  JEM_PRECHECK ( *this );

  data_->store ( Globdat::ELEMENTS, globdat );
}


void ElementSet::store

  ( const String&      name,
    const Properties&  globdat ) const

{
  JEM_PRECHECK ( *this );

  data_->store ( name, globdat );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


ElementSet ElementSet::find

  ( const Properties&  globdat )

{
  return find ( Globdat::ELEMENTS, globdat );
}


ElementSet ElementSet::find

  ( const String&      name,
    const Properties&  globdat )

{
  using jem::dynamicCast;
  using jive::util::ItemSet;

  Ref<Data>  data =

    dynamicCast<Data> ( ItemSet::find( name, globdat ) );

  if ( data && isValidData( data ) )
  {
    return ElementSet ( data );
  }
  else
  {
    return ElementSet ();
  }
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


ElementSet ElementSet::get

  ( const Properties&  globdat,
    const String&      context )

{
  ElementSet  elems = find ( Globdat::ELEMENTS, globdat );

  if ( ! elems )
  {
    throw jem::IllegalInputException (
      context,
      "no elements have been defined"
    );
  }

  return elems;
}


ElementSet ElementSet::get

  ( const String&      name,
    const Properties&  globdat,
    const String&      context )

{
  ElementSet  elems = find ( name, globdat );

  if ( ! elems )
  {
    throw jem::IllegalInputException (
      context,
      "undefined element set: " + name
    );
  }

  return elems;
}


//-----------------------------------------------------------------------
//   isValidData
//-----------------------------------------------------------------------


bool ElementSet::isValidData ( const Ref<Data>& data )
{
  if ( data )
  {
    return NodeSet::Data::getType()->isBaseOf (
      data->getGroupedItems()->getClass ()
    );
  }
  else
  {
    return true;
  }
}


JIVE_END_PACKAGE( fem )
