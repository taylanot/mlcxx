
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


#include <jem/base/rtti.h>
#include <jem/base/Float.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/operators.h>
#include <jem/util/Properties.h>
#include <jive/util/ItemSet.h>
#include <jive/gl/utilities.h>
#include <jive/gl/Names.h>
#include <jive/gl/VectorField.h>


JIVE_BEGIN_PACKAGE( gl )


using jem::IllegalInputException;


//=======================================================================
//   class VectorField
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


VectorField::VectorField

  ( idx_t                rank,
    const Ref<ItemSet>&  items ) :

    rank_  (  rank ),
    items_ ( items )

{
  JEM_PRECHECK ( rank >= 0 && rank <= 3 && items );

  compNames_[0] = "x";
  compNames_[1] = "y";
  compNames_[2] = "z";

  initScaling_ ();
}


VectorField::VectorField

  ( idx_t                rank,
    const Ref<ItemSet>&  items,
    const String         compNames[3] ) :

    rank_  (  rank ),
    items_ ( items )

{
  JEM_PRECHECK ( rank >= 0 && rank <= 3 && items );

  compNames_[0] = compNames[0];
  compNames_[1] = compNames[1];
  compNames_[2] = compNames[2];

  initScaling_ ();
}


VectorField::~VectorField ()
{}


//-----------------------------------------------------------------------
//   configure (without a name)
//-----------------------------------------------------------------------


void VectorField::configure

  ( const Properties&  props,
    const Properties&  dataSets )

{
  props.find ( preScale_,  PropNames::SCALE, 1.0e-20, 1.0e20 );
  props.find ( autoScale,  PropNames::AUTO_SCALE );
  props.find ( scaleStep_, PropNames::SCALE_STEP, 0.0, 1000.0 );

  for ( int i = 0; i < 3; i++ )
  {
    data_[i].configure ( compNames_[i], props, dataSets );
  }
}


//-----------------------------------------------------------------------
//   configure (with a name)
//-----------------------------------------------------------------------


void VectorField::configure

  ( const String&      name,
    const Properties&  props,
    const Properties&  dataSets )

{
  using jem::toValue;

  Ref<Object>  obj;

  if ( ! props.find( obj, name ) )
  {
    return;
  }

  double  x;

  if ( toValue( x, obj ) )
  {
    for ( idx_t i = 0; i < 3; i++ )
    {
      data_[i].init ( x );
    }

    preScale_ = 1.0;

    return;
  }

  String  s;

  if ( toValue( s, obj ) )
  {
    try
    {
      configure_ ( s, dataSets );
    }
    catch ( IllegalInputException& ex )
    {
      ex.setContext ( props.getContext( name ) );
      throw;
    }

    return;
  }

  configure ( props.findProps( name ), dataSets );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void VectorField::getConfig ( const Properties& props ) const
{
  props.set ( PropNames::SCALE,      preScale_  );
  props.set ( PropNames::AUTO_SCALE, autoScale  );
  props.set ( PropNames::SCALE_STEP, scaleStep_ );

  for ( int i = 0; i < 3; i++ )
  {
    data_[i].getConfig ( compNames_[i], props );
  }
}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


void VectorField::getData

  ( double&            scale,
    const Matrix&      vfield,
    const Properties&  globdat,
    const String&      context,
    const AppContext&  apx ) const

{
  ItemSet*  myItems = items_[apx].get ();

  DataSet*  dataSet;


  for ( int i = 0; i < 3; i++ )
  {
    dataSet = data_[i].getDataSet ();

    if ( dataSet && dataSet->getItems( apx ) != myItems )
    {
      dataError_ ( context, *dataSet, apx );
    }

    data_[i].getData ( vfield(i,ALL), globdat, context, apx );
  }

  scale = preScale_;

  if ( autoScale )
  {
    try
    {
      scale *= getScale ( vfield, scaleStep_ );
    }
    catch ( IllegalInputException& ex )
    {
      ex.setContext ( context );
      throw;
    }

    scale *= postScale_;
  }

  vfield *= scale;
}


//-----------------------------------------------------------------------
//   setScaleStep
//-----------------------------------------------------------------------


void VectorField::setScaleStep ( double step )
{
  JEM_PRECHECK ( step >= 0.0 );

  scaleStep_ = step;
}


//-----------------------------------------------------------------------
//   setPostScale
//-----------------------------------------------------------------------


void VectorField::setPostScale ( double scale )
{
  JEM_PRECHECK ( scale >= 0.0 );

  postScale_ = scale;
}


//-----------------------------------------------------------------------
//   initScaling_
//-----------------------------------------------------------------------


inline void VectorField::initScaling_ ()
{
  preScale_  = 1.0;
  autoScale  = true;
  postScale_ = 1.0;
  scaleStep_ = 0.1;
}


//-----------------------------------------------------------------------
//   configure_
//-----------------------------------------------------------------------


void VectorField::configure_

  ( const String&      s,
    const Properties&  dataSets )

{
  using jem::Float;

  Ref<DataSet>  dataSet;
  String        setName;
  double        scale;
  idx_t         i;


  scale = 1.0;
  i     = s.find ( '*' );

  if ( i < 0 )
  {
    setName = s.stripWhite ();
  }
  else
  {
    scale   = Float::parse ( s[slice(BEGIN,i)] );
    setName = s[slice(i + 1,END)].stripWhite ();
  }

  if ( setName.size() == 0 )
  {
    throw IllegalInputException ( JEM_FUNC,
                                  "no data set specified" );
  }

  if ( ! dataSets.find( dataSet, setName ) )
  {
    throw IllegalInputException (
      JEM_FUNC,
      String::format ( "undefined data set: `%s\'", setName )
    );
  }

  for ( i = 0; i < rank_; i++ )
  {
    data_[i].init ( dataSet, compNames_[i] );
  }

  for ( i = rank_; i < 3; i++ )
  {
    data_[i].init ( 0.0 );
  }

  preScale_ = scale;
  autoScale = false;
}


//-----------------------------------------------------------------------
//   dataError_
//-----------------------------------------------------------------------


void VectorField::dataError_

  ( const String&      context,
    const DataSet&     dataSet,
    const AppContext&  apx ) const

{
  String  dsetName = dataSet     .getName     ();
  String  isetName = items_[apx]->getName     ();
  String  itemName = items_[apx]->getItemName ();

  throw IllegalInputException (
    context,
    String::format (
      "data set `%s\' is not associated with %s set `%s\'",
      dsetName,
      itemName,
      isetName
    )
  );
}


JIVE_END_PACKAGE( gl )
