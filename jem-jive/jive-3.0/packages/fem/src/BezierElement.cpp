
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


#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalArgumentException.h>
#include <jive/util/Database.h>
#include <jive/geom/BezierExtension.h>
#include <jive/fem/Names.h>
#include <jive/fem/BezierElement.h>


JEM_DEFINE_CLASS( jive::fem::BezierElement );


JIVE_BEGIN_PACKAGE( fem )


//=======================================================================
//   class BezierElement
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


BezierElement::BezierElement

  ( const ElementSet&     elems,
    const Ref<Database>&  dbase,
    const Ref<IShape>&    shape ) :

    Super ( elems, shape )

{
  JEM_PRECHECK ( dbase );


  if ( elems.getData() != dbase->getRowItems() )
  {
    throw jem::IllegalArgumentException (
      JEM_FUNC,
      String::format (
        "database `%s\' is not associated with element set `%s\'",
        dbase->getName (),
        elems .getName ()
      )
    );
  }

  dbase_          = dbase;
  dbCols_.irows   = dbase->getIntColumn    ( Names::IROWS );
  dbCols_.jcols   = dbase->getIntColumn    ( Names::JCOLS );
  dbCols_.values  = dbase->getFloatColumn  ( Names::VALUES );
  dbCols_.weights = dbase->findFloatColumn ( Names::WEIGHTS );

  bezier_ = nullptr;

  if ( shape )
  {
    shapeChanged_ ();
  }
}


BezierElement::~BezierElement ()
{}


//-----------------------------------------------------------------------
//   loadData
//-----------------------------------------------------------------------


void BezierElement::loadData ( idx_t ielem )
{
  Super::loadData ( ielem );

  idx_t  nnz = dbCols_.irows->rowSize ( ielem );

  if ( nnz != extractor_.irows.size() )
  {
    extractor_.irows .resize ( nnz );
    extractor_.jcols .resize ( nnz );
    extractor_.values.resize ( nnz );
  }

  dbCols_.irows ->getValues ( extractor_.irows,  ielem );
  dbCols_.jcols ->getValues ( extractor_.jcols,  ielem );
  dbCols_.values->getValues ( extractor_.values, ielem );

  if ( dbCols_.weights )
  {
    dbCols_.weights->getValues  ( weights_,   ielem );
    bezier_->setBezierExtractor ( extractor_, weights_ );
  }
  else
  {
    bezier_->setBezierExtractor ( extractor_ );
  }
}


//-----------------------------------------------------------------------
//   setupChanged_
//-----------------------------------------------------------------------


void BezierElement::setupChanged_ ( int flags )
{
  Super::setupChanged_ ( flags );

  if ( flags & NEW_SHAPE_ )
  {
    shapeChanged_ ();
  }
}


//-----------------------------------------------------------------------
//   shapeChanged_
//-----------------------------------------------------------------------


void BezierElement::shapeChanged_ ()
{
  bezier_ = shape->getExtension<BezierExtension> ();

  if ( ! bezier_ )
  {
    throw jem::IllegalArgumentException (
      JEM_FUNC,
      String::format (
        "shape `%s\' does not implement the Bezier extension",
        shape->getName ()
      )
    );
  }

  extractor_.shape = jem::shape ( shape  ->shapeFuncCount(),
                                  bezier_->basisFuncCount() );

  if ( dbCols_.weights )
  {
    weights_.resize ( extractor_.shape[0] );
  }
}


JIVE_END_PACKAGE( fem )
