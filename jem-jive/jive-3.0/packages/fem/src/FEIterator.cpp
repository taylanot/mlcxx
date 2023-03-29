
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
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Properties.h>
#include <jive/geom/ShapeTable.h>
#include <jive/fem/Names.h>
#include <jive/fem/FEIterator.h>


JEM_DEFINE_CLASS( jive::fem::FEIterator );


JIVE_BEGIN_PACKAGE( fem )


//=======================================================================
//   class FEIterator
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


FEIterator::FEIterator ()
{
  shapeFlags_ = 0;
  precision_  = 0;
  updated_    = false;
}


FEIterator::FEIterator

  ( const Ref<ShapeTable>&  shapes,
    const String&           context ) :

    context_    ( context ),
    shapeTable_ ( shapes  )

{
  JEM_PRECHECK ( shapes );

  shapeFlags_ = 0;
  precision_  = 0;
  updated_    = false;
}


FEIterator::~FEIterator ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void FEIterator::resetEvents ()
{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void FEIterator::configure ( const Properties&  props )
{
  const idx_t  maxPrec = shapeTable_->maxPrecision ();
  idx_t        prec    = precision_;

  props.find ( prec, PropNames::PRECISION, 0, maxPrec );

  if ( prec != precision_ )
  {
    setPrecision ( prec );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void FEIterator::getConfig

  ( const Properties&  conf ) const

{
  conf.set ( PropNames::PRECISION, precision_ );
}


//-----------------------------------------------------------------------
//   setShapeFlags
//-----------------------------------------------------------------------


void FEIterator::setShapeFlags ( int flags )
{
  shapeFlags_ = flags;

  invalidate_ ();
}


//-----------------------------------------------------------------------
//   setPrecision
//-----------------------------------------------------------------------


idx_t FEIterator::setPrecision ( idx_t prec )
{
  const idx_t  ubound = shapeTable_->maxPrecision ();

  if      ( prec < 0 )
  {
    prec = 0;
  }
  else if ( prec > ubound )
  {
    prec = ubound;
  }

  if ( precision_ != prec )
  {
    precision_ = prec;

    invalidate_ ();
  }

  return precision_;
}


//-----------------------------------------------------------------------
//   invalidate_
//-----------------------------------------------------------------------


void FEIterator::invalidate_ ()
{
  updated_ = false;

  changeEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   checkPrecision_
//-----------------------------------------------------------------------


void FEIterator::checkPrecision_ ()
{
  using jem::System;

  const idx_t  maxPrec = shapeTable_->maxPrecision ();

  if ( precision_ > maxPrec )
  {
    print ( System::warn(), context_,
            " : precision level (", precision_,
            ") out of range; adjusted to ", maxPrec, "\n" );

    precision_ = maxPrec;
  }
}


//-----------------------------------------------------------------------
//   readFrom_
//-----------------------------------------------------------------------


void FEIterator::readFrom_ ( ObjectInput& in )
{
  decode ( in, context_, shapeFlags_, precision_, shapeTable_ );

  updated_ = false;
}


//-----------------------------------------------------------------------
//   writeTo_
//-----------------------------------------------------------------------


void FEIterator::writeTo_ ( ObjectOutput& out ) const
{
  encode ( out, context_, shapeFlags_, precision_, shapeTable_ );
}


//-----------------------------------------------------------------------
//   emitEvents_
//-----------------------------------------------------------------------


void FEIterator::emitEvents_ ( int events )
{}


JIVE_END_PACKAGE( fem )
