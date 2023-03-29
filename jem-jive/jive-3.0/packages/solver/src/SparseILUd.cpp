
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
#include <jem/base/array/utilities.h>
#include <jem/numeric/sparse/SparseILUd.h>
#include <jem/util/Properties.h>
#include <jive/util/Constraints.h>
#include <jive/solver/Names.h>
#include <jive/solver/SolverInfo.h>
#include <jive/solver/PreconFactory.h>
#include <jive/solver/SparseILUd.h>


JEM_DEFINE_CLASS( jive::solver::SparseILUd );


JIVE_BEGIN_PACKAGE( solver )


//=======================================================================
//   class SparseILUd::Solver_
//=======================================================================


class SparseILUd::Solver_ : public jem::numeric::SparseILUd {};


//=======================================================================
//   class SparseILUd
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*   SparseILUd::TYPE_NAME = "ILUd";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SparseILUd::SparseILUd

  ( const String&        name,
    Ref<AbstractMatrix>  matrix,
    Ref<Constraints>     cons ) :

    Super ( name, matrix, cons )

{
  minSize_ = 0;
  maxFill_ = 1.0;
  quality_ = 1.0;
  solver_  = new Solver_ ();
}


SparseILUd::~SparseILUd ()
{
  delete solver_;
  solver_ = nullptr;
}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


void SparseILUd::getInfo ( const Properties& info ) const
{
  info.set ( SolverInfo::TYPE_NAME, TYPE_NAME );
  info.set ( SolverInfo::MEM_USAGE, solver_->getMemUsage() );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void SparseILUd::configure ( const Properties& props )
{
  using jem::maxOf;

  Super::configure ( props );

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    bool        newConf = false;
    bool        found;

    found   = myProps.find ( minSize_,
                             PropNames::MIN_SIZE,
                             0, maxOf ( minSize_ ) );

    newConf = (newConf || found);

    found   = myProps.find ( maxFill_,
                             PropNames::MAX_FILL,
                             0.0, 10.0 );

    newConf = (newConf || found);

    found   = myProps.find ( quality_,
                             PropNames::QUALITY,
                             0.0, 100.0 );

    if ( newConf )
    {
      setEvents_          ( NEW_CONFIG_ );
      newValuesEvent.emit ( *this );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void SparseILUd::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  Super::getConfig ( props );

  myProps.set ( PropNames::MIN_SIZE, minSize_ );
  myProps.set ( PropNames::MAX_FILL, maxFill_ );
  myProps.set ( PropNames::QUALITY,  quality_ );
}


//-----------------------------------------------------------------------
//   setMinSize
//-----------------------------------------------------------------------


void SparseILUd::setMinSize ( idx_t msize )
{
  JEM_PRECHECK2( msize >= 0, "invalid minimum storage size" );

  setParam_ ( minSize_, msize );
}


//-----------------------------------------------------------------------
//   setMaxFill
//-----------------------------------------------------------------------


void SparseILUd::setMaxFill ( double mfill )
{
  JEM_PRECHECK2 ( mfill >= 0.0, "invalid maximum fill-in factor" );

  setParam_ ( maxFill_, mfill );
}


//-----------------------------------------------------------------------
//   setQuality
//-----------------------------------------------------------------------


void SparseILUd::setQuality ( double qlty )
{
  JEM_PRECHECK2 ( qlty >= 0.0, "invalid quality factor" );

  setParam_ ( quality_, qlty );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Preconditioner> SparseILUd::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  Ref<AbstractMatrix>  matrix;
  Ref<Constraints>     cons;

  if ( decodeParams( matrix, cons, params ) )
  {
    return jem::newInstance<Self> ( name, matrix, cons );
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void SparseILUd::declare ()
{
  PreconFactory::declare ( TYPE_NAME,  & makeNew );
  PreconFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   factor_
//-----------------------------------------------------------------------


void SparseILUd::factor_

  ( int                  events,
    const SparseMatrix&  matrix,
    const BoolVector&    mask,
    const IdxVector&     iperm )

{
  using jem::io::Writer;

  Writer&  debug = jem::System::debug ( myName_ );


  print ( debug, myName_, " : computing ILU factors ...\n" );

  solver_->setMaxFill       ( maxFill_ );
  solver_->setMinSize       ( minSize_ );
  solver_->setDropTol       ( droptol_ );
  solver_->setDiagShift     ( dshift_  );
  solver_->setQuality       ( quality_ );
  solver_->setZeroThreshold ( zeroThreshold_ );

  solver_->factor ( matrix, mask, iperm );

  // Store the current drop tolerance and diagonal shift factor.

  droptol_ = solver_->getDropTol   ();
  dshift_  = solver_->getDiagShift ();

  print ( debug, myName_,
          " : non-zeroes in ILU factors: ",
          (double) solver_->nonZeroCount(), '\n' );
}


//-----------------------------------------------------------------------
//   solve_
//-----------------------------------------------------------------------


void SparseILUd::solve_

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  if ( symmetric_ )
  {
    solver_->symsolve ( lhs, rhs );
  }
  else
  {
    solver_->solve    ( lhs, rhs );
  }
}


JIVE_END_PACKAGE( solver )
