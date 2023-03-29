
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
#include <jem/base/limits.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/utilities.h>
#include <jem/numeric/sparse/SparseILUn.h>
#include <jem/util/Properties.h>
#include <jive/util/Constraints.h>
#include <jive/solver/Names.h>
#include <jive/solver/SolverInfo.h>
#include <jive/solver/PreconFactory.h>
#include <jive/solver/SparseILUn.h>


JEM_DEFINE_CLASS( jive::solver::SparseILUn );


JIVE_BEGIN_PACKAGE( solver )


//=======================================================================
//   class SparseILUn::Solver_
//=======================================================================


class SparseILUn::Solver_ : public jem::numeric::SparseILUn {};


//=======================================================================
//   class SparseILUn
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*   SparseILUn::TYPE_NAME = "ILUn";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SparseILUn::SparseILUn

  ( const String&        name,
    Ref<AbstractMatrix>  matrix,
    Ref<Constraints>     cons ) :

    Super ( name, matrix, cons )

{
  maxFill_ = 0;
  solver_  = new Solver_ ();
}


SparseILUn::~SparseILUn ()
{
  delete solver_;
  solver_ = nullptr;
}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


void SparseILUn::getInfo ( const Properties& info ) const
{
  info.set ( SolverInfo::TYPE_NAME, TYPE_NAME );
  info.set ( SolverInfo::MEM_USAGE, solver_->getMemUsage() );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void SparseILUn::configure ( const Properties& props )
{
  using jem::maxOf;

  Super::configure ( props );

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    bool        newConf = false;
    bool        found;

    found   = myProps.find ( maxFill_,
                             PropNames::MAX_FILL,
                             0, maxOf ( maxFill_ ) );

    newConf = (newConf || found);

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


void SparseILUn::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  Super::getConfig ( props );

  myProps.set ( PropNames::MAX_FILL, maxFill_ );
}


//-----------------------------------------------------------------------
//   setMaxFill
//-----------------------------------------------------------------------


void SparseILUn::setMaxFill ( idx_t mfill )
{
  JEM_PRECHECK2 ( mfill >= 0, "invalid maximum fill-in level" );

  setParam_ ( maxFill_, mfill );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Preconditioner> SparseILUn::makeNew

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


void SparseILUn::declare ()
{
  PreconFactory::declare ( TYPE_NAME,  & makeNew );
  PreconFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   factor_
//-----------------------------------------------------------------------


void SparseILUn::factor_

  ( int                  events,
    const SparseMatrix&  matrix,
    const BoolVector&    mask,
    const IdxVector&     iperm )

{
  using jem::io::Writer;

  Writer&  debug = jem::System::debug ( myName_ );


  print ( debug, myName_, " : computing ILU factors ...\n" );

  solver_->setMaxFill       ( maxFill_ );
  solver_->setDropTol       ( droptol_ );
  solver_->setDiagShift     ( dshift_  );
  solver_->setZeroThreshold ( zeroThreshold_ );

  solver_->factor ( matrix, mask, iperm );

  // Store the current diagonal shift factor.

  dshift_ = solver_->getDiagShift ();

  print ( debug, myName_,
          " : non-zeroes in ILU factors: ",
          (double) solver_->nonZeroCount(), '\n' );
}


//-----------------------------------------------------------------------
//   solve_
//-----------------------------------------------------------------------


void SparseILUn::solve_

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
