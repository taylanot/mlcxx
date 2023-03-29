
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


#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/select.h>
#include <jem/base/array/operators.h>
#include <jem/mp/Context.h>
#include <jem/util/Properties.h>
#include <jive/util/error.h>
#include <jive/util/utilities.h>
#include <jive/util/Constraints.h>
#include <jive/mp/ItemMask.h>
#include <jive/mp/VectorExchanger.h>
#include <jive/algebra/MPVectorSpace.h>
#include <jive/algebra/StdVectorSpace.h>
#include <jive/algebra/MPMatrixObject.h>
#include <jive/solver/Names.h>
#include <jive/solver/Preconditioner.h>
#include <jive/solver/SolverParams.h>
#include <jive/solver/SolverFactory.h>
#include <jive/solver/SolverException.h>
#include <jive/solver/SchurMatrix.h>
#include <jive/solver/SchurDofSpace.h>
#include <jive/solver/SchurSolver.h>


JEM_DEFINE_CLASS( jive::solver::SchurSolver );


JIVE_BEGIN_PACKAGE( solver )


//=======================================================================
//   class SchurSolver
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  SchurSolver::TYPE_NAME = "Schur";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SchurSolver::SchurSolver

  ( const String&         name,
    Ref<Solver>           solver,
    Ref<SchurMatrix>      matrix,
    Ref<VectorSpace>      vspace,
    Ref<VectorExchanger>  vex ) :

    Super      ( name   ),
    solver_    ( solver ),
    matrix_    ( matrix ),
    vspace_    ( vspace ),
    exchanger_ ( vex    )

{
  JEM_PRECHECK ( solver &&
                 matrix &&
                 vspace &&
                 vex    &&
                 matrix->isDistributed() ==
                 vspace->isDistributed() );

  myMode_ = solver_->getMode ();

  solver_->setMode ( myMode_ | PRECON_MODE );
}


SchurSolver::~SchurSolver ()
{}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void SchurSolver::start ()
{
  solver_->start ();
  matrix_->start ();
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void SchurSolver::finish ()
{
  solver_->finish ();
  matrix_->finish ();
}


//-----------------------------------------------------------------------
//   improve
//-----------------------------------------------------------------------


void SchurSolver::improve

  ( const Vector&  lhs,
    const Vector&  rhs )

{
  JEM_PRECHECK ( lhs.size() == rhs.size() );

  using jive::util::sizeError;
  using jive::util::setSlaveDofs;
  using jive::util::evalSlaveDofs;
  using jive::util::evalMasterDofs;

  // This must come first to make sure that everything is up to date.

  SolverScope      scope      ( *this );

  SchurMatrix*     schurMat   = matrix_   .get              ();
  SchurDofSpace*   schurDofs  = schurMat ->getSchurDofSpace ();
  AbstractMatrix*  totalMat   = schurMat ->getTotalMatrix   ();
  Constraints*     totalCons  = schurMat ->getTotalCons     ();

  IdxVector        borderDofs = schurDofs->getBorderDofs    ();

  const idx_t      schurSize  = schurMat ->schurSize        ();
  const idx_t      totalSize  = schurMat ->totalSize        ();

  Vector           totalLhs   ( totalSize );
  Vector           totalRhs   ( totalSize );
  Vector           schurLhs   ( schurSize );
  Vector           schurRhs   ( schurSize );


  if ( lhs.size() != totalSize )
  {
    sizeError ( getContext(), "lhs vector",
                lhs.size(), totalSize );
  }

  if ( myMode_ & PRECON_MODE )
  {
    totalCons = 0;
  }

  if ( totalCons )
  {
    const int  xmode = schurMat ->getExchangeMode ();

    totalLhs = 0.0;

    exchanger_->startOne ( xmode );
    setSlaveDofs         ( totalLhs, *totalCons, -1.0 );
    totalMat  ->matmul   ( totalRhs, totalLhs );
    exchanger_->endOne   ( totalRhs );

    totalRhs += rhs;

    evalMasterDofs ( totalRhs, *totalCons );
  }
  else
  {
    totalRhs = rhs;
  }

  schurLhs = lhs[borderDofs];

  schurMat->getSchurRhs ( schurRhs, totalRhs );
  solver_ ->improve     ( schurLhs, schurRhs );
  schurMat->getTotalLhs ( totalLhs, schurLhs, totalRhs );

  // If not in preconditioning mode and not in lenient mode, then make
  // sure that the total residual norm is small enough.

  if ( ! (myMode_ & (PRECON_MODE | LENIENT_MODE)) )
  {
    const double  ERR_MARGIN = 2.0;
    const int     MAX_ITER   = 10;

    const int     xmode  = schurMat->getExchangeMode ();

    Vector        r      ( totalSize );
    Vector        v      ( totalSize );

    double        rscale = vspace_->norm2 ( totalRhs );
    double        prec0  = solver_->getPrecision ();

    double        prec;
    double        err;
    int           iiter;


    if ( rscale > 0.0 )
    {
      rscale = 1.0 / rscale;
    }

    err   = 0.0;
    prec  = prec0;
    iiter = 0;

    while ( true )
    {
      // Calculate the current residual.

      exchanger_->startOne ( xmode );

      v = totalLhs;

      if ( totalCons )
      {
        evalSlaveDofs ( v, *totalCons, 0.0 );
      }

      totalMat  ->matmul ( r, v );
      exchanger_->endOne ( r );

      if ( totalCons )
      {
        evalMasterDofs ( r, *totalCons );
      }

      r  -= totalRhs;
      err = rscale * vspace_->norm2 ( r );

      iiter++;

      if ( iiter > MAX_ITER  || err <= (ERR_MARGIN * prec0) )
      {
        break;
      }

      print ( jem::System::debug ( myName_ ), myName_,
              " : iter = ", iiter, ", residual = ", err, '\n' );

      // Try again with a greater precision.

      prec = 0.5 * prec * (prec / err);

      if ( prec < 1.0e-6 * prec0 )
      {
        break;
      }

      schurLhs = totalLhs[borderDofs];

      try
      {
        solver_->setPrecision ( prec );
        solver_->improve      ( schurLhs, schurRhs );
        solver_->setPrecision ( prec0 );
      }
      catch ( ... )
      {
        solver_->setPrecision ( prec0 );
        throw;
      }

      schurMat->getTotalLhs ( totalLhs, schurLhs, totalRhs );
    }

    if ( err > (ERR_MARGIN * prec0) )
    {
      throw SolverException (
        getContext (),
        String::format (
          "no convergence after %d trials; relative error = %e",
          iiter,
          err
        )
      );
    }
  }

  if ( totalCons )
  {
    evalSlaveDofs ( totalLhs, *totalCons );
  }

  lhs = totalLhs;
}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


void SchurSolver::getInfo ( const Properties& info ) const
{
  solver_->getInfo ( info );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void SchurSolver::configure ( const Properties& props )
{
  solver_->configure ( props );
  matrix_->configure ( props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void SchurSolver::getConfig ( const Properties& conf ) const
{
  solver_->getConfig ( conf );
  matrix_->getConfig ( conf );
}


//-----------------------------------------------------------------------
//   setMode
//-----------------------------------------------------------------------


void SchurSolver::setMode ( int mode )
{
  myMode_ = mode;

  solver_->setMode ( mode | PRECON_MODE );
}


//-----------------------------------------------------------------------
//   getMode
//-----------------------------------------------------------------------


int SchurSolver::getMode () const
{
  return myMode_;
}


//-----------------------------------------------------------------------
//   setPrecision
//-----------------------------------------------------------------------


void SchurSolver::setPrecision ( double eps )
{
  solver_->setPrecision ( eps );
}


//-----------------------------------------------------------------------
//   getPrecision
//-----------------------------------------------------------------------


double SchurSolver::getPrecision () const
{
  return solver_->getPrecision ();
}


//-----------------------------------------------------------------------
//   getMatrix
//-----------------------------------------------------------------------


AbstractMatrix* SchurSolver::getMatrix () const
{
  return matrix_->getTotalMatrix ();
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


Constraints* SchurSolver::getConstraints () const
{
  return matrix_->getTotalCons ();
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Solver> SchurSolver::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  using jem::newInstance;
  using jem::dynamicCast;
  using jive::util::joinNames;
  using jive::mp::ItemMask;
  using jive::algebra::MPMatrixObj;
  using jive::algebra::MPVectorSpace;
  using jive::algebra::StdVectorSpace;

  Ref<AbstractMatrix>  matrix;

  params.find ( matrix, SolverParams::MATRIX );

  if ( ! matrix )
  {
    return nullptr;
  }

  MPMatrixObj*  mpMat = dynamicCast<MPMatrixObj*> ( matrix );

  // Make sure that the matrix is usable.

  if ( ! mpMat && matrix->isDistributed() )
  {
    return nullptr;
  }

  const String  innerName  =

    joinNames ( name, PropNames::INNER_SOLVER );

  const String  outerName  =

    joinNames ( name, PropNames::OUTER_SOLVER );

  Properties    innerProps = props.makeProps ( innerName );

  Properties    innerParams;
  Properties    outerParams;

  Ref<Precon>   innerSolver;
  Ref<Solver>   outerSolver;


  // By default, use a direct inner solver.

  if ( ! innerProps.contains( "type" ) )
  {
    innerProps.set ( "type", "SparseLU" );
  }

  // Handle the trivial case.

  if ( ! matrix->isDistributed() )
  {
    outerParams = SolverParams::clone ( params );
    innerSolver =

      newPrecon ( innerName, conf, props, params, globdat );

    outerParams.set  ( SolverParams::PRECONDITIONER, innerSolver );

    return newSolver ( outerName, conf, props, outerParams, globdat );
  }

  Ref<VectorExchanger>  exchanger = mpMat->getExchanger ();
  Ref<SchurDofSpace>    schurDofs = newInstance<SchurDofSpace>

    ( exchanger->getDofSpace    (),
      exchanger->getRecvBorders (),
      exchanger->getSendBorders () );

  Ref<VectorSpace>      totalVSpace;
  Ref<VectorSpace>      schurVSpace;
  Ref<VectorSpace>      localVSpace;
  Ref<Constraints>      cons;
  Ref<ItemMask>         mask;


  innerParams.set ( SolverParams::MATRIX, mpMat->getInner() );
  params.find     ( totalVSpace, SolverParams::VECTOR_SPACE );

  localVSpace = totalVSpace;

  if ( ! totalVSpace )
  {
    return nullptr;
  }
  else
  {
    Ref<MPVectorSpace>  mpVSpace =

      dynamicCast<MPVectorSpace> ( totalVSpace );

    if ( mpVSpace )
    {
      localVSpace = mpVSpace->getInner    ();
      mask        = mpVSpace->getItemMask ();
    }

    // The local vector space may not be distributed.

    if ( localVSpace->isDistributed() )
    {
      return nullptr;
    }

    innerParams.set ( SolverParams::VECTOR_SPACE, localVSpace );
  }

  params.find ( cons, SolverParams::CONSTRAINTS );

  if ( cons )
  {
    Ref<Constraints>  innerCons =

      newInstance<Constraints> ( cons->getDofSpace() );

    innerParams.set ( SolverParams::CONSTRAINTS, innerCons );
  }

  Ref<Object>  extra;

  if ( params.find( extra, SolverParams::EXTRA ) )
  {
    innerParams.set ( SolverParams::EXTRA, extra );
    outerParams.set ( SolverParams::EXTRA, extra );
  }

  innerSolver = newPrecon ( innerName, conf, props,
                            innerParams, globdat );

  if ( ! innerSolver )
  {
    return nullptr;
  }

  Ref<SchurMatrix>  schurMatrix = newInstance<SchurMatrix>

    ( joinNames ( name, PropNames::MATRIX ),
      exchanger->getMPContext (),
      matrix,
      innerSolver,
      schurDofs, cons );

  outerParams.set ( SolverParams::MATRIX, schurMatrix );

  if ( ! mask )
  {
    mask = ItemMask::find ( schurDofs->getItems(), globdat );
  }

  if ( ! mask )
  {
    mask = newInstance<ItemMask> ( schurDofs->getItems() );
  }

  schurVSpace = newInstance<StdVectorSpace> ( schurDofs );
  schurVSpace = newInstance<MPVectorSpace>

    ( schurVSpace, exchanger->getMPContext(), mask );

  outerParams.set ( SolverParams::DOF_SPACE,    schurDofs   );
  outerParams.set ( SolverParams::VECTOR_SPACE, schurVSpace );

  outerSolver = newSolver ( outerName, conf, props,
                            outerParams, globdat );

  if ( ! outerSolver )
  {
    return nullptr;
  }
  else
  {
    return newInstance<Self> ( name,
                               outerSolver,
                               schurMatrix,
                               totalVSpace,
                               exchanger );
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void SchurSolver::declare ()
{
  SolverFactory::declare ( TYPE_NAME,  & makeNew );
  SolverFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( solver )
