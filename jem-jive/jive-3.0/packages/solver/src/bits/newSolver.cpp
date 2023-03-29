
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


#include <jive/algebra/VectorSpace.h>
#include <jive/algebra/AbstractMatrix.h>
#include <jive/algebra/SparseMatrixExtension.h>
#include <jive/solver/Names.h>
#include <jive/solver/SolverParams.h>
#include <jive/solver/SolverFactory.h>
#include <jive/solver/GMRES.h>
#include <jive/solver/SparseLU.h>
#include <jive/solver/VerboseSolver.h>


JIVE_BEGIN_PACKAGE( solver )


//-----------------------------------------------------------------------
//   newSolver
//-----------------------------------------------------------------------


Ref<Solver>          newSolver

  ( const String&      name,
    const Properties&  conf,
    const Properties&  params,
    const Properties&  globdat )

{
  using jem::newInstance;
  using jive::algebra::VectorSpace;
  using jive::algebra::SparseMatrixExt;

  Ref<AbstractMatrix>  matrix;
  Ref<VectorSpace>     vspace;
  Properties           props;


  if ( ! params.find( matrix, SolverParams::MATRIX ) )
  {
    return nullptr;
  }

  Properties  solverConf = conf.makeProps ( name );

  params.find ( vspace, SolverParams::VECTOR_SPACE );

  if ( vspace && vspace->isDistributed() )
  {
    solverConf.set ( PropNames::TYPE, GMRES::TYPE_NAME );

    return GMRES   ::makeNew ( name, conf, props, params, globdat );
  }

  if ( matrix->hasExtension<SparseMatrixExt>() )
  {
    solverConf.set ( PropNames::TYPE, SparseLU::TYPE_NAME );

    return SparseLU::makeNew ( name, conf, props, params, globdat );
  }

  if ( vspace )
  {
    solverConf.set ( PropNames::TYPE, GMRES::TYPE_NAME );

    return GMRES   ::makeNew ( name, conf, props, params, globdat );
  }

  return nullptr;
}


Ref<Solver>          newSolver

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  using jem::dynamicCast;

  Ref<Solver>  solver;

  if ( ! props.contains( name ) )
  {
    solver = newSolver ( name, conf, params, globdat );
  }
  else
  {
    solver = SolverFactory::newInstance ( name, conf, props,
                                          params, globdat );

    // Wrap the solver in a VerboseSolver object if the
    // "noiseLevel" property has been set.

    if ( ! dynamicCast<VerboseSolver*>( solver ) )
    {
      Properties  solverProps = props.findProps ( name );

      if ( solverProps.contains( PropNames::NOISE_LEVEL ) )
      {
        solver = newVerboseSolver ( solver );
      }
    }
  }

  return solver;
}


JIVE_END_PACKAGE( solver )
