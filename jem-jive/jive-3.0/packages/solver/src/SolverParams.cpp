
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


#include <jive/util/DofSpace.h>
#include <jive/util/Constraints.h>
#include <jive/algebra/VectorSpace.h>
#include <jive/algebra/ConstrainedMatrix.h>
#include <jive/solver/SolverParams.h>


JIVE_BEGIN_PACKAGE( solver )


//=======================================================================
//   class SolverParams
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  SolverParams::EXTRA              = "extra";
const char*  SolverParams::MATRIX             = "matrix";
const char*  SolverParams::PRECONDITIONER     = "precon";
const char*  SolverParams::RESTRICTOR         = "restrictor";
const char*  SolverParams::DOF_SPACE          = "dofSpace";
const char*  SolverParams::VECTOR_SPACE       = "vectorSpace";
const char*  SolverParams::CONSTRAINTS        = "constraints";
const char*  SolverParams::CONSTRAINED_MATRIX = "constrainedMatrix";


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Properties SolverParams::clone ( const Properties& params )
{
  Properties  cc;

  jem::util::mergeAndReplace ( cc, params );

  return cc;
}


//-----------------------------------------------------------------------
//   newInstance
//-----------------------------------------------------------------------


Properties SolverParams::newInstance

  ( Ref<AbstractMatrix>  matrix,
    Ref<VectorSpace>     vspace,
    Ref<Constraints>     cons,
    Ref<AbstractMatrix>  precon )

{
  Ref<ConstrainedMatrix>  conmat =

    jem::dynamicCast<ConstrainedMatrix> ( matrix );

  Ref<DofSpace>  dofs;

  Properties     params ( "solverParams" );


  if ( conmat && cons == conmat->getConstraints() )
  {
    params.set ( CONSTRAINED_MATRIX, conmat );

    matrix = conmat->getInner ();
  }

  if ( matrix )
  {
    params.set ( MATRIX, matrix );
  }

  if ( vspace )
  {
    params.set ( VECTOR_SPACE, vspace );

    dofs = vspace->getDofSpace ();
  }

  if ( cons )
  {
    params.set ( CONSTRAINTS, cons );

    dofs = cons->getDofSpace ();
  }

  if ( precon )
  {
    params.set ( PRECONDITIONER, precon );
  }

  if ( dofs )
  {
    params.set ( DOF_SPACE, dofs );
  }

  return params;
}


JIVE_END_PACKAGE( solver )
