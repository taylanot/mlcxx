
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


#include <jive/util/utilities.h>
#include <jive/util/Constraints.h>
#include <jive/mp/VectorExchanger.h>
#include <jive/algebra/MPMatrixObject.h>
#include <jive/algebra/SparseMatrixExtension.h>
#include <jive/solver/Names.h>
#include <jive/solver/SolverParams.h>
#include <jive/solver/SchurMatrix.h>
#include <jive/solver/MPRestrictor.h>
#include <jive/solver/SchurRestrictor.h>
#include <jive/solver/SmoothRestrictor.h>
#include <jive/solver/SimpleRestrictor.h>
#include <jive/solver/RestrictorFactory.h>


JIVE_BEGIN_PACKAGE( solver )


using jem::newInstance;


//-----------------------------------------------------------------------
//   newRestrictor
//-----------------------------------------------------------------------


Ref<Restrictor>      newRestrictor

  ( const String&      name,
    const Properties&  conf,
    const Properties&  params,
    const Properties&  globdat )

{
  Ref<Restrictor>  rt = SimpleRestrictor::makeNew

    ( name, conf, Properties(), params, globdat );

  if ( rt )
  {
    Properties  rtConf = conf.makeProps ( name );

    rtConf.set ( PropNames::TYPE,
                 SimpleRestrictor::TYPE_NAME );
  }

  return rt;
}


Ref<Restrictor>      newRestrictor

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  using jem::dynamicCast;
  using jive::util::joinNames;
  using jive::algebra::MPMatrixObj;
  using jive::algebra::MPMatrixExt;
  using jive::algebra::SparseMatrixExt;

  Ref<AbstractMatrix>   matrix;
  Ref<SchurMatrix>      schur;
  Ref<VectorExchanger>  vex;
  Ref<Constraints>      cons;
  Ref<DofSpace>         dofs;
  Ref<Restrictor>       rt;


  params.find ( matrix, SolverParams::MATRIX );
  params.find ( cons,   SolverParams::CONSTRAINTS );

  // Undo the matrix from all "enhancements" and get the original
  // matrix.

  schur = dynamicCast<SchurMatrix> ( matrix );

  if ( schur )
  {
    matrix = schur->getTotalMatrix   ();
    cons   = schur->getTotalCons     ();
    dofs   = schur->getTotalDofSpace ();

    // For smoothing, the vector exchanger for the total DofSpace is
    // needed (see below).

    vex = VectorExchanger::get ( dofs, globdat );
  }
  else if ( matrix->isDistributed() )
  {
    Ref<MPMatrixObj>  mpMat = dynamicCast<MPMatrixObj> ( matrix );

    if ( mpMat )
    {
      matrix = mpMat->getInner     ();
      vex    = mpMat->getExchanger ();
    }
    else
    {
      MPMatrixExt*  mpExt = matrix->getExtension<MPMatrixExt> ();

      if ( mpExt )
      {
        vex = mpExt->getExchanger ();
      }
    }
  }

  if ( ! dofs )
  {
    if ( vex )
    {
      dofs = vex->getDofSpace ();
    }
    else
    {
      params.find ( dofs, SolverParams::DOF_SPACE );
    }
  }

  // Create the parameters for the restriction operator.

  Properties   rtParams ( joinNames( name, "params" ) );
  Ref<Object>  extra;

  if ( matrix )
  {
    rtParams.set ( SolverParams::MATRIX, matrix );
  }

  if ( dofs )
  {
    rtParams.set ( SolverParams::DOF_SPACE, dofs );
  }

  if ( cons )
  {
    rtParams.set ( SolverParams::CONSTRAINTS, cons );
  }

  if ( params.find( extra, SolverParams::EXTRA ) )
  {
    rtParams.set ( SolverParams::EXTRA, extra );
  }

  // Create the (local) restriction operator.

  if ( ! props.contains( name ) )
  {
    rt = newRestrictor ( name, conf, rtParams, globdat );
  }
  else
  {
    Properties  rtProps = props.getProps  ( name );
    Properties  rtConf  = conf .makeProps ( name );

    String      type;

    rtProps.get ( type, PropNames::TYPE );
    rtConf .set ( PropNames::TYPE, type );

    rt = RestrictorFactory

      ::newInstance ( type, name, conf, props, rtParams, globdat );
  }

  if ( rt == nullptr || ! rt->getExtension<LocalRestrictor>() )
  {
    // Try again with the original solver parameters.

    return RestrictorFactory

      ::newInstance ( name, conf, props, params, globdat );
  }

  // Apply smoothing if possible, unless smoothing has been
  // explicitely disabled.

  if ( matrix->getExtension<SparseMatrixExt>() &&
       matrix->isDistributed() == false )
  {
    Properties  rtProps = props.findProps ( name );
    Properties  rtConf  = conf .makeProps ( name );

    bool        smooth  = false;

    rtProps.find ( smooth, PropNames::SMOOTH );
    rtConf .set  ( PropNames::SMOOTH, smooth );

    if ( smooth )
    {
      rt = newInstance<SmoothRestrictor> ( rt, matrix, cons, vex );
    }
  }

  if ( schur )
  {
    rt = newInstance<SchurRestrictor> ( rt, schur );

    // Use the vector exchanger for the interface DofSpace when
    // creating the MPRestrictor below.

    vex = schur->getExchanger ();
  }

  if ( vex )
  {
    rt = newInstance<MPRestrictor> ( rt, vex );
  }

  return rt;
}


JIVE_END_PACKAGE( solver )
