
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


#include <jive/util/ItemSet.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Constraints.h>
#include <jive/mp/Globdat.h>
#include <jive/mp/ItemMask.h>
#include <jive/mp/BorderSet.h>
#include <jive/mp/VectorExchanger.h>
#include <jive/algebra/MPVectorSpace.h>
#include <jive/algebra/MPMatrixObject.h>
#include <jive/solver/SolverParams.h>
#include <jive/implict/utilities.h>


JIVE_BEGIN_PACKAGE( implict )


//-----------------------------------------------------------------------
//   newSolverParams
//-----------------------------------------------------------------------


Properties                  newSolverParams

  ( const Properties&         globdat,
    Ref<AbstractMatrix>       globmat,
    Ref<AbstractMatrix>       precon,
    const Ref<DofSpace>&      dofs )

{
  using jem::newInstance;
  using jive::mp::ItemMask;
  using jive::mp::BorderSet;
  using jive::mp::VectorExchanger;
  using jive::algebra::MPMatrixObj;
  using jive::algebra::MPVectorSpace;
  using jive::solver::SolverParams;

  Ref<VectorSpace>  vspace;
  Ref<Constraints>  cons;


  if ( dofs )
  {
    vspace = VectorSpace::get  ( dofs, globdat );
    cons   = Constraints::find ( dofs, globdat );

    if ( ! globmat->isDistributed() &&
         BorderSet::hasBorders ( dofs->getItems(), globdat ) )
    {
      globmat = newInstance<MPMatrixObj> (
        globmat,
        VectorExchanger::get ( dofs, globdat )
      );
    }

    // Make sure that the matrix and vector space are compatible.

    if ( globmat->isDistributed() && !
         vspace ->isDistributed() )
    {
      vspace = newInstance<MPVectorSpace> (
        vspace,
        mp::Globdat::getMPContext ( globdat ),
        ItemMask::get ( dofs->getItems(), globdat )
      );
    }
  }

  return SolverParams::newInstance ( globmat, vspace, cons, precon );
}


JIVE_END_PACKAGE( implict )
