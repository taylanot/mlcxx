
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

#ifndef JIVE_IMPLICT_SOLVERRUNDATA_H
#define JIVE_IMPLICT_SOLVERRUNDATA_H

#include <jem/util/Properties.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Constraints.h>
#include <jive/model/Model.h>
#include <jive/model/Actions.h>
#include <jive/solver/Solver.h>
#include <jive/implict/import.h>


JIVE_BEGIN_PACKAGE( implict )


using jem::Collectable;


//-----------------------------------------------------------------------
//   class SolverRunData
//-----------------------------------------------------------------------


class SolverRunData : public Collectable
{
 public:

  typedef Collectable     Super;
  typedef SolverRunData   Self;


  explicit                SolverRunData

    ( const String&         ctx );

  void                    init

    ( const Properties&     globdat );

  void                    initSolver

    ( const String&         name,
      const Properties&     conf,
      const Properties&     props,
      const Properties&     globdat );

  void                    getExtVector

    ( const Vector&         fext,
      const Properties&     globdat );

  void                    getIntVector

    ( const Vector&         fint,
      const Properties&     globdat );

  inline void             updateMatrix

    ( const Properties&     globdat );

  void                    updateMatrix

    ( const Vector&         fint,
      const Properties&     globdat );

  inline void             updateModel

    ( const Properties&     globdat );

  void                    updateConstraints

    ( const Properties&     globdat );

  void                    advance

    ( const Properties&     globdat );

  bool                    commit

    ( const Properties&     globdat );

  void                    cancel

    ( const Properties&     globdat );


 public:

  const String            context;

  Ref<Model>              model;
  Ref<DofSpace>           dofs;
  Ref<Constraints>        cons;
  Ref<VectorSpace>        vspace;
  Ref<Solver>             solver;
  Ref<AbstractMatrix>     sysmat;
  Ref<AbstractMatrix>     precon;


 protected:

  virtual                ~SolverRunData  ();

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   updateMatrix
//-----------------------------------------------------------------------


inline void SolverRunData::updateMatrix

  ( const Properties&  globdat )

{
  Vector  fint ( dofs->dofCount() );

  updateMatrix ( fint, globdat );
}


//-----------------------------------------------------------------------
//   updateModel
//-----------------------------------------------------------------------


inline void SolverRunData::updateModel ( const Properties& globdat )
{
  using jive::model::Actions;

  model->takeAction ( Actions::UPDATE, Properties(), globdat );
}


JIVE_END_PACKAGE( implict )

#endif
