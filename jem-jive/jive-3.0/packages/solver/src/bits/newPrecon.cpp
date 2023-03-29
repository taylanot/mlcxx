
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


#include <jem/util/None.h>
#include <jem/util/Properties.h>
#include <jive/solver/Names.h>
#include <jive/solver/Solver.h>
#include <jive/solver/PreconFactory.h>
#include <jive/solver/SolverFactory.h>
#include <jive/solver/DummyPrecon.h>
#include <jive/solver/SolverPrecon.h>


JIVE_BEGIN_PACKAGE( solver )


//-----------------------------------------------------------------------
//   newPrecon
//-----------------------------------------------------------------------


Ref<Precon>          newPrecon

  ( const String&      name,
    const Properties&  conf,
    const Properties&  params,
    const Properties&  globdat )

{
  Properties  nilProps;

  conf.makeProps(name).set ( PropNames::TYPE,
                             DummyPrecon::TYPE_NAME );

  return DummyPrecon::makeNew ( name, conf, nilProps,
                                params, globdat );
}


Ref<Precon>          newPrecon

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  using jem::newInstance;
  using jem::util::isNone;

  Ref<Object>  obj;

  if ( ! props.find( obj, name ) )
  {
    return newPrecon ( name, conf, params, globdat );
  }

  if ( isNone( obj ) )
  {
    Properties  nilProps;

    conf.set ( name, obj );

    return DummyPrecon::makeNew ( name, nilProps, nilProps,
                                  params, globdat );
  }

  Properties  preconProps = props.findProps ( name );

  String      type;

  preconProps.get ( type, PreconFactory::TYPE_PROP );

  if ( ! PreconFactory::exists( type ) &&
         SolverFactory::exists( type ) )
  {
    Ref<Solver>  solver =

      newSolver ( name, conf, props, params, globdat );

    if ( solver )
    {
      return newInstance<SolverPrecon> ( name, solver );
    }
  }

  return PreconFactory::newInstance ( name, conf, props,
                                      params, globdat );
}


JIVE_END_PACKAGE( solver )
