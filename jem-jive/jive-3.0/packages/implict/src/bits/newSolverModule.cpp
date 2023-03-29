
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


#include <jem/util/Properties.h>
#include <jive/app/ModuleFactory.h>
#include <jive/implict/SolverModule.h>


JIVE_BEGIN_PACKAGE( implict )


//-----------------------------------------------------------------------
//   newSolverModule
//-----------------------------------------------------------------------

Ref<SolverModule>     newSolverModule

  ( const String&       name,
    const Properties&   conf,
    const Properties&   props,
    const Properties&   globdat )

{
  using jem::dynamicCast;
  using jive::app::ModuleFactory;

  Ref<Module>        module =

    ModuleFactory::newInstance ( name, conf, props, globdat );

  Ref<SolverModule>  solver = dynamicCast<SolverModule> ( module );

  if ( ! solver )
  {
    Properties  modProps = props.getProps ( name );
    String      modType;

    modProps.get ( modType, ModuleFactory::TYPE_PROP );

    modProps.propertyError (
      ModuleFactory::TYPE_PROP,
      String::format (
        "module of type `%s\' is not a valid solver module",
        modType
      )
    );
  }

  return solver;
}


JIVE_END_PACKAGE( implict )
