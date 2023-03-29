
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
#include <jive/util/utilities.h>
#include <jive/util/Random.h>
#include <jive/graph/Names.h>
#include <jive/graph/PartitionerFactory.h>
#include <jive/graph/FMOptimizer.h>
#include <jive/graph/MLPartitioner.h>
#include <jive/graph/RBPartitioner.h>


JIVE_BEGIN_PACKAGE( graph )


//-----------------------------------------------------------------------
//   newPartitioner
//-----------------------------------------------------------------------


Ref<Partitioner>     newPartitioner

  ( const String&      name,
    const Properties&  conf,
    const Properties&  globdat,
    idx_t              partCount )

{
  using jive::util::joinNames;

  Properties  nilProps;

  if ( partCount < 1 || partCount > 8 )
  {
    conf.makeProps(name).set ( PropNames::TYPE,
                               MLPartitioner::TYPE_NAME );

    return MLPartitioner::makeNew ( name, conf, nilProps, globdat );
  }
  else
  {
    String  pname = joinNames ( name, PropNames::PARTITIONER );
    String  oname = joinNames ( name, PropNames::OPTIMIZER );


    conf.makeProps(name) .set ( PropNames::TYPE,
                                RBPartitioner::TYPE_NAME );
    conf.makeProps(pname).set ( PropNames::TYPE,
                                MLPartitioner::TYPE_NAME );
    conf.makeProps(oname).set ( PropNames::TYPE,
                                FMOptimizer::TYPE_NAME );

    return jem::newInstance<RBPartitioner> (
      name,
      Random::get ( globdat ),
      MLPartitioner::makeNew ( pname, conf, nilProps, globdat ),
      FMOptimizer  ::makeNew ( oname, conf, nilProps, globdat )
    );
  }
}


Ref<Partitioner>     newPartitioner

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat,
    idx_t              partCount )

{
  if ( props.contains( name ) )
  {
    return PartitionerFactory::newInstance ( name,  conf,
                                             props, globdat );
  }
  else
  {
    return newPartitioner ( name, conf, globdat, partCount );
  }
}


JIVE_END_PACKAGE( graph )
