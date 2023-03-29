
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
#include <jive/algebra/Names.h>
#include <jive/algebra/SparseMatrixObject.h>


JIVE_BEGIN_PACKAGE( algebra )


using jem::newInstance;


//-----------------------------------------------------------------------
//   newSparseMatrix
//-----------------------------------------------------------------------


Ref<SparseMatrixObj>  newSparseMatrix

  ( const String&       name,
    const Properties&   conf,
    const Properties&   props,
    const Properties&   params )

{
  Properties       matConf   = conf .makeProps ( name );
  Properties       matProps  = props.findProps ( name );

  String           optimize  = "Runtime";
  bool             symmetric = false;
  SparseMatrixObj
    ::Traits       traits    = 0;
  SparseMatrixObj
    ::Options      options   = 0;


  matProps.find ( symmetric, PropNames::SYMMETRIC );
  matProps.find ( optimize,  PropNames::OPTIMIZE  );

  if ( symmetric )
  {
    traits |= SparseMatrixObj::SYMMETRIC;
  }

  if ( optimize.size() )
  {
    if      ( optimize == "Runtime" )
    {
      options |=  SparseMatrixObj::ENABLE_BLOCKING;
    }
    else if ( optimize == "Memory" )
    {
      options &= ~SparseMatrixObj::ENABLE_BLOCKING;
    }
    else
    {
      matProps.propertyError (
        PropNames::OPTIMIZE,
        String::format (
          "invalid optimization option: %s; "
          "should be `Runtime\', `Memory\' or none",
          optimize
        )
      );
    }
  }

  matConf.set ( PropNames::SYMMETRIC, symmetric );
  matConf.set ( PropNames::OPTIMIZE,  optimize  );

  Ref<SparseMatrixObj>  matrix =

    newInstance<SparseMatrixObj> ( name, traits );

  matrix->setOptions ( options );

  return matrix;
}


JIVE_END_PACKAGE( algebra )
