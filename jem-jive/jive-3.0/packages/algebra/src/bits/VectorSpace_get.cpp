
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


#include <jem/mp/Context.h>
#include <jem/util/Properties.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DofSpace.h>
#include <jive/mp/Globdat.h>
#include <jive/mp/ItemMask.h>
#include <jive/algebra/typedefs.h>
#include <jive/algebra/Globdat.h>
#include <jive/algebra/MPVectorSpace.h>
#include <jive/algebra/StdVectorSpace.h>


JIVE_BEGIN_PACKAGE( algebra )


//=======================================================================
//   class VectorSpace
//=======================================================================

// This function is in a separate source file because it pulls in
// quite a lot of other stuff.

//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<VectorSpace> VectorSpace::get

  ( const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  using jem::newInstance;
  using jive::mp::ItemMask;

  Properties  dofsdat = Globdat::getDataFor ( dofs, globdat );
  Ref<Self>   vspace;


  if ( ! dofsdat.find( vspace, Globdat::VECTOR_SPACE ) )
  {
    vspace = newInstance<StdVectorSpace> ( dofs );

    if ( mp::Globdat::procCount( globdat ) > 1 )
    {
      Ref<MPContext>  mpx   =

        mp::Globdat::getMPContext ( globdat );

      Ref<ItemMask>   imask =

        ItemMask::get ( dofs->getItems(), globdat );

      vspace = newInstance<MPVectorSpace> ( vspace, mpx, imask );
    }

    dofsdat.set ( Globdat::VECTOR_SPACE, vspace );
  }

  return vspace;
}


JIVE_END_PACKAGE( algebra )
