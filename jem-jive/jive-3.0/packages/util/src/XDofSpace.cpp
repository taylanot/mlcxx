
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


#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jive/util/error.h>
#include <jive/util/utilities.h>
#include <jive/util/Globdat.h>
#include <jive/util/ItemSet.h>
#include <jive/util/SparseDofSpace.h>
#include <jive/util/XDofSpace.h>


JEM_DEFINE_CLASS( jive::util::XDofSpace );


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class XDofSpace
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


XDofSpace::~XDofSpace ()
{}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void XDofSpace::reserve ( idx_t )
{}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void XDofSpace::trimToSize ()
{}


//-----------------------------------------------------------------------
//   addTypes
//-----------------------------------------------------------------------


IdxVector XDofSpace::addTypes

  ( const StringVector&  names )

{
  const idx_t  n = names.size ();

  IdxVector    itypes ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    itypes[i] = addType ( names[i] );
  }

  return itypes;
}


//-----------------------------------------------------------------------
//   addDofs
//-----------------------------------------------------------------------


void XDofSpace::addDofs

  ( const IdxVector&  iitems,
    const IdxVector&  itypes )

{
  const idx_t  m = iitems.size ();
  const idx_t  n = itypes.size ();

  for ( idx_t i = 0; i < m; i++ )
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      addDof ( iitems[i], itypes[j] );
    }
  }
}


void XDofSpace::addDofs

  ( idx_t             iitem,
    const IdxVector&  itypes )

{
  idx_t      itmp[1] = { iitem };
  IdxVector  iitems  ( itmp, 1 );

  addDofs ( iitems, itypes );
}


void XDofSpace::addDofs

  ( const IdxVector&  iitems,
    idx_t             itype )

{
  idx_t      itmp[1] = { itype };
  IdxVector  itypes  ( itmp, 1 );

  addDofs ( iitems, itypes );
}


//-----------------------------------------------------------------------
//   eraseDofs
//-----------------------------------------------------------------------


void XDofSpace::eraseDofs ( const IdxVector& idofs )
{
  reorderDofs ( makeEraseReordering( idofs, dofCount() ) );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


Ref<XDofSpace> XDofSpace::find ( const Properties& globdat )
{
  Ref<Self> dofs;

  globdat.find ( dofs, Globdat::DOF_SPACE );

  return dofs;
}


Ref<XDofSpace> XDofSpace::find

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  Ref<Self>  dofs;

  Globdat::findFor ( dofs, Globdat::DOF_SPACE, items, globdat );

  return dofs;
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<XDofSpace> XDofSpace::get

  ( const Properties&  globdat,
    const String&      context )

{
  Ref<Self> dofs;

  if ( ! globdat.find( dofs, Globdat::DOF_SPACE ) )
  {
    Super::get ( globdat, context );
  }

  return dofs;
}


Ref<XDofSpace> XDofSpace::get

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  Properties  itemdat ( Globdat::getDataFor( items, globdat ) );
  Ref<Self>   dofs;

  if ( ! itemdat.find( dofs, Globdat::DOF_SPACE ) )
  {
    dofs = newDofSpace_ ( items );

    itemdat.set ( Globdat::DOF_SPACE, dofs );

    // Make this the primary DOF space if no primary DOF space has
    // been set yet.

    if ( ! globdat.contains( Globdat::DOF_SPACE ) )
    {
      globdat.set ( Globdat::DOF_SPACE, dofs );
    }
  }

  return dofs;
}


//-----------------------------------------------------------------------
//   newDofSpace_
//-----------------------------------------------------------------------


// See FlexDofSpace.cpp


JIVE_END_PACKAGE( util )
