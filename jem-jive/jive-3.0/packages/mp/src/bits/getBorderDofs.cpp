
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


#include <jem/base/Array.h>
#include <jem/util/Flex.h>
#include <jive/util/DofSpace.h>
#include <jive/mp/BorderSet.h>
#include <jive/mp/utilities.h>


JIVE_BEGIN_PACKAGE( mp )


using jem::iarray;
using jem::util::Flex;


//-----------------------------------------------------------------------
//   getBorderDofs
//-----------------------------------------------------------------------


void                getBorderDofs

  ( IdxVector&        idofs,
    IdxVector&        offsets,
    const BorderSet&  borders,
    const DofSpace&   dofs )

{
  const idx_t  typeCount   = dofs.typeCount ();
  const idx_t  borderCount = borders.size   ();

  IdxVector    itypes ( typeCount );
  IdxVector    jdofs  ( typeCount );

  IdxVector    iitems;
  IdxVector    jperm;
  Flex<idx_t>  ibuf;

  idx_t        iitemCount;
  idx_t        jdofCount;
  idx_t        iitem;


  offsets.resize  ( borderCount + 1         );
  iitems .resize  ( borders.maxBorderSize() );
  ibuf   .reserve ( dofs.dofCount()         );

  for ( idx_t ib = 0; ib < borderCount; ib++ )
  {
    offsets[ib] = ibuf.size ();

    iitemCount  = borders.getBorderItems ( iitems, ib );

    for ( idx_t i = 0; i < iitemCount; i++ )
    {
      iitem = iitems[i];

      jdofCount = dofs.getDofsForItem ( jdofs, itypes, iitem );

      if ( jdofCount == 1 )
      {
        ibuf.pushBack ( jdofs[0] );
      }
      else
      {
        jperm.resize   ( jdofCount );
        jperm = iarray ( jdofCount );

        sort ( jperm, itypes );

        for ( idx_t j = 0; j < jdofCount; j++ )
        {
          ibuf.pushBack ( jdofs[jperm[j]] );
        }
      }
    }
  }

  offsets[borderCount] = ibuf.size ();

  idofs.ref ( IdxVector( ibuf.begin(), ibuf.end() ) );
}


JIVE_END_PACKAGE( mp )
