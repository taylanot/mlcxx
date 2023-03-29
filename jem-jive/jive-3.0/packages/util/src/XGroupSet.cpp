
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
#include <jive/util/Reordering.h>
#include <jive/util/XGroupSet.h>


JEM_DEFINE_CLASS( jive::util::XGroupSet );


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class XGroupSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


XGroupSet::XGroupSet ( const String& name ) :

  Super ( name )

{}


XGroupSet::~XGroupSet ()
{}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void XGroupSet::reserve ( idx_t )
{}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void XGroupSet::trimToSize ()
{}


//-----------------------------------------------------------------------
//   addGroups
//-----------------------------------------------------------------------


idx_t XGroupSet::addGroups ( const IdxMatrix& groups )
{
  const idx_t  k = this ->size ();
  const idx_t  n = groups.size (1);

  reserve ( k + n );

  for ( idx_t i = 0; i < n; i++ )
  {
    addGroup ( groups[i] );
  }

  return k;
}


//-----------------------------------------------------------------------
//   eraseGroups
//-----------------------------------------------------------------------


void XGroupSet::eraseGroups ( const IdxVector& igroups )
{
  reorderGroups ( makeEraseReordering( igroups, size() ) );
}


JIVE_END_PACKAGE( util )
