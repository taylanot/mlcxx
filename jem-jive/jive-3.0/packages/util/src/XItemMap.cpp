
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


#include <jem/base/ClassTemplate.h>
#include <jive/util/XItemMap.h>


JEM_DEFINE_CLASS( jive::util::XItemMap );


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class ItemMap
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


XItemMap::~XItemMap ()
{}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void XItemMap::reserve ( idx_t )
{}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void XItemMap::trimToSize ()
{}


//-----------------------------------------------------------------------
//   addItems
//-----------------------------------------------------------------------


idx_t XItemMap::addItems ( idx_t count )
{
  JEM_PRECHECK2 ( count >= 0, "invalid number of items to be added" );

  const idx_t  iitem = size ();

  reserve ( iitem + count );

  for ( idx_t i = 0; i < count; i++ )
  {
    addItem ();
  }

  return iitem;
}


JIVE_END_PACKAGE ( util )
