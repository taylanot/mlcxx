
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


#include <jem/base/IllegalArgumentException.h>
#include <jive/util/ItemGroupSkin.h>


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class ItemGroupSkin
//=======================================================================

//-----------------------------------------------------------------------
//   dataError
//-----------------------------------------------------------------------


void ItemGroupSkin::dataError

  ( const String&     context,
    const String&     itemKind,
    const Ref<Data>&  data )

{
  using jem::IllegalArgumentException;

  if ( data )
  {
    String  s = data->toString ();

    throw IllegalArgumentException (
      context,
      String::format (
        "object `%s\' is not a valid %s group", s, itemKind
      )
    );
  }
  else
  {
    throw IllegalArgumentException (
      context,
      String::format (
        "null object is not a valid %s group", itemKind
      )
    );
  }
}


JIVE_END_PACKAGE( util )
