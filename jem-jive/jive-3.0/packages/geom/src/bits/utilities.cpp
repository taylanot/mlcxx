
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


#include <jive/geom/utilities.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   getGrad2Map
//-----------------------------------------------------------------------


void                getGrad2Map

  ( const IdxVector&  imap,
    const IdxVector&  jmap,
    idx_t             rank )

{
  idx_t  i, j, k;

  for ( i = 0; i < rank; i++ )
  {
    imap[i] = jmap[i] = i;
  }

  k = rank;

  for ( i = 0; i < rank; i++ )
  {
    for ( j = i + 1; j < rank; j++, k++ )
    {
      imap[k] = i;
      jmap[k] = j;
    }
  }
}


//-----------------------------------------------------------------------
//   getQujobi
//-----------------------------------------------------------------------


void                getQujobi

  ( const Matrix&     qu,
    const Matrix&     ja )

{
  const idx_t  m = ja.size(0);
  const idx_t  n = ja.size(1);

  JIVE_GET_QUJOBI( qu, ja, m, n, idx_t )
}


JIVE_END_PACKAGE( geom )
