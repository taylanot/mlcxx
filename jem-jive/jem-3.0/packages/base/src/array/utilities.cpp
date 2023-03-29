
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
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
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */


#include <jem/base/array/utilities.h>


JEM_BEGIN_PACKAGE_BASE
JEM_BEGIN_NAMESPACE( array )


//-----------------------------------------------------------------------
//   dot1Fast
//-----------------------------------------------------------------------


double            dot1Fast

  ( const double*   e,
    idx_t           n )

{
  double  t;

  switch ( n )
  {
  case 0:

    t = 0.0;
    break;

  case 1:

    t = e[0] * e[0];
    break;

  case 2:

    t = e[0] * e[0] + e[1] * e[1];
    break;

  case 3:

    t = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    break;

  default:

    t = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];

    for ( idx_t i = 3; i < n; i++ )
    {
      t += e[i] * e[i];
    }
  }

  return t;
}


//-----------------------------------------------------------------------
//   dot2Fast
//-----------------------------------------------------------------------


double            dot2Fast

  ( const double*   e,
    const double*   f,
    idx_t           n )

{
  double  t;

  switch ( n )
  {
  case 0:

    t = 0.0;
    break;

  case 1:

    t = e[0] * f[0];
    break;

  case 2:

    t = e[0] * f[0] + e[1] * f[1];
    break;

  case 3:

    t = e[0] * f[0] + e[1] * f[1] + e[2] * f[2];
    break;

  default:

    t = e[0] * f[0] + e[1] * f[1] + e[2] * f[2];

    for ( idx_t i = 3; i < n; i++ )
    {
      t += e[i] * f[i];
    }
  }

  return t;
}


JEM_END_NAMESPACE( array )
JEM_END_PACKAGE_BASE
