
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

#ifndef JEM_GL_ERROR_H
#define JEM_GL_ERROR_H

#include <jem/base/String.h>
#include <jem/base/tuple/Tuple.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   error reporting functions
//-----------------------------------------------------------------------


void                      indexError

  ( const String&           where,
    const String&           name,
    idx_t                   index,
    idx_t                   ubound = -1 );

void                      sizeError

  ( const String&           where,
    const String&           name,
    idx_t                   size );

void                      shapeError

  ( const String&           where,
    const String&           name,
    const Tuple<idx_t,2>&   sh );

void                      zeroVectorError

  ( const String&           where,
    const String&           name );


JEM_END_PACKAGE( gl )

#endif
