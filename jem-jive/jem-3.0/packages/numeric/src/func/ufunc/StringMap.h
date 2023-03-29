
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

#ifndef JEM_NUMERIC_FUNC_UFUNC_STRINGMAP_H
#define JEM_NUMERIC_FUNC_UFUNC_STRINGMAP_H

#include <jem/util/HashMap.h>
#include <jem/util/FastAllocator.h>
#include "base.h"


JEM_BEGIN_UFUNC_IMP


using jem::util::makePair;
using jem::util::HashMap;
using jem::util::FastAllocator;


//-----------------------------------------------------------------------
//   typedefs
//-----------------------------------------------------------------------


typedef HashMap < String,
                  idx_t,
                  FastAllocator >   StringMap;

typedef StringMap::Iterator         StringMapIter;


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


inline idx_t         findSymbol

  ( const String&      name,
    const StringMap&   map )

{
  StringMap::ConstIterator  it = map.find ( name );

  if ( it == map.end() )
  {
    return -1_idx;
  }
  else
  {
    return  it->second;
  }
}


JEM_END_UFUNC_IMP

#endif
