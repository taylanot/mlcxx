
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


#include <cstring>
#include <jem/hdf5/Source.h>
#include "jemhdf5.h"


JEM_BEGIN_PACKAGE( hdf5 )


//=======================================================================
//   private functions
//=======================================================================


template <class T>

  static void*              pack_

  ( void*                     vlbuf,
    const util::Flex<T>&      data )

{
  const idx_t  n = data.size ();
  hvl_t*       v = (hvl_t*) vlbuf;

  if ( v->p )
  {
    MemCache::dealloc ( v->p, v->len * sizeof(T) );

    v->p = nullptr;
  }

  v->p   = MemCache::alloc ( (size_t) n * sizeof(T) );
  v->len = (size_t) n;

  std::memcpy ( v->p, data.addr(), (size_t) n * sizeof(T) );

  return (v + 1);
}


//=======================================================================
//   public functions
//=======================================================================


void*                       pack

  ( void*                     vlbuf,
    const String&             data )

{
  const idx_t  n = data.size ();
  hvl_t*       v = (hvl_t*) vlbuf;

  if ( v->p )
  {
    MemCache::dealloc ( v->p, v->len );

    v->p = nullptr;
  }

  v->p   = MemCache::alloc ( (size_t) n );
  v->len = (size_t) n;

  std::memcpy ( v->p, data.addr(), (size_t) n );

  return (v + 1);
}


void*                       pack

  ( void*                     vlbuf,
    const ByteFlex&           data )

{
  return pack_ ( vlbuf, data );
}


void*                       pack

  ( void*                     vlbuf,
    const ShortFlex&          data )

{
  return pack_ ( vlbuf, data );
}


void*                       pack

  ( void*                     vlbuf,
    const IntFlex&            data )

{
  return pack_ ( vlbuf, data );
}


void*                       pack

  ( void*                     vlbuf,
    const LongFlex&           data )

{
  return pack_ ( vlbuf, data );
}


void*                       pack

  ( void*                     vlbuf,
    const LintFlex&           data )

{
  return pack_ ( vlbuf, data );
}


void*                       pack

  ( void*                     vlbuf,
    const IdxFlex&            data )

{
  return pack_ ( vlbuf, data );
}


void*                       pack

  ( void*                     vlbuf,
    const FloatFlex&          data )

{
  return pack_ ( vlbuf, data );
}


void*                       pack

  ( void*                     vlbuf,
    const DoubleFlex&         data )

{
  return pack_ ( vlbuf, data );
}


JEM_END_PACKAGE( hdf5 )

