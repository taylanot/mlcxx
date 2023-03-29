
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
#include <jem/hdf5/Sink.h>
#include "jemhdf5.h"


JEM_BEGIN_PACKAGE( hdf5 )


//=======================================================================
//   private functions
//=======================================================================


template <class T>

  static const void*        unpack_

  ( util::Flex<T>&            data,
    const void*               vlbuf )

{
  const hvl_t*  v = (const hvl_t*) vlbuf;
  const T*      p = (const T*)     v->p;

  data.resize ( (idx_t) v->len );
  std::memcpy ( data.addr(), p, v->len * sizeof(T) );

  return (v + 1);
}


//=======================================================================
//   public functions
//=======================================================================


const void*                 unpack

  ( String&                   data,
    const void*               vlbuf )

{
  const hvl_t*  v = (const hvl_t*) vlbuf;
  const char*   p = (const char*)  v->p;

  data = String ( p, (idx_t) v->len );

  return (v + 1);
}


const void*                 unpack

  ( ByteFlex&                 data,
    const void*               vlbuf )

{
  return unpack_ ( data, vlbuf );
}


const void*                 unpack

  ( ShortFlex&                data,
    const void*               vlbuf )

{
  return unpack_ ( data, vlbuf );
}


const void*                 unpack

  ( IntFlex&                  data,
    const void*               vlbuf )

{
  return unpack_ ( data, vlbuf );
}


const void*                 unpack

  ( LongFlex&                 data,
    const void*               vlbuf )

{
  return unpack_ ( data, vlbuf );
}


const void*                 unpack

  ( LintFlex&                 data,
    const void*               vlbuf )

{
  return unpack_ ( data, vlbuf );
}


const void*                 unpack

  ( IdxFlex&                  data,
    const void*               vlbuf )

{
  return unpack_ ( data, vlbuf );
}


const void*                 unpack

  ( FloatFlex&                data,
    const void*               vlbuf )

{
  return unpack_ ( data, vlbuf );
}


const void*                 unpack

  ( DoubleFlex&               data,
    const void*               vlbuf )

{
  return unpack_ ( data, vlbuf );
}


JEM_END_PACKAGE( hdf5 )

