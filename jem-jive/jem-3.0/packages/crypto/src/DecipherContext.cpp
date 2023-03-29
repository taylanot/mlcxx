
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


#include <jem/base/StringBuffer.h>
#include <jem/crypto/DecipherContext.h>


JEM_BEGIN_PACKAGE( crypto )


//=======================================================================
//   class DecipherContext
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


DecipherContext::~DecipherContext ()
{}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void DecipherContext::update_

  ( ByteVector&  out,
    const void*  data,
    idx_t        size )

{
  JEM_PRECHECK ( size >= 0 );

  idx_t  count;

  out.resize ( sizeHint( size ) );

  count = update ( out.addr(), data, size );

  out.reshape ( count );
}


void DecipherContext::update_

  ( String&      out,
    const void*  data,
    idx_t        size )

{
  StringBuffer  buffer;
  idx_t         count;

  count = update ( buffer.xalloc ( sizeHint( size ) ),
                   data, size );

  buffer.commit ( count );

  out = buffer.toString ();
}


//-----------------------------------------------------------------------
//   finish_
//-----------------------------------------------------------------------


void DecipherContext::finish_

  ( ByteVector&  out,
    const void*  data,
    idx_t        size )

{
  JEM_PRECHECK ( size >= 0 );

  idx_t  count;

  out.resize ( sizeHint( size ) );

  count = finish ( out.addr(), data, size );

  out.reshape ( count );
}


void DecipherContext::finish_

  ( String&      out,
    const void*  data,
    idx_t        size )

{
  JEM_PRECHECK ( size >= 0 );

  StringBuffer  buffer;
  idx_t         count;

  count = finish ( buffer.xalloc ( sizeHint( size ) ),
                   data, size );

  buffer.commit ( count );

  out = buffer.toString ();
}


JEM_END_PACKAGE( crypto )

