
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
#include <jem/crypto/EncipherContext.h>


JEM_BEGIN_PACKAGE( crypto )


//=======================================================================
//   class EncipherContext
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


EncipherContext::~EncipherContext ()
{}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


ByteVector EncipherContext::update

  ( const void*  data,
    idx_t        size )

{
  ByteVector  buffer ( sizeHint( size ) );

  idx_t       count  = update ( buffer.addr(), data, size );

  buffer.reshape ( count );

  return buffer;
}


ByteVector EncipherContext::update ( const ByteVector& data )
{
  return update ( makeContiguous( data ).addr(), data.size() );
}


ByteVector EncipherContext::update ( const String& data )
{
  return update ( data.addr(), data.size() );
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


ByteVector EncipherContext::finish ()
{
  ByteVector  buffer ( sizeHint() );

  idx_t       count  = finish ( buffer.addr() );

  buffer.reshape ( count );

  return buffer;
}


ByteVector EncipherContext::finish

  ( const void*  data,
    idx_t        size )

{
  JEM_PRECHECK ( size >= 0 );

  ByteVector  buffer ( sizeHint( size ) );

  idx_t       count  = finish ( buffer.addr(), data, size );

  buffer.reshape ( count );

  return buffer;
}


ByteVector EncipherContext::finish ( const ByteVector& data )
{
  return finish ( makeContiguous( data ).addr(), data.size() );
}


ByteVector EncipherContext::finish ( const String& data )
{
  return finish ( data.addr(), data.size() );
}


JEM_END_PACKAGE( crypto )

