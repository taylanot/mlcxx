
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


#include <cctype>
#include <jem/base/StringBuffer.h>
#include <jem/hdf5/H5Exception.h>


JEM_BEGIN_PACKAGE( hdf5 )


//=======================================================================
//   class H5Exception
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


H5Exception::H5Exception

  ( const String&  where,
    const String&  what ) :

    Super ( where, what )

{}


H5Exception::H5Exception

  ( const String&  where,
    const String&  what,
    const String&  details ) :

    Super ( where, makeMessage_( what, details ) )

{}


//-----------------------------------------------------------------------
//   name
//-----------------------------------------------------------------------


String H5Exception::name () const
{
  return "jem::hdf5::H5Exception";
}


//-----------------------------------------------------------------------
//   makeMessage_
//-----------------------------------------------------------------------


String H5Exception::makeMessage_

  ( const String&  what,
    const String&  details )

{
  if ( details.size() > 0 )
  {
    StringBuffer  buf;

    buf += what;
    buf += " (";
    buf += (char) tolower ( details[0] );
    buf += details[slice(1,END)];
    buf += ')';

    return buf.toString ();
  }
  else
  {
    return what;
  }
}


JEM_END_PACKAGE( hdf5 )

