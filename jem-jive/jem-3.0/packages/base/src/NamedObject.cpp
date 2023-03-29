
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


#include <jem/base/ClassTemplate.h>
#include <jem/base/StringBuffer.h>
#include <jem/base/NamedObject.h>


JEM_DEFINE_CLASS( jem::NamedObject );


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class NamedObject
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


NamedObject::NamedObject ( const String& name ) :

  myName_ ( name )

{}


NamedObject::~NamedObject ()
{}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String NamedObject::toString () const
{
  if ( ! myName_.size() )
  {
    return Super::toString ();
  }
  else
  {
    const String className = getClass()->getName ();

    return String::format ( "%s ( `%s\' )", className, myName_ );
  }
}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String NamedObject::getContext () const
{
  return makeContext ( "object", myName_ );
}


//-----------------------------------------------------------------------
//   makeContext
//-----------------------------------------------------------------------


String NamedObject::makeContext

  ( const String&  kind,
    const String&  name )

{
  if ( ! name.size() )
  {
    return String ( "unnamed " + kind );
  }
  else
  {
    StringBuffer  buf;

    buf.reserve ( kind.size() + name.size() + 3 );

    buf += kind;
    buf += ' ';
    buf += '`';
    buf += name;
    buf += '\'';

    return buf.toString ();
  }
}


JEM_END_PACKAGE_BASE
