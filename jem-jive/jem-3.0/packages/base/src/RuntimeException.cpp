
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


#include <cstdlib>
#include <jem/base/EnvParams.h>
#include <jem/base/RuntimeException.h>
#include <jem/base/native/System.h>


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class RuntimeException
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


RuntimeException::RuntimeException

  ( const String&  where,
    const String&  what ) :

    Super  ( where, what )

{
  trace_ = NativeSystem::getStackTrace ( 1 );

  if ( ::getenv( EnvParams::ABORT_ON_EXCEPT ) )
  {
    jem::abort ( *this );
  }
}


RuntimeException::RuntimeException ( const Self& rhs ) :

  Super  ( rhs ),
  trace_ ( rhs.trace_ )

{}


//-----------------------------------------------------------------------
//   name()
//-----------------------------------------------------------------------


String RuntimeException::name () const
{
  return "jem::RuntimeException";
}


//-----------------------------------------------------------------------
//   getStackTrace()
//-----------------------------------------------------------------------


String RuntimeException::getStackTrace () const
{
  return trace_;
}


JEM_END_PACKAGE_BASE
