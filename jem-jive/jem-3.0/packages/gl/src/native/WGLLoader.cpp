
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


#include "NativeLoader.h"

extern "C"
{
  #include <windows.h>
  #include <wingdi.h>
}

#include <jem/base/native/winclean.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class WGLLoader
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


WGLLoader::WGLLoader ()
{
  lib_ = nullptr;
}


WGLLoader::~WGLLoader ()
{
  if ( lib_ )
  {
    FreeLibrary ( (HMODULE) lib_ );

    lib_ = nullptr;
  }
}


//-----------------------------------------------------------------------
//   getFuncAddr
//-----------------------------------------------------------------------


void* WGLLoader::getFuncAddr ( const char* name )
{
  void*  p = (void*) wglGetProcAddress ( name );

  if ( (p == 0)           ||
       (p == (void*) 0x1) ||
       (p == (void*) 0x2) ||
       (p == (void*) 0x3) ||
       (p == (void*)  -1) )
  {
    if ( ! lib_ )
    {
      lib_ = (void*) LoadLibraryA ( "opengl32.dll" );
    }

    if ( lib_ )
    {
      p = (void*) GetProcAddress ( (HMODULE) lib_, name );
    }
  }

  return p;
}


JEM_END_PACKAGE( gl )
