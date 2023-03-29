
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


#include <jem/gl/ftable/GLFuncTable1_3.h>
#include "GLFuncLoader.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GLFuncTable1_3Core
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GLFuncTable1_3Core::GLFuncTable1_3Core

  ( GLFuncLoader*  loader )

{
  JEM_LOAD_FUNC ( glActiveTexture           );
  JEM_LOAD_FUNC ( glSampleCoverage          );
  JEM_LOAD_FUNC ( glCompressedTexImage3D    );
  JEM_LOAD_FUNC ( glCompressedTexImage2D    );
  JEM_LOAD_FUNC ( glCompressedTexImage1D    );
  JEM_LOAD_FUNC ( glCompressedTexSubImage3D );
  JEM_LOAD_FUNC ( glCompressedTexSubImage2D );
  JEM_LOAD_FUNC ( glCompressedTexSubImage1D );
  JEM_LOAD_FUNC ( glGetCompressedTexImage   );
}


GLFuncTable1_3Core::~GLFuncTable1_3Core ()
{}


JEM_END_PACKAGE( gl )
