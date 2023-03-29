
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


#include <jem/gl/ftable/GLFuncTable3_3.h>
#include "GLFuncLoader.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GLFuncTable3_3Core
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GLFuncTable3_3Core::GLFuncTable3_3Core

  ( GLFuncLoader*  loader )

{
  JEM_LOAD_FUNC ( glBindFragDataLocationIndexed );
  JEM_LOAD_FUNC ( glGetFragDataIndex            );
  JEM_LOAD_FUNC ( glGenSamplers                 );
  JEM_LOAD_FUNC ( glDeleteSamplers              );
  JEM_LOAD_FUNC ( glIsSampler                   );
  JEM_LOAD_FUNC ( glBindSampler                 );
  JEM_LOAD_FUNC ( glSamplerParameteri           );
  JEM_LOAD_FUNC ( glSamplerParameteriv          );
  JEM_LOAD_FUNC ( glSamplerParameterf           );
  JEM_LOAD_FUNC ( glSamplerParameterfv          );
  JEM_LOAD_FUNC ( glSamplerParameterIiv         );
  JEM_LOAD_FUNC ( glSamplerParameterIuiv        );
  JEM_LOAD_FUNC ( glGetSamplerParameteriv       );
  JEM_LOAD_FUNC ( glGetSamplerParameterIiv      );
  JEM_LOAD_FUNC ( glGetSamplerParameterfv       );
  JEM_LOAD_FUNC ( glGetSamplerParameterIuiv     );
  JEM_LOAD_FUNC ( glQueryCounter                );
  JEM_LOAD_FUNC ( glGetQueryObjecti64v          );
  JEM_LOAD_FUNC ( glGetQueryObjectui64v         );
  JEM_LOAD_FUNC ( glVertexAttribDivisor         );
  JEM_LOAD_FUNC ( glVertexAttribP1ui            );
  JEM_LOAD_FUNC ( glVertexAttribP1uiv           );
  JEM_LOAD_FUNC ( glVertexAttribP2ui            );
  JEM_LOAD_FUNC ( glVertexAttribP2uiv           );
  JEM_LOAD_FUNC ( glVertexAttribP3ui            );
  JEM_LOAD_FUNC ( glVertexAttribP3uiv           );
  JEM_LOAD_FUNC ( glVertexAttribP4ui            );
  JEM_LOAD_FUNC ( glVertexAttribP4uiv           );
}


GLFuncTable3_3Core::~GLFuncTable3_3Core ()
{}


JEM_END_PACKAGE( gl )
