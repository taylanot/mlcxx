
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


#include <jem/gl/ftable/GLFuncTable4_3.h>
#include "GLFuncLoader.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GLFuncTable4_3Core
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GLFuncTable4_3Core::GLFuncTable4_3Core

  ( GLFuncLoader*  loader )

{
  JEM_LOAD_FUNC ( glClearBufferData );
  JEM_LOAD_FUNC ( glClearBufferSubData );
  JEM_LOAD_FUNC ( glDispatchCompute );
  JEM_LOAD_FUNC ( glDispatchComputeIndirect );
  JEM_LOAD_FUNC ( glCopyImageSubData );
  JEM_LOAD_FUNC ( glFramebufferParameteri );
  JEM_LOAD_FUNC ( glGetFramebufferParameteriv );
  JEM_LOAD_FUNC ( glGetInternalformati64v );
  JEM_LOAD_FUNC ( glInvalidateTexSubImage );
  JEM_LOAD_FUNC ( glInvalidateTexImage );
  JEM_LOAD_FUNC ( glInvalidateBufferSubData );
  JEM_LOAD_FUNC ( glInvalidateBufferData );
  JEM_LOAD_FUNC ( glInvalidateFramebuffer );
  JEM_LOAD_FUNC ( glInvalidateSubFramebuffer );
  JEM_LOAD_FUNC ( glMultiDrawArraysIndirect );
  JEM_LOAD_FUNC ( glMultiDrawElementsIndirect );
  JEM_LOAD_FUNC ( glGetProgramInterfaceiv );
  JEM_LOAD_FUNC ( glGetProgramResourceIndex );
  JEM_LOAD_FUNC ( glGetProgramResourceName );
  JEM_LOAD_FUNC ( glGetProgramResourceiv );
  JEM_LOAD_FUNC ( glGetProgramResourceLocation );
  JEM_LOAD_FUNC ( glGetProgramResourceLocationIndex );
  JEM_LOAD_FUNC ( glShaderStorageBlockBinding );
  JEM_LOAD_FUNC ( glTexBufferRange );
  JEM_LOAD_FUNC ( glTexStorage2DMultisample );
  JEM_LOAD_FUNC ( glTexStorage3DMultisample );
  JEM_LOAD_FUNC ( glTextureView );
  JEM_LOAD_FUNC ( glBindVertexBuffer );
  JEM_LOAD_FUNC ( glVertexAttribFormat );
  JEM_LOAD_FUNC ( glVertexAttribIFormat );
  JEM_LOAD_FUNC ( glVertexAttribLFormat );
  JEM_LOAD_FUNC ( glVertexAttribBinding );
  JEM_LOAD_FUNC ( glVertexBindingDivisor );
  JEM_LOAD_FUNC ( glDebugMessageControl );
  JEM_LOAD_FUNC ( glDebugMessageInsert );
  JEM_LOAD_FUNC ( glDebugMessageCallback );
  JEM_LOAD_FUNC ( glGetDebugMessageLog );
  JEM_LOAD_FUNC ( glPushDebugGroup );
  JEM_LOAD_FUNC ( glPopDebugGroup );
  JEM_LOAD_FUNC ( glObjectLabel );
  JEM_LOAD_FUNC ( glGetObjectLabel );
  JEM_LOAD_FUNC ( glObjectPtrLabel );
  JEM_LOAD_FUNC ( glGetObjectPtrLabel );
}


GLFuncTable4_3Core::~GLFuncTable4_3Core ()
{}


JEM_END_PACKAGE( gl )
