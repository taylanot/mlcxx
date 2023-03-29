
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


#include <jem/gl/ftable/GLFuncTable3.h>
#include "GLFuncLoader.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GLFuncTable3Core
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GLFuncTable3Core::GLFuncTable3Core

  ( GLFuncLoader*  loader )

{
  JEM_LOAD_FUNC ( glColorMaski );
  JEM_LOAD_FUNC ( glGetBooleani_v );
  JEM_LOAD_FUNC ( glGetIntegeri_v );
  JEM_LOAD_FUNC ( glEnablei );
  JEM_LOAD_FUNC ( glDisablei );
  JEM_LOAD_FUNC ( glIsEnabledi );
  JEM_LOAD_FUNC ( glEndTransformFeedback );
  JEM_LOAD_FUNC ( glBindBufferRange );
  JEM_LOAD_FUNC ( glBindBufferBase );
  JEM_LOAD_FUNC ( glTransformFeedbackVaryings );
  JEM_LOAD_FUNC ( glGetTransformFeedbackVarying );
  JEM_LOAD_FUNC ( glClampColor );
  JEM_LOAD_FUNC ( glBeginConditionalRender );
  JEM_LOAD_FUNC ( glEndConditionalRender );
  JEM_LOAD_FUNC ( glVertexAttribIPointer );
  JEM_LOAD_FUNC ( glGetVertexAttribIiv );
  JEM_LOAD_FUNC ( glGetVertexAttribIuiv );
  JEM_LOAD_FUNC ( glVertexAttribI1i );
  JEM_LOAD_FUNC ( glVertexAttribI2i );
  JEM_LOAD_FUNC ( glVertexAttribI3i );
  JEM_LOAD_FUNC ( glVertexAttribI4i );
  JEM_LOAD_FUNC ( glVertexAttribI1ui );
  JEM_LOAD_FUNC ( glVertexAttribI2ui );
  JEM_LOAD_FUNC ( glVertexAttribI3ui );
  JEM_LOAD_FUNC ( glVertexAttribI4ui );
  JEM_LOAD_FUNC ( glVertexAttribI1iv );
  JEM_LOAD_FUNC ( glVertexAttribI2iv );
  JEM_LOAD_FUNC ( glVertexAttribI3iv );
  JEM_LOAD_FUNC ( glVertexAttribI4iv );
  JEM_LOAD_FUNC ( glVertexAttribI1uiv );
  JEM_LOAD_FUNC ( glVertexAttribI2uiv );
  JEM_LOAD_FUNC ( glVertexAttribI3uiv );
  JEM_LOAD_FUNC ( glVertexAttribI4uiv );
  JEM_LOAD_FUNC ( glVertexAttribI4bv );
  JEM_LOAD_FUNC ( glVertexAttribI4sv );
  JEM_LOAD_FUNC ( glVertexAttribI4ubv );
  JEM_LOAD_FUNC ( glVertexAttribI4usv );
  JEM_LOAD_FUNC ( glGetUniformuiv );
  JEM_LOAD_FUNC ( glBindFragDataLocation );
  JEM_LOAD_FUNC ( glGetFragDataLocation );
  JEM_LOAD_FUNC ( glUniform1ui );
  JEM_LOAD_FUNC ( glUniform2ui );
  JEM_LOAD_FUNC ( glUniform3ui );
  JEM_LOAD_FUNC ( glUniform4ui );
  JEM_LOAD_FUNC ( glUniform1uiv );
  JEM_LOAD_FUNC ( glUniform2uiv );
  JEM_LOAD_FUNC ( glUniform3uiv );
  JEM_LOAD_FUNC ( glUniform4uiv );
  JEM_LOAD_FUNC ( glTexParameterIiv );
  JEM_LOAD_FUNC ( glTexParameterIuiv );
  JEM_LOAD_FUNC ( glGetTexParameterIiv );
  JEM_LOAD_FUNC ( glGetTexParameterIuiv );
  JEM_LOAD_FUNC ( glClearBufferiv );
  JEM_LOAD_FUNC ( glClearBufferuiv );
  JEM_LOAD_FUNC ( glClearBufferfv );
  JEM_LOAD_FUNC ( glClearBufferfi );
  JEM_LOAD_FUNC ( glGetStringi );
  JEM_LOAD_FUNC ( glIsRenderbuffer );
  JEM_LOAD_FUNC ( glBindRenderbuffer );
  JEM_LOAD_FUNC ( glDeleteRenderbuffers );
  JEM_LOAD_FUNC ( glGenRenderbuffers );
  JEM_LOAD_FUNC ( glRenderbufferStorage );
  JEM_LOAD_FUNC ( glGetRenderbufferParameteriv );
  JEM_LOAD_FUNC ( glIsFramebuffer );
  JEM_LOAD_FUNC ( glBindFramebuffer );
  JEM_LOAD_FUNC ( glDeleteFramebuffers );
  JEM_LOAD_FUNC ( glGenFramebuffers );
  JEM_LOAD_FUNC ( glCheckFramebufferStatus );
  JEM_LOAD_FUNC ( glFramebufferTexture1D );
  JEM_LOAD_FUNC ( glFramebufferTexture2D );
  JEM_LOAD_FUNC ( glFramebufferTexture3D );
  JEM_LOAD_FUNC ( glFramebufferRenderbuffer );
  JEM_LOAD_FUNC ( glGetFramebufferAttachmentParameteriv );
  JEM_LOAD_FUNC ( glGenerateMipmap );
  JEM_LOAD_FUNC ( glBlitFramebuffer );
  JEM_LOAD_FUNC ( glRenderbufferStorageMultisample );
  JEM_LOAD_FUNC ( glFramebufferTextureLayer );
  JEM_LOAD_FUNC ( glMapBufferRange );
  JEM_LOAD_FUNC ( glFlushMappedBufferRange );
  JEM_LOAD_FUNC ( glBindVertexArray );
  JEM_LOAD_FUNC ( glDeleteVertexArrays );
  JEM_LOAD_FUNC ( glGenVertexArrays );
  JEM_LOAD_FUNC ( glIsVertexArray );
}


GLFuncTable3Core::~GLFuncTable3Core ()
{}


//=======================================================================
//   class GLFuncTable3Compat
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GLFuncTable3Compat::GLFuncTable3Compat

  ( GLFuncLoader*  loader )

{}


GLFuncTable3Compat::~GLFuncTable3Compat ()
{}


JEM_END_PACKAGE( gl )
