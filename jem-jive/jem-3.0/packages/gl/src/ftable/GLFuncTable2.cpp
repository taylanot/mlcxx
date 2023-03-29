
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


#include <jem/gl/ftable/GLFuncTable2.h>
#include "GLFuncLoader.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GLFuncTable2Core
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GLFuncTable2Core::GLFuncTable2Core

  ( GLFuncLoader*  loader )

{
  JEM_LOAD_FUNC ( glBlendEquationSeparate );
  JEM_LOAD_FUNC ( glDrawBuffers );
  JEM_LOAD_FUNC ( glStencilOpSeparate );
  JEM_LOAD_FUNC ( glStencilFuncSeparate );
  JEM_LOAD_FUNC ( glStencilMaskSeparate );
  JEM_LOAD_FUNC ( glAttachShader );
  JEM_LOAD_FUNC ( glBindAttribLocation );
  JEM_LOAD_FUNC ( glCompileShader );
  JEM_LOAD_FUNC ( glCreateProgram );
  JEM_LOAD_FUNC ( glCreateShader );
  JEM_LOAD_FUNC ( glDeleteProgram );
  JEM_LOAD_FUNC ( glDeleteShader );
  JEM_LOAD_FUNC ( glDetachShader );
  JEM_LOAD_FUNC ( glDisableVertexAttribArray );
  JEM_LOAD_FUNC ( glEnableVertexAttribArray );
  JEM_LOAD_FUNC ( glGetActiveAttrib );
  JEM_LOAD_FUNC ( glGetActiveUniform );
  JEM_LOAD_FUNC ( glGetAttachedShaders );
  JEM_LOAD_FUNC ( glGetAttribLocation );
  JEM_LOAD_FUNC ( glGetProgramiv );
  JEM_LOAD_FUNC ( glGetProgramInfoLog );
  JEM_LOAD_FUNC ( glGetShaderiv );
  JEM_LOAD_FUNC ( glGetShaderInfoLog );
  JEM_LOAD_FUNC ( glGetShaderSource );
  JEM_LOAD_FUNC ( glGetUniformLocation );
  JEM_LOAD_FUNC ( glGetUniformfv );
  JEM_LOAD_FUNC ( glGetUniformiv );
  JEM_LOAD_FUNC ( glGetVertexAttribdv );
  JEM_LOAD_FUNC ( glGetVertexAttribfv );
  JEM_LOAD_FUNC ( glGetVertexAttribiv );
  JEM_LOAD_FUNC ( glGetVertexAttribPointerv );
  JEM_LOAD_FUNC ( glIsProgram );
  JEM_LOAD_FUNC ( glIsShader );
  JEM_LOAD_FUNC ( glLinkProgram );
  JEM_LOAD_FUNC ( glShaderSource );
  JEM_LOAD_FUNC ( glUseProgram );
  JEM_LOAD_FUNC ( glUniform1f );
  JEM_LOAD_FUNC ( glUniform2f );
  JEM_LOAD_FUNC ( glUniform3f );
  JEM_LOAD_FUNC ( glUniform4f );
  JEM_LOAD_FUNC ( glUniform1i );
  JEM_LOAD_FUNC ( glUniform2i );
  JEM_LOAD_FUNC ( glUniform3i );
  JEM_LOAD_FUNC ( glUniform4i );
  JEM_LOAD_FUNC ( glUniform1fv );
  JEM_LOAD_FUNC ( glUniform2fv );
  JEM_LOAD_FUNC ( glUniform3fv );
  JEM_LOAD_FUNC ( glUniform4fv );
  JEM_LOAD_FUNC ( glUniform1iv );
  JEM_LOAD_FUNC ( glUniform2iv );
  JEM_LOAD_FUNC ( glUniform3iv );
  JEM_LOAD_FUNC ( glUniform4iv );
  JEM_LOAD_FUNC ( glUniformMatrix2fv );
  JEM_LOAD_FUNC ( glUniformMatrix3fv );
  JEM_LOAD_FUNC ( glUniformMatrix4fv );
  JEM_LOAD_FUNC ( glValidateProgram );
  JEM_LOAD_FUNC ( glVertexAttrib1d );
  JEM_LOAD_FUNC ( glVertexAttrib1dv );
  JEM_LOAD_FUNC ( glVertexAttrib1f );
  JEM_LOAD_FUNC ( glVertexAttrib1fv );
  JEM_LOAD_FUNC ( glVertexAttrib1s );
  JEM_LOAD_FUNC ( glVertexAttrib1sv );
  JEM_LOAD_FUNC ( glVertexAttrib2d );
  JEM_LOAD_FUNC ( glVertexAttrib2dv );
  JEM_LOAD_FUNC ( glVertexAttrib2f );
  JEM_LOAD_FUNC ( glVertexAttrib2fv );
  JEM_LOAD_FUNC ( glVertexAttrib2s );
  JEM_LOAD_FUNC ( glVertexAttrib2sv );
  JEM_LOAD_FUNC ( glVertexAttrib3d );
  JEM_LOAD_FUNC ( glVertexAttrib3dv );
  JEM_LOAD_FUNC ( glVertexAttrib3f );
  JEM_LOAD_FUNC ( glVertexAttrib3fv );
  JEM_LOAD_FUNC ( glVertexAttrib3s );
  JEM_LOAD_FUNC ( glVertexAttrib3sv );
  JEM_LOAD_FUNC ( glVertexAttrib4Nbv );
  JEM_LOAD_FUNC ( glVertexAttrib4Niv );
  JEM_LOAD_FUNC ( glVertexAttrib4Nsv );
  JEM_LOAD_FUNC ( glVertexAttrib4Nub );
  JEM_LOAD_FUNC ( glVertexAttrib4Nubv );
  JEM_LOAD_FUNC ( glVertexAttrib4Nuiv );
  JEM_LOAD_FUNC ( glVertexAttrib4Nusv );
  JEM_LOAD_FUNC ( glVertexAttrib4bv );
  JEM_LOAD_FUNC ( glVertexAttrib4d );
  JEM_LOAD_FUNC ( glVertexAttrib4dv );
  JEM_LOAD_FUNC ( glVertexAttrib4f );
  JEM_LOAD_FUNC ( glVertexAttrib4fv );
  JEM_LOAD_FUNC ( glVertexAttrib4iv );
  JEM_LOAD_FUNC ( glVertexAttrib4s );
  JEM_LOAD_FUNC ( glVertexAttrib4sv );
  JEM_LOAD_FUNC ( glVertexAttrib4ubv );
  JEM_LOAD_FUNC ( glVertexAttrib4uiv );
  JEM_LOAD_FUNC ( glVertexAttrib4usv );
  JEM_LOAD_FUNC (glVertexAttribPointer);
}


GLFuncTable2Core::~GLFuncTable2Core ()
{}


//=======================================================================
//   class GLFuncTable2Compat
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GLFuncTable2Compat::GLFuncTable2Compat

  ( GLFuncLoader*  loader )

{}


GLFuncTable2Compat::~GLFuncTable2Compat ()
{}


JEM_END_PACKAGE( gl )
