
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


#include <jem/gl/ftable/GLFuncTable4_1.h>
#include "GLFuncLoader.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GLFuncTable4_1Core
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GLFuncTable4_1Core::GLFuncTable4_1Core

  ( GLFuncLoader*  loader )

{
  JEM_LOAD_FUNC ( glReleaseShaderCompiler );
  JEM_LOAD_FUNC ( glShaderBinary );
  JEM_LOAD_FUNC ( glGetShaderPrecisionFormat );
  JEM_LOAD_FUNC ( glDepthRangef );
  JEM_LOAD_FUNC ( glClearDepthf );
  JEM_LOAD_FUNC ( glGetProgramBinary );
  JEM_LOAD_FUNC ( glProgramBinary );
  JEM_LOAD_FUNC ( glProgramParameteri );
  JEM_LOAD_FUNC ( glUseProgramStages );
  JEM_LOAD_FUNC ( glActiveShaderProgram );
  JEM_LOAD_FUNC ( glCreateShaderProgramv );
  JEM_LOAD_FUNC ( glBindProgramPipeline );
  JEM_LOAD_FUNC ( glDeleteProgramPipelines );
  JEM_LOAD_FUNC ( glGenProgramPipelines );
  JEM_LOAD_FUNC ( glIsProgramPipeline );
  JEM_LOAD_FUNC ( glGetProgramPipelineiv );
  JEM_LOAD_FUNC ( glProgramUniform1i );
  JEM_LOAD_FUNC ( glProgramUniform1iv );
  JEM_LOAD_FUNC ( glProgramUniform1f );
  JEM_LOAD_FUNC ( glProgramUniform1fv );
  JEM_LOAD_FUNC ( glProgramUniform1d );
  JEM_LOAD_FUNC ( glProgramUniform1dv );
  JEM_LOAD_FUNC ( glProgramUniform1ui );
  JEM_LOAD_FUNC ( glProgramUniform1uiv );
  JEM_LOAD_FUNC ( glProgramUniform2i );
  JEM_LOAD_FUNC ( glProgramUniform2iv );
  JEM_LOAD_FUNC ( glProgramUniform2f );
  JEM_LOAD_FUNC ( glProgramUniform2fv );
  JEM_LOAD_FUNC ( glProgramUniform2d );
  JEM_LOAD_FUNC ( glProgramUniform2dv );
  JEM_LOAD_FUNC ( glProgramUniform2ui );
  JEM_LOAD_FUNC ( glProgramUniform2uiv );
  JEM_LOAD_FUNC ( glProgramUniform3i );
  JEM_LOAD_FUNC ( glProgramUniform3iv );
  JEM_LOAD_FUNC ( glProgramUniform3f );
  JEM_LOAD_FUNC ( glProgramUniform3fv );
  JEM_LOAD_FUNC ( glProgramUniform3d );
  JEM_LOAD_FUNC ( glProgramUniform3dv );
  JEM_LOAD_FUNC ( glProgramUniform3ui );
  JEM_LOAD_FUNC ( glProgramUniform3uiv );
  JEM_LOAD_FUNC ( glProgramUniform4i );
  JEM_LOAD_FUNC ( glProgramUniform4iv );
  JEM_LOAD_FUNC ( glProgramUniform4f );
  JEM_LOAD_FUNC ( glProgramUniform4fv );
  JEM_LOAD_FUNC ( glProgramUniform4d );
  JEM_LOAD_FUNC ( glProgramUniform4dv );
  JEM_LOAD_FUNC ( glProgramUniform4ui );
  JEM_LOAD_FUNC ( glProgramUniform4uiv );
  JEM_LOAD_FUNC ( glProgramUniformMatrix2fv );
  JEM_LOAD_FUNC ( glProgramUniformMatrix3fv );
  JEM_LOAD_FUNC ( glProgramUniformMatrix4fv );
  JEM_LOAD_FUNC ( glProgramUniformMatrix2dv );
  JEM_LOAD_FUNC ( glProgramUniformMatrix3dv );
  JEM_LOAD_FUNC ( glProgramUniformMatrix4dv );
  JEM_LOAD_FUNC ( glProgramUniformMatrix2x3fv );
  JEM_LOAD_FUNC ( glProgramUniformMatrix3x2fv );
  JEM_LOAD_FUNC ( glProgramUniformMatrix2x4fv );
  JEM_LOAD_FUNC ( glProgramUniformMatrix4x2fv );
  JEM_LOAD_FUNC ( glProgramUniformMatrix3x4fv );
  JEM_LOAD_FUNC ( glProgramUniformMatrix4x3fv );
  JEM_LOAD_FUNC ( glProgramUniformMatrix2x3dv );
  JEM_LOAD_FUNC ( glProgramUniformMatrix3x2dv );
  JEM_LOAD_FUNC ( glProgramUniformMatrix2x4dv );
  JEM_LOAD_FUNC ( glProgramUniformMatrix4x2dv );
  JEM_LOAD_FUNC ( glProgramUniformMatrix3x4dv );
  JEM_LOAD_FUNC ( glProgramUniformMatrix4x3dv );
  JEM_LOAD_FUNC ( glValidateProgramPipeline );
  JEM_LOAD_FUNC ( glGetProgramPipelineInfoLog );
  JEM_LOAD_FUNC ( glVertexAttribL1d );
  JEM_LOAD_FUNC ( glVertexAttribL2d );
  JEM_LOAD_FUNC ( glVertexAttribL3d );
  JEM_LOAD_FUNC ( glVertexAttribL4d );
  JEM_LOAD_FUNC ( glVertexAttribL1dv );
  JEM_LOAD_FUNC ( glVertexAttribL2dv );
  JEM_LOAD_FUNC ( glVertexAttribL3dv );
  JEM_LOAD_FUNC ( glVertexAttribL4dv );
  JEM_LOAD_FUNC ( glVertexAttribLPointer );
  JEM_LOAD_FUNC ( glGetVertexAttribLdv );
  JEM_LOAD_FUNC ( glViewportArrayv );
  JEM_LOAD_FUNC ( glViewportIndexedf );
  JEM_LOAD_FUNC ( glViewportIndexedfv );
  JEM_LOAD_FUNC ( glScissorArrayv );
  JEM_LOAD_FUNC ( glScissorIndexed );
  JEM_LOAD_FUNC ( glScissorIndexedv );
  JEM_LOAD_FUNC ( glDepthRangeArrayv );
  JEM_LOAD_FUNC ( glDepthRangeIndexed );
  JEM_LOAD_FUNC ( glGetFloati_v );
  JEM_LOAD_FUNC ( glGetDoublei_v );
}


GLFuncTable4_1Core::~GLFuncTable4_1Core ()
{}


JEM_END_PACKAGE( gl )
