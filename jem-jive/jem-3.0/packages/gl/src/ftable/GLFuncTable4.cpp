
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


#include <jem/gl/ftable/GLFuncTable4.h>
#include "GLFuncLoader.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GLFuncTable4Core
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GLFuncTable4Core::GLFuncTable4Core

  ( GLFuncLoader*  loader )

{
  JEM_LOAD_FUNC ( glMinSampleShading );
  JEM_LOAD_FUNC ( glBlendEquationi );
  JEM_LOAD_FUNC ( glBlendEquationSeparatei );
  JEM_LOAD_FUNC ( glBlendFunci );
  JEM_LOAD_FUNC ( glBlendFuncSeparatei );
  JEM_LOAD_FUNC ( glDrawArraysIndirect );
  JEM_LOAD_FUNC ( glDrawElementsIndirect );
  JEM_LOAD_FUNC ( glUniform1d );
  JEM_LOAD_FUNC ( glUniform2d );
  JEM_LOAD_FUNC ( glUniform3d );
  JEM_LOAD_FUNC ( glUniform4d );
  JEM_LOAD_FUNC ( glUniform1dv );
  JEM_LOAD_FUNC ( glUniform2dv );
  JEM_LOAD_FUNC ( glUniform3dv );
  JEM_LOAD_FUNC ( glUniform4dv );
  JEM_LOAD_FUNC ( glUniformMatrix2dv );
  JEM_LOAD_FUNC ( glUniformMatrix3dv );
  JEM_LOAD_FUNC ( glUniformMatrix4dv );
  JEM_LOAD_FUNC ( glUniformMatrix2x3dv );
  JEM_LOAD_FUNC ( glUniformMatrix2x4dv );
  JEM_LOAD_FUNC ( glUniformMatrix3x2dv );
  JEM_LOAD_FUNC ( glUniformMatrix3x4dv );
  JEM_LOAD_FUNC ( glUniformMatrix4x2dv );
  JEM_LOAD_FUNC ( glUniformMatrix4x3dv );
  JEM_LOAD_FUNC ( glGetUniformdv );
  JEM_LOAD_FUNC ( glGetSubroutineUniformLocation );
  JEM_LOAD_FUNC ( glGetSubroutineIndex );
  JEM_LOAD_FUNC ( glGetActiveSubroutineUniformiv );
  JEM_LOAD_FUNC ( glGetActiveSubroutineUniformName );
  JEM_LOAD_FUNC ( glGetActiveSubroutineName );
  JEM_LOAD_FUNC ( glUniformSubroutinesuiv );
  JEM_LOAD_FUNC ( glGetUniformSubroutineuiv );
  JEM_LOAD_FUNC ( glGetProgramStageiv );
  JEM_LOAD_FUNC ( glPatchParameteri );
  JEM_LOAD_FUNC ( glPatchParameterfv );
  JEM_LOAD_FUNC ( glBindTransformFeedback );
  JEM_LOAD_FUNC ( glDeleteTransformFeedbacks );
  JEM_LOAD_FUNC ( glGenTransformFeedbacks );
  JEM_LOAD_FUNC ( glIsTransformFeedback );
  JEM_LOAD_FUNC ( glPauseTransformFeedback );
  JEM_LOAD_FUNC ( glResumeTransformFeedback );
  JEM_LOAD_FUNC ( glDrawTransformFeedback );
  JEM_LOAD_FUNC ( glDrawTransformFeedbackStream );
  JEM_LOAD_FUNC ( glBeginQueryIndexed );
  JEM_LOAD_FUNC ( glEndQueryIndexed );
  JEM_LOAD_FUNC ( glGetQueryIndexediv );
}


GLFuncTable4Core::~GLFuncTable4Core ()
{}


//=======================================================================
//   class GLFuncTable4Compat
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GLFuncTable4Compat::GLFuncTable4Compat

  ( GLFuncLoader*  loader )

{}


GLFuncTable4Compat::~GLFuncTable4Compat ()
{}


JEM_END_PACKAGE( gl )
