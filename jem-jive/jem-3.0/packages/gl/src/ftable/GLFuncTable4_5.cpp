
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


#include <jem/gl/ftable/GLFuncTable4_5.h>
#include "GLFuncLoader.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GLFuncTable4_5Core
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GLFuncTable4_5Core::GLFuncTable4_5Core

  ( GLFuncLoader*  loader )

{
  JEM_LOAD_FUNC ( glClipControl );
  JEM_LOAD_FUNC ( glCreateTransformFeedbacks );
  JEM_LOAD_FUNC ( glTransformFeedbackBufferBase );
  JEM_LOAD_FUNC ( glTransformFeedbackBufferRange );
  JEM_LOAD_FUNC ( glGetTransformFeedbackiv );
  JEM_LOAD_FUNC ( glGetTransformFeedbacki_v );
  JEM_LOAD_FUNC ( glGetTransformFeedbacki64_v );
  JEM_LOAD_FUNC ( glCreateBuffers );
  JEM_LOAD_FUNC ( glNamedBufferStorage );
  JEM_LOAD_FUNC ( glNamedBufferData );
  JEM_LOAD_FUNC ( glNamedBufferSubData );
  JEM_LOAD_FUNC ( glCopyNamedBufferSubData );
  JEM_LOAD_FUNC ( glClearNamedBufferData );
  JEM_LOAD_FUNC ( glClearNamedBufferSubData );
  JEM_LOAD_FUNC ( glMapNamedBuffer );
  JEM_LOAD_FUNC ( glMapNamedBufferRange );
  JEM_LOAD_FUNC ( glUnmapNamedBuffer );
  JEM_LOAD_FUNC ( glFlushMappedNamedBufferRange );
  JEM_LOAD_FUNC ( glGetNamedBufferParameteriv );
  JEM_LOAD_FUNC ( glGetNamedBufferParameteri64v );
  JEM_LOAD_FUNC ( glGetNamedBufferPointerv );
  JEM_LOAD_FUNC ( glGetNamedBufferSubData );
  JEM_LOAD_FUNC ( glCreateFramebuffers );
  JEM_LOAD_FUNC ( glNamedFramebufferRenderbuffer );
  JEM_LOAD_FUNC ( glNamedFramebufferParameteri );
  JEM_LOAD_FUNC ( glNamedFramebufferTexture );
  JEM_LOAD_FUNC ( glNamedFramebufferTextureLayer );
  JEM_LOAD_FUNC ( glNamedFramebufferDrawBuffer );
  JEM_LOAD_FUNC ( glNamedFramebufferDrawBuffers );
  JEM_LOAD_FUNC ( glNamedFramebufferReadBuffer );
  JEM_LOAD_FUNC ( glInvalidateNamedFramebufferData );
  JEM_LOAD_FUNC ( glInvalidateNamedFramebufferSubData );
  JEM_LOAD_FUNC ( glClearNamedFramebufferiv );
  JEM_LOAD_FUNC ( glClearNamedFramebufferuiv );
  JEM_LOAD_FUNC ( glClearNamedFramebufferfv );
  JEM_LOAD_FUNC ( glClearNamedFramebufferfi );
  JEM_LOAD_FUNC ( glBlitNamedFramebuffer );
  JEM_LOAD_FUNC ( glCheckNamedFramebufferStatus );
  JEM_LOAD_FUNC ( glGetNamedFramebufferParameteriv );
  JEM_LOAD_FUNC ( glGetNamedFramebufferAttachmentParameteriv );
  JEM_LOAD_FUNC ( glCreateRenderbuffers );
  JEM_LOAD_FUNC ( glNamedRenderbufferStorage );
  JEM_LOAD_FUNC ( glNamedRenderbufferStorageMultisample );
  JEM_LOAD_FUNC ( glGetNamedRenderbufferParameteriv );
  JEM_LOAD_FUNC ( glCreateTextures );
  JEM_LOAD_FUNC ( glTextureBuffer );
  JEM_LOAD_FUNC ( glTextureBufferRange );
  JEM_LOAD_FUNC ( glTextureStorage1D );
  JEM_LOAD_FUNC ( glTextureStorage2D );
  JEM_LOAD_FUNC ( glTextureStorage3D );
  JEM_LOAD_FUNC ( glTextureStorage2DMultisample );
  JEM_LOAD_FUNC ( glTextureStorage3DMultisample );
  JEM_LOAD_FUNC ( glTextureSubImage1D );
  JEM_LOAD_FUNC ( glTextureSubImage2D );
  JEM_LOAD_FUNC ( glTextureSubImage3D );
  JEM_LOAD_FUNC ( glCompressedTextureSubImage1D );
  JEM_LOAD_FUNC ( glCompressedTextureSubImage2D );
  JEM_LOAD_FUNC ( glCompressedTextureSubImage3D );
  JEM_LOAD_FUNC ( glCopyTextureSubImage1D );
  JEM_LOAD_FUNC ( glCopyTextureSubImage2D );
  JEM_LOAD_FUNC ( glCopyTextureSubImage3D );
  JEM_LOAD_FUNC ( glTextureParameterf );
  JEM_LOAD_FUNC ( glTextureParameterfv );
  JEM_LOAD_FUNC ( glTextureParameteri );
  JEM_LOAD_FUNC ( glTextureParameterIiv );
  JEM_LOAD_FUNC ( glTextureParameterIuiv );
  JEM_LOAD_FUNC ( glTextureParameteriv );
  JEM_LOAD_FUNC ( glGenerateTextureMipmap );
  JEM_LOAD_FUNC ( glBindTextureUnit );
  JEM_LOAD_FUNC ( glGetTextureImage );
  JEM_LOAD_FUNC ( glGetCompressedTextureImage );
  JEM_LOAD_FUNC ( glGetTextureLevelParameterfv );
  JEM_LOAD_FUNC ( glGetTextureLevelParameteriv );
  JEM_LOAD_FUNC ( glGetTextureParameterfv );
  JEM_LOAD_FUNC ( glGetTextureParameterIiv );
  JEM_LOAD_FUNC ( glGetTextureParameterIuiv );
  JEM_LOAD_FUNC ( glGetTextureParameteriv );
  JEM_LOAD_FUNC ( glCreateVertexArrays );
  JEM_LOAD_FUNC ( glDisableVertexArrayAttrib );
  JEM_LOAD_FUNC ( glEnableVertexArrayAttrib );
  JEM_LOAD_FUNC ( glVertexArrayElementBuffer );
  JEM_LOAD_FUNC ( glVertexArrayVertexBuffer );
  JEM_LOAD_FUNC ( glVertexArrayVertexBuffers );
  JEM_LOAD_FUNC ( glVertexArrayAttribBinding );
  JEM_LOAD_FUNC ( glVertexArrayAttribFormat );
  JEM_LOAD_FUNC ( glVertexArrayAttribIFormat );
  JEM_LOAD_FUNC ( glVertexArrayAttribLFormat );
  JEM_LOAD_FUNC ( glVertexArrayBindingDivisor );
  JEM_LOAD_FUNC ( glGetVertexArrayiv );
  JEM_LOAD_FUNC ( glGetVertexArrayIndexediv );
  JEM_LOAD_FUNC ( glGetVertexArrayIndexed64iv );
  JEM_LOAD_FUNC ( glCreateSamplers );
  JEM_LOAD_FUNC ( glCreateProgramPipelines );
  JEM_LOAD_FUNC ( glCreateQueries );
  JEM_LOAD_FUNC ( glGetQueryBufferObjecti64v );
  JEM_LOAD_FUNC ( glGetQueryBufferObjectiv );
  JEM_LOAD_FUNC ( glGetQueryBufferObjectui64v );
  JEM_LOAD_FUNC ( glGetQueryBufferObjectuiv );
  JEM_LOAD_FUNC ( glMemoryBarrierByRegion );
  JEM_LOAD_FUNC ( glGetTextureSubImage );
  JEM_LOAD_FUNC ( glGetCompressedTextureSubImage );
  JEM_LOAD_FUNC ( glGetGraphicsResetStatus );
  JEM_LOAD_FUNC ( glGetnCompressedTexImage );
  JEM_LOAD_FUNC ( glGetnTexImage );
  JEM_LOAD_FUNC ( glGetnUniformdv );
  JEM_LOAD_FUNC ( glGetnUniformfv );
  JEM_LOAD_FUNC ( glGetnUniformiv );
  JEM_LOAD_FUNC ( glGetnUniformuiv );
  JEM_LOAD_FUNC ( glReadnPixels );
  JEM_LOAD_FUNC ( glGetnMapdv );
  JEM_LOAD_FUNC ( glGetnMapfv );
  JEM_LOAD_FUNC ( glGetnMapiv );
  JEM_LOAD_FUNC ( glGetnPixelMapfv );
  JEM_LOAD_FUNC ( glGetnPixelMapuiv );
  JEM_LOAD_FUNC ( glGetnPixelMapusv );
  JEM_LOAD_FUNC ( glGetnPolygonStipple );
  JEM_LOAD_FUNC ( glGetnColorTable );
  JEM_LOAD_FUNC ( glGetnConvolutionFilter );
  JEM_LOAD_FUNC ( glGetnSeparableFilter );
  JEM_LOAD_FUNC ( glGetnHistogram );
  JEM_LOAD_FUNC ( glGetnMinmax );
  JEM_LOAD_FUNC ( glTextureBarrier );
}


GLFuncTable4_5Core::~GLFuncTable4_5Core ()
{}


JEM_END_PACKAGE( gl )
