
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


#include <jem/gl/GLContext.h>
#include <jem/gl/GLException.h>
#include <jem/gl/GLFunctions2.h>
#include "VertexBuffer.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class VertexBuffer
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


VertexBuffer::VertexBuffer ( BufferName name )
{
  name_    = name;
  gtxID_   = 0;
  bufID_   = 0;
  bufSize_ = 0;
}


//-----------------------------------------------------------------------
//   bind
//-----------------------------------------------------------------------


bool VertexBuffer::bind ( GLContext& gtx )
{
  GLFuncs2Core*  funcs = gtx.findFunctions<GLFuncs2Core> ();

  if ( funcs )
  {
    return false;
  }

  funcs->glBindBuffer ( GL_ARRAY_BUFFER, getBufferID( gtx ) );

  return true;
}


//-----------------------------------------------------------------------
//   getBufferID
//-----------------------------------------------------------------------


GLuint VertexBuffer::getBufferID ( GLContext& gtx )
{
  GLuint  id = 0;

  if ( gtx.hasID( gtxID_ ) )
  {
    id = bufID_;
  }
  else
  {
    GLFuncs2Core&  funcs = gtx.getFunctions<GLFuncs2Core> ();

    if ( ! gtx.findGLObject( id, name_ ) )
    {
      id = gtx.newGLObject ( GLObject::BUFFER, name_, nullptr );

      funcs.glBindBuffer   ( GL_ARRAY_BUFFER, id );
    }

    gtxID_   = gtx.getID ();
    bufID_   = id;
    bufSize_ = 0;

    if ( funcs.glIsBuffer( id ) )
    {
      funcs.glBindBuffer         ( GL_ARRAY_BUFFER, id );
      funcs.glGetBufferParameter ( GL_ARRAY_BUFFER, GL_BUFFER_SIZE,
                                   &bufSize_ );
      funcs.glBindBuffer         ( GL_ARRAY_BUFFER, 0 );
    }
    else
    {
      throw GLException (
        JEM_FUNC,
        "unable to generate a vertex buffer; the retrieved "
        "OpenGL object is not a buffer object"
      );
    }
  }

  return id;
}


//-----------------------------------------------------------------------
//   setData
//-----------------------------------------------------------------------


void VertexBuffer::setData

  ( GLContext&      gtx,
    const GLfloat*  data,
    idx_t           count,
    idx_t           offset )

{
  const idx_t    dataSize   = count    * (idx_t) sizeof(*data);
  const idx_t    offsetSize = offset   * (idx_t) sizeof(*data);
  const idx_t    bufferSize = dataSize + offsetSize;

  GLFuncs2Core&  funcs      = gtx.getFunctions<GLFuncs2Core> ();


  funcs.glBindBuffer ( GL_ARRAY_BUFFER, getBufferID( gtx ) );

  if ( (idx_t) bufSize_ < bufferSize )
  {
    // Allocate the buffer memory.

    bufSize_ = (GLsizei) (bufSize_ + bufSize_  / 2);
    bufSize_ = jem::max ( bufSize_, (GLint) bufferSize );

    funcs.glBufferData ( GL_ARRAY_BUFFER, bufSize_,
                         nullptr, GL_DYNAMIC_DRAW );
  }

  // Write the data to the vertex buffer.

  funcs.glBufferSubData ( GL_ARRAY_BUFFER, (GLintptr) offsetSize,
                          (GLsizeiptr) dataSize, data );
  funcs.glBindBuffer    ( GL_ARRAY_BUFFER, 0 );

  checkGLError          ( gtx, JEM_FUNC );
}


JEM_END_PACKAGE( gl )

