
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
#include <jem/gl/GLFunctions3.h>
#include <jem/gl/VertexArrayObject.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class VertexArrayObject
//=======================================================================

//-----------------------------------------------------------------------
//  constructor & destructor
//-----------------------------------------------------------------------


VertexArrayObject::VertexArrayObject ()
{
  gtxID_ = 0;
  vaoID_ = 0;
}


VertexArrayObject::~VertexArrayObject ()
{}


//-----------------------------------------------------------------------
//  bind
//-----------------------------------------------------------------------


bool VertexArrayObject::bind ( GLContext& gtx )
{
  GLFuncs3Core*  funcs = gtx.findFunctions<GLFuncs3Core> ();

  if ( ! funcs )
  {
    return false;
  }

  funcs->glBindVertexArray ( getArrayID( gtx ) );

  return true;
}


//-----------------------------------------------------------------------
//   unbind
//-----------------------------------------------------------------------


void VertexArrayObject::unbind ( GLContext& gtx )
{
  GLFuncs3Core*  funcs = gtx.findFunctions<GLFuncs3Core> ();

  if ( funcs )
  {
    funcs->glBindVertexArray ( 0 );
  }
}


//-----------------------------------------------------------------------
//  unload
//-----------------------------------------------------------------------


void VertexArrayObject::unload ( GLContext& gtx )
{
  gtx.freeGLObject ( this );

  gtxID_ = 0;
  vaoID_ = 0;
}


//-----------------------------------------------------------------------
//   isCreated
//-----------------------------------------------------------------------


bool VertexArrayObject::isCreated ( GLContext& gtx ) const
{
  if ( gtx.hasID( gtxID_ ) )
  {
    return true;
  }
  else
  {
    GLuint  id = 0;

    return gtx.findGLObject ( id, this );
  }
}


//-----------------------------------------------------------------------
//   getArrayID
//-----------------------------------------------------------------------


GLuint VertexArrayObject::getArrayID ( GLContext& gtx )
{
  GLuint  id = 0;

  if ( gtx.hasID( gtxID_ ) )
  {
    id = vaoID_;
  }
  else
  {
    if ( ! gtx.findGLObject( id, this ) )
    {
      id = gtx.newGLObject ( GLObject::VERTEX_ARRAY, this, this );
    }

    gtxID_ = gtx.getID ();
    vaoID_ = id;
  }

  return id;
}


//-----------------------------------------------------------------------
//   enableAttrib
//-----------------------------------------------------------------------


void VertexArrayObject::enableAttrib

  ( GLContext&  gtx,
    GLuint      vbo,
    GLuint      index,
    GLint       size,
    GLenum      type )

{
  GLFuncs3Core&  funcs = gtx.getFunctions<GLFuncs3Core> ();

  funcs.glBindBuffer              ( GL_ARRAY_BUFFER, vbo );
  funcs.glBindVertexArray         ( getArrayID( gtx ) );
  funcs.glEnableVertexAttribArray ( index );
  funcs.glVertexAttribPointer     ( index, size, type,
                                    GL_FALSE, 0, (void*) 0 );
  funcs.glBindVertexArray         ( 0 );
  funcs.glBindBuffer              ( GL_ARRAY_BUFFER, 0 );
  checkGLError                    ( gtx, JEM_FUNC );
}


JEM_END_PACKAGE( gl )
