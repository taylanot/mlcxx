
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
#include <jem/gl/VertexBufferObject.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class VertexBufferObject
//=======================================================================

//-----------------------------------------------------------------------
//  static data
//-----------------------------------------------------------------------


const double  VertexBufferObject::GROWTH_FACTOR_ = 1.3;


//-----------------------------------------------------------------------
//  constructor & destructor
//-----------------------------------------------------------------------


VertexBufferObject::VertexBufferObject ()
{
  size_  = 0;
  gtxID_ = 0;
  vboID_ = 0;
}


VertexBufferObject::~VertexBufferObject ()
{}


//-----------------------------------------------------------------------
//  bind
//-----------------------------------------------------------------------


bool VertexBufferObject::bind ( GLContext& gtx )
{
  GLFuncs2Core*  funcs = gtx.findFunctions<GLFuncs2Core> ();

  if ( ! funcs )
  {
    return false;
  }

  funcs->glBindBuffer ( GL_ARRAY_BUFFER, getBufferID( gtx ) );

  return true;
}


//-----------------------------------------------------------------------
//   unbind
//-----------------------------------------------------------------------


void VertexBufferObject::unbind ( GLContext& gtx )
{
  GLFuncs2Core*  funcs = gtx.findFunctions<GLFuncs2Core> ();

  if ( funcs )
  {
    funcs->glBindBuffer ( GL_ARRAY_BUFFER, 0 );
  }
}


//-----------------------------------------------------------------------
//  update
//-----------------------------------------------------------------------


void VertexBufferObject::update ( GLContext& gtx )
{
  if ( size_ > 0 )
  {
    update_ ( gtx, getBufferID( gtx ) );
  }
}


//-----------------------------------------------------------------------
//  unload
//-----------------------------------------------------------------------


void VertexBufferObject::unload ( GLContext& gtx )
{
  gtx.freeGLObject ( this );

  gtxID_ = 0;
  vboID_ = 0;
}


//-----------------------------------------------------------------------
//   getBufferID
//-----------------------------------------------------------------------


GLuint VertexBufferObject::getBufferID ( GLContext& gtx )
{
  GLuint  id = 0;

  if ( gtx.hasID( gtxID_ ) )
  {
    id = vboID_;
  }
  else
  {
    if ( ! gtx.findGLObject( id, this ) )
    {
      id = gtx.newGLObject ( GLObject::BUFFER, this, this );
    }

    update_ ( gtx, id );

    gtxID_ = gtx.getID ();
    vboID_ = id;
  }

  return id;
}


//-----------------------------------------------------------------------
//  size
//-----------------------------------------------------------------------


idx_t VertexBufferObject::size () const noexcept
{
  return size_;
}


//-----------------------------------------------------------------------
//  resize
//-----------------------------------------------------------------------


void VertexBufferObject::resize ( idx_t size )
{
  JEM_PRECHECK ( size >= 0 );

  if ( size > data_.size() )
  {
    idx_t  cap = (idx_t) (GROWTH_FACTOR_ * (double) data_.size());

    cap = jem::max ( size, cap );

    data_.resize ( cap );
  }

  size_ = size;
}


//-----------------------------------------------------------------------
//  getData
//-----------------------------------------------------------------------


GLfloatVector VertexBufferObject::getData () const
{
  JEM_PRECHECK ( size_ <= data_.size() );

  if ( size_ == data_.size() )
  {
    return data_;
  }
  else
  {
    return data_[slice(BEGIN,size_)];
  }
}


//-----------------------------------------------------------------------
//  update_
//-----------------------------------------------------------------------


void VertexBufferObject::update_

  ( GLContext&  gtx,
    GLuint      idx )

{
  GLFuncs2Core&  funcs   = gtx.getFunctions<GLFuncs2Core> ();

  const GLint    datSize = (GLint) (sizeof(GLfloat) * (size_t) size_);
  GLint          bufSize = 0;


  if ( (idx_t) ((size_t) datSize / sizeof(GLfloat)) != size_ )
  {
    throw GLException (
      JEM_FUNC,
      "vertex buffer size too large; integer overflow"
    );
  }

  funcs.glBindBuffer ( GL_ARRAY_BUFFER, idx );

  funcs.glGetBufferParameter ( GL_ARRAY_BUFFER,
                               GL_BUFFER_SIZE,
                               &bufSize );

  if ( bufSize < datSize )
  {
    GLsizeiptr  newSize =

      (GLsizeiptr) (sizeof(GLfloat) * (size_t) data_.size());

    if ( (idx_t) ((size_t) newSize / sizeof(GLfloat)) != data_.size() )
    {
      throw GLException (
        JEM_FUNC,
        "vertex buffer size too large; integer overflow"
      );
    }

    funcs.glBufferData ( GL_ARRAY_BUFFER, newSize,
                         nullptr, GL_DYNAMIC_DRAW );
  }

  funcs.glBufferSubData ( GL_ARRAY_BUFFER, 0, (GLsizeiptr) datSize,
                          data_.addr() );
  funcs.glBindBuffer    ( GL_ARRAY_BUFFER, 0 );

  checkGLError          ( gtx, JEM_FUNC );
}


JEM_END_PACKAGE( gl )
