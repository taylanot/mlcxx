
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


#include <jem/base/Error.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLException.h>
#include <jem/gl/GLFunctions3.h>
#include <jem/gl/GLObject.h>
#include "GLObjectManager.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GLObjectManager::Item_
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline GLObjectManager::Item_::Item_ ()
{
  next  = nullptr;
  addr  = nullptr;
  type  = GLObject::NONE;
  objID = 0;
  owned = false;
}


//=======================================================================
//   class GLObjectManager
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


GLObjectManager::GLObjectManager ()
{
  head_ = nullptr;
  free_ = nullptr;
}


GLObjectManager::~GLObjectManager ()
{
  // The destructor does *not* free the OpenGL objects as it may
  // be executed outside the correct OpenGL context. The objects
  // will be destroyed anyway when the OpenGL context is destroyed.

  while ( head_ )
  {
    Item_*  next = head_->next;

    delete head_;

    head_ = next;
  }

  while ( free_ )
  {
    Item_*  next = free_->next;

    delete free_;

    free_ = next;
  }
}


//-----------------------------------------------------------------------
//   newObject
//-----------------------------------------------------------------------


GLuint GLObjectManager::newObject

  ( GLContext&          gtx,
    int                 type,
    const void*         addr,
    const Ref<Object>&  owner )

{
  Item_*  item = nullptr;
  GLuint  oid  = 0;


  freeObject ( gtx, addr );

  if ( free_ )
  {
    item  = free_;
    free_ = item->next;
  }
  else
  {
    item = new Item_ ();
  }

  try
  {
    oid = newObject_ ( gtx, type );
  }
  catch ( ... )
  {
    if ( free_ )
    {
      item->next = free_;
      free_      = item;
    }
    else
    {
      delete item;
    }

    throw;
  }

  item->next  =  head_;
  item->addr  =  addr;
  item->type  =  type;
  item->objID =  oid;
  item->owner =  owner;
  item->owned = (owner != nullptr);

  head_ = item;

  return oid;
}


//-----------------------------------------------------------------------
//   findObject
//-----------------------------------------------------------------------


bool GLObjectManager::findObject

  ( GLuint&      oid,
    const void*  addr )

{
  JEM_PRECHECK ( addr );

  Item_*  item = nullptr;

  if ( head_ )
  {
    if ( head_->addr == addr )
    {
      item = head_;
    }
    else
    {
      Item_*  prev = head_;

      while ( prev->next )
      {
        if ( prev->next->addr == addr )
        {
          // The item has been found; move it to the head of the list.

          item       = prev->next;
          prev->next = item->next;
          item->next = head_;
          head_      = item;

          break;
        }

        prev = prev->next;
      }
    }
  }

  if ( item )
  {
    if ( ! item->owned || (item->owner.get() != nullptr) )
    {
      oid = item->objID;

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   freeObject
//-----------------------------------------------------------------------


void GLObjectManager::freeObject

  ( GLContext&   gtx,
    const void*  addr )

{
  JEM_PRECHECK ( addr );

  Item_*  item = head_;
  Item_*  prev = nullptr;

  while ( item )
  {
    Item_*  next = item->next;

    // Delete the specified object and collect garbage.

    if ( (item->addr == addr) ||
         (item->owned && (item->owner.get() == nullptr)) )
    {
      int     type = item->type;
      GLuint  oid  = item->objID;

      item->type  = GLObject::NONE;
      item->objID = 0;
      item->owner = nullptr;
      item->owned = false;

      // Unlink from the list and add to the free list.

      if ( prev )
      {
        prev->next = item->next;
      }
      else
      {
        head_ = item->next;
      }

      item->next = free_;
      free_      = item;

      // Free the underlying OpenGL object.

      freeObject_ ( gtx, type, oid );

      break;
    }
    else
    {
      prev = item;
    }

    item = next;
  }
}


//-----------------------------------------------------------------------
//   freeAllObjects
//-----------------------------------------------------------------------


void GLObjectManager::freeAllObjects ( GLContext& gtx )
{
  while ( head_ )
  {
    Item_*  next = head_->next;

    // Delete the specified object and collect garbage.

    int     type = head_->type;
    GLuint  oid  = head_->objID;

    head_->type  = GLObject::NONE;
    head_->objID = 0;
    head_->owner = nullptr;
    head_->owned = false;

    // Unlink from the list and add to the free list.

    head_->next = free_;
    free_       = head_;

    // Free the underlying OpenGL object.

    freeObject_ ( gtx, type, oid );

    head_ = next;
  }
}


//-----------------------------------------------------------------------
//   newObject_
//-----------------------------------------------------------------------


GLuint GLObjectManager::newObject_

  ( GLContext&  gtx,
    int         type )

{
  GLFuncs1Core&  fn1  = gtx.getFunctions<GLFuncs1Core> ();
  const char*    kind = "";
  GLenum         err  = GL_NO_ERROR;
  GLuint         oid  = 0;


  fn1.glGetError ();

  switch ( type )
  {
  case GLObject::NONE:

    break;

  case GLObject::BUFFER:

    gtx.getFunctions<GLFuncs2Core>().glGenBuffers ( 1, &oid );

    kind = "buffer";

    break;

  case GLObject::FRAME_BUFFER:

    gtx.getFunctions<GLFuncs3Core>().glGenFramebuffers ( 1, &oid );

    kind = "frame buffer";

    break;

  case GLObject::PROGRAM:

    oid = gtx.getFunctions<GLFuncs2Core>().glCreateProgram ();

    kind = "shader program";

    break;

  case GLObject::RENDER_BUFFER:

    gtx.getFunctions<GLFuncs3Core>().glGenRenderbuffers ( 1, &oid );

    kind = "render buffer";

    break;

  case GLObject::TEXTURE:

    gtx.getFunctions<GLFuncs1_1Core>().glGenTextures ( 1, &oid );

    kind = "texture";

    break;

  case GLObject::VERTEX_ARRAY:

    gtx.getFunctions<GLFuncs3Core>().glGenVertexArrays ( 1, &oid );

    kind = "vertex array";

    break;

  default:

    throw Error (
      JEM_FUNC,
      "invalid type identifier: " + String ( type )
    );
  }

  if ( (err = fn1.glGetError()) != GL_NO_ERROR )
  {
    throw GLException (
      JEM_FUNC,
      "failed to create an OpenGL " + String( kind ),
      gtx.errorString ( err )
    );
  }

  return oid;
}


//-----------------------------------------------------------------------
//   freeObject_
//-----------------------------------------------------------------------


void GLObjectManager::freeObject_

  ( GLContext&  gtx,
    int         type,
    GLuint      oid )

{
  switch ( type )
  {
  case GLObject::NONE:

    break;

  case GLObject::BUFFER:

    gtx.getFunctions<GLFuncs2Core>()
         .glDeleteBuffers ( 1, &oid );

    break;

  case GLObject::FRAME_BUFFER:

    gtx.getFunctions<GLFuncs3Core>()
         .glDeleteFramebuffers ( 1, &oid );

    break;

  case GLObject::PROGRAM:

    gtx.getFunctions<GLFuncs2Core>()
         .glDeleteProgram ( oid );

    break;

  case GLObject::RENDER_BUFFER:

    gtx.getFunctions<GLFuncs3Core>()
         .glDeleteRenderbuffers ( 1, &oid );

    break;

  case GLObject::TEXTURE:

    gtx.getFunctions<GLFuncs1_1Core>()
         .glDeleteTextures ( 1, &oid );

    break;

  case GLObject::VERTEX_ARRAY:

    gtx.getFunctions<GLFuncs3Core>()
         .glDeleteVertexArrays ( 1, &oid );

    break;
  }
}


JEM_END_PACKAGE( gl )
