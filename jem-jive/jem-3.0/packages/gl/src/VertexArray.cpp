
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
#include <jem/gl/GLFunctions2.h>
#include <jem/gl/GLFunctions3.h>
#include "VertexArray.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class VertexArray
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const byte  VertexArray::IDENT_[1] = { 42 };


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


VertexArray::VertexArray ( GLContext& gtx ) :

  gtx_   ( gtx ),
  funcs_ ( gtx.getFunctions<GLFuncs2Core>() )

{
  GLFuncs3Core*  funcs3 = gtx.findFunctions<GLFuncs3Core> ();

  count_ = 0;

  if ( funcs3 )
  {
    GLuint  id = 0;

    if ( ! gtx.findGLObject( id, IDENT_ ) )
    {
      id = gtx.newGLObject ( GLObject::VERTEX_ARRAY,
                             IDENT_, nullptr );
    }

    funcs3->glBindVertexArray ( id );
  }

  checkGLError ( gtx, JEM_FUNC );
}


VertexArray::~VertexArray ()
{
  GLFuncs3Core*  funcs3 = gtx_.findFunctions<GLFuncs3Core> ();

  while ( count_ )
  {
    funcs_.glDisableVertexAttribArray ( --count_ );
  }

  if ( funcs3 )
  {
    funcs3->glBindVertexArray ( 0 );
  }
}


//-----------------------------------------------------------------------
//   addAttrib
//-----------------------------------------------------------------------


GLuint VertexArray::addAttrib

  ( GLuint  vbo,
    GLint   size,
    GLenum  type )

{
  GLuint  index = count_++;

  funcs_.glBindBuffer              ( GL_ARRAY_BUFFER, vbo );
  funcs_.glEnableVertexAttribArray ( index );
  funcs_.glVertexAttribPointer     ( index, size, type,
                                     GL_FALSE, 0, (void*) 0 );
  funcs_.glBindBuffer              ( GL_ARRAY_BUFFER, 0 );

  return index;
}


JEM_END_PACKAGE( gl )

