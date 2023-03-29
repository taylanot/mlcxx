
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

#ifndef JEM_GL_RENDER_H
#define JEM_GL_RENDER_H

#include <jem/gl/utilities.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>


JEM_BEGIN_PACKAGE( gl )


class Box;


//-----------------------------------------------------------------------
//   class AttribScope
//-----------------------------------------------------------------------


class AttribScope
{
 public:

  explicit inline         AttribScope

    ( GLContext&            gtx );

  inline                  AttribScope

    ( GLContext&            gtx,
      int                   mask );

  inline                 ~AttribScope  ();

  inline void             push

    ( int                   mask );

  inline void             pop          ();


 private:

  GLContext&              gtx_;
  idx_t                   depth_;

};


//-----------------------------------------------------------------------
//   class EnableScope
//-----------------------------------------------------------------------


class EnableScope
{
 public:

  explicit inline         EnableScope

    ( GLContext&            gtx,
      GLenum                cap );

  inline                  EnableScope

    ( GLContext&            gtx,
      GLenum                cap,
      bool                  choice );

  inline                 ~EnableScope ();


 private:

  GLFuncs1Core*           funcs_;
  GLenum                  cap_;
  GLboolean               state_;

};


//-----------------------------------------------------------------------
//   class DisableScope
//-----------------------------------------------------------------------


class DisableScope
{
 public:

  inline                  DisableScope

    ( GLContext&            gtx,
      GLenum                cap );

  inline                  DisableScope

    ( GLContext&            gtx,
      GLenum                cap,
      bool                  choice );

  inline                 ~DisableScope ();


 private:

  GLFuncs1Core*           funcs_;
  GLenum                  cap_;
  GLboolean               state_;

};


//-----------------------------------------------------------------------
//   class DrawBufferScope
//-----------------------------------------------------------------------


class DrawBufferScope
{
 public:

  explicit inline         DrawBufferScope

    ( GLContext&            gtx,
      GLenum                mode );

  inline                 ~DrawBufferScope ();


 private:

  GLFuncs1Core*           funcs_;
  GLint                   mode_;

};


//-----------------------------------------------------------------------
//   class CanvasScope
//-----------------------------------------------------------------------


class CanvasScope
{
 public:

  explicit inline         CanvasScope

    ( GLContext&            gtx );

  inline                  CanvasScope

    ( GLContext&            gtx,
      GLfloat               clip0,
      GLfloat               clip1 );

                         ~CanvasScope   ();


 public:

  GLsizei                 width;
  GLsizei                 height;
  GLfloat                 nearClip;
  GLfloat                 farClip;


 private:

  void                    init_

    ( GLfloat               clip0,
      GLfloat               clip1 );


 private:

  GLContext&              gtx_;
  bool                    lighting_;
  GLenum                  depthFunc_;

};


//-----------------------------------------------------------------------
//   utility functions
//-----------------------------------------------------------------------


void                      renderRect

  ( GLContext&              gtx,
    const Rect&             rect );

void                      renderRect

  ( GLContext&              gtx,
    GLfloat                 x0,
    GLfloat                 y0,
    GLfloat                 x1,
    GLfloat                 y1 );

void                      renderDisk

  ( GLContext&              gtx,
    GLfloat                 radius  = 1.0F,
    int                     quality = 1 );

void                      renderSlice

  ( GLContext&              gtx,
    GLfloat                 alpha,
    GLfloat                 beta,
    GLfloat                 radius  = 1.0F,
    int                     quality = 1 );

void                      renderArrow3D

  ( GLContext&              gtx,
    const Color&            color,
    GLfloat                 size    = 0.2F );

void                      renderShadow

  ( GLContext&              gtx,
    GLfloat                 x0,
    GLfloat                 y0,
    GLfloat                 x1,
    GLfloat                 y1,
    GLfloat                 w );

void                      renderBoxEdges

  ( GLContext&              gtx,
    const Box&              box );

void                      renderBoxFaces

  ( GLContext&              gtx,
    const Box&              box );




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class AttribScope
//=======================================================================


inline AttribScope::AttribScope ( GLContext& gtx ) :

  gtx_   ( gtx ),
  depth_ ( 0 )

{}


inline AttribScope::AttribScope

  ( GLContext&  gtx,
    int         mask ) :

    gtx_   ( gtx ),
    depth_ ( 1 )

{
  gtx.pushAttribs ( mask );
}


inline AttribScope::~AttribScope ()
{
  for ( ; depth_ > 0; depth_-- )
  {
    gtx_.popAttribs ();
  }
}


inline void AttribScope::push ( int mask )
{
  gtx_.pushAttribs ( mask );

  depth_++;
}


inline void AttribScope::pop ()
{
  if ( depth_ > 0 )
  {
    gtx_.popAttribs ();

    depth_--;
  }
}


//=======================================================================
//   class EnableScope
//=======================================================================


inline EnableScope::EnableScope

  ( GLContext&  gtx,
    GLenum      cap ) :

    funcs_ ( gtx.findFunctions<GLFuncs1Core>() ),
    cap_   ( cap )

{
  if ( funcs_ )
  {
    state_ = funcs_->glIsEnabled ( cap );

    if ( state_ == GL_FALSE )
    {
      funcs_->glEnable ( cap_ );
    }

    checkGLError ( gtx, JEM_FUNC );
  }
}


inline EnableScope::EnableScope

  ( GLContext&  gtx,
    GLenum      cap,
    bool        choice ) :

    funcs_ ( gtx.findFunctions<GLFuncs1Core>() ),
    cap_   ( cap )

{
  if ( funcs_ && choice )
  {
    state_ = funcs_->glIsEnabled ( cap );

    if ( state_ == GL_FALSE )
    {
      funcs_->glEnable ( cap );
    }

    checkGLError ( gtx, JEM_FUNC );
  }
  else
  {
    state_ = GL_TRUE;
  }
}


inline EnableScope::~EnableScope ()
{
  if ( funcs_ && (state_ == GL_FALSE) )
  {
    funcs_->glDisable ( cap_ );
  }
}


//=======================================================================
//   class DisableScope
//=======================================================================


inline DisableScope::DisableScope

  ( GLContext&  gtx,
    GLenum      cap ) :

    funcs_ ( gtx.findFunctions<GLFuncs1Core>() ),
    cap_   ( cap )

{
  if ( funcs_ )
  {
    state_ = funcs_->glIsEnabled ( cap );

    if ( state_ == GL_TRUE )
    {
      funcs_->glDisable ( cap_ );
    }

    checkGLError ( gtx, JEM_FUNC );
  }
  else
  {
    state_ = GL_FALSE;
  }
}


inline DisableScope::DisableScope

  ( GLContext&  gtx,
    GLenum      cap,
    bool        choice ) :

    funcs_ ( gtx.findFunctions<GLFuncs1Core>() ),
    cap_   ( cap )

{
  if ( funcs_ && choice )
  {
    state_ = funcs_->glIsEnabled ( cap );

    if ( state_ == GL_TRUE )
    {
      funcs_->glDisable ( cap );
    }

    checkGLError ( gtx, JEM_FUNC );
  }
  else
  {
    state_ = GL_FALSE;
  }
}


inline DisableScope::~DisableScope ()
{
  if ( funcs_ && (state_ == GL_TRUE) )
  {
    funcs_->glEnable ( cap_ );
  }
}


//=======================================================================
//   class DrawBufferScope
//=======================================================================


inline DrawBufferScope::DrawBufferScope

  ( GLContext&  gtx,
    GLenum      mode ) :

    funcs_ ( gtx.findFunctions<GLFuncs1Core>() )

{
  if ( funcs_ )
  {
    funcs_->glGetParameter ( GL_DRAW_BUFFER, mode_ );
    funcs_->glDrawBuffer   ( mode );
  }
}


inline DrawBufferScope::~DrawBufferScope ()
{
  if ( funcs_ )
  {
    funcs_->glDrawBuffer ( (GLenum) mode_ );
  }
}


//=======================================================================
//   class CanvasScope
//=======================================================================


inline CanvasScope::CanvasScope ( GLContext& gtx ) :

  gtx_ ( gtx )

{
  init_ ( -1.0, 40.0 );
}


inline CanvasScope::CanvasScope

  ( GLContext&  gtx,
    GLfloat     clip0,
    GLfloat     clip1 ) :

    gtx_ ( gtx )

{
  init_ ( clip0, clip1 );
}


JEM_END_PACKAGE( gl )

#endif
