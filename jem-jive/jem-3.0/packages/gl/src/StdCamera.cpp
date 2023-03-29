
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


#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Timer.h>
#include <jem/util/Properties.h>
#include <jem/gl/utilities.h>
#include <jem/gl/Box.h>
#include <jem/gl/Actions.h>
#include <jem/gl/DummyNode.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLException.h>
#include <jem/gl/GLFunctions3_3.h>
#include <jem/gl/GLObject.h>
#include <jem/gl/PropertyNames.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/ShaderProg.h>
#include <jem/gl/StdCamera.h>
#include "VertexArray.h"
#include "ShaderNames.h"
#include "ShaderSource.h"


JEM_DEFINE_CLASS( jem::gl::StdCamera );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class StdCamera::Data_
//=======================================================================


class StdCamera::Data_ : public Object
{
 public:

  static const GLfloat    QUAD_COORDS[18];

                          Data_         ();

  GLuint                  getScreenQuad

    ( GLContext&            gtx );

  GLuint                  getFrameBuffer0

    ( GLContext&            gtx );

  GLuint                  getFrameBuffer1

    ( GLContext&            gtx );

  GLuint                  getTexture0

    ( GLContext&            gtx );

  GLuint                  getTexture1

    ( GLContext&            gtx );


 private:

  void                    initBuffers_

    ( GLContext&            gtx );

  GLint                   getDepthFormat_

    ( GLContext&            gtx )           const;

  bool                    checkChanges_

    ( GLContext&            gtx )           const;


 private:

  Ref<GLObject>           frameBuf0_;
  Ref<GLObject>           frameBuf1_;
  Ref<GLObject>           renderBuf0_;
  Ref<GLObject>           texBuf0_;
  Ref<GLObject>           texBuf1_;

  GLint                   depthFormat_;
  Rect                    vport_;
  int                     gtxID_;

};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const GLfloat StdCamera::Data_::QUAD_COORDS[18] =

  { -1.0F, -1.0F, 0.0F,
     1.0F, -1.0F, 0.0F,
    -1.0F,  1.0F, 0.0F,
    -1.0F,  1.0F, 0.0F,
     1.0F, -1.0F, 0.0F,
     1.0F,  1.0F, 0.0F };


//-----------------------------------------------------------------------
//   constructor & deconstructor
//-----------------------------------------------------------------------


StdCamera::Data_::Data_ ()
{
  frameBuf0_  = newInstance<GLObject> ();
  frameBuf1_  = newInstance<GLObject> ();
  renderBuf0_ = newInstance<GLObject> ();
  texBuf0_    = newInstance<GLObject> ();
  texBuf1_    = newInstance<GLObject> ();
  gtxID_      = 0;
}


//-----------------------------------------------------------------------
//   getScreenQuad
//-----------------------------------------------------------------------


GLuint StdCamera::Data_::getScreenQuad ( GLContext& gtx )
{
  GLFuncs3_3Core&  funcs      = gtx.getFunctions<GLFuncs3_3Core> ();
  const size_t     bufferSize = sizeof(QUAD_COORDS);
  GLuint           vbo;

  if      ( ! gtx.findGLObject( vbo, QUAD_COORDS ) )
  {
    vbo = gtx.newGLObject ( GLObject::BUFFER, QUAD_COORDS, nullptr );
  }
  else if ( funcs.glIsBuffer( vbo ) != GL_TRUE )
  {
    throw GLException ( JEM_FUNC, "GLObject is not a buffer" );
  }

  funcs.glBindBuffer ( GL_ARRAY_BUFFER, vbo );
  funcs.glBufferData ( GL_ARRAY_BUFFER, bufferSize,
                       QUAD_COORDS, GL_STATIC_DRAW );

  return vbo;
}


//-----------------------------------------------------------------------
//   getFrameBuffer0
//-----------------------------------------------------------------------


GLuint StdCamera::Data_::getFrameBuffer0 ( GLContext& gtx )
{
  if ( ! checkChanges_( gtx ) )
  {
    initBuffers_ ( gtx );
  }

  return gtx.getGLObject ( frameBuf0_ );
}


//-----------------------------------------------------------------------
//   getFrameBuffer1
//-----------------------------------------------------------------------


GLuint StdCamera::Data_::getFrameBuffer1 ( GLContext& gtx )
{
  if ( ! checkChanges_( gtx ) )
  {
    initBuffers_ ( gtx );
  }

  return gtx.getGLObject ( frameBuf1_ );
}


//-----------------------------------------------------------------------
//   getTexture0
//-----------------------------------------------------------------------


GLuint StdCamera::Data_::getTexture0 ( GLContext& gtx )
{
  if ( ! checkChanges_( gtx ) )
  {
    initBuffers_ ( gtx );
  }

  return gtx.getGLObject ( texBuf0_ );
}


//-----------------------------------------------------------------------
//   getTexture1
//-----------------------------------------------------------------------


GLuint StdCamera::Data_::getTexture1 ( GLContext& gtx )
{
  if ( ! checkChanges_( gtx ) )
  {
    initBuffers_ ( gtx );
  }

  return gtx.getGLObject ( texBuf1_ );
}


//-----------------------------------------------------------------------
//   initBuffers_
//-----------------------------------------------------------------------


void StdCamera::Data_::initBuffers_ ( GLContext& gtx )
{
  GLFuncs3_3Core&  funcs = gtx.getFunctions<GLFuncs3_3Core> ();

  GLenum  drawBuffers[]  = { GL_COLOR_ATTACHMENT0 };
  GLint   depthFormat    = getDepthFormat_ ( gtx );
  Rect    vport          = gtx.getViewport ();

  GLuint  frameBuf0      = 0;
  GLuint  frameBuf1      = 0;
  GLuint  renderBuf0     = 0;
  GLuint  texBuf0        = 0;
  GLuint  texBuf1        = 0;


  if      ( ! gtx.findGLObject( frameBuf0, frameBuf0_ ) )
  {
    frameBuf0 =

      gtx.newGLObject ( GLObject::FRAME_BUFFER,
                        frameBuf0_, this );
  }
  else if ( funcs.glIsFramebuffer( frameBuf0 ) != GL_TRUE )
  {
    throw GLException ( JEM_FUNC,
                        "GLObject is not a frame buffer" );
  }

  if      ( ! gtx.findGLObject( frameBuf1, frameBuf1_ ) )
  {
    frameBuf1 =

      gtx.newGLObject ( GLObject::FRAME_BUFFER,
                        frameBuf1_, this );
  }
  else if ( funcs.glIsFramebuffer( frameBuf1 ) != GL_TRUE )
  {
    throw GLException ( JEM_FUNC,
                        "GLObject is not a frame buffer" );
  }

  if      ( ! gtx.findGLObject( renderBuf0, renderBuf0_ ) )
  {
    renderBuf0 =

      gtx.newGLObject ( GLObject::RENDER_BUFFER,
                        renderBuf0_, this );
  }
  else if ( funcs.glIsRenderbuffer( renderBuf0 ) != GL_TRUE )
  {
    throw GLException ( JEM_FUNC,
                        "GLObject is not a render buffer" );
  }

  if      ( ! gtx.findGLObject( texBuf0, texBuf0_ ) )
  {
    texBuf0 =

      gtx.newGLObject ( GLObject::TEXTURE,
                        texBuf0_, this );
  }
  else if ( funcs.glIsTexture( texBuf0 ) != GL_TRUE )
  {
    throw GLException ( JEM_FUNC, "GLObject is not a texture" );
  }

  if      ( ! gtx.findGLObject( texBuf1, texBuf1_ ) )
  {
    texBuf1 =

      gtx.newGLObject ( GLObject::TEXTURE,
                        texBuf1_, this );
  }
  else if ( funcs.glIsTexture( texBuf1 ) != GL_TRUE )
  {
    throw GLException ( JEM_FUNC, "GLObject is not a texture" );
  }

  // Create the first frame buffer for rendering separate scenes.

  funcs.glBindFramebuffer     ( GL_FRAMEBUFFER, frameBuf0 );
  funcs.glBindTexture         ( GL_TEXTURE_2D, texBuf0 );
  funcs.glTexImage            ( GL_TEXTURE_2D, 0, GL_RGBA,
                                vport.width, vport.height,
                                0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  funcs.glTexParameter        ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                                GL_NEAREST);
  funcs.glTexParameter        ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                                GL_NEAREST);
  funcs.glTexParameter        ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                                GL_CLAMP_TO_EDGE);
  funcs.glTexParameter        ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                                GL_CLAMP_TO_EDGE);

  funcs.glBindRenderbuffer    ( GL_RENDERBUFFER, renderBuf0 );
  funcs.glRenderbufferStorage ( GL_RENDERBUFFER, depthFormat,
                                vport.width, vport.height );
  funcs.
    glFramebufferRenderbuffer ( GL_FRAMEBUFFER,  GL_DEPTH_ATTACHMENT,
                                GL_RENDERBUFFER, renderBuf0 );
  funcs.glFramebufferTexture  ( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                texBuf0, 0 );
  funcs.glDrawBuffers         ( 1, drawBuffers );

  if ( funcs.glCheckFramebufferStatus( GL_FRAMEBUFFER ) !=
       GL_FRAMEBUFFER_COMPLETE )
  {
    throw GLException (
      JEM_FUNC,
      "incomplete OpenGL frame buffer (off-screen render buffer)"
    );
  }

  checkGLError ( gtx, JEM_FUNC );

  // Create the accumulation frame buffer.

  funcs.glBindFramebuffer     ( GL_FRAMEBUFFER, frameBuf1 );

  funcs.glBindTexture         ( GL_TEXTURE_2D, texBuf1 );
  funcs.glTexImage            ( GL_TEXTURE_2D, 0, GL_RGBA,
                                vport.width, vport.height,
                                0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  funcs.glTexParameter        ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                                GL_NEAREST);
  funcs.glTexParameter        ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                                GL_NEAREST);
  funcs.glTexParameter        ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                                GL_CLAMP_TO_EDGE);
  funcs.glTexParameter        ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                                GL_CLAMP_TO_EDGE);
  funcs.glFramebufferTexture  ( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                texBuf1, 0 );
  funcs.glDrawBuffers         ( 1, drawBuffers );

  if ( funcs.glCheckFramebufferStatus( GL_FRAMEBUFFER ) !=
       GL_FRAMEBUFFER_COMPLETE )
  {
    throw GLException (
      JEM_FUNC,
      "incomplete OpenGL frame buffer (accumulation buffer)"
    );
  }

  checkGLError ( gtx, JEM_FUNC );

  // Remember viewport used in the initialization

  vport_       = vport;
  depthFormat_ = depthFormat;
  gtxID_       = gtx.getID ();
}


//-----------------------------------------------------------------------
//   getDepthFormat_
//-----------------------------------------------------------------------


GLint StdCamera::Data_::getDepthFormat_ ( GLContext& gtx ) const
{
  GLFuncs3_3Core&  funcs       = gtx.getFunctions<GLFuncs3_3Core> ();
  GLint            depthFormat = GL_DEPTH24_STENCIL8;
  GLenum           attachment  = GL_DEPTH_ATTACHMENT;
  GLint            drawFboId   = 0;
  GLint            objectType  = 0;


  funcs.glGetParameter ( GL_DRAW_FRAMEBUFFER_BINDING, drawFboId );

  if ( drawFboId == 0 )
  {
    attachment = GL_DEPTH;
  }

  funcs.glGetFramebufferAttachmentParameter (
    GL_FRAMEBUFFER,
    attachment,
    GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE,
    objectType
  );

  if ( objectType == GL_RENDERBUFFER )
  {
    funcs.glGetRenderbufferParameter (
      GL_RENDERBUFFER,
      GL_RENDERBUFFER_INTERNAL_FORMAT,
      &depthFormat
    );
  }

  checkGLError ( gtx, JEM_FUNC );

  return depthFormat;
}


//-----------------------------------------------------------------------
//   checkChanges_
//-----------------------------------------------------------------------


bool StdCamera::Data_::checkChanges_

  ( GLContext& gtx ) const

{
  bool  result = false;

  if ( gtx.hasID( gtxID_ ) )
  {
    GLint  depthFormat = getDepthFormat_ ( gtx );
    Rect   vport       = gtx.getViewport ();

    result = ((vport_.x      == vport.x)      &&
              (vport_.y      == vport.y)      &&
              (vport_.width  == vport.width)  &&
              (vport_.height == vport.height) &&
              (depthFormat_  == depthFormat));
  }

  return result;
}


//=======================================================================
//   class StdCamera::FSAAScope_
//=======================================================================


class StdCamera::FSAAScope_
{
 public:

  explicit                FSAAScope_

    ( GLContext&            gtx );

                         ~FSAAScope_  ();

  void                    restore     ();


 private:

  GLFuncs3_3Core&         funcs_;
  GLboolean               blend_;
  GLint                   blendSrc_;
  GLint                   blendDst_;
  GLint                   frameBuf_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdCamera::FSAAScope_::FSAAScope_ ( GLContext& gtx ) :

  funcs_ ( gtx.getFunctions<GLFuncs3_3Core>() )

{
  blend_ = funcs_.glIsEnabled ( GL_BLEND );

  funcs_.glGetParameter ( GL_BLEND_SRC_ALPHA,     &blendSrc_ );
  funcs_.glGetParameter ( GL_BLEND_DST_ALPHA,     &blendDst_ );
  funcs_.glGetParameter ( GL_FRAMEBUFFER_BINDING, &frameBuf_ );
}


StdCamera::FSAAScope_::~FSAAScope_ ()
{
  restore ();
}


//-----------------------------------------------------------------------
//   restore()
//-----------------------------------------------------------------------


void StdCamera::FSAAScope_::restore ()
{
  if ( funcs_.glIsFramebuffer( (GLuint) frameBuf_ ) == GL_TRUE )
  {
    funcs_.glBindFramebuffer ( GL_FRAMEBUFFER, (GLuint) frameBuf_ );
  }
  else
  {
    funcs_.glBindFramebuffer ( GL_FRAMEBUFFER, 0 );
    funcs_.glDrawBuffer      ( GL_BACK );
  }

  if ( blend_ )
  {
    funcs_.glEnable  ( GL_BLEND );
  }
  else
  {
    funcs_.glDisable ( GL_BLEND );
  }

  funcs_.glBlendFunc   ( (GLenum) blendSrc_, (GLenum) blendDst_ );
  funcs_.glUseProgram  ( 0 );
  funcs_.glBindTexture ( GL_TEXTURE_2D, 0 );
}


//=======================================================================
//   class StdCamera
//=======================================================================

/*
  Implementation note: some methods of the StdCamera class hold a
  local reference to the child node. This is necessary because the
  child node may be deleted by itself during the method call.
*/

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int      StdCamera::MAX_QUALITY  = 8;

const GLfloat  StdCamera::JITTER_[58]  =

  {
    // 1 sample:

    0.0F, 0.0F,

    // 2 samples:

    0.25F, 0.75F, 0.75F, 0.25F,

    // 3 samples:

    0.5033922635F, 0.8317967229F, 0.7806016275F, 0.2504380877F,
    0.2261828938F, 0.4131553612F,

    // 4 samples:

    0.375F, 0.250F, 0.125F, 0.75F, 0.875F, 0.25F, 0.625F, 0.75F,

    // 5 samples:

    0.5F, 0.5F, 0.3F, 0.1F, 0.7F, 0.9F, 0.9F, 0.3F, 0.1F, 0.7F,

    // 6 samples:

    0.4646464646F, 0.4646464646F, 0.1313131313F, 0.7979797979F,
    0.5353535353F, 0.8686868686F, 0.8686868686F, 0.5353535353F,
    0.7979797979F, 0.1313131313F, 0.2020202020F, 0.2020202020F,

    // 8 samples:

    0.5625F, 0.4375F, 0.0625F, 0.9375F, 0.3125F, 0.6875F,
    0.6875F, 0.8125F, 0.8125F, 0.1875F, 0.9375F, 0.5625F,
    0.4375F, 0.0625F, 0.1875F, 0.3125F
  };

const int      StdCamera::JINDEX_[16] =

  {  0,  2,  // quality 1 (1 sample)
     2,  6,  // quality 2 (2 samples)
     6, 12,  // quality 3 (3 samples)
    12, 20,  // quality 4 (4 samples)
    20, 30,  // quality 5 (5 samples)
    30, 42,  // quality 6 (6 samples)
    42, 58,  // quality 7 (8 samples)
    42, 58   // quality 8 (8 samples)
  };


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdCamera::StdCamera

  ( const String&        name,
    const Ref<GfxNode>&  child ) :

    Super  ( name  ),
    child_ ( child )

{
  if ( ! child_ )
  {
    child_ = newInstance<DummyNode> ( myName_ );
  }

  child_ ->setParent   ( this );
  proj_   .setIdentity ();
  transfm_.setIdentity ();

  data_        = newInstance<Data_> ();
  zoom_        = 0.0;
  newProj_     = true;
  quality_     = 1;
  maxQuality_  = MAX_QUALITY;
  nextQuality_ = (double) quality_;
  lastLowered_ = Time::zero ();
}


StdCamera::~StdCamera ()
{
  child_->clearParent ();
}


//-----------------------------------------------------------------------
//   setChild
//-----------------------------------------------------------------------


void StdCamera::setChild ( const Ref<GfxNode>& child )
{
  child_->clearParent ();

  if ( ! child )
  {
    child_ = newInstance<DummyNode> ( myName_ );
  }
  else
  {
    child_ = child;
  }

  child_->setParent ( this );
}


//-----------------------------------------------------------------------
//   getChild
//-----------------------------------------------------------------------


Ref<GfxNode> StdCamera::getChild () const
{
  return child_;
}


//-----------------------------------------------------------------------
//   findNode
//-----------------------------------------------------------------------


GfxNode* StdCamera::findNode ( const String& name ) const
{
  if ( name == myName_ )
  {
    return const_cast<Self*> ( this );
  }
  else
  {
    return child_->findNode ( name );
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool StdCamera::takeAction

  ( const String&      action,
    const Properties&  params )

{
  return passAction ( child_, action, params );
}


//-----------------------------------------------------------------------
//   takeGLAction
//-----------------------------------------------------------------------


bool StdCamera::takeGLAction

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params )

{
  using util::Timer;

  GLContext&     gtx         = rtx.getGLContext ();

  bool           myResult    = false;
  bool           childResult = false;


  if ( action == (int) GLActions::INIT_VIEW )
  {
    int  xpos,  ypos;
    int  width, height;

    params.get ( xpos,   GLActionParams::VIEW_XPOS   );
    params.get ( ypos,   GLActionParams::VIEW_YPOS   );
    params.get ( width,  GLActionParams::VIEW_WIDTH  );
    params.get ( height, GLActionParams::VIEW_HEIGHT );

    vport_.x      = (GLint)   xpos;
    vport_.y      = (GLint)   ypos;
    vport_.width  = (GLsizei) width;
    vport_.height = (GLsizei) height;

    newProj_ = true;
    myResult = true;
  }

  if ( newProj_ )
  {
    proj_    = newProjection_ ();
    newProj_ = false;
  }

  if ( (action == (int) GLActions::REDRAW) && (quality_ > 1) &&
       gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    GLFuncs1Core*  funcs = gtx.findFunctions<GLFuncs1Core> ();

    if ( options_ & AUTO_QUALITY )
    {
      Timer  tm;

      tm.start    ();
      redrawFSAA_ ( rtx, params );
      finishGLOps ( gtx );
      tm.stop     ();

      adjustQuality_ ( tm.toDouble() );
    }
    else
    {
      redrawFSAA_ ( rtx, params );
    }

    if ( funcs && funcs->glGetError() )
    {
      // The error may very well be caused by the absence of the
      // accumulation buffer. Disable anti-aliasing.

      quality_  = 1;
      options_ &= ~AUTO_QUALITY;
    }

    return true;
  }

  gtx.setTransform ( proj_, GLContext::PROJECTION );
  gtx.setTransform ( transfm_ );

  if ( action == (int) GLActions::REDRAW ||
       action == (int) GLActions::REDRAW_FAST )
  {
    double  lod = 1.0;

    if ( action == (int) GLActions::REDRAW_FAST )
    {
      params.find ( lod, GLActionParams::DETAIL_LEVEL );
    }

    if ( lod > 0.95 )
    {
      if ( (options_ & AUTO_QUALITY) && quality_ == 1 )
      {
        Timer  tm;

        tm.start       ();
        passGLAction   ( child_, rtx, (int) GLActions::REDRAW, params );
        finishGLOps    ( gtx );
        tm.stop        ();

        adjustQuality_ ( tm.toDouble() );
      }
      else
      {
        passGLAction ( child_, rtx, (int)GLActions::REDRAW, params );
      }
    }
    else
    {
      passGLAction ( child_, rtx, action, params );
    }

    return true;
  }

  childResult = passGLAction ( child_, rtx, action, params );

  return (myResult || childResult);
}


//-----------------------------------------------------------------------
//   getBBox
//-----------------------------------------------------------------------


void StdCamera::getBBox ( Box& box ) const
{
  child_->getBBox ( box );
  box.transform   ( transfm_ );
}


//-----------------------------------------------------------------------
//   getBareBBox
//-----------------------------------------------------------------------


void StdCamera::getBareBBox ( Box& box ) const
{
  child_->getBBox ( box );
}


//-----------------------------------------------------------------------
//   getVisibility
//-----------------------------------------------------------------------


void StdCamera::getVisibility

  ( bool&             viz,
    Box&              vbox,
    const Transform&  proj ) const

{
  Transform  t;

  matmul ( t, proj, transfm_ );

  child_->getVisibility ( viz, vbox, t );

  vbox.transform ( transfm_ );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void StdCamera::configure ( const Properties& props )
{
  using util::findBool;

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    double      z;

    if ( myProps.find( z, PropNames::ZOOM ) )
    {
      setZoom ( z );
    }

    if ( myProps.find( quality_, PropNames::QUALITY, 1, MAX_QUALITY ) )
    {
      maxQuality_  = MAX_QUALITY;
      nextQuality_ = (double) quality_;
      lastLowered_ = Time::zero ();
    }

    findBool ( options_,  AUTO_QUALITY, myProps,
               PropNames::AUTO_QUALITY );
  }

  configChild ( child_, props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void StdCamera::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set ( PropNames::ZOOM,    (double) zoom_ );
  myProps.set ( PropNames::QUALITY, quality_ );

  setBool     ( myProps,
                PropNames::AUTO_QUALITY,
                options_,  AUTO_QUALITY );

  child_->getConfig ( props );
}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


void StdCamera::listProps ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set ( PropNames::ZOOM,
                "The (exponential) zoom factor" );
  myProps.set ( PropNames::QUALITY,
                "The quality of the rendered image. A quality "
                "larger than one will activate a full-screen "
                "anti-aliasing (oversampling) algorithm)" );
  myProps.set ( PropNames::AUTO_QUALITY,
                "Whether the quality should be adjusted "
                "dynamically depending on the available "
                "CPU/GPU resources" );

  child_->listProps ( props );
}


//-----------------------------------------------------------------------
//   getTransform
//-----------------------------------------------------------------------


Transform StdCamera::getTransform () const
{
  return transfm_;
}


//-----------------------------------------------------------------------
//   setTransform
//-----------------------------------------------------------------------


void StdCamera::setTransform ( const Transform& tr )
{
  transfm_ = tr;
}


//-----------------------------------------------------------------------
//   getViewport
//-----------------------------------------------------------------------


Rect StdCamera::getViewport () const
{
  return vport_;
}


//-----------------------------------------------------------------------
//   getProjection
//-----------------------------------------------------------------------


Transform StdCamera::getProjection () const
{
  if ( newProj_ )
  {
    return newProjection_ ();
  }
  else
  {
    return proj_;
  }
}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void StdCamera::reset ()
{
  transfm_.setIdentity ();

  zoom_    = 0.0;
  newProj_ = true;
}


//-----------------------------------------------------------------------
//   translateObj
//-----------------------------------------------------------------------


void StdCamera::translateObj

  ( GfxNode&  node,
    GLdouble  dx,
    GLdouble  dy,
    GLdouble  dz ) const

{}


//-----------------------------------------------------------------------
//   setZoom
//-----------------------------------------------------------------------


GLdouble StdCamera::setZoom ( GLdouble z )
{
  if ( ! isTiny( z - zoom_ ) )
  {
    zoom_    = z;
    newProj_ = true;
  }

  return zoom_;
}


//-----------------------------------------------------------------------
//   getZoom
//-----------------------------------------------------------------------


GLdouble StdCamera::getZoom () const
{
  return zoom_;
}


//-----------------------------------------------------------------------
//   setQuality
//-----------------------------------------------------------------------


int StdCamera::setQuality ( int level )
{
  quality_     = clamped ( level, 1, MAX_QUALITY );
  maxQuality_  = MAX_QUALITY;
  nextQuality_ = (double) quality_;
  lastLowered_ = Time::zero ();

  return quality_;
}


//-----------------------------------------------------------------------
//   getQuality
//-----------------------------------------------------------------------


int StdCamera::getQuality () const noexcept
{
  return quality_;
}


//-----------------------------------------------------------------------
//   setOption
//-----------------------------------------------------------------------


void StdCamera::setOption

  ( Option  option,
    bool    yesno )

{
  options_.set ( option, yesno );
}


//-----------------------------------------------------------------------
//   setOptions
//-----------------------------------------------------------------------


void StdCamera::setOptions ( Options options )
{
  options_ = options;
}


//-----------------------------------------------------------------------
//   getOptions
//-----------------------------------------------------------------------


StdCamera::Options StdCamera::getOptions () const noexcept
{
  return options_;
}


//-----------------------------------------------------------------------
//   redrawFSAA_
//-----------------------------------------------------------------------


void StdCamera::redrawFSAA_

  ( RenderContext&     rtx,
    const Properties&  params )

{
  GLContext&       gtx   = rtx.getGLContext                 ();
  GLFuncs3_3Core&  funcs = gtx.getFunctions<GLFuncs3_3Core> ();
  Ref<GfxNode>     child = child_;
  Transform        p     = proj_;


  FSAAScope_       scope ( gtx );

  GLuint           frameBuf0;
  GLuint           frameBuf1;
  GLuint           tex0;
  GLuint           tex1;
  GLdouble         dx, dx0;
  GLdouble         dy, dy0;
  int              i, j, k;


  frameBuf0 = data_->getFrameBuffer0 ( gtx );
  frameBuf1 = data_->getFrameBuffer1 ( gtx );
  tex0      = data_->getTexture0     ( gtx );
  tex1      = data_->getTexture1     ( gtx );

  k  = 2 * (quality_ - 1);

  i  = JINDEX_[k + 0];
  j  = JINDEX_[k + 1];

  dx = 0.0;
  dy = 0.0;

  for ( k = i; k < j; k += 2 )
  {
    int  level = (k - i) / 2;

    funcs.glBindFramebuffer ( GL_FRAMEBUFFER, frameBuf0 );
    funcs.glClear           ( GL_COLOR_BUFFER_BIT |
                              GL_DEPTH_BUFFER_BIT );

    // Jitter the view.

    dx0 = dx;
    dy0 = dy;
    dx  = JITTER_[k + 0];
    dy  = JITTER_[k + 1];

    jitter ( p, (dx - dx0), (dy - dy0) );

    gtx.resetTransforms ();
    gtx.setTransform    ( p, GLContext::PROJECTION );
    gtx.setTransform    ( transfm_ );

    params.set ( GLActionParams::XJITTER, dx );
    params.set ( GLActionParams::YJITTER, dy );

    child->takeGLAction ( rtx, (int) GLActions::REDRAW, params );

    funcs.glBindFramebuffer ( GL_FRAMEBUFFER, frameBuf1 );
    funcs.glEnable          ( GL_BLEND );
    funcs.glBlendFunc       ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

    if ( k == i )
    {
      funcs.glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    }

    drawScreenQuad_ ( gtx, tex0,
                      1.0F / (GLfloat) (level + 1), level );
  }

  scope.restore   ();
  funcs.glClear   ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

  checkGLError    ( gtx, JEM_FUNC );
  drawScreenQuad_ ( gtx, tex1, 1.0F, 0 );

  params.erase    ( GLActionParams::XJITTER );
  params.erase    ( GLActionParams::YJITTER );
}


//-----------------------------------------------------------------------
//   adjustQuality_()
//-----------------------------------------------------------------------


void StdCamera::adjustQuality_ ( double dt )
{
  double  target = 0.1;


  // The target draw time is set lower if the current quality
  // level is higher.

  target *= (1.0 - 0.025 * (double) (quality_ - 1));

  if      ( dt < target )
  {
    // Ramp up the quality level.

    nextQuality_ += (target - dt) * nextQuality_;

    if ( dt          < (0.5 * target) &&
         maxQuality_ <  MAX_QUALITY   &&
         maxQuality_ <= quality_ )
    {
      Time  dt = Time::now() - lastLowered_;

      if ( dt.toDouble() > 20.0 )
      {
        maxQuality_++;
      }
    }

    if ( nextQuality_ >= (double) (quality_ + 1) )
    {
      quality_     = jem::min ( quality_ + 1, maxQuality_ );
      nextQuality_ = (double)   quality_;
    }
  }
  else if ( dt > 4.0 )
  {
    // Drop directly to the lowest quality level.

    quality_     = 1;
    maxQuality_  = jem::max ( 1, maxQuality_ - 4 );
    nextQuality_ = (double) quality_;
    lastLowered_ = Time::now ();
  }
  else if ( dt > target )
  {
    // Ramp down the quality level.

    nextQuality_ -= (dt - target);
    nextQuality_ -= 0.2 * nextQuality_;

    if ( dt > (2.0 * target) && maxQuality_ > 1 )
    {
      maxQuality_  = maxQuality_ - 1;
      lastLowered_ = Time::now ();

      if ( quality_ > maxQuality_ )
      {
        quality_     = maxQuality_;
        nextQuality_ = (double) quality_;
      }
    }

    if ( nextQuality_ <= (double) (quality_ - 1) )
    {
      quality_     = jem::max ( quality_ - 1, 1 );
      nextQuality_ = (double)   quality_;
    }
  }
}


//-----------------------------------------------------------------------
//   drawScreenQuad_()
//-----------------------------------------------------------------------


void StdCamera::drawScreenQuad_

  ( GLContext&  gtx,
    GLuint      texID,
    GLfloat     accumFactor,
    int         level )

{
  GLFuncs3_3Core&  funcs  = gtx.getFunctions<GLFuncs3_3Core> ();

  ShaderProg&      shader =

    ShaderSource::getProgram ( gtx, ShaderNames::ACCUMULATION );

  GLuint           vxBuf  = data_->getScreenQuad ( gtx );

  VertexArray      vao    ( gtx );

  GLint            ilocAccFactor;
  GLint            ilocTex;


  if ( ! shader.bind( gtx ) )
  {
    return;
  }

  ilocAccFactor = shader.getUniformIdx ( gtx, ShaderUnis::ACC_FACTOR );
  ilocTex       = shader.getUniformIdx ( gtx, ShaderUnis::TEXTURE_2D );

  funcs.glUniform       ( ilocAccFactor, accumFactor );
  funcs.glActiveTexture ( GL_TEXTURE0 );
  funcs.glBindTexture   ( GL_TEXTURE_2D, texID );
  funcs.glUniform       ( ilocTex, 0 );
  vao  .addAttrib       ( vxBuf, 3, GL_FLOAT );

  funcs.glDrawArrays (
    GL_TRIANGLES, 0,
    (GLsizei) (sizeof(Data_::QUAD_COORDS) /
               sizeof(Data_::QUAD_COORDS[0]) / 3)
  );

  checkGLError ( gtx, JEM_FUNC );
}


JEM_END_PACKAGE( gl )
