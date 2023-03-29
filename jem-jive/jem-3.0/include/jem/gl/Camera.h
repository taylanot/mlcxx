
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

#ifndef JEM_GL_CAMERA_H
#define JEM_GL_CAMERA_H

#include <jem/gl/Rect.h>
#include <jem/gl/Transform.h>
#include <jem/gl/GfxNode.h>


JEM_BEGIN_PACKAGE( gl )


class RenderContext;


//-----------------------------------------------------------------------
//   class Camera
//-----------------------------------------------------------------------


class Camera : public GfxNode
{
 public:

  JEM_DECLARE_CLASS       ( Camera, GfxNode );


  explicit inline           Camera

    ( const String&           name  = "" );

  virtual bool              isOrtho         () const;

  virtual void              reset           ();

  virtual void              lookAtBox

    ( const Box&              box )                  = 0;

  void                      lookAtObj

    ( const GfxNode&          obj );

  void                      lookAtRect

    ( RenderContext&          rtx,
      const Rect&             rect );

  virtual void              getVisibleBox

    ( Box&                    box )            const = 0;

  virtual void              translateObj

    ( GfxNode&                node,
      GLdouble                dx,
      GLdouble                dy,
      GLdouble                dz )             const = 0;

  virtual void              updateZRange    ();
  virtual Rect              getViewport     () const = 0;
  virtual Transform         getProjection   () const = 0;

  virtual void              jitter

    ( Transform&              proj,
      GLdouble                dx,
      GLdouble                dy )             const = 0;

  virtual GLdouble          setZoom

    ( GLdouble                factor );

  virtual GLdouble          getZoom         () const;

  inline  GLdouble          zoomIn

    ( GLdouble                step = 1.0 );

  inline  GLdouble          zoomOut

    ( GLdouble                step = 1.0 );


 protected:

  virtual                  ~Camera          ();


 private:

  void                      getZRange_

    ( RenderContext&          rtx,
      GLdouble&               clip0,
      GLdouble&               clip1,
      const Rect&             rect );

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline Camera::Camera ( const String& name ) : Super ( name )
{}


//-----------------------------------------------------------------------
//   zoomIn & zoomOut
//-----------------------------------------------------------------------


inline GLdouble Camera::zoomIn ( GLdouble step )
{
  return setZoom( getZoom() + step );
}


inline GLdouble Camera::zoomOut ( GLdouble step )
{
  return setZoom( getZoom() - step );
}


JEM_END_PACKAGE( gl )

#endif
