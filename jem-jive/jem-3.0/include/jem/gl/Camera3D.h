
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

#ifndef JEM_GL_CAMERA3D_H
#define JEM_GL_CAMERA3D_H

#include <jem/gl/StdCamera.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class Camera3D
//-----------------------------------------------------------------------


class Camera3D : public StdCamera
{
 public:

  JEM_DECLARE_CLASS       ( Camera3D, StdCamera );

  static const GLdouble     MIN_ZOOM;
  static const GLdouble     MAX_ZOOM;

  static const GLdouble     MIN_ANGLE;
  static const GLdouble     MAX_ANGLE;
  static const GLdouble     VIEW_ANGLE;

  static const GLdouble     MIN_FAR;
  static const GLdouble     MAX_FAR;
  static const GLdouble     MIN_NEAR;
  static const GLdouble     MAX_NEAR;
  static const GLdouble     FAR_CLIP;
  static const GLdouble     NEAR_CLIP;


  explicit                  Camera3D

    ( const String&           name  = "camera",
      const Ref<GfxNode>&     child = nullptr );

  virtual void              configure

    ( const Properties&       props )                override;

  virtual void              getConfig

    ( const Properties&       props )          const override;

  virtual void              listProps

    ( const Properties&       props )          const override;

  virtual void              reset           ()       override;

  virtual void              lookAtBox

    ( const Box&              box )                  override;

  virtual void              getVisibleBox

    ( Box&                    box )            const override;

  virtual void              translateObj

    ( GfxNode&                node,
      GLdouble                dx,
      GLdouble                dy,
      GLdouble                dz )             const override;

  virtual GLdouble          setZoom

    ( GLdouble                zoom )                 override;

  virtual void              jitter

    ( Transform&              proj,
      GLdouble                dx,
      GLdouble                dy )             const override;

  virtual void              updateZRange    ()       override;

  void                      setZRange

    ( GLdouble                clip0,
      GLdouble                clip1 );

  inline GLdouble           getNearClip     () const noexcept;
  inline GLdouble           getFarClip      () const noexcept;

  void                      setViewAngle

    ( GLdouble                fovy );

  inline GLdouble           getViewAngle    () const noexcept;


 protected:

  virtual                  ~Camera3D        ();

  virtual Transform         newProjection_  () const override;


 private:

  inline GLdouble           getTeta_        () const;


 private:

  GLdouble                  nearClip_;
  GLdouble                  farClip_;
  GLdouble                  fovy_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getNearClip
//-----------------------------------------------------------------------


GLdouble Camera3D::getNearClip () const noexcept
{
  return nearClip_;
}


//-----------------------------------------------------------------------
//   getFarClip
//-----------------------------------------------------------------------


GLdouble Camera3D::getFarClip () const noexcept
{
  return farClip_;
}


//-----------------------------------------------------------------------
//   getViewAngle
//-----------------------------------------------------------------------


GLdouble Camera3D::getViewAngle () const noexcept
{
  return fovy_;
}


JEM_END_PACKAGE( gl )

#endif
