
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

#ifndef JEM_GL_ORTHOCAMERA_H
#define JEM_GL_ORTHOCAMERA_H

#include <jem/gl/StdCamera.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class OrthoCamera
//-----------------------------------------------------------------------


class OrthoCamera : public StdCamera
{
 public:

  JEM_DECLARE_CLASS       ( OrthoCamera, StdCamera );

  static const GLdouble     MIN_ZOOM;
  static const GLdouble     MAX_ZOOM;

  static const GLdouble     RANGE;


  explicit                  OrthoCamera

    ( const String&           name  = "camera",
      const Ref<GfxNode>&     child = nullptr );

  virtual bool              isOrtho         () const override;
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

  void                      setRange

    ( GLdouble                range );

  inline GLdouble           getRange        () const noexcept;
  virtual void              updateZRange    ()       override;

  void                      setZRange

    ( GLdouble                clip0,
      GLdouble                clip1 );

  inline GLdouble           getNearClip     () const noexcept;
  inline GLdouble           getFarClip      () const noexcept;


 protected:

  virtual                  ~OrthoCamera     ();

  virtual Transform         newProjection_  () const override;


 protected:

  GLdouble                  range_;
  GLdouble                  nearClip_;
  GLdouble                  farClip_;

};



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getRange
//-----------------------------------------------------------------------


inline GLdouble OrthoCamera::getRange () const noexcept
{
  return range_;
}


//-----------------------------------------------------------------------
//   getNearClip
//-----------------------------------------------------------------------


GLdouble OrthoCamera::getNearClip () const noexcept
{
  return nearClip_;
}


//-----------------------------------------------------------------------
//   getFarClip
//-----------------------------------------------------------------------


GLdouble OrthoCamera::getFarClip () const noexcept
{
  return farClip_;
}


JEM_END_PACKAGE( gl )

#endif
