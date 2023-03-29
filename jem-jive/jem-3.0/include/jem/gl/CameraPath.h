
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

#ifndef JEM_GL_CAMERAPATH_H
#define JEM_GL_CAMERAPATH_H

#include <jem/base/Time.h>
#include <jem/gl/Box.h>
#include <jem/gl/Transform.h>
#include <jem/gl/WrapperNode.h>


JEM_BEGIN_PACKAGE( gl )


class Camera;
class RenderContext;


//-----------------------------------------------------------------------
//   class CameraPath
//-----------------------------------------------------------------------


class CameraPath : public WrapperNode
{
 public:

  JEM_DECLARE_CLASS       ( CameraPath, WrapperNode );


  explicit                  CameraPath

    ( const Ref<Camera>&      cam );

  virtual bool              takeGLAction

    ( RenderContext&          rtx,
      int                     action,
      const Properties&       params )      override;

  void                      setPath

    ( const Box&              end );

  void                      setPath

    ( const Box&              mid,
      const Box&              end );

  void                      finish       ();

  void                      animate

    ( double                  time = 1.0 );

  static void               animate

    ( const Ref<Camera>&      cam,
      const Box&              end,
      double                  time = 1.0 );

  static void               animate

    ( const Ref<Camera>&      cam,
      const Box&              mid,
      const Box&              end,
      double                  time = 1.0 );


 protected:

  virtual                  ~CameraPath   ();


 private:

  bool                      advance_

    ( RenderContext&          rtx );


 private:

  Ref<Camera>               camera_;
  Transform                 trans0_;
  Time                      start_;
  Box                       path_ [3];
  double                    time_ [3];
  double                    duration_;
  bool                      started_;

};


JEM_END_PACKAGE( gl )

#endif
