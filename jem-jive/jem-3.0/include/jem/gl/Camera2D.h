
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

#ifndef JEM_GL_CAMERA2D_H
#define JEM_GL_CAMERA2D_H

#include <jem/gl/OrthoCamera.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class Camera2D
//-----------------------------------------------------------------------


class Camera2D : public OrthoCamera
{
 public:

  JEM_DECLARE_CLASS       ( Camera2D, OrthoCamera );


  explicit                  Camera2D

    ( const String&           name  = "camera",
      const Ref<GfxNode>&     child = nullptr );

  virtual void              getBBox

    ( Box&                    box )       const override;

  virtual void              configure

    ( const Properties&       props )           override;

  virtual void              getConfig

    ( const Properties&       props )      const override;

  virtual void              listProps

    ( const Properties&       props )      const override;


 protected:

  virtual                  ~Camera2D    ();

};


JEM_END_PACKAGE( gl )

#endif
