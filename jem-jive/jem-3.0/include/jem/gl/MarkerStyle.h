
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

#ifndef JEM_GL_MARKERSTYLE_H
#define JEM_GL_MARKERSTYLE_H

#include <jem/base/Object.h>
#include <jem/gl/gl.h>
#include <jem/gl/import.h>


JEM_BEGIN_PACKAGE( gl )


class Box;
class GLContext;


//-----------------------------------------------------------------------
//   class MarkerStyle
//-----------------------------------------------------------------------


class MarkerStyle : public Object
{
 public:

  JEM_DECLARE_CLASS       ( MarkerStyle, Object );


  virtual void              configure

    ( const Properties&       props );

  virtual void              getConfig

    ( const Properties&       props )        const;

  virtual void              listProps

    ( const Properties&       props )        const;

  virtual void              setFont      ();

  virtual void              beginDraw

    ( GLContext&              gtx,
      const Box&              box,
      idx_t                   count )              = 0;

  virtual void              endDraw

    ( GLContext&              gtx )                = 0;

  virtual void              drawMarker

    ( GLContext&              gtx,
      idx_t                   tag,
      const GLfloat*          pos )                = 0;


 protected:

  virtual                  ~MarkerStyle   ();

};


JEM_END_PACKAGE( gl )

#endif
