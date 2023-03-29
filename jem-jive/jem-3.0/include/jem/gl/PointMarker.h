
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

#ifndef JEM_GL_POINTMARKER_H
#define JEM_GL_POINTMARKER_H

#include <jem/util/Flex.h>
#include <jem/gl/MarkerStyle.h>
#include <jem/gl/ColoredMarker.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class PointMarker
//-----------------------------------------------------------------------


class PointMarker : public MarkerStyle,
                    public ColoredMarker
{
 public:

  JEM_DECLARE_CLASS       ( PointMarker, MarkerStyle );

  static const GLfloat      MAX_SIZE;


  explicit                  PointMarker

    ( GLfloat                 size  = 3.0,
      const Color&            color = Color::black() );

  virtual String            toString      () const override;

  virtual void              configure

    ( const Properties&       props )              override;

  virtual void              getConfig

    ( const Properties&       props )        const override;

  virtual void              listProps

    ( const Properties&       props )        const override;

  virtual void              beginDraw

    ( GLContext&              gtx,
      const Box&              box,
      idx_t                   count )              override;

  virtual void              endDraw

    ( GLContext&              gtx )                override;

  virtual void              drawMarker

    ( GLContext&              gtx,
      idx_t                   tag,
      const GLfloat*          pos )                override;

  virtual void              setColor

    ( const Color&            color )              override;

  virtual Color             getColor      () const override;

  void                      setSize

    ( GLfloat                 size );

  inline GLfloat            getSize       () const noexcept;


 protected:

  virtual                  ~PointMarker   ();


 private:

  jem::util::Flex<GLfloat>  coords_;
  GLfloat                   size_;
  Color                     color_;
  bool                      active_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getSize
//-----------------------------------------------------------------------


inline GLfloat PointMarker::getSize () const noexcept
{
  return size_;
}


JEM_END_PACKAGE( gl )

#endif
