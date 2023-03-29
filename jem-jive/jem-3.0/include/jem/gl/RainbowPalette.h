
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

#ifndef JEM_GL_RAINBOWPALETTE_H
#define JEM_GL_RAINBOWPALETTE_H

#include <jem/gl/Palette.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class RainbowPalette
//-----------------------------------------------------------------------


class RainbowPalette : public Palette
{
 public:

  JEM_DECLARE_CLASS       ( RainbowPalette, Palette );


                            RainbowPalette  ()       noexcept;

  virtual void              getColor

    ( Color&                  col,
      double                  val )            const override;

  virtual double            findValue

    ( double&                 val,
      double                  prec,
      const Color&            col )            const override;

  void                      setRange

    ( double                  lo,
      double                  hi );

  virtual double            getMinValue     () const override;
  virtual double            getMaxValue     () const override;
  virtual GLfloat           getMinAlpha     () const override;

  virtual void              configure

    ( const Properties&       props )                override;

  virtual void              getConfig

    ( const Properties&       props )          const override;

  virtual void              listProps

    ( const Properties&       props )          const override;

  virtual void              fillRectangle

    ( GLContext&              gtx,
      GLint                   width,
      GLint                   height )         const override;


 protected:

  virtual                  ~RainbowPalette  ();


 private:

  double                    minValue_;
  double                    maxValue_;

};


JEM_END_PACKAGE( gl )

#endif
