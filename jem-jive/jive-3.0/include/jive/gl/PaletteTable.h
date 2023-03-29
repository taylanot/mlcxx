
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */

#ifndef JIVE_GL_PALETTETABLE_H
#define JIVE_GL_PALETTETABLE_H

#include <jem/base/Object.h>
#include <jem/gl/Palette.h>
#include <jive/Array.h>
#include <jive/gl/import.h>


namespace jem
{
  namespace util
  {
    class Dictionary;
  }
}


JIVE_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class PaletteTable
//-----------------------------------------------------------------------


class PaletteTable : public Object
{
 public:

  JEM_DECLARE_CLASS       ( PaletteTable, Object );

  typedef
    jem::gl::Palette        Palette;

  typedef Ref<Palette>    (*PaletteCtor)      ();


                            PaletteTable      ();

  void                      configure

    ( const Properties&       props );

  void                      getConfig

    ( const Properties&       props )            const;

  void                      clear             ();

  void                      addPalette

    ( const String&           name,
      const Ref<Palette>&     palette );

  void                      addPalettes

    ( const StringVector&     names,
      const Properties&       conf,
      const Properties&       props );

  Ref<Palette>              findPalette

    ( const String&           name )             const;

  Ref<Palette>              getPalette

    ( const String&           name,
      const String&           context )          const;

  StringVector              listPalettes      () const;

  static Ref<Self>          get

    ( const Properties&       viewdat );

  static void               declarePalette

    ( const String&           type,
      PaletteCtor             ctor );

  static StringVector       listPaletteTypes  ();

  static Ref<Palette>       newPalette

    ( const String&           type );

  static Ref<Palette>       newPalette

    ( const Properties&       conf,
      const Properties&       props );


 protected:

  virtual                  ~PaletteTable      ();


 private:

  static void               initOnce_         ();


 private:

  typedef jem::
    util::Dictionary        Dict_;
  class                     CtorMap_;

  Ref<Dict_>                palettes_;

};


JIVE_END_PACKAGE( gl )

#endif
