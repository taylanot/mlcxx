
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

#ifndef JIVE_GL_MESHVIEW_H
#define JIVE_GL_MESHVIEW_H

#include <jive/gl/Plugin.h>


JIVE_BEGIN_PACKAGE( gl )


class ChainPlugin;


//-----------------------------------------------------------------------
//   class MeshView
//-----------------------------------------------------------------------


class MeshView : public Plugin
{
 public:

  JEM_DECLARE_CLASS       ( MeshView, Plugin );

  static const char*        TYPE_NAME;


  explicit                  MeshView

    ( const String&           name );

  virtual void              init

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       viewdat,
      const AppContext&       apx )                override;

  virtual void              update

    ( const Properties&       viewdat,
      const AppContext&       apx )                override;

  virtual void              shutdown

    ( const Properties&       viewdat,
      const AppContext&       apx )                override;

  virtual void              configure

    ( const Properties&       props,
      const Properties&       viewdat,
      const AppContext&       apx )                override;

  virtual void              getConfig

    ( const Properties&       props )        const override;

  virtual void              printHelp

    ( PrintWriter&            pr )           const override;

  void                      addUserPlugin ();
  inline ChainPlugin*       getPlugins    () const;

  static void               declare       ();

  static Ref<Plugin>        makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props );


 protected:

  virtual                  ~MeshView      ();


 private:

  class                     RunData_;
  class                     Scope_;
  friend class              Scope_;

  Ref<ChainPlugin>          plugins_;
  Ref<RunData_>             rundat_;

};



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getPlugins
//-----------------------------------------------------------------------


inline ChainPlugin* MeshView::getPlugins () const
{
  return plugins_.get ();
}


JIVE_END_PACKAGE( gl )

#endif
