
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

#ifndef JIVE_MODEL_MODEL_H
#define JIVE_MODEL_MODEL_H

#include <jem/base/NamedObject.h>
#include <jive/defines.h>


JIVE_BEGIN_PACKAGE( model )


//-----------------------------------------------------------------------
//   class Model
//-----------------------------------------------------------------------


class Model : public NamedObject
{
 public:

  JEM_DECLARE_CLASS       ( Model, NamedObject );


  explicit                  Model

    ( const String&           name = "" );

                            Model

    ( const Self&             rhs );

  virtual Model*            findModel

    ( const String&           name )       const;

  virtual void              configure

    ( const Properties&       props,
      const Properties&       globdat );

  virtual void              getConfig

    ( const Properties&       conf,
      const Properties&       globdat )    const;

  virtual bool              takeAction

    ( const String&           action,
      const Properties&       params,
      const Properties&       globdat )          = 0;

  virtual String            getContext  () const override;

  void                      store

    ( const Properties&       globdat )    const;

  static Ref<Model>         find

    ( const Properties&       globdat );

  static Ref<Model>         get

    ( const Properties&       globdat,
      const String&           context );

  static Ref<Model>         extract

    ( const Properties&       globdat );


 protected:

  virtual                  ~Model       ();

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


Ref<Model>                  newModel

  ( const String&             name,
    const Properties&         conf,
    const Properties&         props,
    const Properties&         globdat );

Ref<Model>                  explicitNewModel

  ( const String&             name,
    const Properties&         conf,
    const Properties&         props,
    const Properties&         globdat );


JIVE_END_PACKAGE( model )

#endif
