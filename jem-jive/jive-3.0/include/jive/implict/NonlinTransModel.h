
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

#ifndef JIVE_IMPLICT_NONLINTRANSMODEL_H
#define JIVE_IMPLICT_NONLINTRANSMODEL_H

#include <jem/base/Flags.h>
#include <jem/io/Serializable.h>
#include <jem/util/Properties.h>
#include <jive/model/Model.h>
#include <jive/implict/import.h>


JIVE_BEGIN_PACKAGE( implict )


class TimeStepper;


//-----------------------------------------------------------------------
//   class NonlinTransModel
//-----------------------------------------------------------------------


class NonlinTransModel : public Model,
                         public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( NonlinTransModel, Model );

  static const char*        TYPE_NAME;

  enum                      Option
  {
                              MANAGED = 1 << 0
  };

  typedef
    jem::Flags<Option>      Options;


                            NonlinTransModel  ();

                            NonlinTransModel

    ( const String&           name,
      const Ref<Model>&       child,
      Ref<TimeStepper>        stepper,
      Options                 options = 0 );

                            NonlinTransModel

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  void                      init

    ( const Properties&       globdat );

  virtual void              readFrom

    ( ObjectInput&            in )                     override;

  virtual void              writeTo

    ( ObjectOutput&           out )              const override;

  virtual Model*            findModel

    ( const String&           name )             const override;

  virtual void              configure

    ( const Properties&       props,
      const Properties&       globdat )                override;

  virtual void              getConfig

    ( const Properties&       conf,
      const Properties&       globdat )          const override;

  virtual bool              takeAction

    ( const String&           action,
      const Properties&       params,
      const Properties&       globdat )                override;

  virtual String            getContext        () const override;

  void                      setContext

    ( const String&           context );

  static Ref<Model>         makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  static void               declare           ();


 protected:

  virtual                  ~NonlinTransModel  ();


 private:

  void                      init_             ();

  void                      restart_

    ( const Properties&       globdat );

  void                      commit_

    ( const Properties&       globdat );

  void                      newMatrix_

    ( const Properties&       globdat );

  void                      dofsChanged_    ();


 private:

  static const int          STARTED_;
  static const int          ADVANCING_;

  String                    context_;

  Ref<Model>                child_;
  Ref<TimeStepper>          stepper_;
  Ref<MatrixBuilder>        mbuilder_;

  Ref<DofSpace>             dofs_;
  Ref<VectorSpace>          vspace_;

  double                    dtMin_;
  double                    dtMax_;
  double                    norm0_;

  int                       status_;
  Options                   options_;

  Properties                matConf_;
  Properties                matProps_;

};


JEM_DEFINE_FLAG_OPS( NonlinTransModel::Options )


JIVE_END_PACKAGE( implict )

#endif
