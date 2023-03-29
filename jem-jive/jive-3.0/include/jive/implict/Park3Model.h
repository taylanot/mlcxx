
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

#ifndef JIVE_IMPLICT_PARK3MODEL_H
#define JIVE_IMPLICT_PARK3MODEL_H

#include <jem/base/Flags.h>
#include <jem/io/Serializable.h>
#include <jive/Array.h>
#include <jive/model/Model.h>
#include <jive/implict/import.h>


JIVE_BEGIN_PACKAGE( implict )


class TransientMatrix;


//-----------------------------------------------------------------------
//   class Park3Model
//-----------------------------------------------------------------------


class Park3Model : public Model,
                   public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( Park3Model, Model );

  static const char*        TYPE_NAME;

  enum                      Option
  {
                              MANAGED = 1 << 0
  };

  typedef
    jem::Flags<Option>      Options;


                            Park3Model      ();

                            Park3Model

    ( const String&           name,
      const Ref<Model>&       child,
      Options                 options = 0 );

                            Park3Model

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  void                      init

    ( const Properties&       globdat );

  virtual void              readFrom

    ( ObjectInput&            in )                   override;

  virtual void              writeTo

    ( ObjectOutput&           out )            const override;

  virtual Model*            findModel

    ( const String&           name )           const override;

  virtual void              configure

    ( const Properties&       props,
      const Properties&       globdat )              override;

  virtual void              getConfig

    ( const Properties&       conf,
      const Properties&       globdat )        const override;

  virtual bool              takeAction

    ( const String&           action,
      const Properties&       params,
      const Properties&       globdat )              override;

  virtual String            getContext      () const override;

  void                      setContext

    ( const String&           context );

  static Ref<Model>         makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  static void               declare         ();


 protected:

  virtual                  ~Park3Model      ();


 private:

  void                      init_           ();
  void                      restart_        ();

  void                      advance_

    ( const Properties&       globdat );

  void                      commit_

    ( const Properties&       globdat );

   void                     cancel_

    ( const Properties&       globdat );

  void                      getIdxVector_

    ( const Vector&           fint,
      const Properties&       globdat );

  void                      getExtVector_

    ( const Vector&           fext,
      const Properties&       globdat );

  void                      updateCoeffs_   ();

  void                      getHistVector_

    ( const Vector&           h,
      const Vector&           adot,
      const Vector            aold[3] )        const;

  void                      dofsChanged_    ();
  void                      consChanged_    ();
  void                      mat2Changed_    ();


 private:

  static const int          STARTED_;
  static const int          ADVANCING_;

  static const int          U_EXT_VEC_;
  static const int          U_INT_VEC_;

  struct                    Coeffs_
  {
    double                    a[3];
    double                    b[3];
    double                    hb;
    double                    hb2;
  };

  struct                    Globvecs_
  {
    Vector                    u;
    Vector                    udot;
  };

  String                    context_;

  Ref<Model>                child_;
  Ref<DofSpace>             dofs_;
  Ref<Constraints>          cons_;
  Ref<TransientMatrix>      transmat_;

  IdxVector                 slaveDofs_;

  Matrix                    mbuf_;
  Vector                    vbuf_;
  Vector                    fext_;
  Vector                    fint_;
  Vector                    vdot_;
  Vector                    uhist_;
  Vector                    vhist_;

  Vector                    uold_[3];
  Vector                    vold_[3];

  Coeffs_                   coeffs_;
  Globvecs_                 globvecs_;

  double                    dtime_;
  double                    alpha_;
  double                    beta_;

  idx_t                     istep_;
  idx_t                     istep0_;
  int                       status_;
  int                       updated_;
  Options                   options_;

};


JEM_DEFINE_FLAG_OPS( Park3Model::Options )


JIVE_END_PACKAGE( implict )

#endif
