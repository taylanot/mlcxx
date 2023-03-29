
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

#ifndef JIVE_IMPLICT_NEWMARKSTEPPER_H
#define JIVE_IMPLICT_NEWMARKSTEPPER_H

#include <jem/io/Serializable.h>
#include <jive/implict/import.h>
#include <jive/implict/TimeStepper.h>


JIVE_BEGIN_PACKAGE( implict )


//-----------------------------------------------------------------------
//   class NewmarkStepper
//-----------------------------------------------------------------------


class NewmarkStepper : public TimeStepper,
                       public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( NewmarkStepper, TimeStepper );

  static const char*        TYPE_NAME;


                            NewmarkStepper  ();

                            NewmarkStepper

    ( const String&           name,
      int                     order,
      const Ref<DofSpace>&    dofs );

                            NewmarkStepper

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  virtual void              readFrom

    ( ObjectInput&            in )                   override;

  virtual void              writeTo

    ( ObjectOutput&           out )            const override;

  virtual int               configure

    ( const Properties&       props )                override;

  virtual void              getConfig

    ( const Properties&       conf )           const override;

  virtual int               restart

    ( const Properties&       globdat )              override;

  virtual int               advance

    ( const Properties&       globdat )              override;

  virtual void              commit

    ( const Properties&       globdat )              override;

  virtual int               cancel

    ( const Properties&       globdat )              override;

  virtual void              updateState

    ( const Properties&       globdat )              override;

  virtual void              predictState

    ( const Properties&       globdat )              override;

  virtual double            getTimeCoeff

    ( int                     order )          const override;

  virtual int               getMaxOrder     () const override;
  virtual double            getDeltaTime    () const override;

  virtual void              setDeltaTime

    ( double                  dtime )                override;

  virtual idx_t             getTimeStep     () const override;
  virtual DofSpace*         getDofSpace     () const override;
  virtual String            getContext      () const override;

  void                      setContext

    ( const String&           context );

  static Ref<TimeStepper>   makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  static void               declare         ();


 protected:

  virtual                  ~NewmarkStepper  ();


 private:

  void                      init_           ();


 private:

  Ref<DofSpace>             dofs_;

  int                       order_;
  idx_t                     istep_;
  idx_t                     istep0_;

  double                    gamma_;
  double                    dtime_;

  String                    context_;

};


JIVE_END_PACKAGE( implict )

#endif
