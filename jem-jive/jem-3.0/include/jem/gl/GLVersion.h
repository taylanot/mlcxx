
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

#ifndef JEM_GL_GLVERSION_H
#define JEM_GL_GLVERSION_H

#include <jem/base/String.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class GLVersion
//-----------------------------------------------------------------------


class GLVersion
{
 public:

  typedef GLVersion       Self;

  enum                    VersionID
  {
                            VERSION_0,
                            VERSION_1,
                            VERSION_1_CORE,
                            VERSION_1_1,
                            VERSION_1_1_CORE,
                            VERSION_1_2,
                            VERSION_1_2_CORE,
                            VERSION_1_3,
                            VERSION_1_3_CORE,
                            VERSION_1_4,
                            VERSION_1_4_CORE,
                            VERSION_1_5,
                            VERSION_1_5_CORE,
                            VERSION_2,
                            VERSION_2_CORE,
                            VERSION_2_1,
                            VERSION_2_1_CORE,
                            VERSION_3,
                            VERSION_3_CORE,
                            VERSION_3_1,
                            VERSION_3_1_CORE,
                            VERSION_3_2,
                            VERSION_3_2_CORE,
                            VERSION_3_3,
                            VERSION_3_3_CORE,
                            VERSION_4,
                            VERSION_4_CORE,
                            VERSION_4_1,
                            VERSION_4_1_CORE,
                            VERSION_4_2,
                            VERSION_4_2_CORE,
                            VERSION_4_3,
                            VERSION_4_3_CORE,
                            VERSION_4_4,
                            VERSION_4_4_CORE,
                            VERSION_4_5,
                            VERSION_4_5_CORE,
                            VERSION_4_6,
                            VERSION_4_6_CORE,

                            LAST_VERSION = VERSION_4_6_CORE
  };

  enum                    Profile
  {
                            NO_PROFILE,
                            CORE_PROFILE,
                            COMPAT_PROFILE
  };


                          GLVersion        ();

                          GLVersion

    ( VersionID             vid );

                          GLVersion

    ( int                   major,
      int                   minor,
      Profile               profile = NO_PROFILE );

                          GLVersion

    ( const String&         vers,
      Profile               profile = NO_PROFILE );

  void                    setVersion

    ( VersionID             vid );

  void                    setVersion

    ( int                   major,
      int                   minor );

  void                    setVersion

    ( const String&         vers );

  void                    setProfile

    ( Profile               profile );

  bool                    isValid          () const;
  String                  toString         () const;
  inline Profile          getProfile       () const noexcept;
  inline VersionID        getVersionID     () const noexcept;
  inline int              getMajorVersion  () const noexcept;
  inline int              getMinorVersion  () const noexcept;


 private:

  VersionID               getVersionID_    () const;


 private:

  struct                  Version_
  {
    byte                    major;
    byte                    minor;
    Profile                 profile;
  };

  static const Version_   VERSIONS_ [];

  int                     major_;
  int                     minor_;
  Profile                 profile_;
  VersionID               versionID_;

};





//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class GLVersion
//=======================================================================

//-----------------------------------------------------------------------
//   getProfile
//-----------------------------------------------------------------------


inline GLVersion::Profile GLVersion::getProfile () const noexcept
{
  return profile_;
}


//-----------------------------------------------------------------------
//   getMajorVersion
//-----------------------------------------------------------------------


inline int GLVersion::getMajorVersion () const noexcept
{
  return major_;
}


//-----------------------------------------------------------------------
//   getMinorVersion
//-----------------------------------------------------------------------


inline int GLVersion::getMinorVersion () const noexcept
{
  return minor_;
}


//-----------------------------------------------------------------------
//   getVersionID
//-----------------------------------------------------------------------


inline GLVersion::VersionID GLVersion::getVersionID () const noexcept
{
  return versionID_;
}


JEM_END_PACKAGE( gl )

#endif

