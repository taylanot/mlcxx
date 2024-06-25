/**
 * @file progress.h
 * @author Ozgur Taylan Turan
 *
 * Simple Progress thread safe progress bar for my loops
 */



#ifndef __PROGRESS_H
#define __PROGRESS_H

//=============================================================================
// ProgresssBar : A Simple Progress Bar that shows the perc. of your loop
//                It can be used with omp parallel too
//=============================================================================
class ProgressBar
{
public:
  ProgressBar(int total) : tot_(total), curr_(0), what_("Loop") 
  { std::cout << "\n"; }
  ProgressBar(std::string what, int total) : tot_(total), curr_(0), what_(what) 
  { std::cout << "\n"; }

  void Update()
  {
    #pragma omp critical
    {
      ++curr_;
      Show();
    }
  }

  /* void Update(int i) */
  /* { */
  /*   #pragma omp critical */
  /*   { */
  /*     if (curr_ != i) */
  /*     { */
  /*       curr_ = i ; */
  /*       Show(); */
  /*     } */
  /*   } */
  /* } */

private:
  int tot_;
  int curr_;
  std::string what_;

  void Show()
  {
    double prog = static_cast<double>(curr_) / tot_;
    int perc = static_cast<int>(prog * 100.0);
    std::cout << "\r" << std::setw(3) << perc << " %" 
                                              << "  : " << what_ << std::flush;
    if (curr_ == tot_)
      std::cout << "\n\n" ;
    
  }
};


#endif
