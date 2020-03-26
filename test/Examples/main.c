#include "runtime/main.h"
#include "runtime/stop.h"

/* main entry into PROGRAM */
void _QQmain();

/* C main stub */
int main(int argc, char **argv, char **envp)
{
  RTNAME(ProgramStart)(argc, argv, envp);
  _QQmain();
  RTNAME(ProgramEndStatement)();
  return 0;
}
