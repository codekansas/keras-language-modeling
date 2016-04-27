#ifndef SMART_ERRORH
#define SMART_ERRORH
#include <errno.h>
#define SMART_MINERR 1000
#define SM_INCON_ERR 1000
#define SM_ILLSK_ERR 1001
#define SM_ILLMD_ERR 1002
#define SM_ILLPA_ERR 1003
#define SMART_NUMERR 4


extern int  errno;
extern int  smart_errno;        /* If > 0 and <= sys_nerr then refers to  */
                                /* sys_errlist, else if >= smart_errmin */
                                /* and <= smart_errmax, then smart_errlist */
extern char *smart_message;     /* Message to be printed (often filename) */
extern char *smart_routine;     /* Major routine issuing error message */


#define set_error(n,m,r) {  if (n > 0) smart_errno = n;\
                            smart_message = m;\
                            smart_routine = r; }
#define clr_err() smart_errno = errno = 0
#endif /* SMART_ERRORH */
