#ifndef BUFH
#define BUFH
/*        $Header: /home/smart/release/src/h/buf.h,v 11.0 1992/07/21 18:18:32 chrisb Exp $*/

/* structure used for passing around text (buf) which possibly includes
   NULLs.  see buf_util.c for add_buf(). */
typedef struct {
    int size;                      
    int end;
    char *buf;
} SM_BUF;

#endif /* BUFH */
