/* Copyright (c) 2003, 1991, 1990, 1984 Chris Buckley.  */


#include "common.h"
#include "sysfunc.h"
#include "smart_error.h"
#include "trec_eval.h"
#include <ctype.h>

/* Read all retrieved results information from trec_top_file.
Read text tuples from trec_top_file of the form
     030  Q0  ZF08-175-870  0   4238   prise1
     qid iter   docno      rank  sim   run_id
giving TREC document numbers (a string) retrieved by query qid 
(a string) with similarity sim (a float).  The other fields are ignored,
with the exception that the run_id field of the last line is kept and
output.  In particular, note that the rank field is ignored here;
internally ranks are assigned by sorting by the sim field with ties 
broken determinstically (using docno).
Sim is assumed to be higher for the docs to be retrieved first.
File may contain no NULL characters.
Any field following run_id is ignored.
*/

int 
get_top (trec_top_file, all_trec_top)
char *trec_top_file;
ALL_TREC_TOP *all_trec_top;
{
    int fd;
    int size = 0;
    char *trec_top_buf;
    char *ptr;
    char *current_qid;
    char *qid_ptr, *docno_ptr, *sim_ptr;
    char *run_id_ptr = "";
    long i;
    TREC_TOP *current_top = NULL;
    float sim;

    /* Read entire file into memory */
    if (-1 == (fd = open (trec_top_file, 0)) ||
        -1 == (size = lseek (fd, 0L, 2)) ||
        NULL == (trec_top_buf = malloc ((unsigned) size+2)) ||
        -1 == lseek (fd, 0L, 0) ||
        size != read (fd, trec_top_buf, size) ||
	-1 == close (fd)) {

        set_error (SM_ILLPA_ERR, "Cannot read results file", "trec_eval");
        return (UNDEF);
    }

    current_qid = "";

    /* Initialize all_trec_top */
    all_trec_top->num_q_tr = 0;
    all_trec_top->max_num_q_tr = INIT_NUM_QUERIES;
    if (NULL == (all_trec_top->trec_top = Malloc (INIT_NUM_QUERIES,
						     TREC_TOP)))
	return (UNDEF);
    
    if (size == 0)
	return (0);

    /* Append ending newline if not present, Append NULL terminator */
    if (trec_top_buf[size-1] != '\n') {
	trec_top_buf[size] = '\n';
	size++;
    }
    trec_top_buf[size] = '\0';

    ptr = trec_top_buf;

    while (*ptr) {
	/* Get current line */
	/* Get qid */
	while (*ptr != '\n' && isspace (*ptr)) ptr++;
	if (*ptr == '\n') {
	    /* Ignore blank lines (people seem to insist on them!) */
	    ptr++;
	    continue;
	}
	qid_ptr = ptr;
	while (! isspace (*ptr)) ptr++;
	if (*ptr == '\n') {
	    set_error (SM_ILLPA_ERR,"malformed top results line", "trec_eval");
	    return (UNDEF);
	}
	*ptr++ = '\0';
	/* Skip iter */
	while (*ptr != '\n' && isspace (*ptr)) ptr++;
	while (! isspace (*ptr)) ptr++;
	if (*ptr++ == '\n') {
	    set_error (SM_ILLPA_ERR,"malformed top results line", "trec_eval");
	    return (UNDEF);
	}
	/* Get docno */
	while (*ptr != '\n' && isspace (*ptr)) ptr++;
	docno_ptr = ptr;
	while (! isspace (*ptr)) ptr++;
	if (*ptr == '\n') {
	    set_error (SM_ILLPA_ERR,"malformed top results line", "trec_eval");
	    return (UNDEF);
	}
	*ptr++ = '\0';
	/* Skip rank */
	while (*ptr != '\n' && isspace (*ptr)) ptr++;
	while (! isspace (*ptr)) ptr++;
	if (*ptr++ == '\n') {
	    set_error (SM_ILLPA_ERR,"malformed top results line", "trec_eval");
	    return (UNDEF);
	}
	/* Get sim */
	while (*ptr != '\n' && isspace (*ptr)) ptr++;
	sim_ptr = ptr;
	while (! isspace (*ptr)) ptr++;
	if (*ptr == '\n') {
	    set_error (SM_ILLPA_ERR,"malformed top results line", "trec_eval");
	    return (UNDEF);
	}
	*ptr++ = '\0';
	/* Get run_id */
	while (*ptr != '\n' && isspace (*ptr)) ptr++;
	if (*ptr == '\n') {
	    set_error (SM_ILLPA_ERR,"malformed top results line", "trec_eval");
	    return (UNDEF);
	}
	run_id_ptr = ptr;
	while (! isspace (*ptr)) ptr++;
	if (*ptr != '\n') {
	    /* Skip over rest of line */
	    *ptr++ = '\0';
	    while (*ptr != '\n') ptr++;
	}
	*ptr++ = '\0';

	if (0 != strcmp (qid_ptr, current_qid)) {
	    /* Query has changed. Must check if new query or this is more
	       judgements for an old query */
	    for (i = 0; i < all_trec_top->num_q_tr; i++) {
		if (0 == strcmp (qid_ptr, all_trec_top->trec_top[i].qid))
		    break;
	    }
	    if (i >= all_trec_top->num_q_tr) {
		/* New unseen query, add and initialize it */
		if (all_trec_top->num_q_tr >=
		    all_trec_top->max_num_q_tr) {
		    all_trec_top->max_num_q_tr *= 10;
		    if (NULL == (all_trec_top->trec_top = 
				 Realloc (all_trec_top->trec_top,
					  all_trec_top->max_num_q_tr,
					  TREC_TOP)))
			return (UNDEF);
		}
		current_top = &all_trec_top->trec_top[i];
		current_top->qid = qid_ptr;
		current_top->num_text_tr = 0;
		current_top->max_num_text_tr = INIT_NUM_RESULTS;
		if (NULL == (current_top->text_tr =
			     Malloc (INIT_NUM_RESULTS, TEXT_TR)))
		    return (UNDEF);
		all_trec_top->num_q_tr++;
	    }
	    else {
		/* Old query, just switch current_q_index */
		current_top = &all_trec_top->trec_top[i];
	    }
	    current_qid = current_top->qid;
	}
	
	/* Add retrieval docno/sim to current query's list */
	if (current_top->num_text_tr >= 
	    current_top->max_num_text_tr) {
	    /* Need more space */
	    current_top->max_num_text_tr *= 10;
	    if (NULL == (current_top->text_tr = 
			 Realloc (current_top->text_tr,
				  current_top->max_num_text_tr,
				  TEXT_TR)))
		return (UNDEF);
	}
	current_top->text_tr[current_top->num_text_tr].docno = docno_ptr;
	sim = atof (sim_ptr);
	current_top->text_tr[current_top->num_text_tr].sim = sim;
	current_top->text_tr[current_top->num_text_tr++].rank = 0;
    }

    all_trec_top->run_id = run_id_ptr;

    return (1);
}

