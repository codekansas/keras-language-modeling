#ifdef RCSID
static char rcsid[] = "$Header: /home/smart/release/src/libevaluate/tr_eval.c,v 11.0 1992/07/21 18:20:33 chrisb Exp chrisb $";
#endif

/* Copyright (c) 1991, 1990, 1984 - Gerard Salton, Chris Buckley. 

   Permission is granted for use of this file in unmodified form for
   research purposes. Please contact the SMART project to obtain 
   permission for other uses.
*/

#include "common.h"
#include "sysfunc.h"
#include "buf.h"
#include "trec_eval.h"

static SM_BUF internal_output = {0, 0, (char *) 0};
int add_buf_string();

extern SINGLE_MEASURE sing_meas[];
extern PARAMETERIZED_MEASURE param_meas[];
extern MICRO_MEASURE micro_meas[];
extern int num_param_meas, num_sing_meas, num_micro_meas;

int
accumulate_results (query_eval, accum_eval)
TREC_EVAL *query_eval;
TREC_EVAL *accum_eval;
{
    long i,j;
    float *float_query, *float_accum;
    long *long_query, *long_accum;

    if (query_eval->num_ret <= 0)
	return (0);

    accum_eval->num_queries++;

    for (i = 0; i < num_sing_meas; i++) {
	if (sing_meas[i].is_long_flag) {
	    long_query = (long *) (((char *) query_eval) + 
				   sing_meas[i].byte_offset);
	    long_accum = (long *) (((char *) accum_eval) +
				   sing_meas[i].byte_offset);
	    *long_accum += *long_query;
	}
	else {
	    float_query = (float *) (((char *) query_eval) +
				   sing_meas[i].byte_offset);
	    float_accum = (float *) (((char *) accum_eval) +
				   sing_meas[i].byte_offset);
	    *float_accum += *float_query;
	}
    }

    for (i = 0; i < num_param_meas; i++) {
	for (j = 0; j < param_meas[i].num_values; j++) {
	    if (param_meas[i].is_long_flag) {
		long_query = (long *) (((char *) query_eval) + 
				       param_meas[i].byte_offset);
		long_accum = (long *) (((char *) accum_eval) +
				       param_meas[i].byte_offset);
		long_accum[j] += long_query[j];
	    }
	    else {
		float_query = (float *) (((char *) query_eval) +
					 param_meas[i].byte_offset);
		float_accum = (float *) (((char *) accum_eval) +
					 param_meas[i].byte_offset);
		float_accum[j] += float_query[j];
	    }
	}
    }

    return (0);
}

void
print_rel_trec_eval_list (is_single_query_flag, epi, eval, output)
long is_single_query_flag;
EVAL_PARAM_INFO *epi;
TREC_EVAL *eval;
SM_BUF *output;
{
    long i,j;
    char temp_buf[1024];
    char q_buf[20];
    char name_buf[80];
    SM_BUF *out_p;
    long long_eval;
    float float_eval;

    if (output == NULL) {
        out_p = &internal_output;
        out_p->end = 0;
    }
    else
        out_p = output;

    if (is_single_query_flag) {
        (void) sprintf (q_buf, "%.20s", eval[0].qid);
    }
    else {
        (void) sprintf (q_buf, "%s", "all");
        (void) sprintf (temp_buf, "%-15s\t%s\t%ld\n",
			"num_q", q_buf, eval->num_queries);
        if (UNDEF == add_buf_string (temp_buf, out_p))
            return;
    }

    for (i = 0; i < num_sing_meas; i++) {
        if ((! sing_meas[i].print_short_flag) && (! epi->all_flag))
            continue;
        if (sing_meas[i].print_time_flag && (!epi->time_flag))
            continue;
	if (sing_meas[i].print_only_query_flag && (!is_single_query_flag))
            continue;
	if (sing_meas[i].print_only_average_flag && (is_single_query_flag))
            continue;
        if (sing_meas[i].is_long_flag) {
            long_eval = *((long *) (((char *) eval) + 
                                   sing_meas[i].byte_offset));
            if (sing_meas[i].avg_results_flag)
		long_eval /= eval->num_queries;
	    (void) sprintf (temp_buf, "%-15s\t%s\t%ld\n",
			    sing_meas[i].name, q_buf, long_eval);
        }
        else {
            float_eval = *((float *) (((char *) eval) +
                                   sing_meas[i].byte_offset));
            if (sing_meas[i].avg_results_flag)
		float_eval /= eval->num_queries;
	    else if (sing_meas[i].avg_rel_results_flag && eval->num_rel > 0)
		/* average over number of rel docs instead of number queries */
		float_eval /= eval->num_rel;
	    else if (sing_meas[i].gm_results_flag) {
		/* computing geometric mean instead of mean */
		if (!is_single_query_flag && epi->average_complete_flag)
		    /* Must patch up averages for any missing queries, since */
		    /* value of 0 means perfection */
		    float_eval += (eval->num_queries - eval->num_orig_queries)*
			log (MIN_GEO_MEAN);
		float_eval = (float) exp ((double) (float_eval /
						    eval->num_queries));
	    }
	    (void) sprintf (temp_buf, "%-15s\t%s\t%6.4f\n",
			    sing_meas[i].name, q_buf, float_eval);
        }
        if (UNDEF == add_buf_string (temp_buf, out_p))
            return;
    }

    for (i = 0; i < num_param_meas; i++) {
        if ((! param_meas[i].print_short_flag) && (! epi->all_flag))
            continue;
        if (param_meas[i].print_time_flag && (!epi->time_flag))
            continue;
	if (param_meas[i].print_only_query_flag && (!is_single_query_flag))
            continue;
	if (param_meas[i].print_only_average_flag && (is_single_query_flag))
            continue;
        for (j = 0; j < param_meas[i].num_values; j++) {
            sprintf (name_buf, param_meas[i].format_string,
                     param_meas[i].get_param_str (epi, j));
            if (param_meas[i].is_long_flag) {
		long_eval = ((long *) (((char *) eval) + 
					param_meas[i].byte_offset))[j];
                if (param_meas[i].avg_results_flag)
		    long_eval /= eval->num_queries;
		(void) sprintf (temp_buf, "%-15s\t%s\t%ld\n",
                                    name_buf, q_buf, long_eval);
            }
            else {
                float_eval = ((float *) (((char *) eval) +
                                         param_meas[i].byte_offset))[j];
                if (param_meas[i].avg_results_flag)
		    float_eval /= eval->num_queries;
		(void) sprintf (temp_buf, "%-15s\t%s\t%6.4f\n",
				name_buf, q_buf, float_eval);
            }
            if (UNDEF == add_buf_string (temp_buf, out_p))
                return;
        }
    }

    if (! is_single_query_flag) {
	long denom_long_eval;
	for (i = 0; i < num_micro_meas; i++) {
	    if ((! micro_meas[i].print_short_flag) && (! epi->all_flag))
		continue;
	    long_eval = *((long *) (((char *) eval) + 
				micro_meas[i].numerator_byte_offset));
	    denom_long_eval = *((long *) (((char *) eval) + 
				micro_meas[i].denominator_byte_offset));
	    float_eval = (float) long_eval / (float) denom_long_eval;
	    (void) sprintf (temp_buf, "%-15s\t%s\t%6.4f\n",
			    micro_meas[i].name, q_buf, float_eval);
            if (UNDEF == add_buf_string (temp_buf, out_p))
                return;
	}
    }

    if (output == NULL) {
        (void) fwrite (out_p->buf, 1, out_p->end, stdout);
        out_p->end = 0;
    }
}

static long cutoff[] = CUTOFF_VALUES;

void
old_print_trec_eval_list (epi, eval, num_runs, output)
EVAL_PARAM_INFO *epi;
TREC_EVAL *eval;
int num_runs;
SM_BUF *output;
{
    long i,j;
    char temp_buf[1024];
    SM_BUF *out_p;

    if (output == NULL) {
        out_p = &internal_output;
        out_p->end = 0;
    }
    else
        out_p = output;

    /* Print total numbers retrieved/rel for all runs */
    if (UNDEF == add_buf_string("\nQueryid (Num):\t", out_p))
        return;
    for (i = 0; i < num_runs; i++) {
        if (UNDEF == add_buf_string (eval->qid, out_p))
            return;
    }
    if (UNDEF == add_buf_string("\nTotal number of documents over all queries",
                                out_p))
        return;
    if (UNDEF == add_buf_string("\n    Retrieved:", out_p))
        return;
    for (i = 0; i < num_runs; i++) {
        (void) sprintf (temp_buf, "    %5ld", eval[i].num_ret);
        if (UNDEF == add_buf_string (temp_buf, out_p))
            return;
    }
    if (UNDEF == add_buf_string("\n    Relevant: ", out_p))
        return;
    for (i = 0; i < num_runs; i++) {
        (void) sprintf (temp_buf, "    %5ld", eval[i].num_rel);
        if (UNDEF == add_buf_string (temp_buf, out_p))
            return;
    }
    if (UNDEF == add_buf_string("\n    Rel_ret:  ", out_p))
        return;
    for (i = 0; i < num_runs; i++) {
        (void) sprintf (temp_buf, "    %5ld", eval[i].num_rel_ret);
        if (UNDEF == add_buf_string (temp_buf, out_p))
            return;
    }

    /* Print recall precision figures at NUM_RP_PTS recall levels */
    if (UNDEF == add_buf_string
        ("\nInterpolated Recall - Precision Averages:", out_p))
        return;
    for (j = 0; j < NUM_RP_PTS; j++) {
        (void) sprintf (temp_buf, "\n    at %4.2f     ",
                        (float) j / (NUM_RP_PTS - 1));
        if (UNDEF == add_buf_string (temp_buf, out_p))
            return;
        for (i = 0; i < num_runs; i++) {
            (void) sprintf (temp_buf, "  %6.4f ",
			    eval[i].int_recall_precis[j] /eval[i].num_queries);
            if (UNDEF == add_buf_string (temp_buf, out_p))
                return;
        }
   }

    /* Print average recall precision and percentage improvement */
    (void) sprintf (temp_buf,
                   "\nAverage precision (non-interpolated) for all rel docs(averaged over queries)\n                ");
    if (UNDEF == add_buf_string (temp_buf, out_p))
        return;
    for (i = 0; i < num_runs; i++) {
        (void) sprintf (temp_buf, "  %6.4f ",
			eval[i].av_recall_precis / eval[i].num_queries);
        if (UNDEF == add_buf_string (temp_buf, out_p))
            return;
    }
    if (num_runs > 1) {
        (void) sprintf (temp_buf, "\n    %% Change:           ");
        if (UNDEF == add_buf_string (temp_buf, out_p))
            return;
        for (i = 1; i < num_runs; i++) {
            (void) sprintf (temp_buf, "  %6.1f ",
                            (((eval[i].av_recall_precis / eval[i].num_queries)/
                              (eval[0].av_recall_precis / eval[i].num_queries))
                             - 1.0) * 100.0);
            if (UNDEF == add_buf_string (temp_buf, out_p))
                return;
        }
    }
    (void) sprintf (temp_buf, "\nPrecision:");
    if (UNDEF == add_buf_string (temp_buf, out_p))
        return;
    for (j = 0; j < NUM_CUTOFF; j++) {
        (void) sprintf (temp_buf, "\n  At %4ld docs:", cutoff[j]);
        if (UNDEF == add_buf_string (temp_buf, out_p))
            return;
        for (i = 0; i < num_runs; i++) {
            (void) sprintf (temp_buf, "   %6.4f",
			    eval[i].precis_cut[j] / eval[i].num_queries);
            if (UNDEF == add_buf_string (temp_buf, out_p))
                return;
        }
    }

    (void) sprintf (temp_buf, "\nR-Precision (precision after R (= num_rel for a query) docs retrieved):\n    Exact:     ");
    if (UNDEF == add_buf_string (temp_buf, out_p))
        return;
    for (i = 0; i < num_runs; i++) {
        (void) sprintf (temp_buf, "   %6.4f",
			eval[i].R_recall_precis / eval[i].num_queries);
        if (UNDEF == add_buf_string (temp_buf, out_p))
            return;
    }

    if (UNDEF == add_buf_string ("\n", out_p))
	return;
    
    if (output == NULL) {
	(void) fwrite (out_p->buf, 1, out_p->end, stdout);
	out_p->end = 0;
    }
    return;
}

