#ifndef TRECEVALH
#define TRECEVALH

/* Static state info; set at beginning, possibly from program options, */
/* but then remains constant throughout. */
typedef struct {
    long query_flag;              /* 0. If set, evaluation output will be
                                     printed for each query, in addition
                                     to summary at end. */
    long all_flag;                /* 0. If set, all evaluation measures will
                                     be printed instead of just the
                                     final TREC 2 measures. */
    long time_flag;               /* 0. If set, calculate time-based measures*/
    long relation_flag;           /* 1. If set, print in relational form */
    long average_complete_flag;   /* 0. If set, average over the complete set
				     of relevance judgements (qrels), instead
				     of the number of queries 
				     in the intersection of qrels and result */
    long judged_docs_only_flag;   /* 0. If set, throw out all unjudged docs
				     for the retrieved set before calculating
				     any measures. */
    double utility_a;             /* UTILITY_A. Default utility values */
    double utility_b;             /* UTILITY_B. Default utility values */
    double utility_c;             /* UTILITY_C. Default utility values */
    double utility_d;             /* UTILITY_D. Default utility values */
    long num_docs_in_coll;        /* 0. number of docs in collection */
    long relevance_level;         /* 1. In relevance judgements, the level at
				     which a doc is considered relevant for
				     this evaluation */
    long max_num_docs_per_topic;  /* MAXLONG. evaluate only this many docs */
} EVAL_PARAM_INFO;

/* Measure characteristics (how to print them, average them). */
/* List of measures is in measures.c */
/* Three types of measures:
       single measures - single measure and name
       parameterized measures - arrays of a measure, whose measure name
              depends on parameter (eg P5, P10)
       micro measures - measures defined as the micro average over all
              docs retrieved independent of topic. Only calculated
	      and printed for the "all" pseudo-query.
	      Eg  micro_prec = num_rel_ret / num_ret
*/
typedef struct {
    char *name;
    char *long_name;
    unsigned char is_long_flag;      /* otherwise float */
    unsigned char print_short_flag;  /* if set, measure is always printed
					(not just if all_flag set) */
    unsigned char print_time_flag;   /* if set, measure is printed only
					if time_flag is set */
    unsigned char print_only_query_flag; /* if set, measure is printed only
				        when printing individual query output*/
    unsigned char print_only_average_flag; /* if set, measure is printed only
				        when printing overall average output*/
    unsigned char avg_results_flag;  /* if set, average results over queries */
    unsigned char avg_rel_results_flag;/* if set,average results over num_rel*/
    unsigned char gm_results_flag;   /* if set, measure uses geometric mean. ie
				        exponentiate the average before
				        printing */
    long byte_offset;
} SINGLE_MEASURE;

typedef struct {
    char *long_name;
    unsigned char is_long_flag;      /* otherwise float */
    unsigned char print_short_flag;  /* if set, print in short output */
    unsigned char print_time_flag;   /* if set, measure is printed only
					if time_flag is set */
    unsigned char print_only_query_flag; /* if set, measure is printed only
				        when printing individual query output*/
    unsigned char print_only_average_flag; /* if set, measure is printed only
				        when printing overall average output*/
    unsigned char avg_results_flag;  /* if set, average results over queries */
    long byte_offset;
    long num_values;
    char *format_string;
    char *long_format_string;
    char *(*get_param_str) (EVAL_PARAM_INFO *ip, long index);
} PARAMETERIZED_MEASURE;

typedef struct {
    char *name;
    char *long_name;
    unsigned char print_short_flag;  /* if set, measure is always printed
					(not just if all_flag set) */
    long numerator_byte_offset;
    long denominator_byte_offset;
} MICRO_MEASURE;


typedef struct {                    /* For each retrieved document result */
    char *docno;                       /* document id */
    float sim;                         /* score */
    long rank;                         /* rank assigned after breaking ties */
} TEXT_TR;

typedef struct {                    /* For each query in retrieved results */
    char *qid;                         /* query id */
    long num_text_tr;                  /* number of TEXT_TR results for query*/
    long max_num_text_tr;              /* number results space reserved for */
    TEXT_TR *text_tr;                  /* Array of TEXT_TR results */
} TREC_TOP;

typedef struct {                    /* Overall retrieved results */
    char *run_id;                      /* run id */
    long num_q_tr;                     /* Number of TREC_TOP queries */
    long max_num_q_tr;                 /* Num queries space reserved for*/
    TREC_TOP *trec_top;                /* Array of TREC_TOP query results */
} ALL_TREC_TOP;

typedef struct {                    /* For each relevance judgement */
    char *docno;                       /* document id */
    long rel;                          /* document judgement */
} TEXT_QRELS;

typedef struct {                    /* For each query in rel judgements */
    char *qid;                         /* query id */
    long num_text_qrels;               /* number of judged documents */
    long max_num_text_qrels;           /* Num docs space reserved for */
    TEXT_QRELS *text_qrels;            /* Array of judged TEXT_QRELS */
} TREC_QRELS;

typedef struct {                    /* Overall relevance judgements */
    long num_q_qrels;                  /* Number of TREC_QRELS queries */
    long max_num_q_qrels;              /* Num queries space reserved for */
    TREC_QRELS *trec_qrels;            /* Array of TREC_QRELS queries */
} ALL_TREC_QRELS;



#define INIT_NUM_QUERIES 50
#define INIT_NUM_RESULTS 1000
#define INIT_NUM_RELS 2000

/* Non standard values for tr_vec->rel field */
#define RELVALUE_NONPOOL -1
#define RELVALUE_UNJUDGED -2


/* Set retrieval is based on contingency table:
                      relevant  nonrelevant
    retrieved            a          b
    nonretrieved         c          d

    Often you see r == num_rel_ret == a
                  R == num_rel     == a+c
		  n == num_ret     == a+b
		  N == num_docs    == a+b+c+d
    Some of these definitions are used in comments below
*/


/* ----------------------------------------------- */
/* Defined constants that are collection/purpose dependent */

/* Number of cutoffs for recall,precision, and rel_precis measures. */
/* CUTOFF_VALUES gives the number of retrieved docs that these */
/* evaluation mesures are applied at. */
#define NUM_CUTOFF  9
#define CUTOFF_VALUES  {5, 10, 15, 20, 30, 100, 200, 500, 1000}

/* Maximum fallout value, expressed in number of non-rel docs retrieved. */
/* (Make the approximation that number of non-rel docs in collection */
/* is equal to the number of number of docs in collection) */
#define MAX_FALL_RET  142

/* Maximum multiple of R (number of rel docs for this query) to calculate */
/* R-based precision at */
#define MAX_RPREC 2.0

#define MAX_TIME 300.0
#define NUM_TIME_PTS 60

/* Set a maximum number of nonrel docs to be used for preference measures */
#define PREF_TOP_NONREL_NUM 100

/* ----------------------------------------------- */
/* Defined constants that are collection/purpose independent.  If you
   change these, you probably need to change comments and documentation,
   and some variable names may not be appropriate any more! */
#define NUM_RP_PTS  11
#define THREE_PTS {2, 5, 8}    
#define NUM_FR_PTS  11
#define NUM_PREC_PTS 11
#define UTILITY_A 1.0
#define UTILITY_B -1.0
#define UTILITY_C 0.0
#define UTILITY_D 0.0

#define MIN_GEO_MEAN .00001
#define INFAP_EPSILON .00001

typedef struct {
    char  *qid;                     /* query id  */
    long num_queries;               /* Number of queries for this eval */
    long num_orig_queries;          /* Number of queries for this eval without
				       missing values, if using trec_eval -c */
    /* Summary Numbers over all queries */
    long num_rel;                   /* Number of relevant docs */
    long num_ret;                   /* Number of retrieved docs */
    long num_rel_ret;               /* Number of relevant retrieved docs */
    long num_nonrel_judged_ret;     /* Number of non-relevant retrieved
				       judged docs */
    float avg_doc_prec;             /* Average of precision over all
                                       relevant documents (query independent)*/

    /* Measures after num_ret docs */
    float exact_recall;             /* Recall after num_ret docs */
    float exact_precis;             /* Precision after num_ret docs */
    float exact_rel_precis;         /* Relative Precision (or recall) */
                                    /* Defined to be precision / max possible
                                       precision */
    float exact_uap;                /* Unranked Average Precision */
                                    /* Every rel doc in retrieved set gets
				       precision, every nonret rel doc gets 0.
				       Average over all rel docs */
                                    /* Note this = exact_recall *
				       exact_precision for a query */
                                    /* Preferred measure for evaluation of
				       unranked sets of arbitrary size. */
    float exact_rel_uap;            /* Relative Unranked Average Precision */
                                    /* Above, but relativized given size of 
				       retrieved set */
                                    /* If (n<R) set num_rel to n
				       If (n>R) set num_ret to R
				       Then use uap formula */
                                    /* exact_rel_precis ** 2 */
    float exact_utility;            /* From contingency table, by default:
				       UTILITY_A * a + UTILITY_B * b +
				       UTILITY_C * c + UTILITY_D * d.
				       By default, a-b (or r - (n-r)) */
    float recip_rank;               /* reciprical rank of top retrieved
				       relevant document */
    long rank_first_rel;            /* Rank of top retrieved rel doc. Set to
				       0 if none. Unaveraged */

    /* Measures after each document */
    float recall_cut[NUM_CUTOFF];   /* Recall after cutoff[i] docs */

    float precis_cut[NUM_CUTOFF];   /* precision after cutoff[i] docs. If
                                       less than cutoff[i] docs retrieved,
                                       then assume an additional 
                                       cutoff[i]-num_ret non-relevant docs
                                       are retrieved. */
    float rel_precis_cut[NUM_CUTOFF];/* Relative precision after cutoff[i] 
                                       docs. (Note relative precision is
				       identical to relative recall) */
    float uap_cut[NUM_CUTOFF];       /* uap (is recall * precision) after 
                                       cutoff[i] docs. Not recommended  */
    float rel_uap_cut[NUM_CUTOFF];   /* rel_uap at cutoff[i] docs */
    float av_rel_precis;             /* average (integral) of rel_precis
					after each doc. Do not use if
					number of docs retrieved varies */
    float av_rel_uap;                /* average (integral) of rel_uap
					after each doc. Do not use if
					number of docs retrieved varies */


    /* Measures after each rel doc */
    float av_recall_precis;         /* MAP! average(integral) of precision at
                                       all rel doc ranks. THE MAJOR
				       EVALUATION MEASURE FOR RANKED DOCS */
    float int_av_recall_precis;     /* Same as above, but the precision values
                                       have been interpolated, so that prec(X)
                                       is actually MAX prec(Y) for all 
                                       Y >= X   */
    float int_recall_precis[NUM_RP_PTS];/* interpolated precision at 
                                       0.1 increments of recall */
    float int_av3_recall_precis;    /* interpolated average at 3 intermediate 
                                       points */
    float int_av11_recall_precis;   /* interpolated average at NUM_RP_PTS 
                                       intermediate points (recall_level) */

    /* Measures after each non-rel doc */
    float fall_recall[NUM_FR_PTS];  /* max recall after each non-rel doc,
                                       at 11 points starting at 0.0 and
                                       ending at MAX_FALL_RET /num_docs */
    float av_fall_recall;           /* Average of fallout-recall, after each
                                       non-rel doc until fallout of 
                                       MAX_FALL_RET / num_docs achieved */

    /* Measures after R-related cutoffs.  R is the number of relevant
     docs for a particular query, but note that these cutoffs are after
     R docs, whether relevant or non-relevant, have been retrieved.
     R-related cutoffs are really only applicable to a situtation where
     there are many relevant docs per query (or lots of queries). */
    float R_recall_precis;          /* Recall or precision after R docs
                                       (note they are equal at this point) */
    float av_R_precis;              /* Average (or integral) of precision at
                                       each doc until R docs have been 
                                       retrieved */
    float R_prec_cut[NUM_PREC_PTS]; /* Precision measured after multiples of
                                       R docs have been retrieved. 10 
                                       equal points, with max multiple
                                       having value MAX_RPREC */
    float int_R_recall_precis;      /* Interpolated precision after R docs
                                       Prec(X) = MAX(prec(Y)) for all Y>=X */
    float int_av_R_precis;          /* Interpolated */
    float int_R_prec_cut[NUM_PREC_PTS]; /* Interpolated */

    /* Measures after particular time relative to size of eventual retrieved 
       set.  Eg, precision is num_rel_so_far/num_ret
                 relprecision is num_rel_so_far/MIN(num_ret,num_rel)
		 uap is num_rel_so_far**2/(num_ret*MIN(num_ret,num_rel))
                 reluap is relprecision * relprecision */
    float time_num_rel[NUM_TIME_PTS]; /* Number of rel docs in time bucket*/
    float time_num_nrel[NUM_TIME_PTS];/* Number of nrel docs in each bucket*/
    float time_cum_rel[NUM_TIME_PTS]; /* Cumulative time_num_rel */
    float time_precis[NUM_TIME_PTS];  /* First Precision in each bucket */
    float time_relprecis[NUM_TIME_PTS];/* First rel-Precision in each bucket */
    float time_uap[NUM_TIME_PTS];     /* First uap in bucket*/
    float time_reluap[NUM_TIME_PTS];  /* First relative uap in bucket*/
    float time_utility[NUM_TIME_PTS]; /* First Utility (default 1,-1,0,0) 
					 in bucket */
    float av_time_precis;            /* Sum (integral) of time_precis */
    float av_time_relprecis;         /* Sum (integral) of time_relprecis */
    float av_time_uap;               /* Sum (integral) of time_uap */
    float av_time_reluap;            /* Sum (integral) of time_reluap */
    float av_time_utility;           /* Sum (integral) of time_utility */
    float av_time_cum_rel;           /* Sum (integral) of time_cum_rel */

    /* Measures dependent on only judged documents */
    /* Binary Pref relations: fraction of nonrel documents retrieved after 
       each rel doc */
    float bpref;                     /* real BPREF.  Top num_rel nonrel docs */
    float bpref_top5Rnonrel;         /* Top 5 * num_rel nonrel docs */
    float bpref_top10Rnonrel;        /* Top 10 * num_rel nonrel docs */
/*    float bpref_topRnonrel;         *  renamed as bpref */
    float bpref_allnonrel;           /* all judged nonrel docs */
    float bpref_retnonrel;           /* Only retrieved nonrel docs */
    float bpref_topnonrel;           /* Top PREF_TOPNREL_NUM nonrel docs */
    float bpref_top50pRnonrel;       /* Top 50 + num_rel nonrel docs */
    float bpref_top25pRnonrel;       /* Top 25 + num_rel nonrel docs */
    float bpref_top10pRnonrel;       /* Top 10 + num_rel nonrel docs.
                                        Bad version used in SIGIR 2004 paper */
    float old_bpref_top10pRnonrel;   /* bad old version. Top 10 + num_rel 
                                        nonrel docs. Used in SIGIR 2004 paper*/
    float bpref_top25p2Rnonrel;      /* Top 25 + 2 * num_rel nonrel docs */
    float bpref_retall;              /* Only retrieved rel,nonrel docs */
    float bpref_5;                   /* Only top 5 rel, top 5 nonrel */
    float bpref_10;                  /* Only top 10 rel, top 10 nonrel */
    float old_bpref;                 /* Bad old bpref. Top num_rel nonrel docs.
					Only used retrieved nonrel docs.
					Used in TREC 12,13, mention in 
					SIGIR 2004 paper */
    float bpref_num_all;             /* num not retrieved before (all judged)*/
    float bpref_num_ret;             /* num retrieved after */
    long  bpref_num_correct;         /* num correct preferences */
    long  bpref_num_possible;        /* num possible correct preferences */

    /* Measures that allow sampling of judgement pool: Qrels/results divided 
       into unpooled, pooled_but_unjudged, pooled_judged_rel, 
       pooled_judged_nonrel. */
    /* Inf_ap: "Estimating Average Precision with Incomplete and Imperfect 
       Judgments", Emine Yilmaz and Javed A. Aslam.
       My intuition of it: Calculate P at rel doc using higher retrieved judged
       docs, then average in 0's from higher pooled docs. */
    float inf_ap;                    /* Inferred AP.  see Aslam et al, 
					Estimating Average Precision with
					Incomplete Information */

    /* Measures that use Geometric Mean
       avg_Score = exp (SUM (log (MAX (query_score, .00001))) / N)     
       WARNING: Geometric Mean measures special cased for "trec_eval -c".
       Works, but be careful when implementing new measure */
    float gm_ap;                     /* Geometric Mean version of MAP */
    float gm_bpref;                  /* Geometric Mean version of bpref.  Note
					bpref has lots of 0.0 values */

} TREC_EVAL;

#endif /* TRECEVALH */
