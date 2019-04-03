
# v0.0.14

-   <span class="timestamp-wrapper"><span class="timestamp">[2019-03-22 Fri] </span></span> Damped LBFGS algorithm
-   <span class="timestamp-wrapper"><span class="timestamp">[2019-03-22 Fri] </span></span> Gradient only optimization.


# v0.0.13

-   <span class="timestamp-wrapper"><span class="timestamp">[2019-01-02 Wed] </span></span> new example: optimization of lj38 cluster
-   <span class="timestamp-wrapper"><span class="timestamp">[2018-12-21 Fri] </span></span> fixed progress report issue.


# v0.0.12

-   <span class="timestamp-wrapper"><span class="timestamp">[2018-12-18 Tue] </span></span> new parameter to control max allowed evaluations: max\_evaluations
-   <span class="timestamp-wrapper"><span class="timestamp">[2018-12-18 Tue] </span></span> force to set progress callback function in minimize method.


# v0.0.11

-   <span class="timestamp-wrapper"><span class="timestamp">[2018-12-17 Mon] </span></span> assign lbfgs parameters using builder pattern
-   <span class="timestamp-wrapper"><span class="timestamp">[2018-12-16 Sun] </span></span> fixed important issues inherited from liblibfgs (#2, #3)
-   <span class="timestamp-wrapper"><span class="timestamp">[2018-12-15 Sat] </span></span> remove wolfe parameter from LineSearch (use gtol instead)
-   <span class="timestamp-wrapper"><span class="timestamp">[2018-12-14 Fri] </span></span> LineSearchParm is renamed as LineSearch


# v0.0.10

-   <span class="timestamp-wrapper"><span class="timestamp">[2018-12-12 Wed] </span></span> add Orthantwise struct to represent all orthantwise parameters (see: param.owlqn)
-   <span class="timestamp-wrapper"><span class="timestamp">[2018-12-12 Wed] </span></span> use param.orthantwise option to enable/disable OWL-QN algorithm
-   <span class="timestamp-wrapper"><span class="timestamp">[2018-12-11 Tue] </span></span> new line mod


# v0.0.9

-   <span class="timestamp-wrapper"><span class="timestamp">[2018-12-11 Tue] </span></span> add math mod
-   <span class="timestamp-wrapper"><span class="timestamp">[2018-12-10 Mon] </span></span> remove all unsafe codes


# v0.0.8

-   <span class="timestamp-wrapper"><span class="timestamp">[2018-12-10 Mon] </span></span> update quicli to v0.4
-   <span class="timestamp-wrapper"><span class="timestamp">[2018-12-06 Thu] </span></span> add test for OWL-QN
-   <span class="timestamp-wrapper"><span class="timestamp">[2018-12-05 Wed] </span></span> clean up some unsafe codes


# v0.0.7

-   <span class="timestamp-wrapper"><span class="timestamp">[2018-11-26 Mon] </span></span> Rust codes translated from liblbfgs/c using c2rust


# v0.0.6

-   <span class="timestamp-wrapper"><span class="timestamp">[2018-11-11 Sun] </span></span> new construct method with epsilon
    
        let lbfgs = LBFGS::new(fmax);


# v0.0.5

-   <span class="timestamp-wrapper"><span class="timestamp">[2018-11-16 Fri] </span></span> callback interfaces to liblbfgs/c using closures

