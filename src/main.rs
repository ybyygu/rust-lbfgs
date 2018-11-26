// main.rs
// :PROPERTIES:
// :header-args: :tangle src/main.rs
// :END:

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*main.rs][main.rs:1]]
#![feature(libc)]
#![feature(extern_types)]
#![feature(asm)]
#![feature(ptr_wrapping_offset_from)]
//#![feature(const_slice_as_ptr)]

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(mutable_transmutes)]
#![allow(unused_mut)]

pub mod lbfgs;

use libc;
extern "C" {
    #[no_mangle]
    fn printf(_: *const libc::c_char, ...) -> libc::c_int;
    /*
    A user must implement a function compatible with ::lbfgs_evaluate_t (evaluation
    callback) and pass the pointer to the callback function to lbfgs() arguments.
    Similarly, a user can implement a function compatible with ::lbfgs_progress_t
    (progress callback) to obtain the current progress (e.g., variables, function
    value, ||G||, etc) and to cancel the iteration process if necessary.
    Implementation of a progress callback is optional: a user can pass \c NULL if
    progress notification is not necessary.

    In addition, a user must preserve two requirements:
        - The number of variables must be multiples of 16 (this is not 4).
        - The memory block of variable array ::x must be aligned to 16.

    This algorithm terminates an optimization
    when:

        ||G|| < \epsilon \cdot \max(1, ||x||) .

    In this formula, ||.|| denotes the Euclidean norm.
    */
    /* *
     * Start a L-BFGS optimization.
     *
     *  @param  n           The number of variables.
     *  @param  x           The array of variables. A client program can set
     *                      default values for the optimization and receive the
     *                      optimization result through this array. This array
     *                      must be allocated by ::lbfgs_malloc function
     *                      for libLBFGS built with SSE/SSE2 optimization routine
     *                      enabled. The library built without SSE/SSE2
     *                      optimization does not have such a requirement.
     *  @param  ptr_fx      The pointer to the variable that receives the final
     *                      value of the objective function for the variables.
     *                      This argument can be set to \c NULL if the final
     *                      value of the objective function is unnecessary.
     *  @param  proc_evaluate   The callback function to provide function and
     *                          gradient evaluations given a current values of
     *                          variables. A client program must implement a
     *                          callback function compatible with \ref
     *                          lbfgs_evaluate_t and pass the pointer to the
     *                          callback function.
     *  @param  proc_progress   The callback function to receive the progress
     *                          (the number of iterations, the current value of
     *                          the objective function) of the minimization
     *                          process. This argument can be set to \c NULL if
     *                          a progress report is unnecessary.
     *  @param  instance    A user data for the client program. The callback
     *                      functions will receive the value of this argument.
     *  @param  param       The pointer to a structure representing parameters for
     *                      L-BFGS optimization. A client program can set this
     *                      parameter to \c NULL to use the default parameters.
     *                      Call lbfgs_parameter_init() function to fill a
     *                      structure with the default values.
     *  @retval int         The status code. This function returns zero if the
     *                      minimization process terminates without an error. A
     *                      non-zero value indicates an error.
     */
    #[no_mangle]
    fn lbfgs(
        n: libc::c_int,
        x: *mut lbfgsfloatval_t,
        ptr_fx: *mut lbfgsfloatval_t,
        proc_evaluate: lbfgs_evaluate_t,
        proc_progress: lbfgs_progress_t,
        instance: *mut libc::c_void,
        param: *mut lbfgs_parameter_t,
    ) -> libc::c_int;
    /* *
     * Initialize L-BFGS parameters to the default values.
     *
     *  Call this function to fill a parameter structure with the default values
     *  and overwrite parameter values if necessary.
     *
     *  @param  param       The pointer to the parameter structure.
     */
    #[no_mangle]
    fn lbfgs_parameter_init(param: *mut lbfgs_parameter_t);
    /* *
     * Allocate an array for variables.
     *
     *  This function allocates an array of variables for the convenience of
     *  ::lbfgs function; the function has a requreiemt for a variable array
     *  when libLBFGS is built with SSE/SSE2 optimization routines. A user does
     *  not have to use this function for libLBFGS built without SSE/SSE2
     *  optimization.
     *
     *  @param  n           The number of variables.
     */
    #[no_mangle]
    fn lbfgs_malloc(n: libc::c_int) -> *mut lbfgsfloatval_t;
    /* *
     * Free an array of variables.
     *
     *  @param  x           The array of variables allocated by ::lbfgs_malloc
     *                      function.
     */
    #[no_mangle]
    fn lbfgs_free(x: *mut lbfgsfloatval_t);
}
/*
 *      C library of Limited memory BFGS (L-BFGS).
 *
 * Copyright (c) 1990, Jorge Nocedal
 * Copyright (c) 2007-2010 Naoaki Okazaki
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
/* $Id$ */
/*__cplusplus*/
/*
 * The default precision of floating point values is 64bit (double).
 */
/*LBFGS_FLOAT*/
/*
 * Activate optimization routines for IEEE754 floating point values.
 */
/*LBFGS_IEEE_FLOAT*/
pub type lbfgsfloatval_t = libc::c_double;
/* *
 * L-BFGS optimization parameters.
 *  Call lbfgs_parameter_init() function to initialize parameters to the
 *  default values.
 */
#[derive(Copy, Clone)]
#[repr(C)]
pub struct lbfgs_parameter_t {
    pub m: libc::c_int,
    pub epsilon: lbfgsfloatval_t,
    pub past: libc::c_int,
    pub delta: lbfgsfloatval_t,
    pub max_iterations: libc::c_int,
    pub linesearch: libc::c_int,
    pub max_linesearch: libc::c_int,
    pub min_step: lbfgsfloatval_t,
    pub max_step: lbfgsfloatval_t,
    pub ftol: lbfgsfloatval_t,
    pub wolfe: lbfgsfloatval_t,
    pub gtol: lbfgsfloatval_t,
    pub xtol: lbfgsfloatval_t,
    pub orthantwise_c: lbfgsfloatval_t,
    pub orthantwise_start: libc::c_int,
    pub orthantwise_end: libc::c_int,
}
/* *
 * Callback interface to provide objective function and gradient evaluations.
 *
 *  The lbfgs() function call this function to obtain the values of objective
 *  function and its gradients when needed. A client program must implement
 *  this function to evaluate the values of the objective function and its
 *  gradients, given current values of variables.
 *
 *  @param  instance    The user data sent for lbfgs() function by the client.
 *  @param  x           The current values of variables.
 *  @param  g           The gradient vector. The callback function must compute
 *                      the gradient values for the current variables.
 *  @param  n           The number of variables.
 *  @param  step        The current step of the line search routine.
 *  @retval lbfgsfloatval_t The value of the objective function for the current
 *                          variables.
 */
pub type lbfgs_evaluate_t = Option<
    unsafe extern "C" fn(
        _: *mut libc::c_void,
        _: *const lbfgsfloatval_t,
        _: *mut lbfgsfloatval_t,
        _: libc::c_int,
        _: lbfgsfloatval_t,
    ) -> lbfgsfloatval_t,
>;
/* *
 * Callback interface to receive the progress of the optimization process.
 *
 *  The lbfgs() function call this function for each iteration. Implementing
 *  this function, a client program can store or display the current progress
 *  of the optimization process.
 *
 *  @param  instance    The user data sent for lbfgs() function by the client.
 *  @param  x           The current values of variables.
 *  @param  g           The current gradient values of variables.
 *  @param  fx          The current value of the objective function.
 *  @param  xnorm       The Euclidean norm of the variables.
 *  @param  gnorm       The Euclidean norm of the gradients.
 *  @param  step        The line-search step used for this iteration.
 *  @param  n           The number of variables.
 *  @param  k           The iteration count.
 *  @param  ls          The number of evaluations called for this iteration.
 *  @retval int         Zero to continue the optimization process. Returning a
 *                      non-zero value will cancel the optimization process.
 */
pub type lbfgs_progress_t = Option<
    unsafe extern "C" fn(
        _: *mut libc::c_void,
        _: *const lbfgsfloatval_t,
        _: *const lbfgsfloatval_t,
        _: lbfgsfloatval_t,
        _: lbfgsfloatval_t,
        _: lbfgsfloatval_t,
        _: lbfgsfloatval_t,
        _: libc::c_int,
        _: libc::c_int,
        _: libc::c_int,
    ) -> libc::c_int,
>;
unsafe extern "C" fn evaluate(
    mut instance: *mut libc::c_void,
    mut x: *const lbfgsfloatval_t,
    mut g: *mut lbfgsfloatval_t,
    n: libc::c_int,
    step: lbfgsfloatval_t,
) -> lbfgsfloatval_t {
    let mut i: libc::c_int = 0;
    let mut fx: lbfgsfloatval_t = 0.0f64;
    i = 0i32;
    while i < n {
        let mut t1: lbfgsfloatval_t = 1.0f64 - *x.offset(i as isize);
        let mut t2: lbfgsfloatval_t = 10.0f64
            * (*x.offset((i + 1i32) as isize) - *x.offset(i as isize) * *x.offset(i as isize));
        *g.offset((i + 1i32) as isize) = 20.0f64 * t2;
        *g.offset(i as isize) =
            -2.0f64 * (*x.offset(i as isize) * *g.offset((i + 1i32) as isize) + t1);
        fx += t1 * t1 + t2 * t2;
        i += 2i32
    }
    return fx;
}
unsafe extern "C" fn progress(
    mut instance: *mut libc::c_void,
    mut x: *const lbfgsfloatval_t,
    mut g: *const lbfgsfloatval_t,
    fx: lbfgsfloatval_t,
    xnorm: lbfgsfloatval_t,
    gnorm: lbfgsfloatval_t,
    step: lbfgsfloatval_t,
    mut n: libc::c_int,
    mut k: libc::c_int,
    mut ls: libc::c_int,
) -> libc::c_int {
    printf(
        b"Iteration %d:\n\x00" as *const u8 as *const libc::c_char,
        k,
    );
    printf(
        b"  fx = %f, x[0] = %f, x[1] = %f\n\x00" as *const u8 as *const libc::c_char,
        fx,
        *x.offset(0isize),
        *x.offset(1isize),
    );
    printf(
        b"  xnorm = %f, gnorm = %f, step = %f\n\x00" as *const u8 as *const libc::c_char,
        xnorm,
        gnorm,
        step,
    );
    printf(b"\n\x00" as *const u8 as *const libc::c_char);
    return 0i32;
}
unsafe fn main_0(mut argc: libc::c_int, mut argv: *mut *mut libc::c_char) -> libc::c_int {
    let mut i: libc::c_int = 0;
    let mut ret: libc::c_int = 0i32;
    let mut fx: lbfgsfloatval_t = 0.;
    let mut x: *mut lbfgsfloatval_t = lbfgs_malloc(100i32);
    let mut param: lbfgs_parameter_t = lbfgs_parameter_t {
        m: 0,
        epsilon: 0.,
        past: 0,
        delta: 0.,
        max_iterations: 0,
        linesearch: 0,
        max_linesearch: 0,
        min_step: 0.,
        max_step: 0.,
        ftol: 0.,
        wolfe: 0.,
        gtol: 0.,
        xtol: 0.,
        orthantwise_c: 0.,
        orthantwise_start: 0,
        orthantwise_end: 0,
    };
    if x.is_null() {
        printf(
            b"ERROR: Failed to allocate a memory block for variables.\n\x00" as *const u8
                as *const libc::c_char,
        );
        return 1i32;
    } else {
        /* Initialize the variables. */
        i = 0i32;
        while i < 100i32 {
            *x.offset(i as isize) = -1.2f64;
            *x.offset((i + 1i32) as isize) = 1.0f64;
            i += 2i32
        }
        /* Initialize the parameters for the L-BFGS optimization. */
        lbfgs_parameter_init(&mut param);
        /*param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;*/
        /*
           Start the L-BFGS optimization; this will invoke the callback functions
           evaluate() and progress() when necessary.
        */
        ret = lbfgs(
            100i32,
            x,
            &mut fx,
            Some(evaluate),
            Some(progress),
            0 as *mut libc::c_void,
            &mut param,
        );
        /* Report the result. */
        printf(
            b"L-BFGS optimization terminated with status code = %d\n\x00" as *const u8
                as *const libc::c_char,
            ret,
        );
        printf(
            b"  fx = %f, x[0] = %f, x[1] = %f\n\x00" as *const u8 as *const libc::c_char,
            fx,
            *x.offset(0isize),
            *x.offset(1isize),
        );
        lbfgs_free(x);
        return 0i32;
    };
}
pub fn main() {
    let mut args: Vec<*mut libc::c_char> = Vec::new();
    for arg in ::std::env::args() {
        args.push(
            ::std::ffi::CString::new(arg)
                .expect("Failed to convert argument into CString.")
                .into_raw(),
        );
    }
    args.push(::std::ptr::null_mut());
    unsafe {
        ::std::process::exit(main_0(
            (args.len() - 1) as libc::c_int,
            args.as_mut_ptr() as *mut *mut libc::c_char,
        ) as i32)
    }
}
// main.rs:1 ends here
