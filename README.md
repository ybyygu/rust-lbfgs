
# LBFGS

This repo contains my ongoing efforts to port Naoaki Okazaki's C library
[libLBFGS](http://chokkan.org/software/liblbfgs/) to Rust. Check [rust-liblbfgs](https://github.com/ybyygu/rust-liblbfgs) for a working wrapper around the original
C codes.

[![Build Status](https://travis-ci.org/ybyygu/gchemol.svg?branch=master)](https://travis-ci.org/ybyygu/gchemol)
[![GPL3 licensed](https://img.shields.io/badge/license-GPL3-blue.svg)](./LICENSE)


# Motivation

-   Bring native LBFGS implementation to Rust community.
-   Learn how a great optimization algorithm is implemented in real world.
-   Learn how to "replace the jet engine while still flying" [URL](http://jensimmons.com/post/jan-4-2017/replacing-jet-engine-while-still-flying)
-   Make it more maintainable with Rust high level abstraction.
-   Improve it to meet my needs in computational chemistry.

