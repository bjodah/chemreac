Development
===========

As an open-source project other researchers/students are encouraged to improve upon
the code/documentation. The code is released under the permissive `BSD 2-Clause
License <http://opensource.org/licenses/BSD-2-Clause>`_. The source
code repository is found at https://github.com/bjodah/chemreac which is also
where issues and patches ("pull requests") are posted.


Coding standards
----------------
The Python code should comply with `PEP8
<http://legacy.python.org/dev/peps/pep-0008/>`_. It is also checked by
the CI.
For the C++ code, `PyNE's C/C++ style guide <http://pyne.io/devsguide/style_guide.html#c-c-style-guide>`_ is recommended.

Documentation standard
----------------------
The API docs are generated automatically and `numpydoc
<https://pypi.python.org/pypi/numpydoc>`_ is used to 
automatically parse docstrings of functions and classes. 
See the `NumPy/SciPy documentation guide
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_ for more info.

Unit tests
----------
The correctness of the implementation is verified through the use of
(hundreds of) unit tests. Ideally all code should be covered with tests.

Continuous integration
~~~~~~~~~~~~~~~~~~~~~~
.. include:: ../README.rst
    :start-after: .. ci-start
    :end-before: .. ci-end

Test coverage
~~~~~~~~~~~~~

.. image:: https://coveralls.io/repos/bjodah/chemreac/badge.png?branch=master
   :target: https://coveralls.io/r/bjodah/chemreac?branch=master
   :alt: Test coverage

unit testing is a great tool provided they actually cover the majority
of the code base. In order to keep track of the coverage coveralls is
used (click the badge above to see detailed per-line breakdown).

Environment
-----------
The documentation assumes a \*NIX like environment, or at
least familiarity with this environment for tweaking build scripts
etc. Feedback from users and developers on other platforms are most
welcome.

If you are new to github and the like, a good start is SymPy's wiki
entry on `development workflow
<https://github.com/sympy/sympy/wiki/Development-workflow>`_ (minus
the SymPy specific parts obviously).

Example of setting up a development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here is an example to get you started developing chemreac and contributing
to the code:

1. `Fork <https://help.github.com/articles/fork-a-repo/>`_ the repository on github
2. Install the dependencies specified in the README.rst
3. Clone you own repo:

::

    $ git clone https://bitbucket.org/YOUR_GITHUB_USERNAME/chemreac.git

4. Set-up the pre-commit hook.
::

    $ cd .git/hooks
    $ ln -s ../../scripts/pre-commit.sh pre-commit
    $ cd -

5. Create a new branch
::

    $ git checkout -b fix_silly_bug

6. Edit files, add test(s), make sure all tests pass.

7. make a commit, and push your changes:
::

    $ git commit -am "Fixed a silly bug in the complex-thingy."
    $ git push

7. Go to your forked repo on github and create a `pull-request <https://help.github.com/articles/using-pull-requests/>`_ from there.

