# PVS2IMANDRA: A PVS->Imandra transpiler for executable specs

PVS -> Imandra conversion happens in two steps:
1. `pvs2imandra.lisp` is used to produce an S-expression representation of a PVS theory
2. `pvs2imandra.ml` processes this S-expression and converts it into IML (Imandra Modeling Language)

Example:

Consider the following PVS theory `sum2`:

```
sum2: THEORY
BEGIN

  n : VAR nat
  f,g : VAR [nat -> nat]

  sum(f,n) : RECURSIVE nat =
    IF n = 0 THEN
      0
    ELSE
      f(n-1) + sum(f, n - 1)
    ENDIF
  MEASURE n

  sum_plus : LEMMA
    sum((lambda n : f(n) + g(n)), n)
   = sum(f,n) + sum(g,n)

  square(n) : nat = n * n

  sum_of_squares : LEMMA
    6 * sum(square, n+1) = n * (n+1) * (2*n + 1)

  cube(n) : nat = n * n * n

  sum_of_cubes : LEMMA
    4 * sum(cube, n+1) = n*n*(n+1)*(n+1)

END sum2
```

With `sum2` loaded in PVS, we first use `pvs2imandra.lisp` to produce an S-expression representation of the theory:

```
pvs: (load "pvs2imandra.lisp")
pvs: (pvs2imandra-theory (get-theory "sum2"))
(defun sum (:sig ((:to int int) int) int) (f n ) (if (= n 0 ) 0 (+ (f (- n 1 ) ) (sum f (- n 1 ) ) )))
(defun square (:sig (int) int) (n ) (* n n ))
(defun cube (:sig (int) int) (n ) (* (* n n ) n ))
```

We next process these S-expressions in Imandra using `pvs2imandra.ml`:

```
> #mod_use "./pvs2imandra.ml";;
> Pvs2imandra.top "(defun sum (:sig ((:to int int) int) int) (f n ) (if (= n 0 ) 0 (+ (f (- n 1 ) ) (sum f (- n 1 ) ) )))
  ";;
- : Pvs2imandra.Decl.t =
Pvs2imandra.Decl.Defun
 {Pvs2imandra.Decl.name = "sum"; args = ["f";"n"];
  body =
   Pvs2imandra.Expr.If
    (Pvs2imandra.Expr.Apply ("=",
      [Pvs2imandra.Expr.Var "n";
       Pvs2imandra.Expr.Const (Pvs2imandra.Expr.Int 0)]),
    Pvs2imandra.Expr.Const (Pvs2imandra.Expr.Int 0),
    Some
     (Pvs2imandra.Expr.Apply ("+",
       [Pvs2imandra.Expr.Apply ("f",
         [Pvs2imandra.Expr.Apply ("-",
           [Pvs2imandra.Expr.Var "n";
            Pvs2imandra.Expr.Const (Pvs2imandra.Expr.Int 1)])]);
        Pvs2imandra.Expr.Apply ("sum",
         [Pvs2imandra.Expr.Var "f";
          Pvs2imandra.Expr.Apply ("-",
           [Pvs2imandra.Expr.Var "n";
            Pvs2imandra.Expr.Const (Pvs2imandra.Expr.Int 1)])])])))}
```

Note how we now have a structured IML AST for the relevant PVS function.
To convert this to a parseable IML string, we can use `Pvs2imandra.Decl.pp` (with an arbitrary formatter) or `Pvs2imandra.Decl.to_string` (to directly produce a parseable string):

```
> Pvs2imandra.Decl.to_string @@ Pvs2imandra.top "(defun sum (:sig ((:to int int) int) int) (f n ) (if (= n 0 ) 0 (+ (f (- n 1 ) ) (sum f (- n 1 ) ) ))) ";;
- : string =
"let rec sum f n = if (n = 0) then 0 else ((f (n - 1)) + (sum f (n - 1)))"
```

This can then be evaluated directly in Imandra using `Imandra.eval_string`:

```
> Imandra.eval_string @@ Pvs2imandra.Decl.to_string @@ Pvs2imandra.top "(defun sum (:sig ((:to int int) int) int) (f n ) (if (= n 0 ) 0 (+ (f (- n 1 ) ) (sum f (- n 1 ) ) ))) ";;
val sum : (Z.t -> Z.t) -> Z.t -> Z.t = <fun>
```

And we now have an executable IML/OCaml function which we may compute with:

```
> sum (fun x -> x + 1) 100;;
- : Z.t = 5050
```
