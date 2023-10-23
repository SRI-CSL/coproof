(* Convert PVS to Imandra, from Sexpr IR (produced by pvs2imandra.lisp) to IML *)
(* G.Passmore, Imandra Inc. *)

open Sexplib

exception Trans_error of Sexp.t

module Id = struct

  type t = string

  let (=) = (=)

  let pp fmt = CCFormat.fprintf fmt "%s"

end

module Expr = struct

  type t =
    | Var of Id.t
    | Const of const
    | If of t * t * t option
    | Apply of Id.t * t list

  and const =
    | Int of Z.t
    | String of string
    | Rat of Q.t

  let infix_table = [
    "+"; "-"; "*"; "="; "/"; "&&"; "||"; "==>"
  ]

  let rec pp fmt e =
    match e with
    | Var v -> Id.pp fmt v
    | Const (Int n) -> Z.pp_print fmt n
    | Const (String s) -> CCFormat.fprintf fmt "%S" s
    | Const (Rat q) -> Q.pp_print fmt q
    | If (c, a, Some b) ->
      CCFormat.fprintf fmt "@[if @[%a@] then @[@[%a@]@] else @[@[%a@]@]@]"
        pp c pp a pp b
    | If (c, a, None) ->
      CCFormat.fprintf fmt "@[if @[%a@] then @[@[%a@]@]@]"
        pp c pp a
    | Apply (f, [x;y]) when CCList.mem f infix_table ->
      CCFormat.fprintf fmt "@[(%a %a %a)@]"
        pp x Id.pp f pp y
    | Apply (f, args) ->
      CCFormat.fprintf fmt "@[(%a %a)@]"
        Id.pp f CCFormat.(list ~sep:(return " ") pp) args

  let args_of_sexp args =
    let open Sexp in
    let proj a = match a with
      | Atom s -> Some s
      | _ -> None
    in
    CCList.filter_map proj args

  let contains_f name body =
    let rec aux = function
      | Var _ -> false
      | Const _ -> false
      | If (c, a, Some b) ->
        aux c || aux a || aux b
      | If (c, a, None) ->
        aux c || aux a
      | Apply (f, args) ->
        Id.(f = name) || CCList.exists aux args
    in
    aux body

  let rec of_sexp e =
    let open Sexp in
    match e with
    | List [Atom "if"; c; a; b] ->
      If (of_sexp c,
               of_sexp a,
               Some (of_sexp b))
    | List (Atom f :: args) ->
      Apply (f, CCList.map of_sexp args)
    | Atom v ->
      begin try
        let n = Z.of_string v in
        Const (Int n)
      with Invalid_argument _ ->
        Var v
      end
    | _ -> failwith (CCFormat.sprintf "of_sexpr: %a" Sexp.pp e)

end

module Decl = struct

  type t =
    | Defun of {
        name: Id.t;
        args: Id.t list;
        body: Expr.t
      }

  let is_rec name body =
    Expr.contains_f name body

  let pp fmt = function
    | Defun {name; args; body} ->
      CCFormat.fprintf fmt "let%s %a %a = %a@."
        (if is_rec name body then " rec" else "")
        Id.pp name
        CCFormat.(list ~sep:(return " " )Id.pp) args
        Expr.pp body

  let to_string (d:t) : string =
    CCFormat.to_string pp d

end

(* Ex: (pvs2imandra-theory (get-theory "sum")) *)
let ex_0 = "(defun sum (:sig (int) int) (n) (if (= n 0 ) 0 (+ n (sum (- n 1 ) ) )))"

(* Ex: (pv2imandra-theory (get-theory "sum2")) *)
let ex_1 = "(defun square (:sig (int) int) (n) (* n n ))"
let ex_2 = "(defun cube (:sig (int) int) (n) (* (* n n ) n ))"
let ex_3 = "(defun sum (:sig ((:to int int) int) int) (f n) (if (= n 0 ) 0 (+ (f (- n 1 ) ) (sum f (- n 1 ) ) )))"

let top (s:string) : Decl.t =
  let open Sexp in
  match Sexp.of_string s with
  | List [
      Atom "defun";
      Atom fun_name;
      List fun_sig;
      List args;
      body
    ] ->
    Decl.Defun {
      name = fun_name;
      args = Expr.args_of_sexp args;
      body = Expr.of_sexp body;
    }
  | _ -> failwith "top"
