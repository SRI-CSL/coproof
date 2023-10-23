;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;PVS to Imandra Translator
;;
;; based on
;;
;;PVS to Clean Translator (version 0, Jan 20, 2006)
;;
;;  by Ronny Wichers Schreur and Natarajan Shankar
;;
;;Imandra modifications by Grant Passmore (March, 2022)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;Globals: *imandra-record-defns* records Imandra record type definitions
;;         *imandra-nondestructive-hash* records translations hashed by PVS decl
;;         *imandra-destructive-hash* records destructive translation
;;         *livevars-table* (shadowed) maintains update analysis
;;Top level function is pvs2imandra(expr, context) which initializes globals and
;; calls pvs2imandra*.  The main cases are applications, which lead to
;;pvs-defn-application and update-expr which branches according to destructive and
;;non-destructive updates.  Unfinished work includes modules and datatypes.

;; --------------------------------------------------------------------
;; PVS
;; Copyright (C) 2006, SRI International.  All Rights Reserved.

;; This program is free software; you can redistribute it and/or
;; modify it under the terms of the GNU General Public License
;; as published by the Free Software Foundation; either version 2
;; of the License, or (at your option) any later version.

;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.

;; You should have received a copy of the GNU General Public License
;; along with this program; if not, write to the Free Software
;; Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
;; --------------------------------------------------------------------

(in-package :pvs)

(defvar *livevars-table* nil)
(defvar *imandra-record-defns* nil)

(defun pvs2imandra-primitive-op (name)
  name)

(defmacro pvs2imandra-error (msg &rest args)
  `(format t ,msg ,@args))

(defmacro pvsimandra_update (array index value)
  `(let ((update-op (if (and *destructive?* *livevars-table*)
			(format nil "pvsDestructiveUpdate")
			(format nil "pvsNonDestructiveUpdate"))))
       (format nil  "~a ~a ~a ~a" update-op ,array ,index ,value)))

(defvar *imandra-nondestructive-hash* (make-hash-table :test #'eq))
(defvar *imandra-destructive-hash* (make-hash-table :test #'eq))

(defstruct imandra-info
  id type definition analysis)

(defmacro imandra-hashtable ()
  `(if *destructive?* *imandra-destructive-hash* *imandra-nondestructive-hash*))

(defun imandra_id (op)
  (let ((hashentry (gethash (declaration op) (imandra-hashtable))))
    (when hashentry (imandra-info-id hashentry))))

(defun imandra_nondestructive_id (op)
  (let ((hashentry (gethash (declaration op) *imandra-nondestructive-hash*)))
    (when hashentry (imandra-info-id hashentry))))

(defun imandra_type (op)
  (let ((hashentry (gethash (declaration op) (imandra-hashtable))))
    (when hashentry (imandra-info-type hashentry))))

(defun imandra_definition (op)
  (let ((hashentry (gethash (declaration op) (imandra-hashtable))))
    (when hashentry (imandra-info-definition hashentry))))

(defun imandra_analysis (op)
  (let ((hashentry (gethash (declaration op) (imandra-hashtable))))
    (when hashentry (imandra-info-analysis hashentry))))

(defun mk-imandra-funcall (fun args)
  (format nil "(~a ~{~a ~})" fun args))

(defun pvs2imandra (expr &optional context)
  (let* ((*current-context* (or context *current-context*))
	 (*generate-tccs* 'none))
    (pvs2imandra* expr nil nil)))

(defmethod pvs2imandra* ((expr number-expr) bindings livevars)
  (declare (ignore bindings livevars))
  (number expr))

(defmacro pvs2imandra_tuple (args)
  `(format nil "(~{~a~^, ~})" ,args))

(defmethod pvs2imandra* ((expr tuple-expr) bindings livevars)
  (let ((args (pvs2imandra* (exprs expr) bindings livevars)))
    (pvs2imandra_tuple args)))

(defmethod pvs2imandra* ((expr record-expr) bindings livevars)
  (let* ((sorted-assignments (sort-assignments (assignments expr)))
	       (formatted-fields
	        (loop for entry in sorted-assignments
		            collect (format nil "~a = ~a"
			                          (caar (arguments entry))
			                          (pvs2imandra* (expression entry)
			                                        bindings livevars)))))
    (format nil "{~{~a~^, ~}}" formatted-fields)))

(defun matchlist (index length dummy)
  (if (eql index 0)
	(if (eql length 0)
	    (list dummy)
	    (cons dummy (enlist (1- length))))
      (cons '_ (matchlist (1- index)(1- length) dummy))))

(defun enlist (n)
  (if (eql n 0)
      nil
      (cons '_ (enlist (1- n)))))

(defmethod pvs2imandra* ((expr projection-application) bindings livevars)
  (let* ((ll (length (exprs expr)))
	 (dummy (gentemp "DDD"))
	 (match-list (pvs2imandra_tuple (matchlist (index expr) ll dummy)))
	 (expr-list (pvs2imandra* expr bindings livevars)))
    `(let ,match-list = ,expr-list in ,dummy)))
	


(defmethod pvs2imandra*  ((expr field-application) bindings livevars)
  "Create a FieldName"
  (let* ((clarg (pvs2imandra* (argument expr) bindings livevars))
	 (id (pvs2imandra-id (id expr) :lower)))
    (format nil "~a.~a" clarg id)))

(defmethod pvs2imandra* ((expr list) bindings livevars)
  (if (consp expr)
      (cons (pvs2imandra* (car expr) bindings
			(append (updateable-vars (cdr expr)) livevars))
	    (pvs2imandra* (cdr expr) bindings  ;;need car's freevars
			(append (updateable-vars (car expr)) ;;f(A, A WITH ..)
				livevars)))
      nil))

(defmethod pvs2imandra* ((expr application) bindings livevars)
  (with-slots (operator argument) expr
    (if (constant? operator)
	(if (pvs2cl-primitive? operator)
	    (pvs2imandra-primitive-app expr bindings livevars)
	    (if (datatype-constant? operator)
		(mk-funapp (pvs2imandra-resolution operator)
			   (pvs2imandra* (arguments expr) bindings livevars))
		(pvs2imandra-defn-application  expr bindings livevars)))
	(let ((imandra-op (pvs2imandra* operator bindings
				    (append (updateable-vars
					     argument)
					    livevars)))
	      (imandra-arg (pvs2imandra* argument bindings
				     (append
				      (updateable-free-formal-vars operator)
				      livevars))))
	  (if (imandra-updateable? (type operator))
	      (format nil "(pvsSelect ~a ~a)"
		imandra-op imandra-arg)
	      (mk-imandra-funcall imandra-op (list imandra-arg)))))))

(defun pvs2imandra-primitive-app (expr bindings livevars)
  (format nil "(~a ~{~a ~})"
    (pvs2imandra-primitive-op (operator expr))
    (pvs2imandra* (arguments expr) bindings livevars)))

(defun constant-formals (module)
  (loop for x in (formals module)
			 when (formal-const-decl? x)
			 collect (make-constant-from-decl x)))

(defun pvs2imandra-defn-application (expr bindings livevars)
  (with-slots (operator argument) expr
    (pvs2imandra-resolution operator)
    (let* ((actuals (expr-actuals (module-instance operator)))
	         (op-decl (declaration operator))
	         (args (arguments expr))
	         (imandra-args (pvs2imandra* (append actuals args) bindings livevars))
	         (op-bound-id (cdr (assoc op-decl bindings :key #'declaration))))
      (if *destructive?*
	        (let* ((defns (def-axiom op-decl))
		             (defn (when defns (args2 (car (last (def-axiom op-decl))))))
		             (def-formals (when (lambda-expr? defn)
				                        (bindings defn)))
		             (module-formals (unless (eq (module op-decl) (current-theory))
				                           (constant-formals (module op-decl))))
		             (alist (append (pairlis module-formals actuals)
				                        (when def-formals
				                          (pairlis def-formals args))))
		             (analysis (imandra_analysis operator))
		             (check (unless op-bound-id
			                    (check-output-vars analysis alist livevars))))
	          (format nil "(~a ~{~a ~})"
	                  (or op-bound-id
		                    (if check
		                        (imandra_id operator)
		                      (imandra_nondestructive_id operator)))
	                  imandra-args)
	          )
	      (format nil "(~a ~{~a ~})"
	              ;;should this be imandra_nondestructive_id ?
	              (or op-bound-id (imandra_id operator)) imandra-args)))))

(defun pvs2imandra-resolution (op)
  (let* ((op-decl (declaration op)))
    (pvs2imandra-declaration op-decl)))

(defun pvs2imandra-declaration (op-decl)
  (let ((nd-hashentry (gethash op-decl *imandra-nondestructive-hash*)))
    (when (null nd-hashentry)
      (let ((op-id (format nil "~a" (pvs2imandra-id (id op-decl)))))
	      (setf (gethash op-decl *imandra-nondestructive-hash*)
	            (make-imandra-info :id op-id))
	      (let* ((defns (def-axiom op-decl))
	             (defn (when defns (args2 (car (last (def-axiom op-decl))))))
	             (def-formals (when (lambda-expr? defn)
			                        (bindings defn)))
	             (def-body (if (lambda-expr? defn) (expression defn) defn))
	             (module-formals (constant-formals (module op-decl)))
	             (range-type (if def-formals (range (type op-decl))
			                       (type op-decl))))
	        (pvs2imandra-resolution-nondestructive
           op-decl
           (append module-formals def-formals)
					 def-body range-type))))))

(defun pvs2imandra-resolution-nondestructive (op-decl formals body range-type)
  (let* ((*destructive?* nil)
	       (bind-ids (pvs2imandra-make-bindings formals nil))
	       (cl-body (pvs2imandra* body
			                          (pairlis formals bind-ids)
			                          nil))
	       (cl-type (if (null formals)
		                  (format nil "~a" (pvs2imandra-type range-type))
		                (format nil "(:sig ~a ~a)"
			                      (loop for var in formals
			                            collect (format nil "~a" (pvs2imandra-type (type var))))
			                      (pvs2imandra-type range-type))))
	       (cl-defn (if (null bind-ids)
		                  (format nil "() ~a" cl-body)
		                (format nil "(~{~a ~}) ~a" bind-ids cl-body)))
	       (hash-entry (gethash op-decl *imandra-nondestructive-hash*)))
    (format t "(defun ~a ~a ~a)~%"
            (pvs2imandra-id (id op-decl)) cl-type cl-defn)
    (setf (imandra-info-type hash-entry)
	        cl-type
	        (imandra-info-definition hash-entry)
	        cl-defn
	        )))

(defmethod pvs2imandra* ((expr name-expr) bindings livevars)
  (let* ((decl (declaration expr))
	 (bnd (assoc  decl bindings :key #'declaration)))
    (assert (not (and bnd (const-decl? decl))))
    (if bnd
	(cdr bnd)
	(if (const-decl? decl)
	    (pvs2imandra-constant expr decl bindings livevars)
	    (let ((undef (undefined expr "Hit untranslateable expression ~a")))
	      `(funcall ',undef))))))

(defun pvs2imandra-constant (expr op-decl bindings livevars)
  (let* ((defns (def-axiom op-decl))
	 (defn (when defns (args2 (car (last (def-axiom op-decl))))))
	 (def-formals (when (lambda-expr? defn)
			(bindings defn))))
    (pvs2imandra-resolution expr)
    (if def-formals 
	(let ((eta-expansion
	       (make!-lambda-expr def-formals
		 (make!-application* expr
		   (loop for bd in def-formals
		      collect (mk-name-expr bd))))))
	  (pvs2imandra* eta-expansion bindings livevars))
	(let* ((actuals (expr-actuals (module-instance expr)))
	       (imandra-actuals (pvs2imandra* actuals bindings livevars)))
	  (format nil "(~a ~{ ~a~})" (imandra_nondestructive_id expr)
		  imandra-actuals)))))



(defun pvs2imandra-lambda (bind-decls expr bindings) ;;removed livevars
  (let* ((*destructive?* nil)
	 (bind-ids (pvs2imandra-make-bindings bind-decls bindings))
	 (cl-body (pvs2imandra* expr
			   (append (pairlis bind-decls bind-ids)
				   bindings)
			   nil)))
    (format nil "(lambda (~{~a ~}) ~a)" bind-ids cl-body)))

(defmethod pvs2imandra* ((expr lambda-expr) bindings livevars)
  (declare (ignore livevars))
  (let ((type (type expr))
	(imandra-expr (pvs2imandra-lambda (bindings expr) (expression expr) bindings)))
    (if (and (imandra-updateable? type)
	     (funtype? type))
	(format nil "(Function ~a ~a)" (array-bound type) imandra-expr)
	imandra-expr)))


(defmethod pvs2imandra* ((expr if-expr) bindings livevars)
  (cond ((branch? expr)
	 (let ((condition (condition expr))
	       (then-part (then-part expr))
	       (else-part (else-part expr)))
	 `(if ,(pvs2imandra* condition bindings
			   (append (updateable-vars then-part)
				   (append (updateable-vars else-part)
					   livevars)))
	      ,(pvs2imandra* (then-part expr) bindings livevars)
	      ,(pvs2imandra* (else-part expr) bindings livevars))))
	(t (call-next-method))))

(defmethod pvs2imandra* ((expr cases-expr) bindings livevars)
  (format nil "case ~a of ~{~%  ~a~}"
    (pvs2imandra* (expression expr) bindings livevars)
    (pvs2imandra-cases (selections expr)(else-part expr) bindings livevars)))

(defun pvs2imandra-cases (selections else-part bindings livevars)
  (let ((selections-imandra
	 (loop for entry in selections
	       collect
	       (let* ((bind-decls (args entry))
		      (bind-ids (pvs2imandra-make-bindings bind-decls bindings)))
		 (format nil "~a ~{~a ~} -> ~a"
			 (pvs2imandra* (constructor entry) bindings livevars)
			 bind-ids
			 (pvs2imandra* (expression entry)
				     (append (pairlis bind-decls bind-ids) bindings)
				     livevars))))))
    (if else-part
	(format nil "~a ~% _ -> ~a"
	  selections-imandra
	  (pvs2imandra* (expression else-part) bindings livevars))
	selections-imandra)))

(defmethod pvs2imandra* ((expr update-expr) bindings livevars)
  (if (imandra-updateable? (type (expression expr)))
      (if (and *destructive?*
	       (not (some #'maplet? (assignments expr))))
	  (let* ((expression (expression expr))
		 (assignments (assignments expr))
		 (*livevars-table* 
		  (no-livevars? expression livevars assignments))
		 )
	    ;;very unrefined: uses all
	    ;;freevars of eventually updated expression.
	    (cond (*livevars-table* ;; check-assign-types
		   (push-output-vars (car *livevars-table*)
				     (cdr *livevars-table*))
		   (pvs2imandra-update expr
				  bindings livevars))
		  (t
		   (when (and *eval-verbose* (not *livevars-table*))
		     (format t "~%Update ~s translated nondestructively.
 Live variables ~s present" expr livevars))
		   (pvs2imandra-update  expr
						    bindings livevars))))
	  (pvs2imandra-update expr bindings livevars))
      (pvs2imandra* (translate-update-to-if! expr)
		  bindings livevars)))

(defun pvs2imandra-update
    (expr bindings livevars)
  (with-slots (type expression assignments) expr
    (let* ((assign-exprs (mapcar #'expression assignments))
	   (exprvar (gentemp "E"))
	   (imandra-expr (pvs2imandra* expression bindings
				(append (updateable-free-formal-vars
					 assign-exprs)
					;;assign-args can be ignored
					livevars))))
      (format nil "#! ~a"
    (pvs2imandra-update* (type expression)
				   imandra-expr exprvar
				   (mapcar #'arguments assignments)
				   assign-exprs
				   bindings
				   (append (updateable-vars expression)
					   livevars)
				   (list (list exprvar imandra-expr)))))))

(defun pvs2imandra-assign-rhs (assignments bindings livevars)
  (when (consp assignments)
      (let ((imandra-assign-expr (pvs2imandra* (expression (car assignments))
					   bindings
					   (append (updateable-vars
						    (arguments (car assignments)))
						   (append (updateable-vars (cdr assignments))
					   livevars))))
	    (*lhs-args* nil))
	(cons imandra-assign-expr
	      (pvs2imandra-assign-rhs (cdr assignments) bindings
				    (append (updateable-free-formal-vars
					     (expression (car assignments)))
					    livevars))))))

;;recursion over updates in an update expression
(defun pvs2imandra-update*
    (type expr exprvar
	  assign-args assign-exprs bindings livevars accum)
  (if (consp assign-args)
      (let* ((*lhs-args* nil)
	     (assign-exprvar (gentemp "R"))
	     (imandra-assign-expr
	      (pvs2imandra* (car assign-exprs)
			  bindings
			  (append (updateable-vars (cdr assign-exprs))
				  (append (updateable-vars (cdr assign-args))
					  livevars))))
	     (newexprvar (gentemp "N"))
	     (new-accum (pvs2imandra-update-nd-type
		       type exprvar newexprvar
		       (car assign-args)
		       assign-exprvar
		       bindings
		       (append (updateable-free-formal-vars (car assign-exprs))
			       (append (updateable-vars (cdr assign-exprs))
				       (append (updateable-vars (cdr assign-args))
					       livevars)))
		       accum))
	     (lhs-bindings (nreverse *lhs-args*))
	     (cdr-imandra-output
	      (pvs2imandra-update*
	       type expr
	       newexprvar
	       (cdr assign-args)(cdr assign-exprs) bindings
	       (append (updateable-free-formal-vars (car assign-exprs))
		       livevars) 
		       new-accum )))
	(format nil "~a = ~a ~%~:{~a = ~a~%~} ~a"
	  assign-exprvar imandra-assign-expr
		lhs-bindings
		 cdr-imandra-output))
      (format nil "~:{~a = ~a~%~} = ~a" (nreverse accum) exprvar)))

;;recursion over nested update arguments in a single update.
(defun pvs2imandra-update-nd-type (type expr newexprvar args assign-expr
				 bindings livevars accum)
  (if (consp args)
      (pvs2imandra-update-nd-type* type expr newexprvar (car args) (cdr args) assign-expr
				 bindings livevars accum)
      (cons (list newexprvar assign-expr) accum)))

(defmethod pvs2imandra-update-nd-type* ((type funtype) expr newexprvar arg1 restargs
				      assign-expr bindings livevars accum)
  (let* ((arg1var (gentemp "L"))
	 (imandra-arg1 (pvs2imandra*  (car arg1) bindings
				  (append (updateable-vars restargs)
					  livevars))))
    (push (list arg1var imandra-arg1) *lhs-args*)
    (if (consp restargs)
	(let* ((exprvar (gentemp "E"))
	       (exprval (format nil "pvsSelect ~a ~a" expr arg1var))
	       (newexprvar2 (gentemp "N"))
	       (newaccum
		(pvs2imandra-update-nd-type 
		 (range type) exprvar newexprvar2
		 restargs assign-expr bindings livevars
		 (cons (list exprvar exprval) accum))))
	  (cons (list newexprvar (pvsimandra_update expr arg1var newexprvar2))
		newaccum))
	(cons (list newexprvar (pvsimandra_update expr arg1var assign-expr))
	      accum))))


(defmethod pvs2imandra-update-nd-type* ((type recordtype) expr newexprvar arg1 restargs
				      assign-expr bindings livevars accum)
  (let ((id (pvs2imandra-id (id (car arg1)))))
    (if (consp restargs)
	(let* ((exprvar (gentemp "E"))
	       (new-expr (format nil "~a.~a" expr id))
	       (field-type (type (find id (fields type) :key #'id) ))
	       (newexprvar2 (gentemp "N"))
	       (newaccum (pvs2imandra-update-nd-type field-type exprvar newexprvar2
						   restargs assign-expr bindings
						   livevars
						   (cons (list exprvar new-expr) accum))))
	  (cons (list newexprvar (format nil "{~a & ~a = ~a}" expr id newexprvar2)) newaccum))
	(cons (list newexprvar (format nil "{~a & ~a = ~a}" expr id assign-expr))
	      accum))))

(defmethod pvs2imandra-update-nd-type* ((type adt-type-name) expr newexprvar arg1 restargs
				      assign-expr bindings livevars accum)
  (let ((id (pvs2imandra-id (id (car arg1)))))
    (break "TODO: update-nd-type")
    (if (consp restargs)
	(let* ((exprvar (gentemp "E"))
	       (new-expr (format nil "~a.~a" expr id))
	       (field-type (type (find id (fields type) :key #'id) ))
	       (newexprvar2 (gentemp "N"))
	       (newaccum (pvs2imandra-update-nd-type field-type exprvar newexprvar2
						   restargs assign-expr bindings
						   livevars
						   (cons (list exprvar new-expr) accum))))
	  (cons (list newexprvar (format nil "{~a & ~a = ~a}" expr id newexprvar2)) newaccum))
	(cons (list newexprvar (format nil "{~a & ~a = ~a}" expr id assign-expr))
	      accum))))

(defmethod pvs2imandra-update-nd-type* ((type subtype) expr newexprvar arg1 restargs
				      assign-expr bindings livevars accum)
  (pvs2imandra-update-nd-type* (find-supertype type) expr newexprvar arg1 restargs
			     assign-expr bindings livevars accum))

(defmethod pvs2imandra-type ((type recordtype) &optional tbindings)
  (with-slots (print-type) type
    (if (type-name? print-type)
	(let ((entry (assoc (declaration print-type) *imandra-record-defns*)))
	  (if entry (cadr entry)	;return the imandra-rectype-name
	      (let* ((formatted-fields (loop for fld in (fields type)
					  collect
					    (format nil "~a :: !~a" (pvs2imandra-id (id fld))
						    (pvs2imandra-type (type fld)))))
		     (imandra-rectype (format nil "{ ~{~a~^, ~} }" formatted-fields))
		     (imandra-rectype-name (gentemp (format nil "pvs~a" (pvs2imandra-id (id print-type))))))
		(push (list (declaration print-type) imandra-rectype-name imandra-rectype)
		      *imandra-record-defns*)
		imandra-rectype-name)))
	(pvs2imandra-error "~%Record type ~a must be declared." type))))

(defmethod pvs2imandra-type ((type tupletype) &optional tbindings)
  (format nil "~{~a~^, ~}" (loop for elemtype in (types type)
				   collect (pvs2imandra-type elemtype))))

(defmethod pvs2imandra-type ((type funtype) &optional tbindings)
  (if (imandra-updateable? type)
      (format nil "(PvsArray ~a)" (pvs2imandra-type (range type)))
      (format nil "(:to ~a ~a)"
	(pvs2imandra-type (domain type))
	(pvs2imandra-type (range type)))))

(defmethod pvs2imandra-type ((type subtype) &optional tbindings)
  (cond ((subtype-of? type *integer*)
	 "int")
	((subtype-of? type *real*)
	 "rational")
	(t (pvs2imandra-type (find-supertype type)))))

(defun pvs2imandra-id (id &optional (case :all))
  (let ((idstr (substitute #\p #\? (string (op-to-id id)))))
    (intern
     (case case
       (:lower (string-downcase idstr :start 0 :end 1))
       (:upper (string-upcase idstr :start 0 :end 1))
       (t idstr)))))

(defmethod pvs2imandra-type ((type type-name) &optional tbindings)
  (or (cdr (assoc type tbindings :test #'tc-eq))
      (let ((decl (declaration type)))
	(if (formal-type-decl? decl)
	    (pvs2imandra-id (id type) :lower)
	    (case (id decl)
	      (integer '|int|)
	      (real '|real|)
	      (character '|char|)
	      (boolean '|bool|)
	      (t (pvs2imandra-id (id type) :upper)))))))

;;; Note that bindings is an assoc-list, used to check if id is already
;;; in use.
(defun pvs2imandra-make-bindings (bind-decls bindings &optional nbindids)
  (if (null bind-decls)
      (nreverse nbindids)
    (let* ((bb (car bind-decls))
	         (id (pvs2imandra-id (id bb)))
	         (newid (if (rassoc (id bb) bindings)
			                (pvs2cl-newid id bindings)
			              id)))
	    (pvs2imandra-make-bindings (cdr bind-decls)
				                         bindings (cons newid nbindids)))))

(defmethod imandra-updateable? ((texpr tupletype))
  (imandra-updateable? (types texpr)))

(defmethod imandra-updateable? ((texpr funtype)) ;;add enum types, subrange.
  (and (or (simple-below? (domain texpr))(simple-upto? (domain texpr)))
       (imandra-updateable? (range texpr))))

(defmethod imandra-updateable? ((texpr recordtype))
  (imandra-updateable? (mapcar #'type (fields texpr))))

(defmethod imandra-updateable? ((texpr subtype))
  (imandra-updateable? (find-supertype texpr)))

(defmethod imandra-updateable? ((texpr list))
  (or (null texpr)
      (and (imandra-updateable? (car texpr))
	   (imandra-updateable? (cdr texpr)))))

(defmethod imandra-updateable? ((texpr t))
  t)

(defun pvs2imandra-theory (theory)
  (let* ((theory (get-theory theory))
	 (*current-context* (context theory)))
    (cond ((datatype? theory)
	         (pvs2imandra-datatype theory))
	        (t (loop for decl in (theory theory)
                   do
                   (cond ((type-eq-decl? decl)
			                    (let ((dt (find-supertype (type-value decl))))
			                      (when (adt-type-name? dt)
				                      (pvs2imandra-constructors (constructors dt) dt))))
			                   ((datatype? decl)
			                    (let ((adt (adt-type-name decl)))
			                      (pvs2imandra-constructors (constructors adt) adt)))
			                   ((const-decl? decl)
			                    (unless (eval-info decl)
			                      (progn
				                      (pvs2imandra-declaration decl))))
			                   (t nil)))))))

(defun pvs2imandra-datatype (dt)
  (let* ((typevars (mapcar #'(lambda (fm)
			                         (pvs2imandra-datatype-formal fm dt))
		                       (formals dt)))
	       (constructors (pvs2imandra-constructors
			                  (constructors dt) dt
			                  (mapcar #'cons (formals dt) typevars))))
    (format nil "::~a~{ ~a~} = ~{~a~^ | ~}"
            (pvs2imandra-id (id dt)) typevars constructors)))

(defun pvs2imandra-datatype-formal (formal dt)
  (if (formal-type-decl? formal)
      (let ((id-str (string (pvs2imandra-id (id formal)))))
	(if (lower-case-p (char id-str 0))
	    (pvs2imandra-id (id formal))
	    (make-new-variable (string-downcase id-str :end 1) dt)))
      (break "What to do with constant formals?")))

(defun pvs2imandra-constructors (constrs datatype &optional tvars)
  (pvs2imandra-constructors* constrs datatype tvars))

(defun pvs2imandra-constructors* (constrs datatype tvars)
  (when constrs
    (cons (pvs2imandra-constructor (car constrs) datatype tvars)
	  (pvs2imandra-constructors* (cdr constrs) datatype tvars))))

;;; Maps to ConstructorDef
(defun pvs2imandra-constructor (constr datatype tvars)
  (format nil "~a~{ ~a~}" (pvs2imandra-id (id constr))
	  (mapcar #'(lambda (arg) (pvs2imandra-type (type arg) tvars))
	    (arguments constr))))

(defun clear-imandra-hash ()
  (clrhash *imandra-nondestructive-hash*)
  (clrhash *imandra-destructive-hash*))

(defun generate-imandra-for-pvs-file (filename &optional force?)
  (when force? (clear-imandra-hash))
  (let ((theories (cdr (gethash filename (current-pvs-files)))))
    ;; Sets the hash-tables
    (dolist (theory theories)
      (pvs2imandra-theory theory))
    (with-open-file (output (format nil "~a.icl" filename)
			    :direction :output
			    :if-exists :supersede
			    :if-does-not-exist :create)
      (format output "// Imandra file generated from ~a.pvs~2%" filename)
      (format output
	  "// In general for a definiton foo in an ~
               unparameterized~%// theory th, the names are:~
           ~%//    foo  - takes no arguments, returns a unary closure~
           ~%//   _foo  - the nondestructive version of the function~
           ~%//    foo! - the destructive version of the function")
      (format output
	  "// If the definition appears in a parameterized theory th, ~
               additional functions are generated ~%// that take arguments ~
               corresponding to the theory parameters, take names are:~
           ~%//    th_foo  - takes no arguments, returns a unary closure~
           ~%//   _th_foo  - the nondestructive version of the function~
           ~%//    th_foo! - the destructive version of the function")
      (format output
	  "~%// Function names must be unique, so a number may be appended, ~
            and the type~%// is included for functions associated with ~
            datatypes.~%// For these functions, the mappings are given here.")
      (format output "~%module ~a" filename)
      (dolist (theory theories)
	(dolist (decl (theory theory))
	  (let ((ndes-info (gethash decl *imandra-nondestructive-hash*))
		(des-info (gethash decl *imandra-destructive-hash*)))
	    (when ndes-info
	      (let ((id (imandra-info-id ndes-info)))
		;; First the signature
		(format output "~%~a:: ~a" id (imandra-info-type ndes-info))
		;; Then the defn
		(format output "~%~a ~a" id (imandra-info-definition ndes-info))))
	    (when des-info
	      (let ((id (imandra-info-id des-info)))
		;; First the signature
		(format output "~%~a:: ~a" id (imandra-info-type des-info))
		;; Then the defn
		(format output "~%~a ~a" id (imandra-info-definition des-info))))))))))
