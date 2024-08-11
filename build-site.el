;; Set the package installation directory so that packages aren't stored in the
;; ~/.emacs.d/elpa path.
(require 'package)
(setq package-user-dir (expand-file-name "./.packages"))
(setq package-archives '(("melpa" . "https://melpa.org/packages/")
                         ("elpa" . "https://elpa.gnu.org/packages/")))

;; Initialize the package system
(package-initialize)
(unless package-archive-contents
  (package-refresh-contents))

;; Install dependencies
(package-install 'htmlize)

;; Load the publishing system
(require 'ox-publish)

;; Define the publishing project
(setq org-publish-project-alist
      (list
       (list "my-org-site"
	     :recursive t
	     :base-directory "./content"
	     :publishing-directory "./public"
	     :publishing-function 'org-html-publish-to-html
	     :with-author nil       ;; Don't include author name
	     :with-creator t        ;; Include Emacs and Org versions in footer
	     :with-toc t            ;; Include a table of contents
	     :section-numbers nil   ;; Don't include section numbers
	     :time-stamp-file nil   ;; Don't include time stamp in file
	     )
       (list "images"               ;; Make sure images get exported inline
         :base-directory "./content"
         :base-extension "jpg\\|gif\\|png"
         :publishing-directory "./public"
         :publishing-function 'org-publish-attachment)))

(setq org-html-validation-link nil             ;; Don't show validation link
      org-html-head-include-scripts nil	       ;; Use our own scripts
      org-html-head-include-default-style nil  ;; Use our own styles
      org-html-head "<link rel=\"stylesheet\" href=\"https://cdn.simplecss.org/simple.min.css\" />"
      ;; org-html-head "<link rel=\"stylesheet\" href=\"https://unpkg.com/sakura.css/css/sakura.css\" >"
      )

;; The cache sometimes doesn't reload pages that really should be reloaded
;; Not sure which is relevant, both seem to have an effect
(setq org-element-use-cache nil
      org-publish-cache nil
      )

(org-publish-all t)

;; (require 'simple-httpd)
;; (setq httpd-root "./public/")
;; (httpd-start)

(message "Build complete!")
