import { Editor } from '@tiptap/core';
import StarterKit from '@tiptap/starter-kit';
import FloatingMenu from '@tiptap/extension-floating-menu';
import BubbleMenu from '@tiptap/extension-bubble-menu';
import Image from '@tiptap/extension-image';

export default ( { element, content = '', type = 'edit' } ) => {
	let editor;

	return {
		/**
		 * ------------------------------------
		 * Attributes
		 * ------------------------------------
		 */
		_updated_at: null,
		/**
		 * ------------------------------------
		 * Initialize
		 * ------------------------------------
		 */
		init() {
			if ( type == 'edit' ) {
				Alpine.store( 'tiptapEditor', this );
			}
			this.initEditor();
		},
		// init editor
		initEditor() {
			const _this = this;
			editor = new Editor( {
				element: element,
				extensions: [
					StarterKit.configure({
						link: {
							openOnClick: false,
							autolink: true,
						}
					}),
					Image.configure({
						inline: true
					}),
					FloatingMenu.configure( {
						element: document.querySelector( '.tiptap-floating-menu' )
					} ),
					BubbleMenu.configure( {
						element: document.querySelector( '.tiptap-bubble-menu' )
					} ),
				],
				content: content,
				onUpdate( { editor } ) {
					_this._updated_at = Date.now();
				},
				onSelectionUpdate( { editor } ) {
					_this._updated_at = Date.now();
				},
			} );
		},
		displaySavedData( el, savedContent = '' ) {
			return new Editor( {
				element: el,
				extensions: [
					StarterKit.configure({
						link: {
							openOnClick: false,
							autolink: true,
						}
					}),
					Image.configure({
						inline: true
					}),
				],
				editable: false,
				content: savedContent
			} );
		},
		/**
		 * ------------------------------------
		 * Operations
		 * ------------------------------------
		 */
		// toggle bold
		toggleBold() {
			editor?.chain().focus().toggleBold().run();
		},
		// toggle italic
		toggleItalic() {
			editor?.chain().focus().toggleItalic().run();
		},
		// toggle highlight
		toggleHighlight() {
			editor?.chain().focus().toggleHighlight().run();
		},
		// toggle code
		toggleCode() {
			editor?.chain().focus().toggleCode().run();
		},
		// toggle strike
		toggleStrike() {
			editor?.chain().focus().toggleStrike().run();
		},
		// toggle underline
		toggleUnderline() {
			editor?.chain().focus().toggleUnderline().run();
		},
		// toggle block quote
		toggleBlockquote() {
			editor?.chain().focus().toggleBlockquote().run();
		},
		// toggle bullet list
		toggleBulletList() {
			editor.commands?.toggleBulletList();
		},
		// toggle ordered list
		toggleOrderedList() {
			editor.commands?.toggleOrderedList();
		},
		// toggle heading
		toggleHeading( opt = { level: 1 } ) {
			editor.chain().focus().toggleHeading( opt ).run();
		},
		// toggle heading
		toggleCodeBlock() {
			editor.chain().focus().toggleCodeBlock().run();
		},
		// set paragraph
		setParagraph() {
			editor?.commands.setParagraph();
		},
		// set link
		setLink() {
			if ( !editor ) return;

			const previousUrl = editor.getAttributes('link').href;
			const url = window.prompt('URL', previousUrl);

			// cancelled
			if (url === null) {
				return;
			}

			// empty
			if (url === '') {
				editor.chain().focus().extendMarkRange('link').unsetLink().run();

				return;
			}

			// update link
			try {
				editor.chain().focus().extendMarkRange('link').setLink({ href: url }).run();
			} catch (e) {
				alert(e.message);
			}
		},
		unsetLink() {
			editor?.chain().focus().unsetLink().run();
		},
		// toggle heading
		setHeading( opt = { level: 1 } ) {
			editor?.commands.setHeading( opt );
		},
		// set code block
		setCodeBlock() {
			editor?.chain().focus().setCodeBlock().run();
		},
		// redo
		redo() {
			editor?.chain().focus().redo().run();
		},
		// check can redo
		canRedo( param ) {
			return editor?.can().redo();
		},
		// undo
		undo() {
			editor?.chain().focus().undo().run();
		},
		// check can undo
		canUndo( param ) {
			return editor?.can().undo();
		},
		// get content
		getTextContent() {
			return editor?.getText();
		},
		// get html content
		getHtmlContent() {
			return editor?.getHTML();
		},
		// set content
		setContent( content ) {
			editor?.commands.setContent( content );
		},
		// check if it's active or not.
		isActive( type, opts = {} ) {
			if ( opts != null ) {
				return editor.isActive( type, opts );
			}
			return editor.isActive( type );
		},
		// download handler
		download( event ) {
			const button = event.currentTarget;
			const docType = button.dataset.docType;
			const docName = button.dataset.docName || 'document';

			let content = editor?.getHTML();

			// Fix relative URLs
			const baseUrl = window.location.origin;
			const parser = new DOMParser();
			const doc = parser.parseFromString( content, 'text/html' );

			const tags = [ 'a', 'img', 'video', 'audio', 'source' ];
			tags.forEach( tag => {
				doc.querySelectorAll( tag ).forEach( element => {
					const attr = tag === 'a' ? 'href' : 'src';
					const url = element.getAttribute( attr );
					if ( url && !url.startsWith( 'http' ) ) {
						element.setAttribute( attr, new URL( url, baseUrl ).href );
					}
				} );
			} );

			content = doc.documentElement.outerHTML;

			if ( docType === 'pdf' ) {
				const contentWithMargin = `<div style="margin: 20px;">${ content }</div>`;
				return html2pdf()
					.set( {
						filename: docName
					} )
					.from( contentWithMargin )
					.toPdf()
					.save();
			}

			const html = `
					<html ${ this.doctype === 'doc'
		? 'xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word" xmlns="http://www.w3.org/TR/REC-html40"'
		: ''
}>
					<head>
						<meta charset="utf-8" />
						<title>${ docName }</title>
					</head>
					<body style="margin: 20px;">
						${ content }
					</body>
					</html>`;

			const url = `${ docType === 'doc'
				? 'data:application/vnd.ms-word;charset=utf-8'
				: 'data:text/plain;charset=utf-8'
			},${ encodeURIComponent( html ) }`;

			const downloadLink = document.createElement( 'a' );
			document.body.appendChild( downloadLink );
			downloadLink.href = url;
			downloadLink.download = `${ docName }.${ docType }`;
			downloadLink.click();

			document.body.removeChild( downloadLink );
		},
		// get selected text content
		getSelectedContent() {
			const { view, state } = editor;
			const { from, to } = view.state.selection;
			return state.doc.textBetween( from, to, '' );
		},
		// replace selected text
		replaceSelectedContent( range, content ) {
			editor.commands.insertContentAt( range, content );
		},
		// get selected range
		getSelectedRange() {
			const { view, state } = editor;
			return view.state.selection;
		}
	};
};
