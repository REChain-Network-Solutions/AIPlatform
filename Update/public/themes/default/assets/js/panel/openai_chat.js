// cspell:disable

/**
 * @typedef {Object} AiResponse
 * @property {{slug: string, label: string}} model - The AI model information
 * @property {HTMLElement} bubbleEl - The bubble element
 * @property {HTMLElement} chatContentEl - The chat content element
 * @property {HTMLElement} chatContentContainerEl - The chat content container element
 * @property {HTMLElement | null} acceptButtonEl - The accept button element
 * @property {HTMLElement | null} regenerateButtonEl - The regenrate button element
 * @property {boolean} responseStreaming - Whether response is streaming
 * @property {number} responseId - Id of the message coming from backend
 * @property {string[]} response - Array of response strings
 * @property {AbortController | null} abortController - Abort controller
 * @property {number} animatingWordIndex - The index of animating word
 * @property {Set<HTMLElement>} animatedElements - Set of already animated elements
 * @property {number} lastAnimatedElOffsetTop - Offset top of the last animated element
 */

/**
 * @type {AiResponse[]}
 */
let aiResponses = [];
let selectedPrompt = -1;
let promptsData = [];
let favData = [];
let searchString = '';
let pdf = undefined;
let pdfName = '';
let pdfPath = '';
let filterType = 'all';
let prompt_images = [];
let navigatingInChatsHistory = false;
let selectedHistoryPrompt = -1;

/**
 * Credits: Joydeep Bhowmik https://dev.to/joydeep23/adding-keys-our-dom-diffing-algorithm-4d7g
 */
class LiquidVDOM {
	getnodeType( node ) {
		if ( node.nodeType == 1 ) return node.tagName.toLowerCase();
		else return node.nodeType;
	}

	clean( node ) {
		for ( let n = 0; n < node.childNodes.length; n++ ) {
			let child = node.childNodes[ n ];
			if ( child.nodeType === 8 ) {
				// Only remove comment nodes
				node.removeChild( child );
				n--;
			} else if ( child.nodeType === 1 ) {
				// Element node
				if ( child.hasAttribute( 'key' ) ) {
					let key = child.getAttribute( 'key' );
					child.key = key;
					child.removeAttribute( 'key' );
				}
				this.clean( child );
			}
		}
	}

	parseHTML( str ) {
		let parser = new DOMParser();
		let doc = parser.parseFromString( str, 'text/html' );
		this.clean( doc.body );
		return doc.body;
	}

	attrbutesIndex( el ) {
		var attributes = {};
		if ( el.attributes == undefined ) return attributes;
		for ( var i = 0, atts = el.attributes, n = atts.length; i < n; i++ ) {
			attributes[ atts[ i ].name ] = atts[ i ].value;
		}
		return attributes;
	}

	patchAttributes( vdom, dom ) {
		let vdomAttributes = this.attrbutesIndex( vdom );
		let domAttributes = this.attrbutesIndex( dom );
		if ( vdomAttributes == domAttributes ) return;
		Object.keys( vdomAttributes ).forEach( ( key, i ) => {
			//if the attribute is not present in dom then add it
			if ( !dom.getAttribute( key ) ) {
				dom.setAttribute( key, vdomAttributes[ key ] );
			} //if the atrtribute is present than compare it
			else if ( dom.getAttribute( key ) ) {
				if ( vdomAttributes[ key ] != domAttributes[ key ] ) {
					dom.setAttribute( key, vdomAttributes[ key ] );
				}
			}
		} );
		Object.keys( domAttributes ).forEach( ( key, i ) => {
			//if the attribute is not present in vdom than remove it
			if ( !vdom.getAttribute( key ) ) {
				dom.removeAttribute( key );
			}
		} );
	}

	hasTheKey( dom, key ) {
		let keymatched = false;
		for ( let i = 0; i < dom.children.length; i++ ) {
			if ( key == dom.children[ i ].key ) {
				keymatched = true;
				break;
			}
		}
		return keymatched;
	}

	patchKeys( vdom, dom ) {
		//remove unmatched keys from dom
		for ( let i = 0; i < dom.children.length; i++ ) {
			let dnode = dom.children[ i ];
			let key = dnode.key;
			if ( key ) {
				if ( !this.hasTheKey( vdom, key ) ) {
					dnode.remove();
				}
			}
		}
		//adding keys to dom
		for ( let i = 0; i < vdom.children.length; i++ ) {
			let vnode = vdom.children[ i ];
			let key = vnode.key;
			if ( key ) {
				if ( !this.hasTheKey( dom, key ) ) {
					//if key is not present in dom then add it
					let nthIndex = [].indexOf.call(
						vnode.parentNode.children,
						vnode,
					);
					if ( dom.children[ nthIndex ] ) {
						dom.children[ nthIndex ].before( vnode.cloneNode( true ) );
					} else {
						dom.append( vnode.cloneNode( true ) );
					}
				}
			}
		}
	}

	diff( vdom, dom ) {
		//if dom has no childs then append the childs from vdom
		if ( dom.hasChildNodes() == false && vdom.hasChildNodes() == true ) {
			for ( let i = 0; i < vdom.childNodes.length; i++ ) {
				//appending
				dom.append( vdom.childNodes[ i ].cloneNode( true ) );
			}
		} else {
			this.patchKeys( vdom, dom );
			//if dom has extra child
			if ( dom.childNodes.length > vdom.childNodes.length ) {
				let count = dom.childNodes.length - vdom.childNodes.length;
				if ( count > 0 ) {
					for ( ; count > 0; count-- ) {
						dom.childNodes[ dom.childNodes.length - count ].remove();
					}
				}
			}
			//now comparing all childs
			for ( let i = 0; i < vdom.childNodes.length; i++ ) {
				//if the node is not present in dom append it
				if ( dom.childNodes[ i ] == undefined ) {
					dom.append( vdom.childNodes[ i ].cloneNode( true ) );
					// console.log("appenidng",vdom.childNodes[i])
				} else if (
					this.getnodeType( vdom.childNodes[ i ] ) ==
					this.getnodeType( dom.childNodes[ i ] )
				) {
					//if same node type
					//if the nodeType is text
					if ( vdom.childNodes[ i ].nodeType == 3 ) {
						//we check if the text content is not same
						if (
							vdom.childNodes[ i ].textContent !=
							dom.childNodes[ i ].textContent
						) {
							//replace the text content
							dom.childNodes[ i ].textContent =
								vdom.childNodes[ i ].textContent;
						}
					} else {
						this.patchAttributes(
							vdom.childNodes[ i ],
							dom.childNodes[ i ],
						);
					}
				} else {
					//replace
					dom.childNodes[ i ].replaceWith(
						vdom.childNodes[ i ].cloneNode( true ),
					);
				}
				if ( vdom.childNodes[ i ].nodeType != 3 ) {
					this.diff( vdom.childNodes[ i ], dom.childNodes[ i ] );
				}
			}
		}
	}
}

const liquidVDOM = new LiquidVDOM();

function unwrapWords( node ) {
	if (
		node.nodeName === 'PRE' ||
		node.nodeName === 'CODE' ||
		node.nodeName === 'A' ||
		node.nodeName === 'TR' ||
		node.classList?.contains( 'katex' )
	) return;

	if ( node.classList?.contains( 'done-signal' ) ) {
		return node.remove();
	}

	if ( node.nodeType === 3 ) {
		return;
	}

	if ( node.classList?.contains( 'animated-el' ) ) {
		const textNode = document.createTextNode( node.textContent );
		node.parentNode.replaceChild( textNode, node );
		return;
	}

	const childNodes = [ ...node.childNodes ];
	childNodes.forEach( child => unwrapWords( child ) );
}

function generateUUID() {
	return ([ 1e7 ]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
		(c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
	);
}

/**
 * @param {Object} param0
 * @param {AiResponse} param0.responseObj
 * @param {boolean} param0.withoutDone
 */
function getAiResponseString( { responseObj = null, withoutDone = true } ) {
	if ( !responseObj ) {
		responseObj = aiResponses[0];
	}

	if ( !responseObj ) return '';

	const string = responseObj.response
		.join( '' )
		.trim()
		.replace( /<br\s*\/?>/g, '\n' );

	if ( withoutDone ) {
		return string.replace( '[DONE]', '' );
	}

	return string;
}

function fixUnclosedMarkdownSyntax( string ) {
	let text = string;

	let boldMatch = text.match( /\*\*(?:(?!\*\*).)*$/ );
	if ( boldMatch ) {
		text = text + '**';
	}

	let italicMatch = text.match( /\*(?:(?!\*).)*$/ );
	if ( italicMatch ) {
		text = text + '*';
	}

	let codeBlockMatch = text.match( /```(?:(?!```).)*$/ );
	if ( codeBlockMatch ) {
		text = text + '```';
	}

	let inlineCodeMatch = text.match( /`(?:(?!`).)*$/ );
	if ( inlineCodeMatch ) {
		text = text + '`';
	}

	let strikeMatch = text.match( /~~(?:(?!~~).)*$/ );
	if ( strikeMatch ) {
		text = text + '~~';
	}

	return text;
}

/**
 * @param {string} string
 * @param {object} options
 * @param {boolean} options.readyForAnimation
 */
function formatString( string, options = {} ) {
	if ( !( 'markdownit' in window ) ) return;

	string = fixUnclosedMarkdownSyntax( string );

	string = string
		.replace(
			/(?<=\[START_REASONING\])(?:.*?\n\n.*?)(?=\[END_REASONING\]|$)/gs,
			match => match.replace( /\n\n/g, '\n' ),
		)
		.replace( '[START_REASONING]', '>' )
		.replace( '[END_REASONING]', '\n' )
		.replaceAll( '\\(', '$' )
		.replaceAll( '\\)', '$' )
		.replaceAll( '\\[', '$$' )
		.replaceAll( '\\]', '$$' );

	const renderer = window.markdownit( {
		breaks: true,
		highlight: ( str, lang ) => {
			const language = lang && lang !== '' ? lang : 'md';
			const codeString = str;

			const highlighted = Prism.highlight(
				codeString,
				( Prism.languages[ language ] != null ? Prism.languages[ language ] : ( language === 'blade' ? Prism.languages.html : Prism.languages.markup ) ),
				language,
			);

			return `<pre class="${ options.readyForAnimation ? 'animated-el' : '' } !whitespace-pre-wrap rounded [direction:ltr] max-w-full !w-full language-${ language }"><code data-lang="${ language }" class="language-${ language }">${ highlighted }</code></pre>`;
		},
	} );

	if ( 'katex' in window && 'markdownItKatex' in window ) {
		renderer.use( markdownItKatex );
	}

	renderer.use( function ( md ) {
		// Add data-fslightbox attribute to images
		const defaultRender = md.renderer.rules.image;

		md.renderer.rules.image = function ( tokens, idx, options, env, self ) {
			const token = tokens[ idx ];
			// Find the src attribute to use as the href
			const srcIndex = token.attrIndex( 'src' );
			const src = srcIndex >= 0 ? token.attrs[ srcIndex ][ 1 ] : '';

			// Render the image with default renderer
			const imageHtml = defaultRender( tokens, idx, options, env, self );

			// Wrap in anchor tag with href to the image source
			return `<a href="${ src }" target="_blank" data-fslightbox="gallery">${ imageHtml }</a>`;
		};

		// Detect and wrap HTML code blocks
		md.core.ruler.before( 'block', 'detect_html_blocks', function ( state ) {
			const src = state.src;

			// Look for HTML that starts with <html> or <!DOCTYPE html> and isn't already in code blocks
			const htmlStartRegex = /(?:^|\n)(?!```)(<!DOCTYPE\s+html[^>]*>[\s\S]*?<html[^>]*>|<html[^>]*>)/gi;
			let match;
			let modifiedSrc = src;
			let offset = 0;

			while ( ( match = htmlStartRegex.exec( src ) ) !== null ) {
				const startPos = match.index + offset;
				const htmlStart = match[0];

				// Check if this HTML start is already inside a code block
				const beforeContent = modifiedSrc.substring( 0, startPos );
				const codeBlockCount = ( beforeContent.match( /```/g ) || [] ).length;
				const isInsideCodeBlock = codeBlockCount % 2 === 1;

				if ( !isInsideCodeBlock ) {
					// Check if the HTML content doesn't already have ``` at the beginning
					const startsWithCodeBlock = htmlStart.trim().startsWith( '```' );

					if ( !startsWithCodeBlock ) {
						// Ensure the code block starts on a new line
						const precedingChar = startPos > 0 ? modifiedSrc.charAt( startPos - 1 ) : '\n';
						const needsNewline = precedingChar !== '\n';

						// Start wrapping immediately with opening code block
						const codeBlockStart = (needsNewline ? '\n' : '') + '```html\n';

						// Insert the opening code block
						modifiedSrc = modifiedSrc.substring( 0, startPos ) + codeBlockStart + modifiedSrc.substring( startPos );

						// Update offset
						offset += codeBlockStart.length;

						// Now look for the closing </html> tag in the modified content
						const afterStart = modifiedSrc.substring( startPos + codeBlockStart.length );
						const htmlEndMatch = afterStart.match( /<\/html\s*>/i );

						if ( htmlEndMatch ) {
							const endPos = startPos + codeBlockStart.length + htmlEndMatch.index + htmlEndMatch[0].length;

							// Check if there's already a newline after </html>
							const nextChar = endPos < modifiedSrc.length ? modifiedSrc.charAt( endPos ) : '';
							const hasFollowingNewline = nextChar === '\n';

							// Insert closing code block after </html> with proper newlines
							const codeBlockEnd = '\n```' + (hasFollowingNewline ? '' : '\n');
							modifiedSrc = modifiedSrc.substring( 0, endPos ) + codeBlockEnd + modifiedSrc.substring( endPos );

							// Update offset for next iterations
							offset += codeBlockEnd.length;
						}
					}
				}
			}

			if ( modifiedSrc !== src ) {
				state.src = modifiedSrc;
			}
		} );

		// Wrap words with animated-el spans for animation
		if ( options.readyForAnimation ) {
			md.core.ruler.after( 'inline', 'wrap_words', function ( state ) {
				state.tokens.forEach( function ( blockToken ) {
					if ( blockToken.type !== 'inline' ) return;

					const inlineElements = [ 'strong', 'em', 's', 'u', 'a', 'i', 'b', 'code', 'del', 'ins', 'mark', 'sub', 'sup' ];
					let insideInlineElement = false;

					blockToken.children.forEach( function ( token ) {
						if ( token.type === 'text' && !insideInlineElement ) {
							// Split text into words and wrap each with span
							const words = token.content.split( /(\s+)/ );
							let wrappedContent = '';

							words.forEach( word => {
								if ( word.trim() !== '' ) {
									wrappedContent += `<span class="animated-el ${ word === '[DONE]' ? 'done-signal' : '' }">${ word }</span>`;
								} else {
									wrappedContent += word; // Preserve whitespace
								}
							} );

							// Convert to HTML token
							token.type = 'html_inline';
							token.content = wrappedContent;
						}

						// Track if we're inside inline elements and add animated-el class
						if ( token.type.endsWith( '_open' ) ) {
							const tagName = token.tag;
							if ( inlineElements.includes( tagName ) ) {
								insideInlineElement = true;
								// Add animated-el class to inline elements
								if ( !token.attrGet || !token.attrGet( 'class' ) ) {
									token.attrSet( 'class', 'animated-el' );
								} else {
									const existingClass = token.attrGet( 'class' );
									token.attrSet( 'class', existingClass + ' animated-el' );
								}
							}
						}

						if ( token.type.endsWith( '_close' ) ) {
							const tagName = token.tag;
							if ( inlineElements.includes( tagName ) ) {
								insideInlineElement = false;
							}
						}
					} );
				} );
			} );
		}

		md.core.ruler.after( 'inline', 'convert_elements', function ( state ) {
			state.tokens.forEach( function ( blockToken ) {
				if ( blockToken.type !== 'inline' ) return;

				let fullContent = '';

				blockToken.children.forEach( token => {
					let { content, type } = token;

					switch ( type ) {
						case 'link_open':
							content = `<a ${ token.attrs.map( ( [ key, value ] ) => `${ key }="${ value }"` ).join( ' ' ) }>`;
							break;
						case 'link_close':
							content = '</a>';
							break;
					}

					fullContent += content;
				} );

				if (
					fullContent.includes( '<ol>' ) ||
					fullContent.includes( '<ul>' )
				) {
					const listToken = new state.Token( 'html_inline', '', 0 );
					listToken.content = fullContent.trim();
					listToken.markup = 'html';
					listToken.type = 'html_inline';

					blockToken.children = [ listToken ];
				}
			} );
		} );

		md.core.ruler.after( 'inline', 'convert_links', function ( state ) {
			state.tokens.forEach( function ( blockToken ) {
				if ( blockToken.type !== 'inline' ) return;
				blockToken.children.forEach( function ( token, idx ) {
					const { content } = token;
					if ( content.includes( '<a ' ) ) {
						const linkRegex = /(.*)(<a\s+[^>]*\s+href="([^"]+)"[^>]*>([^<]*)<\/a>?)(.*)/;
						const linkMatch = content.match( linkRegex );

						if ( linkMatch ) {
							const [ , before, , href, text, after ] = linkMatch;

							const beforeToken = new state.Token( 'text', '', 0 );
							beforeToken.content = before;

							const newToken = new state.Token( 'link_open', 'a', 1, );
							newToken.attrs = [
								[ 'href', href ],
								[ 'target', '_blank' ],
							];
							const textToken = new state.Token( 'text', '', 0 );
							textToken.content = text;
							const closingToken = new state.Token( 'link_close', 'a', -1, );

							const afterToken = new state.Token( 'text', '', 0 );
							afterToken.content = after;

							blockToken.children.splice( idx, 1, beforeToken, newToken, textToken, closingToken, afterToken, );
						}
					}
				} );
			} );
		} );
	} );

	// Add a renderer rule to handle emphasize and strong markup at the end of a string without closing markers
	// renderer.use( function ( md ) {
	// 	md.core.ruler.after( 'inline', 'fix_unclosed_markup', function ( state ) {
	// 		state.tokens.forEach( function ( blockToken ) {
	// 			if ( blockToken.type !== 'inline' ) return;

	// 			blockToken.children.forEach( ( token, idx ) => {
	// 				const { content } = token;

	// 				// Check for unclosed markup at the end of the content
	// 				if ( token.type === 'text' ) {
	// 					// Replace multiple patterns in sequence
	// 					let newContent = content;

	// 					// Remove trailing *** (three or more asterisks)
	// 					newContent = newContent.replace( /\*{3,}$/, '' );

	// 					// Update content if modified
	// 					if ( newContent !== content ) {
	// 						token.content = newContent;
	// 					}
	// 				}
	// 			} );
	// 		} );
	// 	} );
	// } );

	let renderedString = renderer.render( renderer.utils.unescapeAll( string ) );

	return renderedString;
}

const throttledRefreshFsLightbox = _.throttle( () => {
	if ( 'refreshFsLightbox' in window ) {
		refreshFsLightbox();
	}
}, 250 );

function hideTempNote() {
	const tempChatNote = document.getElementById('temp-chat-note');
	if (tempChatNote) {
		tempChatNote.style.display = 'none';
	}
}

function switchGenerateButtonsStatus( generating ) {
	const generateBtn = document.querySelector( '#send_message_button' );
	const stopBtn = document.querySelector( '#stop_button' );

	generateBtn.disabled = generating;
	generateBtn.classList.toggle( 'hidden', generating );
	generateBtn.classList.toggle( 'submitting', generating );

	stopBtn.classList.toggle( 'active', generating );
	stopBtn.disabled = !generating;
}

/**
 *
 * @param {AiResponse} responseObj
 * @param {HTMLElement} el
 */
function setAnimatingWordY( responseObj, el ) {
	if ( !el || el?.classList?.contains('done-signal') ) return;

	let { offsetTop } = el;

	if ( offsetTop <= responseObj.lastAnimatedElOffsetTop ) return;

	responseObj.lastAnimatedElOffsetTop = offsetTop;

	if ( el.nodeName === 'TR' ) {
		offsetTop = offsetTop + ( el.closest( 'table' )?.offsetTop || 0 );
	}

	responseObj.bubbleEl.style.setProperty( '--animating-word-y', `${ offsetTop }px`, );
}

/**
 *
 * @param {AiResponse} responseObj
 * @param {HTMLElement} el
 */
function onWordAnimationFinish( responseObj, el ) {
	const isDoneSignal = el.classList.contains( 'done-signal' );

	el.classList.replace( 'animating', 'animated' );

	if ( !responseObj.responseStreaming && isDoneSignal ) {
		responseObj.bubbleEl.classList.replace( 'animating-words', 'animating-words-done' );

		responseObj.animatingWordIndex = -1;

		switchGenerateButtonsStatus( aiResponses.every(res => res.responseStreaming) );

		_.defer( () => {
			unwrapWords( responseObj.chatContentEl );
		} );
	}

	setAnimatingWordY(responseObj, el);
}

/**
 * @param {AiResponse} responseObj
 */
function animateNewElements(responseObj) {
	const allAnimatableElements = responseObj.chatContentEl.querySelectorAll('.animated-el, li, hr, tr');

	allAnimatableElements.forEach((el, index) => {
		// Skip if already animated
		if (responseObj.animatedElements.has(el)) {
			return;
		}

		// Mark as animated immediately
		responseObj.animatedElements.add(el);

		// Simple staggered animation with timeout
		setTimeout(() => {
			responseObj.bubbleEl.classList.replace( 'loading', 'streaming-started' );

			el.animate([ { opacity: 1 } ], {
				duration: 500,
				easing: 'ease',
				fill: 'forwards',
			})
				.onfinish = onWordAnimationFinish(responseObj, el);
		}, index * 20);
	});
}

/**
 * Reset animation state for new responses
 * @param {AiResponse} responseObj
 */
function resetAnimationState( responseObj ) {
	responseObj.animatedElements = new Set();
	responseObj.animatingWordIndex = -1;
}

/**
 * @param {AiResponse} responseObj
*/
function onAiResponse( responseObj ) {
	const contentEl = responseObj.chatContentEl;
	const responseString = getAiResponseString( { responseObj, withoutDone: false } );
	const formattedResponse = formatString( responseString, {
		readyForAnimation: true
	} );
	const responseHTML = liquidVDOM.parseHTML( formattedResponse );

	liquidVDOM.clean( contentEl );
	liquidVDOM.diff( responseHTML, contentEl );

	responseObj.bubbleEl.classList.toggle( 'streaming-on', responseObj.responseStreaming );

	animateNewElements( responseObj );

	throttledRefreshFsLightbox();

	switchGenerateButtonsStatus( aiResponses.every(res => res.responseStreaming) );
}

/**
 *
 * @param {AiResponse} responseObj
 */
async function handleCanvasResponseStore( responseObj ) {
	try {
		const res = await fetch('/tiptap-content-store', {
			method: 'post',
			headers: {
				'Accept': 'application/json',
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({
				'message_id': responseObj.responseId,
				'content': formatString( getAiResponseString({ responseObj }) ),
				'type': 'output'
			})
		});
		const resData = await res.json();

		if (res.status === 401) {
			return;
		}

		if (!res.ok || resData.status == 'error') {
			toastr.error(resData.message || magicai_localize.could_not_save_the_canvas_data);
		}
	} catch (error) {
		console.error(error);
	}
}

function onBeforePageUnload( e ) {
	e.preventDefault();
	e.returnValue = '';
}

/**
 * @param {Event} event
 */
async function onAcceptResponseButtonClick( event ) {
	event.preventDefault();

	const button = event.currentTarget;
	const messageId = button.getAttribute('data-message-id');
	const model = button.getAttribute('data-model');
	const bubbleEl = document.querySelector(`.lqd-chat-ai-bubble[data-message-id="${messageId}"]`);
	const multiAiResposeWrap = bubbleEl.closest('.multi-model-response-wrap');

	const formData = new FormData();
	formData.append('_token', document.querySelector('input[name=_token]')?.value);
	formData.append('messageId', messageId);

	await fetch(
		'/dashboard/user/multimodel/accept-response',
		{
			method: 'POST',
			body: formData
		}
	);

	multiAiResposeWrap.parentNode.insertBefore(bubbleEl, multiAiResposeWrap);
	multiAiResposeWrap.remove();

	const chatModelChangeEvent = new CustomEvent('chat-model-change', {
		detail: { model }
	});
	document.dispatchEvent(chatModelChangeEvent);
}

/**
 * @param {Event} event
 */
async function onRegenerateResponseButtonClick( event ) {
	event.preventDefault();

	const button = event.currentTarget;
	const messageId = button.getAttribute('data-message-id');
	const model = button.getAttribute('data-model');

	console.log(event);
}

function getFrontModelEl() {
	let chatbotFrontModel = document.querySelector( '#chatbot_front_model' );

	if ( !chatbotFrontModel ) {
		chatbotFrontModel = document.createElement('select');
		chatbotFrontModel.id = 'chatbot_front_model';
		chatbotFrontModel.style.display = 'none';

		const defaultOption = document.createElement('option');
		defaultOption.value = '';
		defaultOption.textContent = magicai_localize.default_model || 'Default Model';
		chatbotFrontModel.appendChild(defaultOption);

		document.body.appendChild(chatbotFrontModel);
	}

	return chatbotFrontModel;
}

function createAiResponses() {
	const aiBubbleTemplateEl = document.querySelector( '#chat_ai_bubble' );
	const canvasEditBtnTemplate = document.querySelector('#canvas_edit_btn_block');
	const canvasModeActivated = document.querySelector('#create_canvas_button.active');
	const chatsContainer = document.querySelector( '.chats-container' );
	const chatbotFrontModel = getFrontModelEl();
	const multiModelsSelected = chatbotFrontModel.selectedOptions.length > 1;
	let appendAiBubblesTo = chatsContainer;

	if ( multiModelsSelected ) {
		const multiAiResposeWrap = document.createElement('div');

		multiAiResposeWrap.classList.add('multi-model-response-wrap', 'grid', 'grid-cols-1', 'lg:grid-cols-2', 'gap-x-6');

		chatsContainer.insertAdjacentElement('beforeend', multiAiResposeWrap );

		appendAiBubblesTo = multiAiResposeWrap;
	}

	Array.from(chatbotFrontModel.selectedOptions).forEach(option => {
		const slug = option.value;
		const label = option.innerText.replace(/\n/g, '').trim();
		const bubbleEl = aiBubbleTemplateEl.content.cloneNode(true).firstElementChild;
		const chatContentContainerEl = bubbleEl.querySelector('.chat-content-container');
		const chatContentEl = bubbleEl.querySelector('.chat-content');
		let acceptButtonEl = null;
		let regenerateButtonEl = null;

		bubbleEl.setAttribute('data-model', slug);
		bubbleEl.classList.add('loading', 'animating-words', multiModelsSelected ? 'w-full' : 'w-auto');

		if (category.slug === 'ai_chat_image') {
			bubbleEl.querySelector('.chat-content-container')?.classList?.add('flex', 'items-center');
			bubbleEl.querySelector('.lqd-typing')?.remove();
			bubbleEl.querySelector('button')?.remove();
		}

		if (multiModelsSelected) {
			const multiModelMessageHeadTemplate = document.querySelector('#multi-model-response-head');
			const multiModelMessageFootTemplate = document.querySelector('#multi-model-response-foot');

			if (multiModelMessageHeadTemplate) {
				const headEl = multiModelMessageHeadTemplate.content.cloneNode(true).firstElementChild;
				const nameEl = headEl.querySelector('.multi-model-response-name');

				nameEl.innerText = label;
				nameEl.setAttribute('title', label);

				chatContentContainerEl.insertAdjacentElement('afterbegin', headEl);
			}
			if (multiModelMessageFootTemplate) {
				const footEl = multiModelMessageFootTemplate.content.cloneNode(true).firstElementChild;
				chatContentContainerEl.insertAdjacentElement('beforeend', footEl);
			}

			acceptButtonEl = bubbleEl.querySelector('.multi-model-response-accept');
			regenerateButtonEl = bubbleEl.querySelector('.multi-model-response-regenerate');

			if (acceptButtonEl) {
				acceptButtonEl.addEventListener('click', onAcceptResponseButtonClick);
				acceptButtonEl.setAttribute('data-model', slug);
			}
			if (regenerateButtonEl) {
				regenerateButtonEl.addEventListener('click', onRegenerateResponseButtonClick);
				regenerateButtonEl.setAttribute('data-model', slug);
			}
		}

		if (canvasModeActivated && canvasEditBtnTemplate) {
			const canvasEditBtn = canvasEditBtnTemplate.content.cloneNode(true).firstElementChild;
			chatContentContainerEl.insertAdjacentElement('afterbegin', canvasEditBtn);
		}

		aiResponses.push({
			model: {
				slug,
				label
			},
			bubbleEl,
			chatContentEl,
			chatContentContainerEl,
			acceptButtonEl,
			regenerateButtonEl,
			responseStreaming: true,
			response: [],
			animatingWordIndex: -1,
			animatedElements: new Set(),
			lastAnimatedElOffsetTop: 0
		});

		appendAiBubblesTo.append(bubbleEl);
	});
}

/**
 *
 * @param {string} type
 * @param {any} images
 * @param {AiResponse} responseObj
 * @param {string | null} sharedMessageUUID
 */
function sendRequest(type, images, responseObj, sharedMessageUUID = null) {
	const formData = new FormData();
	const tempChatButton = document.querySelector( '#temp_chat_button' );
	const realtime = document.getElementById( 'realtime' );
	const chatBrandVoice = document.getElementById( 'chat_brand_voice' );
	const brandVoiceProd = document.getElementById( 'brand_voice_prod' );
	const assistant = document.getElementById( 'assistant' );
	const canvasModeActivated = document.querySelector('#create_canvas_button.active');
	const promptInput = document.getElementById( 'prompt' );
	const chat_id = $( '#chat_id' ).val();
	const throttledOnAiResponse = _.throttle( onAiResponse, 100 );
	const abortController = new AbortController();
	let receivedMessageId = false;

	formData.append( 'template_type', type );
	formData.append( 'prompt', promptInput.value );
	formData.append( 'chat_id', chat_id );
	formData.append( 'category_id', category.id );
	formData.append( 'images', images == undefined ? '' : images );
	formData.append( 'pdfname', pdfName == undefined ? '' : pdfName );
	formData.append( 'pdfpath', pdfPath == undefined ? '' : pdfPath );
	formData.append( 'realtime', realtime?.checked ? 1 : 0 );
	formData.append( 'chat_brand_voice', chatBrandVoice?.value || '' );
	formData.append( 'brand_voice_prod', brandVoiceProd?.value || '' );
	formData.append( 'chatbot_front_model',  responseObj.model.slug);
	formData.append( 'assistant', assistant.value || '' );

	if ( sharedMessageUUID ) {
		formData.append( 'shared_message_uuid', sharedMessageUUID);
	}

	if ( tempChatButton && tempChatButton.classList.contains( 'active' ) ) {
		formData.append( 'temp_chat_button', '1' );
	}

	responseObj.abortController = abortController;

	fetchEventSource( '/dashboard/user/generator/generate-stream', {
		method: 'POST',
		headers: {
			'X-CSRF-TOKEN': $( 'meta[name="csrf-token"]' ).attr( 'content' ),
		},
		body: formData,
		signal: responseObj.abortController.signal,
		onmessage: async e => {
			const txt = e.data;

			if ( !receivedMessageId ) {
				const eventData = e.event
					.split( '\n' )
					.reduce( ( acc, line ) => {
						if ( line.startsWith( 'message' ) ) {
							acc.type = 'message';
							acc.data = e.data;
						}

						return acc;
					}, {} );

				if ( eventData.type === 'message' ) {
					const responseId = eventData.data;

					receivedMessageId = true;

					responseObj.responseId = responseId;

					responseObj.bubbleEl.setAttribute('data-message-id', responseId);
					responseObj.acceptButtonEl?.setAttribute('data-message-id', responseId);
					responseObj.regenerateButtonEl?.setAttribute('data-message-id', responseId);
				}

				return;
			}

			if ( txt == null ) return;

			const responseIndex = aiResponses.findIndex(response => response.responseId === responseObj.responseId);
			const isDoneSignal = txt === '[DONE]';

			if ( isDoneSignal ) {
				messages.push( {
					role: 'assistant',
					content: getAiResponseString({ responseObj }),
				} );

				if ( messages.length >= 6 ) {
					messages.splice( 1, 2 );
				}

				// if it's done signal, add a space before
				responseObj.response.push( `${isDoneSignal ? ' ' : ''}${txt}` );
				responseObj.responseStreaming = false;
				responseObj.abortController = null;

				throttledOnAiResponse( responseObj );

				if ( canvasModeActivated ) {
					handleCanvasResponseStore( responseObj );
				}

				if ( responseIndex === aiResponses.length - 1 ) {
					window.removeEventListener( 'beforeunload', onBeforePageUnload );

					changeChatTitle( responseObj.responseId );
				}

				return;
			}

			responseObj.response.push( txt );

			throttledOnAiResponse( responseObj );
		},
		onerror: err => {
			window.removeEventListener( 'beforeunload', onBeforePageUnload );

			switchGenerateButtonsStatus( false );

			responseObj.abortController = null;

			responseObj.responseStreaming = false;
			responseObj.response.push( `${magicai_localize.error}: ${err.message}` );

			throttledOnAiResponse( responseObj );

			throw err;
		},
	} );
}

async function startGenerateRequest( ev ) {
	'use strict';

	ev?.preventDefault();

	const promptInput = document.getElementById( 'prompt' );
	const promptInputValue = promptInput.value;

	if ( !promptInputValue.trim() ) {
		return toastr.error( magicai_localize?.please_fill_message || 'Please fill the message field', );
	}

	const generateBtn = document.querySelector( '#send_message_button' );
	const chatbotFrontModel = getFrontModelEl();
	const chatsWrapper = document.querySelector( '.chats-wrap' );
	const chatsContainer = $( '.chats-container' );
	const userBubbleTemplate = document.querySelector( '#chat_user_bubble' ).content.cloneNode( true );
	const chatType = document.querySelector( '#chatType' )?.value;
	const mainUpscaleSrc = document.querySelector( '#mainupscale_src' );
	const suggestions = document.querySelector( '#sugg' );
	const chat_id = $( '#chat_id' ).val();
	const multiModelsSelected = chatbotFrontModel.selectedOptions.length > 1;
	const sharedMessageUUID = generateUUID();

	chatsWrapper.classList.remove( 'conversation-not-started' );
	chatsWrapper.classList.add( 'conversation-started' );

	Alpine.store( 'realtimeChatStatus' )?.setConversationStarted( true );

	if ( generateBtn.classList.contains( 'submitting' ) ) return;

	// Clean up any existing animations before resetting
	aiResponses.forEach(responseObj => {
		resetAnimationState( responseObj );
	});

	aiResponses = [];

	window.addEventListener( 'beforeunload', onBeforePageUnload );

	switchGenerateButtonsStatus( true );

	hideTempNote();

	userBubbleTemplate.querySelector( '.chat-content' ).innerHTML = promptInputValue;

	handlePromptHistory( promptInputValue );

	chatsContainer.append( userBubbleTemplate );

	if ( mainUpscaleSrc ) {
		mainUpscaleSrc.style.display = 'none';
	}
	if ( suggestions ) {
		suggestions.style.display = 'none';
	}

	for ( let i = 0; i < prompt_images.length; i++ ) {
		const chatImageBubbleTemplate = document.querySelector( '#chat_user_image_bubble' ).content.cloneNode( true );

		chatImageBubbleTemplate.querySelector( 'a' ).href = prompt_images[ i ];
		chatImageBubbleTemplate.querySelector( '.img-content' ).src = prompt_images[ i ];

		chatsContainer.append( chatImageBubbleTemplate );
	}

	throttledRefreshFsLightbox();

	createAiResponses();

	scrollConversationArea( { smooth: true } );

	if ( prompt_images.length == 0 ) {
		messages.push( {
			role: 'user',
			content: promptInputValue,
		} );
	} else {
		messages.push( {
			role: 'user',
			content: promptInputValue,
		} );
	}

	if ( category.slug == 'ai_chat_image' ) {
		let image_formData = new FormData();

		image_formData.append( 'prompt', promptInputValue );
		image_formData.append( 'chatHistory', JSON.stringify( messages ) );

		let response = await $.ajax( {
			url: '/dashboard/user/openai/image/generate',
			type: 'POST',
			data: image_formData,
			processData: false,
			contentType: false,
		} );

		const chatImageBubbleTemplate = document.querySelector( '#chat_bot_image_bubble' ).content.cloneNode( true );

		chatImageBubbleTemplate.querySelector( 'a' ).href = response.path;
		chatImageBubbleTemplate.querySelector( '.img-content' ).src = response.path;

		chatsContainer.append( chatImageBubbleTemplate );

		messages.push( {
			role: 'assistant',
			content: '',
		} );

		if ( messages.length >= 6 ) {
			messages.splice( 1, 2 );
		}

		saveResponseAsync( promptInputValue, '', chat_id, '', '', '', response.path );

		switchGenerateButtonsStatus( false );

		window.removeEventListener( 'beforeunload', onBeforePageUnload );

		throttledRefreshFsLightbox();

		scrollConversationArea();

		return;
	}

	pdfName = '';
	pdfPath = '';

	if ( prompt_images.length ) {
		let temp = [ ...prompt_images ];

		prompt_images = [];

		updatePromptImages();

		$.ajax( {
			type: 'POST',
			url: '/images/upload',
			data: {
				images: temp,
			},
			success: result => {
				aiResponses.forEach(responseObj => sendRequest('vision', result.path, responseObj, multiModelsSelected ? sharedMessageUUID : null));
			},
		} );

		return;
	}

	aiResponses.forEach(responseObj => sendRequest(chatType ?? 'chatbot', null, responseObj, multiModelsSelected ? sharedMessageUUID : null));

	promptInput.value = '';
	promptInput.style.height = '';
}

function reduceOnStop() {
	aiResponses.forEach(responseObj => {
		responseObj.abortController?.abort();
		responseObj.abortController = null;
		responseObj.responseStreaming = false;

		fetch('/dashboard/user/generator/reduce-tokens/chat', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').getAttribute('content')
			},
			body: JSON.stringify({
				streamed_text: getAiResponseString({ responseObj }),
				streamed_message_id: responseObj.responseId
			})
		});
	});
}

function stopGenerateRequest() {
	switchGenerateButtonsStatus( false );

	reduceOnStop();
}

function updateChatButtons() {
	const generateBtn = document.querySelector( '#send_message_button' );
	const stopBtn = document.querySelector( '#stop_button' );
	const promptInput = document.querySelector( '#prompt' );
	const acceptButtonEls = document.querySelectorAll('.multi-model-response-accept');
	const regenerateButtonEls = document.querySelectorAll('.multi-model-response-regenerate');

	generateBtn?.removeEventListener( 'click', startGenerateRequest );
	stopBtn?.removeEventListener( 'click', stopGenerateRequest );
	acceptButtonEls.forEach(el => el.removeEventListener('click', onAcceptResponseButtonClick));
	regenerateButtonEls.forEach(el => el.removeEventListener('click', onRegenerateResponseButtonClick));

	if ( promptInput ) {
		promptInput.addEventListener( 'keypress', ev => {
			if ( ev.code == 'Enter' && !ev.shiftKey ) {
				ev.preventDefault();
				$( '.lqd-chat-record-trigger' ).show();
				return startGenerateRequest();
			}
		} );
	}

	generateBtn?.addEventListener( 'click', startGenerateRequest );
	stopBtn?.addEventListener( 'click', stopGenerateRequest );
	acceptButtonEls.forEach(el => el.addEventListener('click', onAcceptResponseButtonClick));
	regenerateButtonEls.forEach(el => el.addEventListener('click', onRegenerateResponseButtonClick));
}

function updateFav( id ) {
	$.ajax( {
		type: 'post',
		url: '/dashboard/user/openai/chat/update-prompt',
		data: {
			id: id,
		},
		success: function ( data ) {
			favData = data;
			updatePrompts( promptsData );
		},
		error: function () { },
	} );
}

function updatePrompts( data ) {
	const $prompts = $( '#prompts' );

	$prompts.empty();

	if ( data.length == 0 ) {
		$( '#no_prompt' ).removeClass( 'hidden' );
	} else {
		$( '#no_prompt' ).addClass( 'hidden' );
	}

	for ( let i = 0; i < data.length; i++ ) {
		let isFav = favData.filter( item => item.item_id == data[ i ].id ).length;

		let title = data[ i ].title.toLowerCase();
		let prompt = data[ i ].prompt.toLowerCase();
		let searchStr = searchString.toLowerCase();

		if ( data[ i ].id == selectedPrompt ) {
			if ( title.includes( searchStr ) || prompt.includes( searchStr ) ) {
				if ( ( filterType == 'fav' && isFav != 0 ) || filterType != 'fav' ) {
					let prompt = document.querySelector( '#selected_prompt' ).content.cloneNode( true );
					const favbtn = prompt.querySelector( '.favbtn' );
					prompt.querySelector( '.prompt_title' ).innerHTML = data[ i ].title;
					prompt.querySelector( '.prompt_text' ).innerHTML = data[ i ].prompt;
					favbtn.setAttribute( 'id', data[ i ].id );

					if ( isFav != 0 ) {
						favbtn.classList.add( 'active' );
					} else {
						favbtn.classList.remove( 'active' );
					}

					$prompts.append( prompt );
				} else {
					selectedPrompt = -1;
				}
			} else {
				selectedPrompt = -1;
			}
		} else {
			if ( title.includes( searchStr ) || prompt.includes( searchStr ) ) {
				if (
					( filterType == 'fav' && isFav != 0 ) ||
					filterType != 'fav'
				) {
					let prompt = document
						.querySelector( '#unselected_prompt' )
						.content.cloneNode( true );
					const favbtn = prompt.querySelector( '.favbtn' );
					prompt.querySelector( '.prompt_title' ).innerHTML =
						data[ i ].title;
					prompt.querySelector( '.prompt_text' ).innerHTML =
						data[ i ].prompt;
					favbtn.setAttribute( 'id', data[ i ].id );

					if ( isFav != 0 ) {
						favbtn.classList.add( 'active' );
					} else {
						favbtn.classList.remove( 'active' );
					}

					$prompts.append( prompt );
				}
			}
		}
	}
	let favCnt = favData.length;
	let perCnt = data.length;

	if ( favCnt == 0 ) {
		$( '#fav_count' )[ 0 ].innerHTML = '';
	} else {
		$( '#fav_count' )[ 0 ].innerHTML = favCnt;
	}

	if ( perCnt == 0 || perCnt == undefined ) {
		$( '#per_count' )[ 0 ].innerHTML = '';
	} else {
		$( '#per_count' )[ 0 ].innerHTML = perCnt;
	}
}

function searchStringChange( e ) {
	searchString = $( '#search_str' ).val();
	updatePrompts( promptsData );
}

function openNewImageDlg( e ) {
	$( '#selectImageInput' ).click();
}

function updatePromptImages() {
	$( '#chat_images' ).empty();
	if ( prompt_images.length == 0 ) {
		$( '#chat_images' ).removeClass( 'active' );
		$( '.split_line' ).addClass( 'hidden' );
		return;
	}
	$( '#chat_images' ).addClass( 'active' );
	$( '.split_line' ).removeClass( 'hidden' );
	for ( let i = 0; i < prompt_images.length; i++ ) {
		let new_image = document
			.querySelector( '#prompt_image' )
			.content.cloneNode( true );
		$( new_image.querySelector( 'img' ) ).attr( 'src', prompt_images[ i ] );
		$( new_image.querySelector( '.prompt_image_close' ) ).attr( 'index', i );
		$( document.querySelector( '#chat_images' ) ).append( new_image );
	}
	let new_image_btn = document
		.querySelector( '#prompt_image_add_btn' )
		.content.cloneNode( true );
	document.querySelector( '#chat_images' ).append( new_image_btn );
	$( '.promt_image_btn' ).on( 'click', function ( e ) {
		e.preventDefault();
		$( '#chat_add_image' ).click();
	} );
	$( '.prompt_image_close' ).on( 'click', function () {
		prompt_images.splice( $( this ).attr( 'index' ), 1 );
		updatePromptImages();
	} );
}

function addImagetoChat( data ) {
	if ( prompt_images.filter( item => item == data ).length == 0 ) {
		prompt_images.push( data );
		updatePromptImages();
	}
}

function initChat() {
	var mediaRecorder;
	var chunks = [];
	var stream_;

	prompt_images = [];

	$( '#scrollable_content' ).animate( { scrollTop: 1000 }, 200 );

	// Start recording when the button is pressed
	$( '#voice_record_button' ).click( function () {
		chunks = [];
		navigator.mediaDevices
			.getUserMedia( { audio: true } )
			.then( function ( stream ) {
				stream_ = stream;
				mediaRecorder = new MediaRecorder( stream );
				$( '#voice_record_button' ).addClass( 'inactive' );
				$( '#voice_record_stop_button' ).addClass( 'active' );
				mediaRecorder.ondataavailable = function ( e ) {
					chunks.push( e.data );
				};
				mediaRecorder.start();
			} )
			.catch( function ( err ) {
				console.log( 'The following error occurred: ' + err );
				toastr.warning( 'Audio is not allowed' );
			} );

		$( '#voice_record_stop_button' ).click( function ( e ) {
			e.preventDefault();
			$( '#voice_record_button' ).removeClass( 'inactive' );
			$( '#voice_record_stop_button' ).removeClass( 'active' );
			mediaRecorder.onstop = function () {
				var blob = new Blob( chunks, { type: 'audio/mp3' } );

				var formData = new FormData();
				var fileOfBlob = new File( [ blob ], 'audio.mp3' );
				formData.append( 'file', fileOfBlob );

				chunks = [];

				$.ajax( {
					url: '/dashboard/user/openai/chat/transaudio',
					type: 'POST',
					data: formData,
					contentType: false,
					processData: false,
					success: function ( response ) {
						if ( response.length >= 4 ) {
							$( '#prompt' ).val( response );
						}
					},
					error: function () {
						// Handle the error response
					},
				} );
			};
			mediaRecorder.stop();
			stream_
				.getTracks() // get all tracks from the MediaStream
				.forEach( track => track.stop() ); // stop each of them
		} );
	} );

	$( '#btn_add_new_prompt' ).on( 'click', function ( e ) {
		prompt_title = $( '#new_prompt_title' ).val();
		prompt = $( '#new_prompt' ).val();

		if ( prompt_title.trim() == '' ) {
			toastr.warning( 'Please input title' );
			return;
		}

		if ( prompt.trim() == '' ) {
			toastr.warning( 'Please input prompt' );
			return;
		}

		$.ajax( {
			type: 'post',
			url: '/dashboard/user/openai/chat/add-prompt',
			data: {
				title: prompt_title,
				prompt: prompt,
			},
			success: function ( data ) {
				promptsData = data;
				updatePrompts( data );
				$( '.custom__popover__back' ).addClass( 'hidden' );
				$( '#custom__popover' ).removeClass( 'custom__popover__wrapper' );
			},
			error: function () { },
		} );
	} );

	$( '#add_btn' ).on( 'click', function ( e ) {
		$( '#custom__popover' ).addClass( 'custom__popover__wrapper' );
		$( '.custom__popover__back' ).removeClass( 'hidden' );
		e.stopPropagation();
	} );

	$( '.custom__popover__back' ).on( 'click', function () {
		$( this ).addClass( 'hidden' );
		$( '#custom__popover' ).removeClass( 'custom__popover__wrapper' );
	} );

	$( '#prompt_library' ).on( 'click', function ( e ) {
		e.preventDefault();

		$( '#prompts' ).empty();

		$.ajax( {
			type: 'post',
			url: '/dashboard/user/openai/chat/prompts',
			success: function ( data ) {
				filterType = 'all';
				promptsData = data.promptData;
				favData = data.favData;
				updatePrompts( data.promptData );
				$( '#modal' ).addClass( 'lqd-is-active' );
				$( '.modal__back' ).removeClass( 'hidden' );
			},
			error: function () { },
		} );
		e.stopPropagation();
	} );

	$( '.modal__back' ).on( 'click', function () {
		$( this ).addClass( 'hidden' );
		$( '#modal' ).removeClass( 'lqd-is-active' );
	} );

	$( document ).on( 'click', '.prompt', function () {
		const $promptInput = $( '#prompt' );
		selectedPrompt = Number( $( this.querySelector( '.favbtn' ) ).attr( 'id' ) );
		$promptInput.val(
			promptsData.filter( item => item.id == selectedPrompt )[ 0 ].prompt,
		);
		$( '.modal__back' ).addClass( 'hidden' );
		$( '#modal' ).removeClass( 'lqd-is-active' );
		selectedPrompt = -1;
		$promptInput.css( 'height', '5px' );
		$promptInput.css( 'height', $promptInput[ 0 ].scrollHeight + 'px' );
	} );

	$( document ).on( 'click', '.filter_btn', function () {
		$( '.filter_btn' ).removeClass( 'active' );
		$( this ).addClass( 'active' );
		filterType = $( this ).attr( 'filter' );
		updatePrompts( promptsData );
	} );

	$( document ).on( 'click', '.favbtn', function ( e ) {
		updateFav( Number( $( this ).attr( 'id' ) ) );
		e.stopPropagation();
	} );

	$( '#chat_add_image' ).click( function () {
		$( '#selectImageInput' ).click();
	} );

	$( '#selectImageInput' ).change( function () {
		if ( this.files && this.files[ 0 ] ) {
			for ( let i = 0; i < this.files.length; i++ ) {
				let reader = new FileReader();
				// Existing image handling code
				reader.onload = function ( e ) {
					var img = new Image();
					img.src = e.target.result;
					img.onload = function () {
						var canvas = document.createElement( 'canvas' );
						var ctx = canvas.getContext( '2d' );
						canvas.height = ( img.height * 200 ) / img.width;
						canvas.width = 200;
						ctx.drawImage( img, 0, 0, canvas.width, canvas.height );
						var base64 = canvas.toDataURL( 'image/png' );
						addImagetoChat( base64 );
					};
				};
				reader.readAsDataURL( this.files[ i ] );
			}
		}
		document.getElementById( 'mainupscale_src' ) &&
			( document.getElementById( 'mainupscale_src' ).style.display = 'none' );
	} );

	$( '#upscale_src' ).change( function () {
		if ( this.files && this.files[ 0 ] ) {
			for ( let i = 0; i < this.files.length; i++ ) {
				let reader = new FileReader();

				reader.onload = function ( e ) {
					addImagetoChat( e.target.result );
					// $( "#selectImageInput" ).val( '' );
				};
				reader.readAsDataURL( this.files[ i ] );
			}
		}
		document.getElementById( 'mainupscale_src' ) &&
			( document.getElementById( 'mainupscale_src' ).style.display = 'none' );
	} );

	document
		.querySelectorAll( '.lqd-chat-ai-bubble .chat-content' )
		.forEach( el => {
			el.classList.remove( '!whitespace-pre-wrap', 'whitespace-pre-wrap' );
			el.style.whiteSpace = 'normal';

			if ( el.classList.contains('is-html') ) {
				const turndownService = new TurndownService();
				const markdown = turndownService.turndown(el);

				el.innerHTML = markdown;
			}

			el.innerHTML = formatString( el.innerHTML );
			throttledRefreshFsLightbox();
		} );
}

async function saveResponseAsync( input, response, chat_id, imagePath, pdfName, pdfPath, outputImage = '', model = '', ) {
	var formData = new FormData();

	if ( !response ) {
		response = '';
	}

	formData.append( 'chat_id', chat_id );
	formData.append( 'input', input );
	formData.append( 'response', response );
	formData.append( 'images', imagePath );
	formData.append( 'pdfName', pdfName );
	formData.append( 'pdfPath', pdfPath );
	formData.append( 'outputImage', outputImage );
	formData.append( 'model', model );

	try {
		const result = await jQuery.ajax( {
			url: '/dashboard/user/openai/chat/low/chat_save',
			type: 'POST',
			headers: {
				'X-CSRF-TOKEN': '{{ csrf_token() }}',
			},
			data: formData,
			contentType: false,
			processData: false,
		} );
		if ( result.status === 'error' ) {
			toastr.error( result.message, 'Error' );
		}

		return result;
	} catch ( error ) {
		if ( error.responseJSON && error.responseJSON.message ) {
			toastr.error( error.responseJSON.message, 'Error' );
		} else {
			toastr.error( 'An unexpected error occurred. Please try again.', 'Error' );
		}
	}
	return false;
}

/*

DO NOT FORGET TO ADD THE CHANGES TO BOTH FUNCTION makeDocumentReadyAgain and the document ready function on the top!!!!

*/
function makeDocumentReadyAgain() {
	const chatsWrapper = document.querySelector( '.chats-wrap' );
	const chatBubbles = chatsWrapper?.querySelectorAll( '.lqd-chat-ai-bubble, .lqd-chat-user-bubble' );

	_.defer( () => {
		setChatsCssVars();
		updateChatButtons();
	} );

	$( document ).ready( function () {
		'use strict';

		const chat_id = $( '#chat_id' ).val();
		$( `#chat_${ chat_id }` )
			.addClass( 'active' )
			.siblings()
			.removeClass( 'active' );

		scrollConversationArea();

		handlePromptHistoryNavigate();
	} );

	if ( chatBubbles ) {
		chatsWrapper.classList.toggle( 'conversation-not-started', chatBubbles.length <= 1 );
		chatsWrapper.classList.toggle( 'conversation-started', chatBubbles.length > 1 );
	}
}

function handlePromptHistory( prompt ) {
	const promptHistory = localStorage.getItem( 'promptHistory' );

	if ( !promptHistory ) {
		return localStorage.setItem( 'promptHistory', JSON.stringify( [ prompt ] ) );
	}

	const promptHistoryArray = JSON.parse( promptHistory );

	if ( promptHistoryArray.includes( prompt ) ) {
		promptHistoryArray.splice( promptHistoryArray.indexOf( prompt ), 1 );
	}

	promptHistoryArray.push( prompt );

	localStorage.setItem( 'promptHistory', JSON.stringify( promptHistoryArray ) );
}

function removePromptHistoryHandler() {
	const promptInput = document.querySelector( '.lqd-chat-form #prompt' );

	promptInput?.removeEventListener( 'keydown', onPromptInputKeyUpDown );
}

function handlePromptHistoryNavigate() {
	const promptInput = document.querySelector( '.lqd-chat-form #prompt' );

	if ( !promptInput ) return;

	promptInput.addEventListener( 'keydown', onPromptInputKeyUpDown );
}

function onPromptInputKeyUpDown( e ) {
	const promptInput = e.target;
	const promptHistory = localStorage.getItem( 'promptHistory' ) || '[]';
	const promptHistoryArray = JSON.parse( promptHistory );

	if ( promptHistoryArray.length === 0 ) return;

	if ( promptInput.value !== '' && !navigatingInChatsHistory ) {
		return;
	}

	const arrowsPressed = e.key === 'ArrowUp' || e.key === 'ArrowDown';

	if ( e.key === 'ArrowUp' ) {
		navigatingInChatsHistory = true;

		if ( selectedHistoryPrompt === -1 ) {
			selectedHistoryPrompt = promptHistoryArray.length - 1;
		} else {
			selectedHistoryPrompt = Math.max( 0, selectedHistoryPrompt - 1 );
		}

		promptInput.value = promptHistoryArray[ selectedHistoryPrompt ];
	}

	if ( e.key === 'ArrowDown' ) {
		navigatingInChatsHistory = true;

		if ( selectedHistoryPrompt === -1 ) {
			selectedHistoryPrompt = 0;
		} else {
			selectedHistoryPrompt = Math.min(
				promptHistoryArray.length - 1,
				selectedHistoryPrompt + 1,
			);
		}

		promptInput.value = promptHistoryArray[ selectedHistoryPrompt ];
	}

	if ( !arrowsPressed ) {
		navigatingInChatsHistory = false;
		selectedHistoryPrompt = -1;
	}
}

handlePromptHistoryNavigate();

function escapeHtml( html ) {
	var text = document.createTextNode( html );
	var div = document.createElement( 'div' );
	div.appendChild( text );
	return div.innerHTML;
}

function openChatAreaContainer( chat_id, website_url = null ) {

	chatid = chat_id;
	$( `#chat_${ chat_id }` ).addClass( 'active' ).siblings().removeClass( 'active' );

	var formData = new FormData();

	formData.append( 'chat_id', chat_id );

	if ( website_url != null && website_url != '' ) {
		formData.append( 'website_url', website_url );
	}

	let openChatAreaContainerUrl = $( '#openChatAreaContainerUrl' ).val();

	$.ajax( {
		type: 'post',
		url: openChatAreaContainerUrl,
		data: formData,
		contentType: false,
		processData: false,
		success: function ( data ) {
			removePromptHistoryHandler();

			$( '#load_chat_area_container > .lqd-card-body' ).html( data.html );

			initChat();

			messages = [
				{
					role: 'assistant',
					content: prompt_prefix,
				},
			];

			data.lastThreeMessage.forEach( message => {
				messages.push( {
					role: 'user',
					content: message.input,
				} );
				messages.push( {
					role: 'assistant',
					content: message.output,
				} );
			} );

			makeDocumentReadyAgain();
			if ( data.lastThreeMessage != '' ) {
				document.getElementById( 'mainupscale_src' ) &&
					( document.getElementById( 'mainupscale_src' ).style.display =
						'none' );
				document.getElementById( 'sugg' ) &&
					( document.getElementById( 'sugg' ).style.display = 'none' );
			}
			setTimeout( function () {
				scrollConversationArea();
			}, 750 );
		},
		error: function ( data ) {
			var err = data.responseJSON.errors;
			if ( err ) {
				$.each( err, function ( index, value ) {
					toastr.error( value );
				} );
			} else {
				toastr.error( data.responseJSON.message );
			}
		},
	} );

	return false;
}

function startNewChat( category_id, local, website_url = null ) {
	const formData = new FormData();
	const chatsWrapper = document.querySelector( '.chats-wrap' );
	formData.append( 'category_id', category_id );

	// let website_url = $("#website_url")?.val();
	let createChatUrl = $( '#createChatUrl' )?.val();

	if ( website_url != null && website_url != '' ) {
		formData.append( 'website_url', website_url );
	}

	let link = '/dashboard/user/openai/chat/start-new-chat';

	if ( createChatUrl ) {
		link = createChatUrl;
	}

	return $.ajax( {
		type: 'post',
		url: link,
		data: formData,
		contentType: false,
		processData: false,
		success: function ( data ) {
			removePromptHistoryHandler();

			chatid = data.chat.id;

			chatsWrapper.classList.remove( 'conversation-started' );
			chatsWrapper.classList.add( 'conversation-not-started' );

			$( '#load_chat_area_container > .lqd-card-body' ).html( data.html );
			$( '#chat_sidebar_container' ).html( data.html2 );

			initChat();

			messages = [
				{
					role: 'assistant',
					content: prompt_prefix,
				},
			];

			makeDocumentReadyAgain();

			setTimeout( function () {
				scrollConversationArea();
			}, 750 );
		},
		error: function ( data ) {
			var err = data.responseJSON.errors;
			if ( err ) {
				$.each( err, function ( index, value ) {
					toastr.error( value );
				} );
			} else {
				toastr.error( data.responseJSON.message );
			}
		},
	} );
}

function deleteAllConv( category_id ) {
	if ( confirm( 'Are you sure you want to remove all chats?' ) ) {
		if ( category_id == 0 ) {
			toastr.error( 'Please select a category' );
			return false;
		}

		var formData = new FormData();
		formData.append( 'category_id', category_id );
		let link = '/dashboard/user/openai/chat/clear-chats';
		$.ajax( {
			type: 'post',
			url: link,
			data: formData,
			contentType: false,
			processData: false,
			success: function ( data ) {
				// refresh page
				location.reload();
			},
			error: function ( data ) {
				var err = data.responseJSON.errors;
				if ( err ) {
					$.each( err, function ( index, value ) {
						toastr.error( value );
					} );
				} else {
					toastr.error( data.responseJSON.message );
				}
			},
		} );
		return false;
	}
}

function startNewDocChat( file, type ) {
	'use strict';

	let category_id = $( '#chat_search_word' ).data( 'category-id' );

	var formData = new FormData();
	formData.append( 'category_id', category_id );
	formData.append( 'doc', pdf );
	formData.append( 'type', type );

	Alpine.store( 'appLoadingIndicator' ).show();
	$( '.lqd-upload-doc-trigger' ).attr( 'disabled', true );

	$.ajax( {
		type: 'post',
		url: '/dashboard/user/openai/chat/start-new-doc-chat',
		data: formData,
		contentType: false,
		processData: false,
		success: function ( data ) {
			removePromptHistoryHandler();
			Alpine.store( 'appLoadingIndicator' ).hide();
			$( '.lqd-upload-doc-trigger' ).attr( 'disabled', false );
			$( '#selectDocInput' ).val( '' );
			chatid = data.chat.id;
			$( '#load_chat_area_container > .lqd-card-body' ).html( data.html );
			$( '#chat_sidebar_container' ).html( data.html2 );

			initChat();
			messages = [
				{
					role: 'assistant',
					content: prompt_prefix,
				},
			];
			makeDocumentReadyAgain();
			setTimeout( function () {
				$( '.conversation-area' )
					.stop()
					.animate(
						{ scrollTop: $( '.conversation-area' ).outerHeight() },
						200,
					);
			}, 750 );

			toastr.success( magicai_localize.analyze_file_finish );
		},
		error: function ( data ) {
			Alpine.store( 'appLoadingIndicator' ).hide();
			$( '.lqd-upload-doc-trigger' ).attr( 'disabled', false );
			$( '#selectDocInput' ).val( '' );
			var err = data.responseJSON.errors;
			if ( err ) {
				$.each( err, function ( index, value ) {
					toastr.error( value );
				} );
			} else {
				toastr.error( data.responseJSON.message );
			}
		},
	} );
	return false;
}

function searchChatFunction() {
	'use strict';

	const categoryId = $( '#chat_search_word' ).data( 'category-id' );
	const formData = new FormData();
	formData.append(
		'_token',
		document.querySelector( 'input[name=_token]' )?.value,
	);
	formData.append(
		'search_word',
		document.getElementById( 'chat_search_word' ).value,
	);
	formData.append( 'category_id', categoryId );

	$.ajax( {
		type: 'POST',
		url: '/dashboard/user/openai/chat/search',
		data: formData,
		contentType: false,
		processData: false,
		success: function ( result ) {
			$( '#chat_sidebar_container' ).html( result.html );
			$( document ).trigger( 'ready' );
		},
	} );
}

/**
 * @param {object} opts
 * @param {'end' | number} opts.y
 * @param {boolean} opts.smooth
 */
function scrollConversationArea( opts = {} ) {
	const options = {
		y: 'end',
		smooth: false,
		...opts,
	};
	const el = document.querySelector( '.conversation-area' );

	if ( !el ) return;

	const y = options.y === 'end' ? el.scrollHeight + 200 : options.y;

	el.scrollTo( {
		top: Math.round( y ),
		left: 0,
		behavior: options.smooth ? 'smooth' : 'auto',
	} );
}

function saveResponse( input, response, chat_id, imagePath = '', pdfName = '', pdfPath = '', outputImage = '' ) {
	var formData = new FormData();
	formData.append( 'chat_id', chat_id );
	formData.append( 'input', input );
	formData.append( 'response', response );
	formData.append( 'images', imagePath );
	formData.append( 'pdfName', pdfName );
	formData.append( 'pdfPath', pdfPath );
	formData.append( 'outputImage', outputImage );
	jQuery.ajax( {
		url: '/dashboard/user/openai/chat/low/chat_save',
		type: 'POST',
		headers: {
			'X-CSRF-TOKEN': '{{ csrf_token() }}',
		},
		data: formData,
		contentType: false,
		processData: false,
	} );
	return false;
}

function addText( text ) {
	var promptElement = document.getElementById( 'prompt' );
	var currentText = promptElement.value;
	var newText = currentText + text;
	promptElement.value = newText;
}

function dropHandler( ev, id ) {
	// Prevent default behavior (Prevent file from being opened)

	ev.preventDefault();
	const input = document.querySelector( `#${ id }` );
	const fileNameEl =
		input?.previousElementSibling?.querySelector( '.file-name' );

	if ( !input ) return;

	input.files = ev.dataTransfer.files;

	if ( fileNameEl ) {
		fileNameEl.innerText = ev.dataTransfer.files[ 0 ].name;
	}

	for ( let i = 0; i < ev.dataTransfer.files.length; i++ ) {
		let reader = new FileReader();
		// Existing image handling code
		reader.onload = function ( e ) {
			var img = new Image();
			img.src = e.target.result;
			img.onload = function () {
				var canvas = document.createElement( 'canvas' );
				var ctx = canvas.getContext( '2d' );
				canvas.height = ( img.height * 200 ) / img.width;
				canvas.width = 200;
				ctx.drawImage( img, 0, 0, canvas.width, canvas.height );
				var base64 = canvas.toDataURL( 'image/png' );
				addImagetoChat( base64 );
			};
		};
		reader.readAsDataURL( input.files[ i ] );
	}
	document.getElementById( 'mainupscale_src' ).style.display = 'none';
}

function dragOverHandler( ev ) {
	// Prevent default behavior (Prevent file from being opened)
	ev.preventDefault();
}

function handleFileSelect( id ) {
	$( '#' + id )
		.prev()
		.find( '.file-name' )
		.text( $( '#' + id )[ 0 ].files[ 0 ].name );
}

function exportAsPdf() {
	var win = window.open(
		`/dashboard/user/openai/chat/generate-pdf?id=${ chatid }`,
		'_blank',
	);
	win.focus();
}

function exportAsWord() {
	var win = window.open(
		`/dashboard/user/openai/chat/generate-word?id=${ chatid }`,
		'_blank',
	);
	win.focus();
}

function exportAsTxt() {
	var win = window.open(
		`/dashboard/user/openai/chat/generate-txt?id=${ chatid }`,
		'_blank',
	);
	win.focus();
}

$( document ).ready( function () {
	'use strict';

	initChat();

	scrollConversationArea();

	_.defer( updateChatButtons );

	function saveChatNewTitle( chatId, newTitle ) {
		var formData = new FormData();
		formData.append( 'chat_id', chatId );
		formData.append( 'title', newTitle );

		$.ajax( {
			type: 'post',
			url: '/dashboard/user/openai/chat/rename-chat',
			data: formData,
			contentType: false,
			processData: false,
		} );
		return false;
	}

	function deleteChatItem( chatId, chatTitle ) {
		if ( confirm( `Are you sure you want to remove ${ chatTitle }?` ) ) {
			var formData = new FormData();
			formData.append( 'chat_id', chatId );

			const chatTrigger = $( `#${ chatId }` );
			const chatIsActive = chatTrigger.hasClass( 'active' );
			let nextChatToActivate = chatTrigger.prevAll( ':visible' ).first();
			const chatsContainer = document.querySelector( '.chats-container' );

			if ( nextChatToActivate.length === 0 ) {
				nextChatToActivate = chatTrigger.nextAll( ':visible' ).first();
			}

			$.ajax( {
				type: 'post',
				url: '/dashboard/user/openai/chat/delete-chat',
				data: formData,
				contentType: false,
				processData: false,
				success: function ( data ) {
					//Remove chat li
					chatTrigger.hide();
					if ( chatIsActive ) {
						if ( chatsContainer ) {
							chatsContainer.innerHTML = '';
						}
						nextChatToActivate
							.children( '.chat-list-item-trigger' )
							.click();
					}
					toastr.success(
						magicai_localize.conversation_deleted_successfully,
					);
				},
				error: function ( data ) {
					var err = data.responseJSON.errors;
					if ( err ) {
						$.each( err, function ( index, value ) {
							toastr.error( value );
						} );
					} else {
						toastr.error( data.responseJSON.message );
					}
				},
			} );
			return false;
		}
	}

	$( '#chat_sidebar_container' ).on( 'click', '.chat-item-delete', ev => {
		const button = ev.currentTarget;
		const parent = button.closest( 'li' );
		const chatId = parent.getAttribute( 'id' );
		const chatTitle = parent.querySelector( '.chat-item-title' ).innerText;
		deleteChatItem( chatId, chatTitle );
	} );

	$( '#chat_sidebar_container' ).on( 'click', '.chat-item-update-title', ev => {
		const button = ev.currentTarget;
		const parent = button.closest( '.chat-list-item' );
		const title = parent.querySelector( '.chat-item-title' );
		const chatId = parent.getAttribute( 'id' );
		const currentText = title.innerText;

		function setEditMode( mode ) {
			if ( mode === 'editStart' ) {
				parent.classList.add( 'edit-mode' );

				title.setAttribute( 'data-current-text', currentText );
				title.setAttribute( 'contentEditable', true );
				title.focus();
				window.getSelection().selectAllChildren( title );
			} else if ( mode === 'editEnd' ) {
				parent.classList.remove( 'edit-mode' );

				title.removeAttribute( 'contentEditable' );
				title.removeAttribute( 'data-current-text' );
			}
		}

		function keydownHandler( ev ) {
			const { key } = ev;
			const escapePressed = key === 'Escape';
			const enterPressed = key === 'Enter';

			if ( !escapePressed && !enterPressed ) return;

			ev.preventDefault();

			if ( escapePressed ) {
				title.innerText = currentText;
			}

			if ( enterPressed ) {
				saveChatNewTitle( chatId, title.innerText );
			}

			setEditMode( 'editEnd' );
			document.removeEventListener( 'keydown', keydownHandler );
		}

		// if alreay editting then turn the edit button to a save button
		if ( title.hasAttribute( 'contentEditable' ) ) {
			setEditMode( 'editEnd' );
			document.removeEventListener( 'keydown', keydownHandler );
			return saveChatNewTitle( chatId, title.innerText );
		}

		$( '.chat-list-ul .edit-mode' ).each( ( i, el ) => {
			const title = el.querySelector( '.chat-item-title' );
			title.innerText = title.getAttribute( 'data-current-text' );
			title.removeAttribute( 'data-current-text' );
			title.removeAttribute( 'contentEditable' );
			el.classList.remove( 'edit-mode' );
		} );

		setEditMode( 'editStart' );

		document.addEventListener( 'keydown', keydownHandler );
	});

	$( '#chat_sidebar_container' ).on( 'click', '.chat-item-pin', ev => {
		const button = ev.currentTarget;
		const parent = button.closest( '.chat-list-item' );
		const chatId = parent.getAttribute( 'id' );
		const isPinned = parent.classList.contains( 'pin-mode' );

		function togglePinMode() {
			parent.classList.toggle( 'pin-mode' );
			const pinIcon = button.querySelector( '.tabler-pin' );
			const pinnedIcon = button.querySelector( '.tabler-pinned' );
			if ( pinIcon && pinnedIcon ) {
				pinIcon.classList.toggle( 'hidden' );
				pinnedIcon.classList.toggle( 'hidden' );
			}
		}
		togglePinMode();

		$.ajax( {
			type: 'post',
			url: '/dashboard/user/openai/chat/pin-conversation',
			data: JSON.stringify( { pinned: !isPinned, chat_id: chatId } ),
			contentType: 'application/json',
			success: function ( data ) {
				toastr.success( isPinned ? magicai_localize.conversation_unpinned : magicai_localize.conversation_pinned );
			},
			error: ( xhr, status, error ) => {
				console.error( 'Error updating pin status:', error );
				togglePinMode();
				toastr.error( magicai_localize.conversation_pin_error );
			},
		} );
	});

	$( '#chat_search_word' ).on( 'keyup', function () {
		return searchChatFunction();
	} );

	$( 'body' ).on( 'input', '#prompt', ev => {
		const el = ev.target;
		el.style.height = '5px';
		el.style.height = el.scrollHeight + 'px';
		const recordTrigger = $( '.lqd-chat-record-trigger' );
		const chatsWrapper = $( '.chats-wrap' );

		// check if value is not empty and then hide .lqd-chat-record-trigger and .lqd-chat-record-stop-trigger elements
		if (
			el.value &&
			el.value !== '' &&
			!( Array.isArray( el.value ) && el.value.length === 0 ) &&
			!(
				typeof el.value === 'object' &&
				Object.keys( el.value ).length === 0
			)
		) {
			recordTrigger.hide();
			chatsWrapper.addClass( 'prompt-filled' );
		} else {
			recordTrigger.show();
			chatsWrapper.removeClass( 'prompt-filled' );
		}
	} );

	$( '#selectDocInput' ).change( function () {
		if ( this.files && this.files[ 0 ] ) {
			let reader = new FileReader();
			pdf = this.files[ 0 ];

			toastr.success( magicai_localize.analyze_file_begin );

			startNewDocChat( pdf, this.files[ 0 ].type );

			document.getElementById( 'mainupscale_src' ) &&
				( document.getElementById( 'mainupscale_src' ).style.display =
					'none' );
		}
	} );

	window.addEventListener( 'beforeunload', function ( e ) {
		reduceOnStop();
	} );
} );

$( 'body' ).on( 'click', '.chat-download', event => {
	const button = event.currentTarget;
	const docType = button.dataset.docType;
	const docName = button.dataset.docName || 'document';

	const container = document.querySelector( '.chats-container' );
	let content = container?.parentElement?.innerHTML;
	let html;

	if ( !content ) return;

	if ( docType === 'pdf' ) {
		return html2pdf()
			.set( {
				filename: docName,
			} )
			.from( content )
			.toPdf()
			.save();
	}

	if ( docType === 'txt' ) {
		html = container.innerText;
	} else {
		html = `
	<html ${ this.doctype === 'doc'
		? 'xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word" xmlns="http://www.w3.org/TR/REC-html40"'
		: ''
}>
	<head>
		<meta charset="utf-8" />
		<title>${ docName }</title>
	</head>
	<body>
		${ content }
	</body>
	</html>`;
	}

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
} );

function changeChatTitle( responseId ) {
	const $lqdChatUserBubblesLength = document.querySelectorAll( '.lqd-chat-user-bubble' ).length;

	if ( $lqdChatUserBubblesLength != 1 ) return;

	$.ajax( {
		type: 'post',
		url: '/dashboard/change-chat-title',
		data: {
			streamed_message_id: responseId,
		},
		success: function ( data ) {
			if ( data.changed ) {
				const chatTitleEl = document.querySelector(
					`#chat_${ data.chat_id } .chat-item-title`,
				);

				if ( !chatTitleEl ) return;

				const newTitle = data.new_title.replaceAll( ' ', '\u00a0' );
				const newTitleStringArray = newTitle.split( '' );

				chatTitleEl.innerText = '';

				const interval = setInterval( () => {
					chatTitleEl.innerText += newTitleStringArray.shift();

					if ( !newTitleStringArray.length ) {
						clearInterval( interval );
					}
				}, 30 );
			}
		},
	} );
}

function setChatsCssVars() {
	const chatsWrapper = document.querySelector( '.chats-wrap' );
	const chatsContainer = document.querySelector( '.chats-container' );
	const chatsHead = document.querySelector( '.lqd-chat-head' );
	const chatsForm = document.querySelector( '.lqd-chat-form' );
	const conversationArea = document.querySelector( '.conversation-area' );

	if (
		chatsWrapper &&
		chatsContainer &&
		chatsHead &&
		chatsForm &&
		conversationArea
	) {
		chatsWrapper.style.setProperty(
			'--chats-container-height',
			`${ conversationArea.offsetHeight - chatsHead.offsetHeight - chatsForm.offsetHeight }px`,
		);
	}
}

( () => {
	setChatsCssVars();

	window.addEventListener( 'resize', _.debounce( setChatsCssVars, 150 ) );
} )();

